# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import json
import logging
import os
import sys
import time
import tifffile

from tqdm import tqdm
from pathlib import Path

import torch
from monai.losses import PatchAdversarialLoss, PerceptualLoss
from monai.networks.nets import PatchDiscriminator
from monai.config import print_config
from monai.utils import set_determinism
from torch.nn import L1Loss, MSELoss
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from utils_tif import KL_loss, define_instance, prepare_tif_dataloader_vae, setup_ddp, normalize_batch_for_display

import numpy as np
from torchvision.utils import make_grid


def main():
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Training")
    parser.add_argument(
        "-e",
        "--environment-file",
        default="./config/environment_tif.json",
        help="environment json file that stores environment path",
    )
    parser.add_argument(
        "-c",
        "--config-file",
        default="./config/config_train_16g_cond.json",
        help="config json file that stores hyper-parameters",
    )
    parser.add_argument("-g", "--gpus", default=1, type=int, help="number of gpus per node")
    args = parser.parse_args()

    # Step 0: configuration
    ddp_bool = args.gpus > 1  # whether to use distributed data parallel
    if ddp_bool:
        rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        dist, device = setup_ddp(rank, world_size)
    else:
        rank = 0
        world_size = 1
        device = 0

    torch.cuda.set_device(device)
    print(f"Using {device}")

    print_config()
    torch.backends.cudnn.benchmark = True
    torch.set_num_threads(4)
    torch.autograd.set_detect_anomaly(True)

    env_dict = json.load(open(args.environment_file, "r"))["vae"]
    config_dict = json.load(open(args.config_file, "r"))

    # Définir les arguments
    for k, v in env_dict.items():
        setattr(args, k, v)
    for k, v in config_dict.items():
        setattr(args, k, v)

    run_dir = args.run_dir
    args.model_dir = os.path.join(run_dir, "trained_weights")
    args.tfevent_path = os.path.join(run_dir, "tfevent")

    if rank == 0:
        Path(args.model_dir).mkdir(parents=True, exist_ok=True)
        Path(args.tfevent_path).mkdir(parents=True, exist_ok=True)

    set_determinism(42)

    best_epoch_saved = None  # On garde en mémoire le nom de la dernière epoch enregistrée

    # Step 1: set data loader
    size_divisible = 2 ** (len(args.autoencoder_def["channels"]) - 1)
    train_loader, val_loader = prepare_tif_dataloader_vae(
        args,
        args.autoencoder_train["batch_size"],
        args.autoencoder_train["patch_size"],
        sample_axis=args.sample_axis,
        randcrop=True,
        rank=rank,
        world_size=world_size,
        cache=1.0,
        download=False,
        size_divisible=size_divisible,
    )

    print(f"[INFO] Nombre d'images dans le train : {len(train_loader.dataset)}")
    print(f"[INFO] Nombre d'images dans la validation : {len(val_loader.dataset)}")

    # Step 2: Define Autoencoder KL network and discriminator
    autoencoder = define_instance(args, "autoencoder_def").to(device)
    discriminator_norm = "INSTANCE"
    discriminator = PatchDiscriminator(
        spatial_dims=args.spatial_dims,
        num_layers_d=3,
        channels=32,
        in_channels=1,
        out_channels=1,
        norm=discriminator_norm,
    ).to(device)
    if ddp_bool:
        # When using DDP, BatchNorm needs to be converted to SyncBatchNorm.
        discriminator = torch.nn.SyncBatchNorm.convert_sync_batchnorm(discriminator)

    trained_g_path = os.path.join(args.model_dir, "autoencoder.pt")
    trained_d_path = os.path.join(args.model_dir, "discriminator.pt")
    trained_g_path_last = os.path.join(args.model_dir, "autoencoder_last.pt")
    trained_d_path_last = os.path.join(args.model_dir, "discriminator_last.pt")

    if rank == 0:
        Path(args.model_dir).mkdir(parents=True, exist_ok=True)

    # if args.resume_ckpt:
    #     map_location = {"cuda:%d" % 0: "cuda:%d" % rank}
    #     try:
    #         autoencoder.load_state_dict(torch.load(trained_g_path, map_location=map_location, weights_only=True))
    #         print(f"Rank {rank}: Load trained autoencoder from {trained_g_path}")
    #     except:
    #         print(f"Rank {rank}: Train autoencoder from scratch.")

    #     try:
    #         discriminator.load_state_dict(torch.load(trained_d_path, map_location=map_location, weights_only=True))
    #         print(f"Rank {rank}: Load trained discriminator from {trained_d_path}")
    #     except:
    #         print(f"Rank {rank}: Train discriminator from scratch.")

    if ddp_bool:
        autoencoder = DDP(
            autoencoder,
            device_ids=[device],
            output_device=rank,
            find_unused_parameters=True,
        )
        discriminator = DDP(
            discriminator,
            device_ids=[device],
            output_device=rank,
            find_unused_parameters=True,
        )

    # Step 3: training config
    if "recon_loss" in args.autoencoder_train and args.autoencoder_train["recon_loss"] == "l2":
        intensity_loss = MSELoss()
        if rank == 0:
            print("Use l2 loss")
    else:
        intensity_loss = L1Loss()
        if rank == 0:
            print("Use l1 loss")
    adv_loss = PatchAdversarialLoss(criterion="least_squares")
    loss_perceptual = PerceptualLoss(spatial_dims=args.spatial_dims, network_type="squeeze")
    loss_perceptual.to(device)

    adv_weight = 0.5
    perceptual_weight = args.autoencoder_train["perceptual_weight"]
    # kl_weight: important hyper-parameter.
    #     If too large, decoder cannot recon good results from latent space.
    #     If too small, latent space will not be regularized enough for the diffusion model
    kl_weight = args.autoencoder_train["kl_weight"]

    optimizer_g = torch.optim.Adam(params=autoencoder.parameters(), lr=args.autoencoder_train["lr"] * world_size)
    optimizer_d = torch.optim.Adam(params=discriminator.parameters(), lr=args.autoencoder_train["lr"] * world_size)

    # initialize tensorboard writer
    if rank == 0:
        tensorboard_writer = SummaryWriter(args.tfevent_path)

    # Chargement d'un checkpoint si demandé
    if args.resume_ckpt:
        checkpoint_path = args.checkpoint_dir
        print(checkpoint_path)

        if os.path.exists(checkpoint_path):
            print(f"[INFO] Chargement du checkpoint depuis {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=torch.device(f"cuda:{device}"))

            # Pour DDP : .module sinon .state_dict directement
            if ddp_bool:
                autoencoder.module.load_state_dict(checkpoint["autoencoder_state_dict"])
                discriminator.module.load_state_dict(checkpoint["discriminator_state_dict"])
            else:
                autoencoder.load_state_dict(checkpoint["autoencoder_state_dict"])
                discriminator.load_state_dict(checkpoint["discriminator_state_dict"])

            optimizer_g.load_state_dict(checkpoint["optimizer_g_state_dict"])
            optimizer_d.load_state_dict(checkpoint["optimizer_d_state_dict"])

            start_epoch = checkpoint["epoch"] + 1
            best_val_recon_epoch_loss = checkpoint["best_val_loss"]
            total_step = checkpoint["total_step"]

            print(f"[INFO] Reprise à l'epoch {start_epoch} | best_val_loss = {best_val_recon_epoch_loss:.4f} | total_step = {total_step}")
        else:
            raise FileNotFoundError(f"[ERREUR] Le checkpoint demandé n'existe pas : {checkpoint_path}")
    else:
        print(f"[INFO] Entraînement depuis zéro (aucun checkpoint chargé).")
        start_epoch = 0
        best_val_recon_epoch_loss = 100.0
        total_step = 0



    # Step 4: training
    autoencoder_warm_up_n_epochs = 5
    max_epochs = args.autoencoder_train["max_epochs"]
    val_interval = args.autoencoder_train["val_interval"]

    for epoch in range(start_epoch, max_epochs):
        start_time = time.time()
        autoencoder.train()
        discriminator.train()

        if ddp_bool:
            train_loader.sampler.set_epoch(epoch)
            val_loader.sampler.set_epoch(epoch)

        display_epoch = epoch + 1 if start_epoch == 0 else epoch

        for step, batch in enumerate(tqdm(train_loader, desc=f"Epoch {display_epoch}/{max_epochs}")):
            images = batch.to(device)

            optimizer_g.zero_grad(set_to_none=True)
            reconstruction, z_mu, z_sigma = autoencoder(images)

            recons_loss = intensity_loss(reconstruction, images)
            kl_loss = KL_loss(z_mu, z_sigma)
            p_loss = loss_perceptual(reconstruction.float(), images.float())
            loss_g = recons_loss + kl_weight * kl_loss + perceptual_weight * p_loss

            if epoch > autoencoder_warm_up_n_epochs:
                logits_fake = discriminator(reconstruction.contiguous().float())[-1]
                generator_loss = adv_loss(logits_fake, target_is_real=True, for_discriminator=False)
                loss_g = loss_g + adv_weight * generator_loss

            loss_g.backward()
            optimizer_g.step()

            if epoch > autoencoder_warm_up_n_epochs:
                optimizer_d.zero_grad(set_to_none=True)
                logits_fake = discriminator(reconstruction.contiguous().detach())[-1]
                loss_d_fake = adv_loss(logits_fake, target_is_real=False, for_discriminator=True)
                logits_real = discriminator(images.contiguous().detach())[-1]
                loss_d_real = adv_loss(logits_real, target_is_real=True, for_discriminator=True)
                discriminator_loss = (loss_d_fake + loss_d_real) * 0.5
                loss_d = adv_weight * discriminator_loss

                loss_d.backward()
                optimizer_d.step()

            if rank == 0:
                total_step += 1
                tensorboard_writer.add_scalar("train_recon_loss_iter", recons_loss, total_step)

                 # 📸 Affichage périodique d'un triplet (original | recon | diff)
                if step == 0 and rank == 0:
                    with torch.no_grad():
                        img = images[0].unsqueeze(0).detach().cpu()
                        recon = reconstruction[0].unsqueeze(0).detach().cpu()
                        diff = torch.abs(img - recon)

                        img_disp = torch.rot90(normalize_batch_for_display(img), k=3, dims=[2, 3])[0]
                        recon_disp = torch.rot90(normalize_batch_for_display(recon), k=3, dims=[2, 3])[0]
                        diff_disp = torch.rot90(normalize_batch_for_display(diff), k=3, dims=[2, 3])[0]

                        triplet = torch.cat([img_disp, recon_disp, diff_disp], dim=2)

                        tensorboard_writer.add_image("train_triplets", triplet, global_step=total_step)

        # Validation
        if (epoch) % val_interval == 0:
            autoencoder.eval()
            val_recon_epoch_loss = 0
            triplets = []

            # 📦 Paramètres d'enregistrement
            start_epoch_to_save = 10
            save_every = 5
            save_root = Path(os.path.join(run_dir, "validation_samples"))
            do_save_images = rank == 0 and epoch >= start_epoch_to_save and epoch % save_every == 0
            do_save_weights = rank == 0 and epoch >= start_epoch_to_save

            if do_save_images:
                epoch_dir = save_root / f"epoch_{epoch}"
                dir_original = epoch_dir / "originale"
                dir_recon = epoch_dir / "reconstruction"
                dir_diff = epoch_dir / "diff"
                dir_original.mkdir(parents=True, exist_ok=True)
                dir_recon.mkdir(parents=True, exist_ok=True)
                dir_diff.mkdir(parents=True, exist_ok=True)

            for step, batch in enumerate(val_loader):
                images = batch.to(device)

                with torch.no_grad():
                    reconstruction, z_mu, z_sigma = autoencoder(images)
                    recons_loss = intensity_loss(reconstruction.float(), images.float()) + \
                                perceptual_weight * loss_perceptual(reconstruction.float(), images.float())

                val_recon_epoch_loss += recons_loss.item()

                if rank == 0:
                    # On prend la 1ère image du batch
                    img = images[0].squeeze().detach().cpu()         # [H, W]
                    recon = reconstruction[0].squeeze().detach().cpu()
                    diff = torch.abs(img - recon)

                    # Sauvegarde brute en .tif (centrée/réduite), avec rotation
                    if do_save_images:
                        tifffile.imwrite(dir_original / f"step{step:03}.tif", torch.rot90(img, k=3, dims=[0, 1]).numpy())
                        tifffile.imwrite(dir_recon / f"step{step:03}.tif", torch.rot90(recon, k=3, dims=[0, 1]).numpy())
                        tifffile.imwrite(dir_diff / f"step{step:03}.tif", torch.rot90(diff, k=3, dims=[0, 1]).numpy())

                    # Pour TensorBoard
                    img_disp = torch.rot90(normalize_batch_for_display(img.unsqueeze(0).unsqueeze(0)), k=3, dims=[2, 3])[0]
                    recon_disp = torch.rot90(normalize_batch_for_display(recon.unsqueeze(0).unsqueeze(0)), k=3, dims=[2, 3])[0]
                    diff_disp = torch.rot90(normalize_batch_for_display(diff.unsqueeze(0).unsqueeze(0)), k=3, dims=[2, 3])[0]

                    triplet = torch.cat([img_disp, recon_disp, diff_disp], dim=2)
                    triplets.append((step, triplet))


            val_recon_epoch_loss = val_recon_epoch_loss / (step + 1)

            if do_save_weights:
                print(f"Epoch {epoch} val_loss: {val_recon_epoch_loss:.4f} | Time: {time.time() - start_time:.1f}s")

                # Sauvegarde des modèles
                if ddp_bool:
                    torch.save(autoencoder.module.state_dict(), trained_g_path_last)
                    torch.save(discriminator.module.state_dict(), trained_d_path_last)
                else:
                    torch.save(autoencoder.state_dict(), trained_g_path_last)
                    torch.save(discriminator.state_dict(), trained_d_path_last)

                if val_recon_epoch_loss < best_val_recon_epoch_loss:
                    best_val_recon_epoch_loss = val_recon_epoch_loss

                    # Nettoyage des anciens meilleurs poids/checkpoint
                    if best_epoch_saved is not None:
                        files_to_remove = [
                            os.path.join(args.model_dir, f"checkpoint_epoch{best_epoch_saved}.pth"),
                            os.path.join(args.model_dir, f"autoencoder_epoch{best_epoch_saved}.pth"),
                            os.path.join(args.model_dir, f"discriminator_epoch{best_epoch_saved}.pth"),
                        ]
                        for f in files_to_remove:
                            if os.path.exists(f):
                                os.remove(f)

                    # Mise à jour du numéro d'epoch courant
                    best_epoch_saved = epoch

                    # Sauvegarde des nouveaux meilleurs poids
                    if ddp_bool:
                        torch.save(autoencoder.module.state_dict(), os.path.join(args.model_dir, f"autoencoder_epoch{epoch}.pth"))
                        torch.save(discriminator.module.state_dict(), os.path.join(args.model_dir, f"discriminator_epoch{epoch}.pth"))
                    else:
                        torch.save(autoencoder.state_dict(), os.path.join(args.model_dir, f"autoencoder_epoch{epoch}.pth"))
                        torch.save(discriminator.state_dict(), os.path.join(args.model_dir, f"discriminator_epoch{epoch}.pth"))

                    # Sauvegarde du checkpoint complet
                    checkpoint_path = os.path.join(args.model_dir, f"checkpoint_epoch{epoch}.pth")
                    torch.save({
                        'epoch': epoch,
                        'autoencoder_state_dict': autoencoder.module.state_dict() if ddp_bool else autoencoder.state_dict(),
                        'discriminator_state_dict': discriminator.module.state_dict() if ddp_bool else discriminator.state_dict(),
                        'optimizer_g_state_dict': optimizer_g.state_dict(),
                        'optimizer_d_state_dict': optimizer_d.state_dict(),
                        'best_val_loss': best_val_recon_epoch_loss,
                        'total_step': total_step,
                    }, checkpoint_path)

                    print(f"✅ Meilleurs modèles enregistrés pour l'epoch {epoch}.")

            tensorboard_writer.add_scalar("val_recon_loss", val_recon_epoch_loss, epoch)

            # ✅ Un tag par image, avec epoch en global_step : reste groupé sous "val_triplets"
            for step_idx, triplet in triplets:
                tensorboard_writer.add_image(f"val_triplets/step{step_idx:03}", triplet, global_step=epoch)



if __name__ == "__main__":
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format="[%(asctime)s.%(msecs)03d][%(levelname)5s](%(name)s) - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    main()
