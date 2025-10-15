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
from pathlib import Path

import tifffile
import torch
import torch.nn.functional as F
from monai.config import print_config
from monai.inferers import LatentDiffusionInferer
from monai.networks.schedulers import DDPMScheduler
from monai.utils import first, set_determinism
from torch.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from utils_tif_no_augment import define_instance, normalize_batch_for_display, prepare_tif_dataloader_ldm, setup_ddp


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

    env_dict = json.load(open(args.environment_file))["ldm"]
    config_dict = json.load(open(args.config_file))

    for k, v in env_dict.items():
        setattr(args, k, v)
    for k, v in config_dict.items():
        setattr(args, k, v)

    args.model_dir = os.path.join(args.run_dir, "trained_weights")
    args.tfevent_path = os.path.join(args.run_dir, "tfevent")

    if rank == 0:
        Path(args.model_dir).mkdir(parents=True, exist_ok=True)
        Path(args.tfevent_path).mkdir(parents=True, exist_ok=True)

    set_determinism(42)

    best_epoch_saved = None  # On garde en mémoire le nom de la dernière epoch enregistrée

    # Step 1: set data loader
    size_divisible = 2 ** (len(args.autoencoder_def["channels"]) + len(args.diffusion_def["channels"]) - 2)
    train_loader, val_loader = prepare_tif_dataloader_ldm(
        args,
        args.diffusion_train["batch_size"],
        args.diffusion_train["patch_size"],
        sample_axis=args.sample_axis,
        randcrop=True,
        rank=rank,
        world_size=world_size,
        cache=0.0,
        download=False,
        size_divisible=size_divisible,
        amp=True,
    )

    # initialize tensorboard writer
    if rank == 0:
        Path(args.tfevent_path).mkdir(parents=True, exist_ok=True)
        tensorboard_path = args.tfevent_path
        tensorboard_writer = SummaryWriter(tensorboard_path)

    # Step 2: Define Autoencoder KL network and diffusion model
    # Load Autoencoder KL network
    autoencoder = define_instance(args, "autoencoder_def").to(device)

    condition_projector = torch.nn.Linear(4, args.diffusion_def["cross_attention_dim"]).to(device)

    trained_g_path = args.autoencoder_path

    map_location = {"cuda:%d" % 0: "cuda:%d" % rank}
    autoencoder.load_state_dict(torch.load(trained_g_path, map_location=map_location, weights_only=True))
    print(f"Rank {rank}: Load trained autoencoder from {trained_g_path}")

    # Compute Scaling factor
    # As mentioned in Rombach et al. [1] Section 4.3.2 and D.1, the signal-to-noise ratio (induced by the scale of the latent space) can affect the results obtained with the LDM,
    # if the standard deviation of the latent space distribution drifts too much from that of a Gaussian.
    # For this reason, it is best practice to use a scaling factor to adapt this standard deviation.
    # _Note: In case where the latent space is close to a Gaussian distribution, the scaling factor will be close to one,
    # and the results will not differ from those obtained when it is not used._

    with torch.no_grad():
        with autocast("cuda", enabled=True):
            check_data = first(train_loader)[0]
            z = autoencoder.encode_stage_2_inputs(check_data.to(device))

    scale_factor = 1 / torch.std(z)
    print(f"Rank {rank}: local scale_factor: {scale_factor}")
    if ddp_bool:
        dist.barrier()
        dist.all_reduce(scale_factor, op=torch.distributed.ReduceOp.AVG)
    print(f"Rank {rank}: final scale_factor -> {scale_factor}")

    # Define Diffusion Model
    unet = define_instance(args, "diffusion_def").to(device)

    trained_diffusion_path = os.path.join(args.model_dir, "diffusion_unet.pt")
    trained_diffusion_path_last = os.path.join(args.model_dir, "diffusion_unet_last.pt")

    start_epoch = 0
    if args.resume_ckpt:
        start_epoch = args.start_epoch
        map_location = {"cuda:%d" % 0: "cuda:%d" % rank}
        try:
            unet.load_state_dict(torch.load(trained_diffusion_path, map_location=map_location, weights_only=True))
            print(
                f"Rank {rank}: Load trained diffusion model from",
                trained_diffusion_path,
            )
        except:
            print(f"Rank {rank}: Train diffusion model from scratch.")

    scheduler = DDPMScheduler(
        num_train_timesteps=args.NoiseScheduler["num_train_timesteps"],
        schedule="scaled_linear_beta",
        beta_start=args.NoiseScheduler["beta_start"],
        beta_end=args.NoiseScheduler["beta_end"],
    )

    if ddp_bool:
        autoencoder = DDP(
            autoencoder,
            device_ids=[device],
            output_device=rank,
            find_unused_parameters=True,
        )
        unet = DDP(unet, device_ids=[device], output_device=rank, find_unused_parameters=True)

    # We define the inferer using the scale factor:
    inferer = LatentDiffusionInferer(scheduler, scale_factor=scale_factor)

    # Step 3: training config
    optimizer_diff = torch.optim.Adam(
        list(unet.parameters()) + list(condition_projector.parameters()), lr=args.diffusion_train["lr"] * world_size
    )
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer_diff,
        milestones=args.diffusion_train["lr_scheduler_milestones"],
        gamma=0.1,
    )

    # Step 4: training
    max_epochs = args.diffusion_train["max_epochs"]
    val_interval = args.diffusion_train["val_interval"]
    autoencoder.eval()
    scaler = GradScaler("cuda")
    total_step = 0
    best_val_recon_epoch_loss = 100.0
    start_time = time.time()

    for epoch in range(start_epoch, max_epochs):
        print(f"\n📅 Début de l'epoch {epoch}/{max_epochs - 1}")

        unet.train()
        lr_scheduler.step()
        if ddp_bool:
            train_loader.sampler.set_epoch(epoch)
            val_loader.sampler.set_epoch(epoch)

        for step, (images, condition_images) in enumerate(train_loader):
            if step == 0:
                print(f"🚀 Premier batch de l'epoch {epoch}")
            if step % 100 == 0:
                print(f"🌀 Epoch {epoch} - Batch {step}/{len(train_loader)}")

            images = images.to(device)
            condition_images = condition_images.to(device)

            # Encode les images dentées en latent
            condition_latent = autoencoder.encode_stage_2_inputs(condition_images)  # [B, 4, H, W]

            # Reformater pour l'attention
            B, C, H, W = condition_latent.shape
            condition_seq = condition_latent.permute(0, 2, 3, 1).reshape(B, H * W, C)  # [B, N, 4]
            condition_context = condition_projector(condition_seq)  # [B, N, 128]

            optimizer_diff.zero_grad(set_to_none=True)

            with autocast("cuda", enabled=True):
                # Generate random noise
                noise_shape = [images.shape[0]] + list(z.shape[1:])
                noise = torch.randn(noise_shape, dtype=images.dtype).to(device)

                # test mix random noise + coupe dentée
                # alpha = 0.7
                # pure_noise = torch.randn_like(condition_latent)
                # noise = torch.sqrt(torch.tensor(alpha)) * pure_noise + torch.sqrt(torch.tensor(1 - alpha)) * condition_latent
                # # Renormalisation locale par image (batch-wise)
                # noise = (noise - noise.mean(dim=(1,2,3), keepdim=True)) / noise.std(dim=(1,2,3), keepdim=True)

                # Create timesteps
                timesteps = torch.randint(
                    0,
                    inferer.scheduler.num_train_timesteps,
                    (images.shape[0],),
                    device=images.device,
                ).long()

                # Get model prediction
                if ddp_bool:
                    inferer_autoencoder = autoencoder.module
                else:
                    inferer_autoencoder = autoencoder
                noise_pred = inferer(
                    inputs=images,
                    autoencoder_model=inferer_autoencoder,
                    diffusion_model=unet,
                    noise=noise,
                    timesteps=timesteps,
                    condition=condition_context,
                    mode="crossattn",
                )

                loss = F.mse_loss(noise_pred.float(), noise.float())

            scaler.scale(loss).backward()
            scaler.step(optimizer_diff)
            scaler.update()

            # write train loss for each batch into tensorboard
            if rank == 0:
                total_step += 1
                tensorboard_writer.add_scalar("train_diffusion_loss_iter", loss, total_step)

            # Visualisation
            if step == 0 and rank == 0:
                with torch.no_grad():
                    # Pour reproductibilité
                    torch.manual_seed(42)
                    np.random.seed(42)

                    cond = condition_images[0].unsqueeze(0).detach().cpu()
                    img = images[0].unsqueeze(0).detach().cpu()

                    # print("img shape:", img.shape)

                    cond_disp = torch.rot90(normalize_batch_for_display(cond), k=3, dims=[2, 3])[0]
                    img_disp = torch.rot90(normalize_batch_for_display(img), k=3, dims=[2, 3])[0]

                    # Generate random noise
                    noise_shape = [images.shape[0]] + list(z.shape[1:])
                    noise = torch.randn(noise_shape, dtype=images.dtype).to(device)

                    # Visu à t = 100, 500 et 999
                    t_values = [100, 500, 999]

                    # Grille 4x2 : [cond, GT], puis [noise_img@t, recon@t] pour t=100,500,999
                    grid_rows = []

                    # Ligne 1 : Condition + GT
                    row0 = torch.cat([cond_disp, img_disp], dim=2)
                    grid_rows.append(row0)

                    # Lignes 2-4 : bruitée + reconstruction
                    for t_int in t_values:
                        t = torch.tensor([t_int], device=device).long()
                        z_t = inferer.scheduler.add_noise(original_samples=z, noise=noise, timesteps=t)

                        # Calcul du bruit réel injecté à t
                        alphas_cumprod = inferer.scheduler.alphas_cumprod.to(device)
                        alpha_t = alphas_cumprod[t].reshape(-1, 1, 1, 1).type_as(z)

                        # Bruitée
                        noise_img = inferer_autoencoder.decode_stage_2_outputs(z_t).cpu()
                        noise_img_disp = torch.rot90(normalize_batch_for_display(noise_img), k=3, dims=[2, 3])[0]

                        # Reconstruction
                        noise_pred = unet(z_t, timesteps=t, context=condition_context[0:1])
                        z_hat = (z_t - torch.sqrt(1 - alpha_t) * noise_pred) / torch.sqrt(alpha_t)
                        recon = inferer_autoencoder.decode_stage_2_outputs(z_hat).cpu()
                        recon_disp = torch.rot90(normalize_batch_for_display(recon), k=3, dims=[2, 3])[0]

                        # Assemble la ligne : [noise, recon]
                        row = torch.cat([noise_img_disp, recon_disp], dim=2)
                        grid_rows.append(row)

                    # Concatène les lignes pour obtenir une image (4 lignes × 2 colonnes)
                    full_visu = torch.cat(grid_rows, dim=1)
                    tensorboard_writer.add_image("train_visu_4x2_noise_recon", full_visu, global_step=total_step)

        # Validation
        if (epoch) % val_interval == 0:
            print(f"\n🔍 Début de la validation à l'epoch {epoch}")

            autoencoder.eval()
            unet.eval()
            val_recon_epoch_loss = 0
            quatuors = []
            val_steps_used = 0

            # Paramètres d'enregistrement
            start_epoch_to_save = 10
            save_every = 2
            save_root = Path(os.path.join(args.run_dir, "validation_samples"))
            do_save_images = rank == 0 and epoch >= start_epoch_to_save and epoch % save_every == 0
            do_save_weights = rank == 0 and epoch >= start_epoch_to_save

            if do_save_images:
                epoch_dir = save_root / f"epoch_{epoch}"
                dir_image = epoch_dir / "edente"
                dir_synth = epoch_dir / "edente_synth"
                dir_image.mkdir(parents=True, exist_ok=True)
                dir_synth.mkdir(parents=True, exist_ok=True)

            with torch.no_grad():
                with autocast("cuda", enabled=True):
                    # valeur du bruit et de l'image latente
                    if rank == 0 and epoch == 0:
                        stats_file = Path(args.run_dir) / "stat_debug_alpha.txt"
                        sum_mean_noise, sum_std_noise = 0.0, 0.0
                        sum_mean_cond, sum_std_cond = 0.0, 0.0
                        sum_mean_comb, sum_std_comb = 0.0, 0.0
                        n = 0

                    for step, (images, condition_images) in enumerate(val_loader):
                        if step % 7 != 0:
                            continue  # saute cette itération, gain de temps

                        print(f"🔬 Validation epoch {epoch} - step {step}/{len(val_loader)}")

                        images = images.to(device)
                        condition_images = condition_images.to(device)

                        # Encode les images dentées en latent
                        condition_latent = autoencoder.encode_stage_2_inputs(condition_images)  # [B, 4, H, W]

                        # Reformater en [B, N, 4] pour l'attention (cross_attention_dim = 4)
                        B, C, H, W = condition_latent.shape
                        condition_seq = condition_latent.permute(0, 2, 3, 1).reshape(B, H * W, C)  # [B, N, 4]
                        condition_context = condition_projector(condition_seq)  # [B, N, 128]

                        # Generate random noise
                        # noise_shape = [images.shape[0]] + list(z.shape[1:])
                        # noise = torch.randn(noise_shape, dtype=images.dtype).to(device)

                        alpha = 0.7
                        pure_noise = torch.randn_like(condition_latent)
                        noise_mixed = (
                            torch.sqrt(torch.tensor(alpha)) * pure_noise
                            + torch.sqrt(torch.tensor(1 - alpha)) * condition_latent
                        )

                        # Stats AVANT normalisation
                        if rank == 0:
                            sum_mean_noise += pure_noise.mean().item()
                            sum_std_noise += pure_noise.std().item()
                            sum_mean_cond += condition_latent.mean().item()
                            sum_std_cond += condition_latent.std().item()
                            sum_mean_comb += noise_mixed.mean().item()
                            sum_std_comb += noise_mixed.std().item()
                            n += 1

                        # Renormalisation APRES
                        noise = (noise_mixed - noise_mixed.mean(dim=(1, 2, 3), keepdim=True)) / noise_mixed.std(
                            dim=(1, 2, 3), keepdim=True
                        )

                        timesteps = torch.randint(
                            0,
                            inferer.scheduler.num_train_timesteps,
                            (images.shape[0],),
                            device=images.device,
                        ).long()

                        if ddp_bool:
                            inferer_autoencoder = autoencoder.module
                        else:
                            inferer_autoencoder = autoencoder

                        noise_pred = inferer(
                            inputs=images,
                            autoencoder_model=inferer_autoencoder,
                            diffusion_model=unet,
                            noise=noise,
                            timesteps=timesteps,
                            condition=condition_context[0:1, ...],
                            mode="crossattn",
                        )
                        val_loss = F.mse_loss(noise_pred.float(), noise.float())
                        val_recon_epoch_loss += val_loss
                        val_steps_used += 1

                        if rank == 0:
                            synthetic_images = inferer.sample(
                                input_noise=noise[0:1, ...],
                                autoencoder_model=inferer_autoencoder,
                                diffusion_model=unet,
                                scheduler=scheduler,
                                conditioning=condition_context[0:1, ...],
                                mode="crossattn",
                            )

                            # Récupération des 4 images pour TensorBoard
                            cond = condition_images[0].unsqueeze(0).detach().cpu()
                            img = images[0].unsqueeze(0).detach().cpu()
                            recon = synthetic_images[0].unsqueeze(0).detach().cpu()
                            noise_cpu = noise[0].unsqueeze(0).detach().cpu()

                            # Décodage du bruit en image
                            noise_img = torch.clamp(
                                inferer_autoencoder.decode_stage_2_outputs(noise_cpu.to(device)), min=-5, max=5
                            ).cpu()

                            if do_save_images:
                                tifffile.imwrite(
                                    dir_image / f"step{step:03}.tif", torch.rot90(img.cpu(), k=3, dims=[0, 1]).numpy()
                                )
                                tifffile.imwrite(
                                    dir_synth / f"step{step:03}.tif", torch.rot90(recon.cpu(), k=3, dims=[0, 1]).numpy()
                                )

                            # Pour TensorBoard
                            cond_disp = torch.rot90(normalize_batch_for_display(cond), k=3, dims=[2, 3])[0]
                            img_disp = torch.rot90(normalize_batch_for_display(img), k=3, dims=[2, 3])[0]
                            noise_disp = torch.rot90(normalize_batch_for_display(noise_img), k=3, dims=[2, 3])[0]
                            recon_disp = torch.rot90(normalize_batch_for_display(recon), k=3, dims=[2, 3])[0]
                            quatuor = torch.cat([cond_disp, img_disp, noise_disp, recon_disp], dim=2)
                            quatuors.append((step, quatuor))

            val_recon_epoch_loss = val_recon_epoch_loss / val_steps_used
            tensorboard_writer.add_scalar("val_diffusion_loss", val_recon_epoch_loss, epoch)

            # Un tag par image, avec epoch pour tensorboard
            for step_idx, quatuor in quatuors:
                tensorboard_writer.add_image(f"val_quatuor/step{step_idx:03}", quatuor, global_step=epoch)

            if rank == 0 and n > 0:
                with open(stats_file, "w") as f:
                    f.write(f"📊 Moyennes sur {n} batchs de validation — bruit = pure_noise + condition_latent\n\n")
                    f.write(f"alpha = {alpha}\n\n")
                    f.write(f"pure_noise       : mean = {sum_mean_noise / n:.4f}, std = {sum_std_noise / n:.4f}\n")
                    f.write(f"condition_latent : mean = {sum_mean_cond / n:.4f}, std = {sum_std_cond / n:.4f}\n")
                    f.write(f"combined_noise   : mean = {sum_mean_comb / n:.4f}, std = {sum_std_comb / n:.4f}\n")

            # Bloc de sauvegarde des poids
            if do_save_weights:
                print(f"Epoch {epoch} val_loss: {val_recon_epoch_loss:.4f} | Time: {time.time() - start_time:.1f}s")

                if ddp_bool:
                    torch.save(unet.module.state_dict(), trained_diffusion_path_last)
                else:
                    torch.save(unet.state_dict(), trained_diffusion_path_last)

                if val_recon_epoch_loss < best_val_recon_epoch_loss:
                    best_val_recon_epoch_loss = val_recon_epoch_loss

                    if best_epoch_saved is not None:
                        files_to_remove = [
                            os.path.join(args.model_dir, f"checkpoint_epoch{best_epoch_saved}.pth"),
                            os.path.join(args.model_dir, f"diffusion_unet_epoch{best_epoch_saved}.pth"),
                        ]
                        for f in files_to_remove:
                            if os.path.exists(f):
                                os.remove(f)

                    best_epoch_saved = epoch

                    # Save new best
                    torch.save(
                        unet.module.state_dict() if ddp_bool else unet.state_dict(),
                        os.path.join(args.model_dir, f"diffusion_unet_epoch{epoch}.pth"),
                    )

                    torch.save(
                        {
                            "epoch": epoch,
                            "unet_state_dict": unet.module.state_dict() if ddp_bool else unet.state_dict(),
                            "condition_projector_state_dict": condition_projector.state_dict(),
                            "optimizer_state_dict": optimizer_diff.state_dict(),
                            "best_val_loss": best_val_recon_epoch_loss,
                            "total_step": total_step,
                        },
                        os.path.join(args.model_dir, f"checkpoint_epoch{epoch}.pth"),
                    )

                    print(f"✅ Meilleurs modèles enregistrés pour l'epoch {epoch}.")


if __name__ == "__main__":
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format="[%(asctime)s.%(msecs)03d][%(levelname)5s](%(name)s) - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    main()

    os.system("sudo shutdown -h now")
