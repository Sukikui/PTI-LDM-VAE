import json
import logging
import tifffile
from pathlib import Path
import torch
import numpy as np
from torchsummary import summary
from monai.config import print_config
from monai.utils import set_determinism
from utils_tif import define_instance
from PIL import Image
import sys
from monai.data import Dataset, DataLoader
from monai.transforms import (
    Compose,
    LoadImage,
    EnsureChannelFirst,
    Resize,
    ScaleIntensity,
    ToTensor,
    NormalizeIntensity,
    EnsureType,
)


from torchvision.utils import make_grid


# === 🔧 À MODIFIER ICI =====================================================

# Chemin vers le fichier de poids
weights_path = Path("12_06_2025_vae_pour_ldm_edente_edente/trained_weights/checkpoint_epoch73.pth")

# Dossier contenant les .tif d'entrée
input_dir = Path("data_cs_1_dm_encastre_tif_04_06_2025_lot_inference_res_150/dente")

# Description manuelle pour le nom du dossier de sortie
description = "test_dente_reu_13_06_2025"

# Fichiers de config
environment_file = Path("../config/environment_tif.json")
config_file = Path("./config/config_train_16g.json")

# ===========================================================================


class LocalNormalizeByMask:
    def __call__(self, img):
        if isinstance(img, torch.Tensor):
            img = img.numpy()
        mask = img != 0
        pixels = img[mask]
        mean = pixels.mean()
        std = pixels.std() if pixels.std() > 1e-5 else 1.0
        img_norm = (img - mean) / std
        img_norm[~mask] = 0.0
        return img_norm.astype(np.float32)

def normalize_batch_for_display(tensor, low=2, high=98):
    np_img = tensor.detach().cpu().numpy()
    normed = []

    for b in range(np_img.shape[0]):
        normed_channels = []
        for c in range(np_img.shape[1]):
            slice_ = np_img[b, c]
            mask = slice_ != 0
            if np.any(mask):
                pixels = slice_[mask]
                min_val = np.percentile(pixels, low)
                max_val = np.percentile(pixels, high)
                slice_norm = np.zeros_like(slice_)
                slice_norm[mask] = np.clip((pixels - min_val) / (max_val - min_val + 1e-8), 0, 1)
            else:
                slice_norm = np.zeros_like(slice_)
            slice_norm[slice_norm < 1e-3] = 0.0
            normed_channels.append(slice_norm)
        normed.append(np.stack(normed_channels))

    normed_tensor = torch.tensor(np.stack(normed))
    return normed_tensor

def prepare_tif_dataloader_vae(image_paths, batch_size, patch_size):
    if len(image_paths) == 0:
        raise FileNotFoundError("Aucune image .tif trouvée dans les chemins fournis")

    compute_dtype = torch.float32  # ou float16 si amp
    transforms = Compose([
        LoadImage(image_only=True),
        EnsureChannelFirst(),
        Resize(patch_size),
        LocalNormalizeByMask(),
        EnsureType(dtype=compute_dtype),
    ])

    inf_ds = Dataset(data=image_paths, transform=transforms)
    inf_loader = DataLoader(inf_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    return inf_loader


# def load_and_preprocess_tif(image_path):
#     img = tifffile.imread(image_path).astype(np.float32)
#     if img.ndim == 2:
#         img = np.expand_dims(img, axis=0)  # [1, H, W]
#     img = np.expand_dims(img, axis=0)      # [1, 1, H, W]
#     img = LocalNormalizeByMask()(img)      # numpy array
#     return torch.tensor(img, dtype=torch.float32)



def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print_config()
    set_determinism(42)

    env_dict = json.load(open(environment_file, "r"))
    config_dict = json.load(open(config_file, "r"))

    class Args: pass
    args = Args()
    for k, v in env_dict.items():
        setattr(args, k, v)
    for k, v in config_dict.items():
        setattr(args, k, v)

    # Définir noms
    nom_dossier = weights_path.parent.parent.name
    nom_fichier = weights_path.name
    epoch = nom_fichier.split("epoch")[-1].split(".")[0]

    # Créer dossiers
    out_dir = Path(f"inference_{nom_dossier}_{epoch}_{description}")
    out_tif = out_dir / "résultats_tif"
    out_png = out_dir / "résultats_png"
    out_tif.mkdir(parents=True, exist_ok=True)
    out_png.mkdir(parents=True, exist_ok=True)

    # Charger modèle
    autoencoder = define_instance(args, "autoencoder_def").to(device)

    # Chargement depuis un checkpoint complet
    checkpoint = torch.load(weights_path, map_location=device)
    autoencoder.load_state_dict(checkpoint["autoencoder_state_dict"])

    # summary(autoencoder, (1,256,256))

    autoencoder.eval()


    # # Charger les images
    # image_paths = sorted(input_dir.glob("*.tif"))
    # print(f"🖼️ {len(image_paths)} images trouvées.")

    # Charger les images vues à l'entraînement pour verif 
    import random  # ajoute en haut du fichier si ce n’est pas déjà fait

    image_paths = sorted(input_dir.glob("*.tif"))
    random.seed(42)  # pour reproductibilité
    image_paths = random.sample(image_paths, 20)


    # Charger les images via le même pipeline que pour le train/val
    inf_loader = prepare_tif_dataloader_vae(
        image_paths=image_paths,
        batch_size=args.autoencoder_train["batch_size"],
        patch_size=args.autoencoder_train["patch_size"],
    )

    # Inference sur chaque batch
    for i, batch in enumerate(inf_loader):
        with torch.no_grad():
            input_tensor = batch.to(device)
            print(input_tensor.shape)
            exit()
            recon, _, _ = autoencoder(input_tensor)
            input_tensor = input_tensor.cpu()
            recon = recon.cpu()

        # Pour chaque image dans le batch
        for j in range(input_tensor.shape[0]):
            input_np = input_tensor[j, 0].numpy()
            recon_np = recon[j, 0].numpy()

            # Enregistrer le TIF concaténé (original + reconstruction)
            concat_tif = np.concatenate([input_np, recon_np], axis=1)  # concaténation horizontale (W)
            tifffile.imwrite(out_tif / f"image{(i * args.autoencoder_train['batch_size']) + j}.tif", concat_tif)

            # Enregistrer le PNG concaténé
            input_disp = normalize_batch_for_display(input_tensor[[j]])[0]
            recon_disp = normalize_batch_for_display(recon[[j]])[0]
            concat = torch.cat([input_disp, recon_disp], dim=2)
            array = (concat.numpy()[0] * 255).astype(np.uint8)
            Image.fromarray(array).save(out_png / f"image{(i * args.autoencoder_train['batch_size']) + j}.png")

    print(f"✅ Inférence terminée. Résultats dans : {out_dir}")



if __name__ == "__main__":
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format="[%(asctime)s.%(msecs)03d][%(levelname)5s](%(name)s) - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    main()
