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

from glob import glob
from monai.transforms import (
    Compose,
    LoadImage,
    EnsureChannelFirst,
    Resize,
    ScaleIntensity,
    ToTensor,
    NormalizeIntensity,
    EnsureType,
    LoadImaged,
    EnsureChannelFirstd,
    ResizeD,
    EnsureTyped,
)
from monai.data import Dataset
from augmentation_utils import get_albumentations_transform, DataLoader
from datetime import timedelta
from monai.bundle import ConfigParser
import torch.distributed as dist

import numpy as np
import torch

import tifffile
from torch.utils.data import Dataset
import albumentations as A
import numpy as np
import os


def setup_ddp(rank, world_size):
    print(f"Running DDP diffusion example on rank {rank}/world_size {world_size}.")
    print(f"Initing to IP {os.environ['MASTER_ADDR']}")
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        timeout=timedelta(seconds=36000),
        rank=rank,
        world_size=world_size,
    )  # gloo, nccl
    dist.barrier()
    device = torch.device(f"cuda:{rank}")
    return dist, device


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

class ApplyLocalNormd:
    def __init__(self, keys):
        self.keys = keys
        self.norm = LocalNormalizeByMask()

    def __call__(self, data):
        for k in self.keys:
            data[k] = torch.tensor(self.norm(data[k]))
        return data

class ToTuple:
    def __init__(self, keys):
        self.keys = keys

    def __call__(self, data):
        return tuple(data[k] for k in self.keys)


def prepare_tif_dataloader_ldm(
    args,
    batch_size,
    patch_size,
    amp=False,
    sample_axis=None,
    randcrop=False,
    rank=0,
    world_size=1,
    cache=1.0,
    download=False,
    size_divisible=1,
    num_center_slice=None,
):
    augment = getattr(args, 'augment', False)
    dir_edente = os.path.join(args.data_base_dir, "edente")
    dir_dente = os.path.join(args.data_base_dir, "dente")

    tif_paths_edente = sorted(glob(os.path.join(dir_edente, "*.tif")))
    tif_paths_dente = sorted(glob(os.path.join(dir_dente, "*.tif")))

    if len(tif_paths_edente) == 0 or len(tif_paths_dente) == 0:
        raise FileNotFoundError(f"Aucune image trouvée dans {dir_edente} ou {dir_dente}")
    if len(tif_paths_edente) != len(tif_paths_dente):
        raise ValueError("Les dossiers denté et édenté doivent contenir le même nombre d'images.")

    paired_data = [{"image": e, "condition_image": d} for e, d in zip(tif_paths_edente, tif_paths_dente)]
    split_idx = int(0.9 * len(paired_data))
    train_data = paired_data[:split_idx]
    val_data = paired_data[split_idx:]

    if augment:
        albumentations_transform = get_albumentations_transform()

        class AugAlb:
            def __call__(self, data):
                img = data['image'].squeeze(0).numpy()
                cond = data['condition_image'].squeeze(0).numpy()
                aug = albumentations_transform(image=img, condition_image=cond)
                data['image'] = torch.from_numpy(aug['image'][None, ...])
                data['condition_image'] = torch.from_numpy(aug['condition_image'][None, ...])
                return data

        aug_monai = AugAlb()
    else:
        aug_monai = lambda x: x

    transforms = Compose([
        LoadImaged(keys=["image", "condition_image"]),
        EnsureChannelFirstd(keys=["image", "condition_image"]),
        ResizeD(keys=["image", "condition_image"], spatial_size=patch_size),
        EnsureTyped(keys=["image", "condition_image"], dtype=torch.float32),
        ApplyLocalNormd(keys=["image", "condition_image"]),
        aug_monai,
        ToTuple(keys=["image", "condition_image"]),  # ⬅️ ici on retourne un tuple
    ])

    train_ds = Dataset(data=train_data, transform=transforms)
    val_ds = Dataset(data=val_data, transform=transforms)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    if rank == 0:
        sample = next(iter(train_loader))
        print(f"[LDM] Image shape: {sample[0].shape}, Condition shape: {sample[1].shape}")

    return train_loader, val_loader



def prepare_tif_dataloader_vae(
    args,
    batch_size,
    patch_size,
    amp=False,
    sample_axis=None,
    randcrop=False,
    rank=0,
    world_size=1,
    cache=1.0,
    download=False,
    size_divisible=1,
    num_center_slice=None,
):
    augment = getattr(args, 'augment', False)
    data_dir = os.path.join(args.data_base_dir, "edente")  
    tif_paths = sorted(glob(os.path.join(data_dir, "*.tif")))

    if len(tif_paths) == 0:
        raise FileNotFoundError(f"Aucune image .tif trouvée dans {data_dir}")

    # # Réduction temporaire du dataset pour tests rapides
    # tif_paths = tif_paths[:len(tif_paths) // 4]

    # Séparer train/val
    split_idx = int(0.9 * len(tif_paths))
    train_paths = tif_paths[:split_idx]
    val_paths = tif_paths[split_idx:]

    if augment:
        albumentations_transform = get_albumentations_transform()

        class AugAlb:
            def __call__(self, img):
                img_np = img.squeeze(0).numpy()
                aug = albumentations_transform(image=img_np)
                return torch.from_numpy(aug['image'][None, ...])

        aug_monai = AugAlb()
    else:
        aug_monai = lambda x: x

    transforms = Compose([
        LoadImage(image_only=True),
        EnsureChannelFirst(),
        Resize(patch_size),
        LocalNormalizeByMask(),
        aug_monai,
        EnsureType(dtype=torch.float32),
    ])

    train_ds = Dataset(data=train_paths, transform=transforms)
    val_ds = Dataset(data=val_paths, transform=transforms)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True
    )

    if rank == 0:
        print(f"Image shape {train_ds[0].shape}")

    return train_loader, val_loader


def define_instance(args, instance_def_key):
    parser = ConfigParser(vars(args))
    parser.parse(True)
    return parser.get_parsed_content(instance_def_key, instantiate=True)


def KL_loss(z_mu, z_sigma):
    kl_loss = 0.5 * torch.sum(
        z_mu.pow(2) + z_sigma.pow(2) - torch.log(z_sigma.pow(2)) - 1,
        dim=list(range(1, len(z_sigma.shape))),
    )
    return torch.sum(kl_loss) / kl_loss.shape[0]


def normalize_batch_for_display(tensor, low=2, high=98):
    """
    Normalise un batch d'images [B, C, H, W] en [0, 1] pour l'affichage TensorBoard.
    Le fond (valeurs == 0) reste noir, et les faibles valeurs reconstruites (< 1e-3) sont forcées à 0.
    """
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
            slice_norm[slice_norm < 1e-3] = 0.0  # suppression du bruit de fond
            normed_channels.append(slice_norm)
        normed.append(np.stack(normed_channels))  # [C, H, W]

    normed_tensor = torch.tensor(np.stack(normed))  # [B, C, H, W]
    return normed_tensor
