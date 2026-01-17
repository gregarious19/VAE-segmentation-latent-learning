# data_loader.py
import os
import glob
import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F


class NiiLazyDataset(Dataset):
    """
    Lazy-loading dataset for paired NIfTI files:
        *.img.nii.gz
        *.label.nii.gz
    """

    def __init__(self, root_dir, transform=None, downsample_factor=1.0):
        self.root_dir = root_dir
        self.transform = transform
        self.downsample_factor = downsample_factor

        self.img_files = sorted(
            glob.glob(os.path.join(root_dir, "*.img.nii.gz"))
        )

        if len(self.img_files) == 0:
            raise RuntimeError(f"No .img.nii.gz files found in {root_dir}")


    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path = self.img_files[idx]
        label_path = img_path.replace(".img.nii.gz", ".label.nii.gz")

        img = nib.load(img_path).get_fdata(dtype=np.float32)
        label = nib.load(label_path).get_fdata(dtype=np.float32)

        img = torch.from_numpy(img)[None]      # [1, D, H, W]
        label = torch.from_numpy(label)[None]

        if self.downsample_factor != 1.0:
            scale = self.downsample_factor

            img = F.interpolate(
                img[None],
                scale_factor=scale,
                mode="trilinear",
                align_corners=False,
            )[0]

            label = F.interpolate(
                label[None],
                scale_factor=scale,
                mode="nearest",
            )[0]
        label = (label > 0).float()

        return img, label



def get_dataloader(
    root_dir,
    batch_size=1,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    downsample_factor=1.0,
):
    dataset = NiiLazyDataset(
        root_dir,
        downsample_factor=downsample_factor
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )



# -----------------------
# Example usage
# -----------------------
if __name__ == "__main__":
    train_loader = get_dataloader("data", batch_size=1)
    val_loader = get_dataloader("val", batch_size=1, shuffle=False)
    test_loader = get_dataloader("test", batch_size=1, shuffle=False)

    x, y = next(iter(train_loader))
    print("Image shape:", x.shape)
    print("Label shape:", y.shape)
