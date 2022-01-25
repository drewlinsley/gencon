import os
import sys
import time
import argparse
import numpy as np
from glob import glob

from db import db

import torch
from torch.nn import functional as F

from tqdm import tqdm
from omegaconf import OmegaConf
from src.gcp_models import UNet3D


def weights_update(model, checkpoint):
    model_dict = model.state_dict()
    pretrained_dict = {}
    for k, v in checkpoint['state_dict'].items():
        k = k.replace("net.", "")
        if k in model_dict:
            pretrained_dict[k] = v
        else:
            print(k)
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    return model


def get_segmentation(
        vol,
        ckpt_path,
        in_channels,
        out_channels,
        keep_channels,
        normalize=True,
        plot_intermediate=False,
        stride=[60, 128, 128],  # Stride
        cube_size=[64, 256, 256],
        device="cuda",  # "cuda"
        padded=4):  # (8, , ),  # Half-Pad size
    """Apply the FFN routines using fGRUs."""
    vol_shape = vol.shape
    model_cls = getattr(UNet3D, "ResidualUNet3D")
    model = model_cls(
        in_channels=in_channels,
        out_channels=out_channels).to(device)
    model = weights_update(
        model=model,
        checkpoint=torch.load(ckpt_path))
    model_shape = cube_size  # list(vol.shape)
    if padded:
        model_shape = [x + padded * 2 for x in model_shape]
    model_shape = model_shape + [2]
    label_shape = np.copy(model_shape)
    label_shape[-1] = 1

    # FOR SYNAPSE, NORMALIZE VOL
    if normalize:
        vol /= 255.

    # Loop through volume
    d_ind_start = np.arange(0, vol_shape[0], stride[0]).astype(int)[:-1]
    h_ind_start = np.arange(0, vol_shape[1], stride[1]).astype(int)[:-1]
    w_ind_start = np.arange(0, vol_shape[2], stride[2]).astype(int)[:-1]
    ribbons = torch.zeros([out_channels] + [x for x in vol_shape][:-1], device="cpu", dtype=torch.float)
    num_completed = 0
    for d_start in tqdm(d_ind_start, total=len(d_ind_start), desc="Processing", leave=False):
        d_end = d_start + cube_size[0]
        for h_start in h_ind_start:
            h_end = h_start + cube_size[1]
            for w_start in w_ind_start:
                w_end = w_start + cube_size[2]
                cube = vol[d_start: d_end, h_start: h_end, w_start: w_end]
                if padded:
                    cube = np.pad(cube, ((padded, padded), (padded, padded), (padded, padded), (0, 0)), "wrap")

                # Run pytorch model
                cube = torch.from_numpy(cube).to(device).permute(3, 0, 1, 2)[None]
                with torch.no_grad():
                    preds = model.forward(cube)
                preds = preds[0].detach().cpu()
                if padded:
                    preds = preds[:, padded: -padded, padded: -padded, padded: -padded]
                ribbons[:, d_start: d_end, h_start: h_end, w_start: w_end] = np.maximum(
                    ribbons[:, d_start: d_end, h_start: h_end, w_start: w_end],
                    preds)
                num_completed += 1
    ribbons = F.softmax(ribbons, 0)  # Softmax over channels
    ribbons = ribbons[keep_channels]
    if plot_intermediate:
        from matplotlib import pyplot as plt
        plt.subplot(121)
        plt.imshow(ribbons[0, 32])
        plt.subplot(122)
        plt.imshow(ribbons[1, 32])
        plt.show()
    return ribbons


if __name__ == '__main__':
    conf = "configs/W-Q.yml"
    conf = OmegaConf.load(conf)
    ckpt_path = "../gcp_models/results/2022-01-22/21-08-22/results/2022-01-22/21-08-22/cont-learn/1ptzic6o/checkpoints/"
    ckpt = glob(os.path.join(ckpt_path, "*.ckpt"))
    ckpt = sorted(ckpt, key=os.path.getmtime)[-1]
    in_channels = 2
    out_channels = 3
    keep_channels = np.arange(1, 3).astype(int)  # ribbon and conventional

    # Get coordinates from the db
    # x, y, z = 0, 0, 0
    # vol = np.load("/media/data_cifs/projects/prj_connectomics/wong/mag1_membranes/x{}/y{}/z{}/110629_k0725_mag1_x{}_y{}_z{}.npy".format(x, y, z, x, y, z))
    # seg = get_segmentation(vol, ckpt_path)

    coords = db.pull_main_seg_coors()
    for coord in tqdm(coords, total=len(coords), desc="DB coords"):
        x, y, z = coord["x"], coord["y"], coord["z"]
        vol = np.load("/media/data_cifs/projects/prj_connectomics/wong/mag1_membranes/x{}/y{}/z{}/110629_k0725_mag1_x{}_y{}_z{}.npy".format(x, y, z, x, y, z))
        seg = get_segmentation(vol, ckpt, in_channels, out_channels, keep_channels)
        path = conf.storage.ribbon_path_str.format(x, y, z, x, y, z)
        os.makedirs(os.path.sep.join(path.split(os.path.sep)[:-1]), exist_ok=True)
        np.save(path, seg)

