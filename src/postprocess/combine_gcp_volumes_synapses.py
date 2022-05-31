import os
import sys
import time
import argparse
import numpy as np
from glob import glob

import webknossos as wk
from webknossos.dataset import COLOR_CATEGORY, SEGMENTATION_CATEGORY
from webknossos.dataset.properties import LayerViewConfiguration

from db import db

# import torch
# from torch.nn import functional as F

from tqdm import tqdm
from omegaconf import OmegaConf
# from src.gcp_models import UNet3D

from skimage.transform import resize
from joblib import Parallel, delayed


def resize_wrapper(inp, shape):
    return resize(inp, shape, anti_aliasing=True, preserve_range=True, order=0)


if __name__ == '__main__':
    debug = False
    dryrun = False
    n_jobs = -1
    conf = "configs/W-Q.yml"
    conf = OmegaConf.load(conf)
    ds_input = conf.ds.path
    scale = tuple(conf.ds.scale)
    image_layer = conf.ds.img_layer
    token = conf.token
    path_str = conf.storage.ribbon_path_str
    image_shape = conf.ds.vol_shape
    resize_mod = conf.ds.resize_mod
    gcp_dtype = np.uint8

    with wk.webknossos_context(url="https://webknossos.org", token=token):
        # Get native resolution
        ds = wk.Dataset(ds_input, scale=scale, exist_ok=True)
        volume_shape = ds._properties.data_layers[0].bounding_box.size
        images = ds.get_layer(image_layer).get_mag("1")

        try:
            ds.delete_layer("ribbon_synapses")
            print("Deleting prior ribbon layer.")
        except:
            print("No prior ribbon layer found.")

        try:
            ds.delete_layer("conventional_synapses")
            print("Deleting prior conventional layer.")
        except:
            print("No prior conventional layer found.")

        layer_ribbon = ds.add_layer(
            "ribbon_synapses",
            COLOR_CATEGORY,
            gcp_dtype,
            largest_segment_id=np.iinfo(np.uint8).max,
            num_channels=1,
            exist_ok=True
        )
        layer_ribbon.add_mag(1)
        layer_ribbon.default_view_configuration = LayerViewConfiguration(color=(255, 0, 0))
        mag1_ribbon = layer_ribbon.get_mag("1")

        layer_conventional = ds.add_layer(
            "conventional_synapses",
            COLOR_CATEGORY,
            gcp_dtype,
            largest_segment_id=np.iinfo(np.uint8).max,
            num_channels=1,
            exist_ok=True
        )
        layer_conventional.add_mag(1)
        layer_conventional.default_view_configuration = LayerViewConfiguration(color=(0, 255, 0))
        mag1_conventional = layer_conventional.get_mag("1")

        # Loop through coordinates
        res_shape = np.asarray(image_shape)
        # res_shape = res_shape.astype(float) / np.asarray(resize_mod).astype(float)
        res_shape = res_shape.astype(int)
        coords = db.pull_main_seg_coors()
        coords = np.asarray([[coord["x"], coord["y"], coord["z"]] for coord in coords])
        unique_zs = np.unique(coords[:, -1])
        volume = np.zeros((volume_shape), dtype=np.float32)
        with Parallel(n_jobs=n_jobs, prefer="threads") as parallel:
            for z in tqdm(unique_zs, total=len(unique_zs), desc="Z axis progress", leave=True, position=1):
                sel_coords = coords[coords[:, -1] == z]
                for coord in tqdm(sel_coords, total=len(sel_coords), desc="Processing", leave=False, position=2):
                    # Go through and load each vol, select that area from the ds, then maximum the two
                    x, y, z = coord
                    if dryrun:
                        print(x, y, z)
                        continue
                    path = path_str.format(x, y, z, x, y, z)
                    preds = np.load("{}.npy".format(path))
                    preds = (preds * 255.).astype(gcp_dtype).transpose(1, 2, 3, 0)
                    # preds = (preds * 255.).astype(gcp_dtype).transpose(0, 3, 2, 1)
                
                    # Now resize
                    # res_pred = parallel(delayed(resize_wrapper)(pred, res_shape[1:].tolist() + [preds.shape[-1]]) for pred in preds)
                    res_pred = parallel(delayed(resize_wrapper)(pred, res_shape[1:].tolist()) for pred in tqdm(preds, total=len(preds), leave=False, desc="Resizing", position=3))

                    # preds = resize(
                    #     preds,
                    #     res_shape[1:].tolist() + [preds.shape[-1]],
                    #     anti_aliasing=True,
                    #     preserve_range=True,
                    #     order=0)

                    # Now transpose to xyzc
                    preds = np.asarray(res_pred).astype(gcp_dtype)
                    preds = preds.transpose(3, 1, 2, 0)
                    # preds = preds.transpose(3, 2, 1, 0)
                    # preds = (preds * 255.).astype(gcp_dtype).transpose(0, 2, 3, 1)

                    shape = preds.shape[1:]
                
                    # existing_ribbon = mag1_ribbon.read((x, y, z), shape)[0]
                    # existing_conventional = mag1_conventional.read((x, y, z), shape)[0]

                    # preds_ribbon = np.maximum(preds[0], existing_ribbon)
                    # preds_conventional = np.maximum(preds[1], existing_conventional)

                    preds_ribbon, preds_conventional = preds[0], preds[1]  # These were transposed originally
                    if debug:
                        existing_images = images.read((x, y, z), shape)[0]
                        from matplotlib import pyplot as plt
                        imn = 160
                        plt.subplot(141)
                        plt.imshow(existing_images[..., 32])
                        plt.subplot(142)
                        plt.imshow(preds_ribbon[..., 32])
                        plt.subplot(143)
                        plt.imshow(preds_conventional[..., 32])
                        plt.subplot(144)
                        plt.imshow(existing_images[..., 32].astype(np.float32) + preds_ribbon[..., 32].astype(np.float32) + preds_conventional[..., 32].astype(np.float32))
                        plt.show()

                    mag1_ribbon.write(preds_ribbon, (x, y, z))
                    mag1_conventional.write(preds_conventional, (x, y, z))

        # if not dryrun:
        #     mag1_ribbon.compress()
        #     mag1_conventional.compress()
        #     layer_ribbon.downsample(compress=True)
        #     layer_conventional.downsample(compress=True)

