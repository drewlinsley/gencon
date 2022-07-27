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
    # conf = "configs/GJD2.yml"
    conf = sys.argv[1]
    if conf is None:
        raise RuntimeError("No config file provided.")
    conf = OmegaConf.load(conf)
    ds_input = conf.ds.path
    scale = tuple(conf.ds.scale)
    image_layer = conf.ds.img_layer
    token = conf.token
    path_str = conf.storage.muller_path_str
    image_shape = conf.ds.vol_shape
    resize_mod = conf.ds.resize_mod
    gcp_dtype = np.uint8

    with wk.webknossos_context(url="https://webknossos.org", token=token):
        # Get native resolution
        ds = wk.Dataset(ds_input, scale=scale, exist_ok=True)
        volume_shape = ds._properties.data_layers[0].bounding_box.size
        images = ds.get_layer(image_layer).get_mag("1")

        try:
            ds.delete_layer("muller")
            print("Deleting prior muller layer.")
        except:
            print("No prior muller layer found.")

        layer_muller = ds.add_layer(
            "muller",
            COLOR_CATEGORY,
            gcp_dtype,
            largest_segment_id=np.iinfo(np.uint8).max,
            num_channels=1,
            exist_ok=True
        )
        layer_muller.add_mag(1)
        layer_muller.default_view_configuration = LayerViewConfiguration(color=(255, 0, 0))
        mag1_muller = layer_muller.get_mag("1")

        # Loop through coordinates
        res_shape = np.asarray(image_shape)
        # res_shape = res_shape.astype(float) / np.asarray(resize_mod).astype(float)
        res_shape = res_shape.astype(int)
        coords = db.pull_main_seg_coors()
        coords = np.asarray([[coord["x"], coord["y"], coord["z"]] for coord in coords])
        unique_zs = np.unique(coords[:, -1])
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
                
                    # Now resize
                    res_pred = parallel(delayed(resize_wrapper)(pred, res_shape[1:].tolist()) for pred in tqdm(preds, total=len(preds), leave=False, desc="Resizing", position=3))

                    # Now transpose to xyzc
                    preds = np.asarray(res_pred).astype(gcp_dtype)
                    preds = preds.transpose(3, 1, 2, 0)
                    preds = preds.squeeze(0)
                    shape = preds.shape[1:]
                    if debug:
                        existing_images = images.read((x, y, z), shape)[0]
                        from matplotlib import pyplot as plt
                        imn = 160
                        plt.subplot(141)
                        plt.imshow(existing_images[..., 32])
                        plt.subplot(142)
                        plt.imshow(preds[..., 32])
                        plt.subplot(143)
                        plt.imshow(existing_images[..., 32].astype(np.float32) + preds[..., 32].astype(np.float32))
                        plt.show()

                    mag1_muller.write(preds, (x, y, z))

