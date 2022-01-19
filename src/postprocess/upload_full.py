import os
import sys
from time import gmtime, strftime
import numpy as np
import webknossos as wk
from webknossos.dataset import COLOR_CATEGORY, SEGMENTATION_CATEGORY
from webknossos.dataset.properties import LayerViewConfiguration
from skimage.segmentation import relabel_sequential as rs
from skimage.transform import resize
from omegaconf import OmegaConf
from tqdm import tqdm
from glob import glob
from joblib import Parallel, delayed


def write_cube(add_segs, data, coord, image_shape, zstart, zend, zwritestart):
    h, w = coord
    cube = data[h: h + image_shape[1], w: w + image_shape[2], zstart: zend]
    add_segs.write(cube, offset=(h, w, zwritestart))


def main(conf, n_jobs=1):
    """Upload a partially processed volume."""
    conf = OmegaConf.load(conf)
    name = conf.name
    ds_input = conf.ds.path
    image_layer = conf.ds.img_layer
    scale = tuple(conf.ds.scale)
    image_shape = conf.ds.vol_shape[::-1]
    mem_path_str = conf.storage.mem_path_str
    seg_path_str = conf.storage.seg_path_str
    merge_path = conf.storage.merge_seg_path
    res_shape = np.asarray(image_shape)
    token = conf.token

    seg_dtype = np.uint32
    mem_dtype = np.uint8
    img_dtype = np.uint8
    with wk.webknossos_context(url="https://webknossos.org", token=token):

        # Get native resolution
        ds = wk.Dataset(ds_input, scale=scale, exist_ok=True)
        try:
            ds.delete_layer("segmentation")
            print("Deleting prior segmentation layer.")
        except:
            print("No prior segmentation layer found.")

        # Compress and downsample images
        color_layer = ds.get_layer(image_layer)
        color_mag = color_layer.get_mag("1")
        color_mag.compress()
        color_layer.downsample(compress=True)

        # Compress and downsample segmentationns
        segmentation_layer = ds.get_layer("segmentation")
        segmentation_mag = segmentation_layer.get_mag("1")
        segmentation_mag.compress() 
        segmentation_layer.downsample(compress=True)

        # Upload
        url = ds.upload()
        print(f"Successfully uploaded {url}")


if __name__ == '__main__':
    conf = sys.argv[1]
    main(conf)

