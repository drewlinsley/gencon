import sys
import os
import webknossos as wk
from src.process import run_gcp_model
from skimage.transform import resize
import numpy as np
from db import db
from omegaconf import OmegaConf


def main(conf, resize_order=3):
    """Perform 0-shot membrane and neurite segmentation on a volume."""
    conf = OmegaConf.load(conf)
    ds_input = conf.ds.path
    ds_layer = conf.ds.img_layer
    scale = conf.ds.scale
    image_shape = conf.ds.vol_shape
    resize_mod = conf.ds.resize_mod
    force_coord = conf.ds.force_coord
    muller_ckpt = conf.inference.muller_ckpt
    in_channels = 2
    out_channels = 3
    keep_channels = [1]  # np.arange(1, 3).astype(int)  # ribbon and conventional

    # Get coordinates from DB
    if force_coord is not None and force_coord != False:
        x, y, z = force_coord
    else:
        next_coordinate = db.get_coordinate()
        if next_coordinate is None:
            # No need to process this point
            raise RuntimeException('No more coordinates found!')
        x, y, z = next_coordinate['x'], next_coordinate['y'], next_coordinate['z']  # noqa
    muller_path = conf.storage.muller_path_str.format(x, y, z, x, y, z)
    mem_path = conf.storage.mem_path_str.format(x, y, z, x, y, z)
    os.makedirs(os.path.sep.join(muller_path.split(os.path.sep)[:-1]), exist_ok=True)

    # Get images and membranes
    vol = np.load("{}.npy".format(mem_path))
    vol = np.concatenate((vol[..., [0]], vol[..., 1:].mean(-1, keepdims=True)), -1)

    print("Detecting mullers")
    seg = run_gcp_model.get_segmentation(vol, muller_ckpt, in_channels, out_channels, keep_channels)
    np.save(muller_path, seg)

    if force_coord is None:
        # Finish coordinate in DB
        db.finish_coordinate(x, y, z)
    print("Finished volume at {} {} {}".format(x, y, z))


if __name__ == '__main__':
    conf = sys.argv[1]
    main(conf)

