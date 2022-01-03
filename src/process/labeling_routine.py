import sys
import os
import webknossos as wk
from src.process import ribbon_detection
# from src.process import muller_detection
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
    ribbon_ckpt = conf.inference.ribbon_ckpt
    muller_ckpt = conf.inference.muller_ckpt

    # Get coordinates from DB
    if force_coord is not None:
        x, y, z = force_coord
    else:
        next_coordinate = db.get_coordinate()
        if next_coordinate is None:
            # No need to process this point
            raise RuntimeException('No more coordinates found!')
        x, y, z = next_coordinate['x'], next_coordinate['y'], next_coordinate['z']  # noqa
    ribbon_path = conf.storage.ribbon_path_str.format(x, y, z, x, y, z)
    muller_path = conf.storage.muller_path_str.format(x, y, z, x, y, z)
    mem_path = conf.storage.mem_path_str.format(x, y, z, x, y, z)
    os.makedirs(os.path.sep.join(ribbon_path.split(os.path.sep)[:-1]), exist_ok=True)
    os.makedirs(os.path.sep.join(muller_path.split(os.path.sep)[:-1]), exist_ok=True)

    # Get images and membranes
    img = np.load("{}.npy".format(mem_path))

    print("Detecting mullers")
    muller = ribbon_detection.get_segmentation(
        vol=img,
        ckpt_path=muller_ckpt,
        normalize=True)
    np.save(muller_path, muller)

    print("Detecting ribbons")
    ribbon = ribbon_detection.get_segmentation(
        vol=img,
        ckpt_path=ribbon_ckpt,
        normalize=True)
    np.save(ribbon_path, ribbon)

    if force_coord is None:
        # Finish coordinate in DB
        db.finish_coordinate(x, y, z)
    print("Finished volume at {} {} {}".format(x, y, z))


if __name__ == '__main__':
    conf = sys.argv[1]
    main(conf)

