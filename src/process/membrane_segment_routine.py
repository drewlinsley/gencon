import sys
import os
import webknossos as wk
from src.process import neurite_segmentation
from src.process import membrane_segmentation
from skimage.transform import resize
from skimage.segmentation import relabel_sequential as rs
import numpy as np
from db import db
from omegaconf import OmegaConf
from matplotlib import pyplot as plt
import lycon


def main(conf, resize_order=3):
    """Perform 0-shot membrane and neurite segmentation on a volume."""
    conf = OmegaConf.load(conf)
    ds_input = conf.ds.path
    ds_layer = conf.ds.img_layer
    project = conf.project_directory
    scale = tuple(conf.ds.scale)
    image_shape = conf.ds.vol_shape
    resize_mod = conf.ds.resize_mod
    force_coord = conf.ds.force_coord
    membrane_ckpt = conf.inference.membrane_ckpt
    ffn_ckpt = conf.inference.ffn_ckpt
    ffn_model = conf.inference.ffn_model
    move_threshold = conf.inference.move_threshold
    segment_threshold = conf.inference.segment_threshold
    mem_seed_threshold = conf.inference.membrane_seed_threshold
    truncation = conf.inference.truncation
    allow_skip = conf.inference.skip_membranes_if_completed
    if conf.ds.membrane_slice:
        membrane_slice = [x for x in conf.ds.membrane_slice]
    else:
        membrane_slice = conf.ds.membrane_slice

    # Get coordinates from DB
    if force_coord:
        x, y, z = force_coord
    else:
        next_coordinate = db.get_coordinate()
        if next_coordinate is None:
            # No need to process this point
            raise RuntimeError('No more coordinates found!')
        x, y, z = next_coordinate['x'], next_coordinate['y'], next_coordinate['z']  # noqa
    mem_path = conf.storage.mem_path_str.format(x, y, z, x, y, z)
    seg_path = conf.storage.seg_path_str.format(x, y, z, x, y, z)
    os.makedirs(os.path.sep.join(mem_path.split(os.path.sep)[:-1]), exist_ok=True)
    os.makedirs(os.path.sep.join(seg_path.split(os.path.sep)[:-1]), exist_ok=True)

    # Get images
    ds = wk.Dataset(ds_input, scale=scale, exist_ok=True)
    layer = ds.get_layer(ds_layer)
    mag1 = layer.get_mag("1")
    print("Extracting images")
    cube_in = mag1.read((x, y, z), image_shape[::-1])[0]

    # Segment membranes
    if os.path.exists("{}.npy".format(mem_path)) and allow_skip:
        pass
    else:
        # Resize images
        res_shape = np.asarray(image_shape)
        res_shape = res_shape.astype(float) / np.asarray(resize_mod).astype(float)
        res_shape = res_shape.astype(int)
        print("Resizing images")
        if not np.all(res_shape[::-1][:-1] == np.asarray(cube_in.shape[:-1])):
            res_cube_in = resize(cube_in, res_shape[::-1][:-1], anti_aliasing=True, preserve_range=True, order=resize_order)
        else:
            res_cube_in = cube_in
        res_cube_in = res_cube_in.astype(np.float32)
        # from matplotlib import pyplot as plt;plt.subplot(121);plt.imshow(res_cube_in[32]);plt.subplot(122);plt.imshow(cube_in[32]);plt.show()
        del cube_in, mag1, layer, ds# clean up
        res_cube_in = res_cube_in.transpose(2, 0, 1)

        print("Segmenting membranes with input vol size: {} and data type: {}".format(res_cube_in.shape, res_cube_in.dtype))  # noqa
        vol_mem = membrane_segmentation.get_segmentation(
            vol=res_cube_in,
            membrane_ckpt=membrane_ckpt,
            membrane_slice=membrane_slice,  # [140, 384, 384],
            normalize=True)
        # vol_mem *= 255.
        # from matplotlib import pyplot as plt;plt.subplot(121);plt.imshow(vol_mem[32, ..., 0]);plt.subplot(122);plt.imshow(vol_mem[32, ..., 1]);plt.show()
        np.save(mem_path, vol_mem)
    print("Finished volume at {} {} {}".format(x, y, z))

    if not force_coord:
        # Finish coordinate in DB
        db.finish_coordinate(x, y, z)


if __name__ == '__main__':
    conf = sys.argv[1]
    main(conf)

