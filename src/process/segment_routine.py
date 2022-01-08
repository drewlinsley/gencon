import sys
import os
import webknossos as wk
from src.process import neurite_segmentation
from src.process import membrane_segmentation
from skimage.transform import resize
import numpy as np
from db import db
from omegaconf import OmegaConf


def main(conf, resize_order=3):
    """Perform 0-shot membrane and neurite segmentation on a volume."""
    conf = OmegaConf.load(conf)
    ds_input = conf.ds.path
    ds_layer = conf.ds.img_layer
    scale = tuple(conf.ds.scale)
    image_shape = conf.ds.vol_shape
    resize_mod = conf.ds.resize_mod
    force_coord = conf.ds.force_coord
    membrane_ckpt = conf.inference.membrane_ckpt
    membrane_slice = [x for x in conf.ds.membrane_slice]
    ffn_ckpt = conf.inference.ffn_ckpt
    ffn_model = conf.inference.ffn_model

    # Get coordinates from DB
    if force_coord is not None:
        x, y, z = force_coord
    else:
        next_coordinate = db.get_coordinate()
        if next_coordinate is None:
            # No need to process this point
            raise RuntimeException('No more coordinates found!')
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

    # Resize images
    res_shape = np.asarray(image_shape)
    res_shape = res_shape.astype(float) / np.asarray(resize_mod).astype(float)
    res_shape = res_shape.astype(int)
    print("Resizing images")
    if not np.all(res_shape[::-1][:-1] == np.asarray(cube_in.shape[:-1])):
        res_cube_in = resize(cube_in, res_shape[::-1][:-1], anti_aliasing=True, preserve_range=True, order=resize_order)
    else:
        res_cube_in = cube_in
    # from matplotlib import pyplot as plt;plt.subplot(121);plt.imshow(res_cube_in[32]);plt.subplot(122);plt.imshow(cube_in[32]);plt.show()
    del cube_in, mag1, layer, ds# clean up
    res_cube_in = res_cube_in.transpose(2, 0, 1)

    # Segment membranes
    print("Segmenting membranes")
    vol_mem = membrane_segmentation.get_segmentation(
        vol=res_cube_in,
        membrane_ckpt=membrane_ckpt,
        membrane_slice=membrane_slice,  # [140, 384, 384],
        normalize=True)
    # vol_mem *= 255.
    # from matplotlib import pyplot as plt;plt.subplot(121);plt.imshow(vol_mem[32, ..., 0]);plt.subplot(122);plt.imshow(vol_mem[32, ..., 1]);plt.show()
    del res_cube_in
    np.save(mem_path, vol_mem)

    # Segment neurites
    print("Flood filling membranes")
    segs = neurite_segmentation.get_segmentation(
        vol=vol_mem,
        ffn_ckpt=ffn_ckpt,
        move_threshold=0.7,
        ffn_model=ffn_model)  # Takes uint8 inputs

    # Resize and transpose segments
    # segs = resize(seg.transpose(1, 2, 0), image_shape[::-1][:-1], anti_aliasing=False, preserve_range=True, order=1)
    segs = segs.transpose(1, 2, 0)  # Keep low-res for post-processing
    # from matplotlib import pyplot as plt;plt.subplot(131);plt.imshow(cube_in[..., 32]);plt.subplot(132);plt.imshow(vol_mem[32]);plt.subplot(133);plt.imshow(segs[..., 32]);plt.show()
    np.save(seg_path, segs)

    if force_coord is None:
        # Finish coordinate in DB
        db.finish_coordinate(x, y, z)
    print("Finished volume at {} {} {}".format(x, y, z))


if __name__ == '__main__':
    conf = sys.argv[1]
    main(conf)

