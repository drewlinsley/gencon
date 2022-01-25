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
    membrane_slice = [x for x in conf.ds.membrane_slice]
    ffn_ckpt = conf.inference.ffn_ckpt
    ffn_model = conf.inference.ffn_model
    move_threshold = conf.inference.move_threshold
    segment_threshold = conf.inference.segment_threshold

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
    res_cube_in = res_cube_in.astype(np.float32)
    # from matplotlib import pyplot as plt;plt.subplot(121);plt.imshow(res_cube_in[32]);plt.subplot(122);plt.imshow(cube_in[32]);plt.show()
    del cube_in, mag1, layer, ds# clean up
    res_cube_in = res_cube_in.transpose(2, 0, 1)

    # Segment membranes
    print("Segmenting membranes with input vol size: {} and data type: {}".format(res_cube_in.shape, res_cube_in.dtype))  # noqa
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
        # seed_policy="PolicyMembraneAndPeaks",
        mem_seed_thresh=0.8,
        move_threshold=move_threshold,
        segment_threshold=segment_threshold,
        ffn_model=ffn_model)  # Takes uint8 inputs

    # Resize and transpose segments
    # segs = resize(seg.transpose(1, 2, 0), image_shape[::-1][:-1], anti_aliasing=False, preserve_range=True, order=1)
    segs = segs.transpose(1, 2, 0)  # Keep low-res for post-processing
    # from matplotlib import pyplot as plt;plt.subplot(131);plt.imshow(cube_in[..., 32]);plt.subplot(132);plt.imshow(vol_mem[32]);plt.subplot(133);plt.imshow(segs[..., 32]);plt.show()
    np.save(seg_path, segs)

    if force_coord is None:
        # Finish coordinate in DB
        db.finish_coordinate(x, y, z)

    try:
    # Add a visualization of a middle slice for sanity check
        mid = segs.shape[-1] // 2
        figpath = os.path.join(project, "sanitycheck.png")
        f = plt.figure()
        plt.subplot(131)
        plt.imshow(vol_mem[mid, ..., 0], cmap="Greys_r")
        plt.title("image")
        plt.axis("off")
        plt.subplot(132)
        plt.imshow(vol_mem[mid, ..., 1])
        plt.title("membranes")
        plt.axis("off")
        plt.subplot(133)
        plt.imshow(rs(segs[..., mid])[0])
        plt.title("segments")
        plt.axis("off")
        plt.savefig(figpath)
        plt.close(f)
    except:
        print("Failed to save sanity check visualization.")
    print("Finished volume at {} {} {}".format(x, y, z))


if __name__ == '__main__':
    conf = sys.argv[1]
    main(conf)

