import sys
import os
import webknossos as wk
from src.process import neurite_segmentation
from src.process import membrane_segmentation
from skimage.transform import resize
from skimage.segmentation import relabel_sequential as rs
import numpy as np
from omegaconf import OmegaConf
from matplotlib import pyplot as plt
import lycon
try:
    from db import db
except:
    print("Failed to import db scripts.")


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
        vol_mem = np.load("{}.npy".format(mem_path))
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
        del res_cube_in

    # TEST
    # vol_mem = resize(vol_mem, res_shape[::-1][:-1], anti_aliasing=True, preserve_range=True, order=resize_order)

    # Truncate the membranes to force oversegmentation
    mems = np.copy(vol_mem)
    vol_mem_mem = vol_mem[..., 1:].max(-1, keepdims=True)
    vol_mem = np.concatenate((vol_mem[..., [0]], vol_mem_mem), -1)
    del vol_mem_mem

    if truncation:
        vol_mem[..., 1] = (1 - (np.clip(255 - vol_mem[..., 1], 0, truncation) / truncation)) * 255.

    # Segment neurites
    print("Flood filling membranes")
    segs, probs = [], []
    scales = [1.]  # , 1.25, 0.75]
    for scale in scales:
        if scale != 1:
            h, w = vol_mem.shape[-3], vol_mem.shape[-2]
            size = [int(h * scale), int(w * scale)]
            res_data = [lycon.resize(x, width=size[1], height=size[0], interpolation=lycon.Interpolation.CUBIC) for x in vol_mem]
            it_vol_mem = np.stack((res_data), 0)
        else:
            it_vol_mem = vol_mem
        seg, prob = neurite_segmentation.get_segmentation(
            vol=it_vol_mem,
            ffn_ckpt=ffn_ckpt,
            # seed_policy="PolicyMembraneAndPeaks",
            mem_seed_thresh=mem_seed_threshold,
            move_threshold=move_threshold,
            segment_threshold=segment_threshold,
            ffn_model=ffn_model)  # Takes uint8 inputs
        segs.append(seg)
        probs.append(prob)
    if len(probs) > 1:
        # Resegment the average probability maps
        probs = np.stack(probs).mean(0)
    else:
        segs = segs[0]

    # Resize and transpose segments
    # segs = resize(seg.transpose(1, 2, 0), image_shape[::-1][:-1], anti_aliasing=False, preserve_range=True, order=1)
    segs = segs.transpose(1, 2, 0)  # Keep low-res for post-processing
    # from matplotlib import pyplot as plt;plt.subplot(131);plt.imshow(cube_in[..., 32]);plt.subplot(132);plt.imshow(vol_mem[32]);plt.subplot(133);plt.imshow(segs[..., 32]);plt.show()
    np.save(seg_path, segs)

    if not force_coord:
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

