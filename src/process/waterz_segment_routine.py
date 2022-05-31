import sys
import os
from src.process import neurite_segmentation
from src.process import membrane_segmentation
from skimage.transform import resize
from skimage.segmentation import relabel_sequential as rs
from skimage.morphology import remove_small_objects
import numpy as np
from db import db
from omegaconf import OmegaConf
from matplotlib import pyplot as plt
import cc3d
import waterz


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
    pp_seg_path = conf.storage.postproc_seg_path_str.format(x, y, z, x, y, z)
    os.makedirs(os.path.sep.join(mem_path.split(os.path.sep)[:-1]), exist_ok=True)
    os.makedirs(os.path.sep.join(seg_path.split(os.path.sep)[:-1]), exist_ok=True)
    os.makedirs(os.path.sep.join(pp_seg_path.split(os.path.sep)[:-1]), exist_ok=True)

    # Get images
    assert os.path.exists("{}.npy".format(mem_path)), "For Watez seg you must first process membranes in a webknossos-compatable env."
    vol_mem = np.load("{}.npy".format(mem_path))

    # Get segmentations
    assert os.path.exists("{}.npy".format(seg_path)), "For Watez seg you must first process instance segs in a webknossos-compatable env."
    segs = np.load("{}.npy".format(seg_path))
    segs = segs.transpose(2, 0, 1)

    # Waterz seg the membranes to fill in gaps
    mems = vol_mem[..., 1:]
    seg_gt = None # segmentation ground truth. If available, the prediction is evaluated against this ground truth and Rand and VI scores are produced.
    aff_thresholds = [0.01, 0.1, 0.2, 0.3]
    seg_thresholds = [0.1]
    w = waterz.waterz(
        mems.transpose(3, 0, 1, 2).astype(np.float32) / 255.,
        seg_thresholds,
        merge_function='aff50_his256',
        aff_threshold=aff_thresholds)

    # Combine the two
    segs = rs(segs)[0]
    insertion = (segs == 0).astype(np.uint32)
    fixed_seg = (w[0] * insertion)
    fixed_seg = fixed_seg + ((fixed_seg != 0).astype(np.uint32) * segs.max())
    fixed_seg = fixed_seg + segs

    # Now postprocess
    # plt.subplot(121);plt.imshow(fixed_seg[120]);plt.subplot(122);plt.imshow(cc3d.connected_components(fixed_seg, connectivity=26)[120]);plt.show()
    fixed_seg = cc3d.connected_components(fixed_seg, connectivity=26)
    fixed_seg = remove_small_objects(fixed_seg, min_size=100, connectivity=26)

    # Resize and transpose segments
    fixed_seg = fixed_seg.transpose(1, 2, 0)  # Keep low-res for post-processing
    np.save(pp_seg_path, fixed_seg)

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

