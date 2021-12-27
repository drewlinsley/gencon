import os
import time
import logging
import argparse
import itertools
import nibabel as nib
import numpy as np
from config import Config
from google.protobuf import text_format
from ffn.inference import inference
from ffn.inference import inference_pb2
from membrane.models import l3_fgru_constr as fgru
from utils.hybrid_utils import recursive_make_dir
from utils.hybrid_utils import pad_zeros
from utils.hybrid_utils import _bump_logit_map
from utils.hybrid_utils import rdirs
from copy import deepcopy
from tqdm import tqdm
import functools
import staintools
from skimage.transform import resize


logger = logging.getLogger()
logger.setLevel(logging.INFO)
AUGS = ['uniform', 'pixel', 'rot270', 'rot90', 'rot180', 'flip_left', 'flip_right', 'depth_flip']  # ['pixel']  # 'depth_flip']  # , 'rot90', 'rot180', 'rot270']
AUGS = ['flip_left', 'flip_right', 'depth_flip']
AUGS = []
ROTS = []  # ['rot90', 'rot180']  # 'rot90', 'rot180', 'rot270']
TEST_TIME_AUGS = functools.reduce(
    lambda x, y: list(
        itertools.combinations(AUGS, y)) + x,
    list(range(len(AUGS) + 1)), [])[:-1]
# PAUGS = []
# for aug in TEST_TIME_AUGS:
#     t = np.array([1 if 'rot' in x else 0 for x in TEST_TIME_AUGS]).sum()
#     if t <= 1:
#         PAUGS += [aug]
# TEST_TIME_AUGS = PAUGS
PAUGS = deepcopy(TEST_TIME_AUGS)
for rot in ROTS:
    it_augs = []
    for idx in range(len(TEST_TIME_AUGS)):
        ita = list(TEST_TIME_AUGS[idx])
        if 'depth_flip' not in ita:
            it_augs += [[rot] + ita]
    PAUGS += it_augs
TEST_TIME_AUGS = [list(p) for p in PAUGS]


def get_membranes_nii(config, seed, path_extent, pull_from_db, return_membrane=False):
    if not pull_from_db:
        seed = seed
    else:
        seed = db.get_next_synapse_coordinate()
        if seed is None:
            raise RuntimeError('No more coordinantes to process!')

    coords, idxs = [], []
    empty = False
    vol = np.zeros((np.array(config.shape) * np.array(path_extent)), dtype=np.float32)  # shape * path_extent
    test_vol = np.zeros((np.array(config.shape) * np.array(path_extent)), dtype=np.float32)
    shape = config.shape
    membrane = True
    for z in range(path_extent[0]):
        for y in range(path_extent[1]):
            for x in range(path_extent[2]):
                for dr in config.mem_dirs:
                    # coord = [seed['x'] + x, seed['y'] + y, seed['z'] + z]
                    coord = [seed['x'] + x, seed['y'] + y, seed['z'] + z]
                    vp = config.path_str.replace("%s", "{}")
                    vp = vp.format(pad_zeros(coord[0], 4), pad_zeros(coord[1], 4), pad_zeros(coord[2], 4), pad_zeros(coord[0], 4), pad_zeros(coord[1], 4), pad_zeros(coord[2], 4))
                    ims = np.fromfile(vp, dtype=np.uint8).reshape(128, 128, 128)
                    test_vol[
                        z * shape[0]: z * shape[0] + shape[0],  # nopep8
                        y * shape[1]: y * shape[1] + shape[1],  # nopep8
                        x * shape[2]: x * shape[2] + shape[2]] = ims
                    if "%s" in dr:
                        dr = dr.replace("%s", "{}")
                        tp = dr.format(pad_zeros(coord[0], 4), pad_zeros(coord[1], 4), pad_zeros(coord[2], 4), pad_zeros(coord[0], 4), pad_zeros(coord[1], 4), pad_zeros(coord[2], 4))
                    else:
                        tp = os.path.join(
                            dr,
                            "x{}".format(pad_zeros(coord[0], 4)),
                            "y{}".format(pad_zeros(coord[1], 4)),
                            "z{}".format(pad_zeros(coord[2], 4)),
                            "110629_k0725_mag1_x{}_y{}_z{}.nii".format(
                                pad_zeros(coord[0], 4),
                                pad_zeros(coord[1], 4),
                                pad_zeros(coord[2], 4)))
                    if os.path.exists(tp):
                        zp = nib.load(tp)
                        h = zp.dataobj
                        v = h.get_unscaled()
                        zp.uncache()
                        del zp, h
                        vol[
                            z * shape[0]: z * shape[0] + shape[0],  # nopep8
                            y * shape[1]: y * shape[1] + shape[1],  # nopep8
                            x * shape[2]: x * shape[2] + shape[2]] = v
                        break
                    else:
                        membrane = None
    if membrane is not None:
        # vol = np.zeros((np.array(config.shape) * np.array(path_extent)), dtype=np.float32)  # shape * path_extent
        # membrane = build_vol(vol=vol, vols=vols, coords=idxs, shape=config.shape)
        membrane = vol
        membrane[np.isnan(membrane)] = 0
        assert membrane.max() > 1, 'Membrane is scaled to [0, 1]. Fix this!'
        if return_membrane:
            return membrane
    else:
        return False


def get_membranes(config, seed, pull_from_db, return_membrane=False):
    if not pull_from_db:
        seed = seed
    else:
        seed = db.get_next_synapse_coordinate()
        if seed is None:
            raise RuntimeError('No more coordinantes to process!')
    try:
        path = config.read_mem_str % (
            pad_zeros(seed['x'], 4),
            pad_zeros(seed['y'], 4),
            pad_zeros(seed['z'], 4),
            pad_zeros(seed['x'], 4),
            pad_zeros(seed['y'], 4),
            pad_zeros(seed['z'], 4))
        membrane = np.load('{}.npy'.format(path))
    except:
        path = config.write_mem_str % (
            pad_zeros(seed['x'], 4),
            pad_zeros(seed['y'], 4),
            pad_zeros(seed['z'], 4),
            pad_zeros(seed['x'], 4),
            pad_zeros(seed['y'], 4),
            pad_zeros(seed['z'], 4))
        membrane = np.load('{}.npy'.format(path))
    assert membrane.max() > 1  # , 'Membrane is scaled to [0, 1]. Fix this!'
    if return_membrane:
        return membrane
    # Check vol/membrane scale
    # vol = (vol / 255.).astype(np.float32)
    membrane[np.isnan(membrane)] = 0.
    vol = np.stack((vol, membrane), -1)[None] / 255.
    return vol, None


def augment(vo, augs):
    """Augment volume with augmentation au."""
    for au in augs:
        if au is 'rot90':
            vo = np.rot90(vo, 1, (2, 3))
        elif au is 'rot180':
            vo = np.rot90(vo, 2, (2, 3))
        elif au is 'rot270':
            vo = np.rot90(vo, 3, (2, 3))
        elif au is 'lr_flip':
            vo = vo[..., ::-1]
        elif au is 'ud_flip':
            vo = vo[..., ::-1, :]
        elif au is 'depth_flip':
            vo = vo[:, ::-1]
        elif au is 'noise':
            vo += np.random.rand(*vo.shape) * 1e-1
            vo = np.clip(vo, 0, 1)
    return vo


def undo_augment(vo, augs, debug_mem=None):
    """Augment volume with augmentation au."""
    for au in augs:
        if au is 'rot90':
            vo = np.rot90(vo, -1, (2, 3))
        elif au is 'rot180':
            vo = np.rot90(vo, -2, (2, 3))
        elif au is 'rot270':
            vo = np.rot90(vo, -3, (2, 3))
        elif au is 'lr_flip':
            vo = vo[..., ::-1, :]  # Note: 3-channel volumes
        elif au is 'ud_flip':
            vo = vo[..., ::-1, :, :]
        elif au is 'depth_flip':
            vo = vo[:, ::-1]
        elif au is 'noise':
            pass
    return vo


def get_segmentation(
        idx,
        data_path=None,
        move_threshold=None,  # 0.7,
        segment_threshold=None,  # 0.5,
        validate=False,
        seed=None,
        savetype='.nii',
        shift_z=None,
        shift_y=None,
        shift_x=None,
        x=None,
        y=None,
        z=None,
        membrane_type='probability',
        ffn_transpose=(0, 1, 2),
        prev_coordinate=None,
        membrane_only=False,
        segment_only=False,
        merge_segment_only=False,
        seg_vol=None,
        voxel_size=None,
        deltas='[15, 15, 3]',  # '[27, 27, 6]'
        target_voxel=np.asarray([13.2, 13.2, 26]),
        seed_policy='PolicyMembrane',  # 'PolicyPeaks'
        membrane_slice=[64, 384, 384],  # 576
        membrane_overlap_factor=[0.5, 0.5, 0.5],  # [0.875, 2./3., 2./3.],
        res_shape=None,
        z_shape=None,
        path_extent=None,  # [1, 1, 1],
        rotate=False):
    """Apply the FFN routines using fGRUs."""
    # TEST_TIME_AUGS = None
    config = Config()
    assert move_threshold is not None
    assert segment_threshold is not None
    assert voxel_size is not None
    voxel_size = np.asarray([int(x) for x in voxel_size.split(",")])
    path_extent = np.asarray([int(x) for x in path_extent.split(",")])
    model_shape = (config.shape * path_extent)
    mpath = '/localscratch/middle_cube_membranes_for_ffn_training'
    membrane_slice = None  # [60, 120*2, 120*2]

    # Move this to the DB
    vol = np.load("/media/data_cifs/connectomics/datasets/middle_cube.npy")  # filtered_wong_berson.npz")
    vol_shape = np.asarray([x for x in vol.shape])

    # Figure out how to resize from voxel_size -> target_size
    if res_shape is None:
        target_ratio = float(target_voxel[-1]) / float(target_voxel[0])
        inp_ratio = float(voxel_size[-1]) / float(voxel_size[0])
        mod = target_ratio / inp_ratio
        res_shape = (vol_shape[1:] * mod).astype(int)
        if res_shape[0] % 2:
            depth = 3
            check = True
            while check:
                res_shape -= 1
                split = res_shape[0] / 2
                for idx in range(2):
                    if np.floor(split) == np.ceil(split):
                        split /= 2
                    else:
                        continue
                    if idx == 1:
                        check = False
    if z_shape is None:
        z_shape = res_shape[0]  # set to a reasonable value to process a bunch at a time
    crop_shape = None  # [640, 640]
    if crop_shape is not None:
        config.shape = np.asarray([z_shape] + crop_shape)
    else:
        config.shape = np.asarray([z_shape] + res_shape.tolist())
    model_shape = config.shape

    # Resize and process the volume
    vol = resize(vol.transpose(1, 2, 0), res_shape, anti_aliasing=True, preserve_range=True, order=3).transpose(2, 0, 1)
    vol = vol.astype(np.float32) / 255.
    vol = vol[:z_shape]
    if crop_shape is not None:
        vol = vol[:, :crop_shape[0], :crop_shape[1]]
    _vol = vol.shape
    print(('seed: %s' % seed))
    print(('mpath: %s' % mpath))
    print(('volume size: (%s, %s, %s)' % (
        _vol[0],
        _vol[1],
        _vol[2])))

    # 2. Predict its membranes
    membranes = fgru.main(
        test=vol,
        evaluate=True,
        adabn=True,
        gpu_device='/gpu:0',
        test_input_shape=np.concatenate((
            model_shape, [1])).tolist(),
        test_label_shape=np.concatenate((
            model_shape, [3])).tolist(),
        checkpoint=config.membrane_ckpt)
    membranes = np.concatenate(membranes, 0).max(-1)  # mean

    vol = vol.transpose(ffn_transpose)  # ).astype(np.uint8)
    membranes = np.stack(
        (vol, membranes), axis=-1).astype(np.float32) * 255.
    np.save(mpath, membranes)
    # from matplotlib import pyplot as plt; slc = 32;plt.subplot(121);plt.imshow(vol[slc]);plt.subplot(122);plt.imshow(membranes[slc, ..., 1]);plt.show()
    print('Saved membrane volume to %s' % mpath)
    return membranes


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--idx',
        dest='idx',
        type=int,
        default=0,
        help='Segmentation version.')
    parser.add_argument(
        '--seed',
        dest='seed',
        type=str,
        default='14,15,18',
        help='Center volume for segmentation.')
    parser.add_argument(
        '--voxel_size',
        dest='voxel_size',
        type=str,
        default='5,5,40',
        help='Center volume for segmentation.')
    parser.add_argument(
        '--res_shape',
        dest='res_shape',
        type=str,
        default=None,
        help='Shape for resizing x/y.')
    parser.add_argument(
        '--z_shape',
        dest='z_shape',
        type=int,
        default=128,
        help='Number of z-slices to segment.')
    parser.add_argument(
        '--seed_policy',
        dest='seed_policy',
        type=str,
        default='PolicyMembrane',
        help='Policy for finding FFN seeds.')
    parser.add_argument(
        '--path_extent',
        dest='path_extent',
        type=str,
        default='1,2,6',
        help='Provide extent of segmentation in 128^3 volumes.')
    parser.add_argument(
        '--segment_threshold',
        dest='segment_threshold',
        type=float,
        default=0.6,  # 0.6
        help='Segment threshold..')
    parser.add_argument(
        '--membrane_slice',
        dest='membrane_slice',
        type=str,
        default=None,
        help='Membrane chunking along z axis.')
    parser.add_argument(
        '--rotate',
        dest='rotate',
        action='store_true',
        help='Rotate the input data.')
    args = parser.parse_args()
    start = time.time()
    get_segmentation(**vars(args))
    end = time.time()
    print(('Segmentation took {}'.format(end - start)))

