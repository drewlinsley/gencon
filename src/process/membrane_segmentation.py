import os
import time
import argparse
import itertools
import numpy as np
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
# import staintools
from skimage.transform import resize


AUGS = ['uniform', 'pixel', 'rot270', 'rot90', 'rot180', 'flip_left', 'flip_right', 'depth_flip']  # ['pixel']  # 'depth_flip']  # , 'rot90', 'rot180', 'rot270']
AUGS = ['flip_left', 'flip_right', 'depth_flip']
AUGS = []
ROTS = []  # ['rot90', 'rot180']  # 'rot90', 'rot180', 'rot270']
TEST_TIME_AUGS = functools.reduce(
    lambda x, y: list(
        itertools.combinations(AUGS, y)) + x,
    list(range(len(AUGS) + 1)), [])[:-1]

PAUGS = deepcopy(TEST_TIME_AUGS)
for rot in ROTS:
    it_augs = []
    for idx in range(len(TEST_TIME_AUGS)):
        ita = list(TEST_TIME_AUGS[idx])
        if 'depth_flip' not in ita:
            it_augs += [[rot] + ita]
    PAUGS += it_augs
TEST_TIME_AUGS = [list(p) for p in PAUGS]


def augment(vo, augs):
    """Augment volume with augmentation au."""
    for au in augs:
        if au == 'rot90':
            vo = np.rot90(vo, 1, (2, 3))
        elif au == 'rot180':
            vo = np.rot90(vo, 2, (2, 3))
        elif au == 'rot270':
            vo = np.rot90(vo, 3, (2, 3))
        elif au == 'lr_flip':
            vo = vo[..., ::-1]
        elif au == 'ud_flip':
            vo = vo[..., ::-1, :]
        elif au == 'depth_flip':
            vo = vo[:, ::-1]
        elif au == 'noise':
            vo += np.random.rand(*vo.shape) * 1e-1
            vo = np.clip(vo, 0, 1)
    return vo


def undo_augment(vo, augs, debug_mem=None):
    """Augment volume with augmentation au."""
    for au in augs:
        if au == 'rot90':
            vo = np.rot90(vo, -1, (2, 3))
        elif au == 'rot180':
            vo = np.rot90(vo, -2, (2, 3))
        elif au == 'rot270':
            vo = np.rot90(vo, -3, (2, 3))
        elif au == 'lr_flip':
            vo = vo[..., ::-1, :]  # Note: 3-channel volumes
        elif au == 'ud_flip':
            vo = vo[..., ::-1, :, :]
        elif au == 'depth_flip':
            vo = vo[:, ::-1]
        elif au == 'noise':
            pass
    return vo


def get_segmentation(
        vol,
        membrane_ckpt,
        normalize=True,
        membrane_slice=[64, 384, 384],  # 576
        membrane_overlap_factor=[0.5, 0.5, 0.5],  # [0.875, 2./3., 2./3.],
        aggregate_membranes=False,
        res_shape=None,
        crop_shape=None,
        z_shape=None):
    """Apply the FFN routines using fGRUs."""
    # TEST_TIME_AUGS = None
    vol_shape = np.asarray([x for x in vol.shape])
    model_shape = vol_shape

    # Normalize and prep the volume
    if normalize:
        vol = vol.astype(np.float32) / 255.
    if crop_shape is not None:
        vol = vol[:, :crop_shape[0], :crop_shape[1]]
    _vol = vol.shape

    # Predict its membranes
    if membrane_slice:
        assert isinstance(
            membrane_slice, list), 'Make membrane_slice a list.'
        # Split up membrane along z-axis into k-voxel chunks
        # Include an overlap so that you have 1 extra slice per dim
        adj_membrane_slice = (np.array(
            membrane_slice) * membrane_overlap_factor).astype(int)
        z_splits = np.arange(
            adj_membrane_slice[0],
            model_shape[0],
            adj_membrane_slice[0])
        y_splits = np.arange(
            adj_membrane_slice[1],
            model_shape[1],
            adj_membrane_slice[1])
        x_splits = np.arange(
            adj_membrane_slice[2],
            model_shape[2],
            adj_membrane_slice[2])
        vols = []
        for z_idx in z_splits:
            for y_idx in y_splits:
                for x_idx in x_splits:
                    if z_idx == 0:
                        zu = z_idx
                        zo = z_idx + membrane_slice[0]
                    else:
                        zu = z_idx - adj_membrane_slice[0]
                        zo = zu + membrane_slice[0]
                    if y_idx == 0:
                        yu = y_idx
                        yo = y_idx + membrane_slice[1]
                    else:
                        yu = y_idx - adj_membrane_slice[1]
                        yo = yu + membrane_slice[1]
                    if x_idx == 0:
                        xu = x_idx
                        xo = x_idx + membrane_slice[2]
                    else:
                        xu = x_idx - adj_membrane_slice[2]
                        xo = xu + membrane_slice[2]
                    vols += [vol[
                        zu.astype(int): zo.astype(int),
                        yu.astype(int): yo.astype(int),
                        xu.astype(int): xo.astype(int)]]
        try:
            vol_stack = np.stack(vols)
        except Exception:
            print((
                'Mismatch in volume_size/membrane slicing {}'.format(
                    [vs.shape for vs in vols])))
            os._exit(1)
        del vols  # Garbage collect
        membranes = fgru.main(
            test=vol_stack,
            evaluate=True,
            adabn=True,
            gpu_device='/gpu:0',
            test_input_shape=np.concatenate((
                membrane_slice, [1])).tolist(),
            test_label_shape=np.concatenate((
                membrane_slice, [3])).tolist(),
            checkpoint=membrane_ckpt)
        if aggregate_membranes:
            membranes = [mbr.max(-1) for mbr in membranes]  # Save some memory on the concatenation
        else:
            membranes = [mbr[..., :3] for mbr in membranes]  # Save some memory on the concatenation
        membranes = np.concatenate(membranes, 0)

    if membrane_slice:  #  is not None:
        # Reconstruct, accounting for overlap
        # membrane_model_shape = tuple(list(_vol) + [3])
        if aggregate_membranes:
            rmembranes = np.zeros(_vol, dtype=np.float32)
        else:
            rmembranes = np.zeros(list(_vol) + [3], dtype=np.float32)
        count = 0
        vols = []
        normalization = np.zeros_like(rmembranes)
        if TEST_TIME_AUGS is not None:
            bump_map = _bump_logit_map(membranes[count].shape)
            bump_map = 1 - bump_map / bump_map.min()
        else:
            bump_map = 1.
        for z_idx in z_splits:
            for y_idx in y_splits:
                for x_idx in x_splits:
                    if z_idx == 0:
                        zu = z_idx
                        zo = z_idx + membrane_slice[0]
                    else:
                        zu = z_idx - adj_membrane_slice[0]
                        zo = zu + membrane_slice[0]
                    if y_idx == 0:
                        yu = y_idx
                        yo = y_idx + membrane_slice[1]
                    else:
                        yu = y_idx - adj_membrane_slice[1]
                        yo = yu + membrane_slice[1]
                    if x_idx == 0:
                        xu = x_idx
                        xo = x_idx + membrane_slice[2]
                    else:
                        xu = x_idx - adj_membrane_slice[2]
                        xo = xu + membrane_slice[2]
                    rmembranes[
                        zu: zo,
                        yu: yo,
                        xu: xo] += membranes[count] * bump_map
                    if normalization is not None:
                        normalization[
                            zu: zo,
                            yu: yo,
                            xu: xo] += bump_map  # 1.
                    count += 1
        if normalization is not None:
            rmembranes = rmembranes / normalization
            rmembranes = np.nan_to_num(rmembranes, nan=0., copy=False)
            rmembranes[np.isnan(rmembranes)] = 0.
        membranes = rmembranes  # [None]
    else:
        membranes = fgru.main(
            test=vol,
            evaluate=True,
            adabn=True,
            gpu_device='/gpu:0',
            test_input_shape=np.concatenate((
                model_shape, [1])).tolist(),
            test_label_shape=np.concatenate((
                model_shape, [3])).tolist(),
            checkpoint=membrane_ckpt)
        if aggregate_membranes:
            membranes = np.concatenate(membranes, 0).max(-1)  # mean
        else:
            membranes = np.concatenate(membranes, 0)
    if aggregate_membranes:
        membranes = np.stack(
            (vol, membranes), axis=-1).astype(np.float32) * 255.
    else:
        membranes = np.concatenate(
            (vol[..., None], membranes), axis=-1).astype(np.float32) * 255.
    # np.save(mpath, membranes)
    # from matplotlib import pyplot as plt; slc = 32;plt.subplot(121);plt.imshow(vol[slc]);plt.subplot(122);plt.imshow(membranes[slc, ..., 1]);plt.show()
    # print('Saved membrane volume to %s' % mpath)
    return membranes

