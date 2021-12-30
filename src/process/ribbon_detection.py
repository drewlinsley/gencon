import os
import sys
import time
import argparse
import numpy as np
from membrane.models import l3_fgru_constr_adabn_synapse as unet
from tqdm import tqdm
from omegaconf import OmegaConf


def process_cubes(cubes, ckpt_path, device, coords, padded=False):
    """Get synapse preds for cubes."""
    model_shape = list(cubes[0].shape)
    debug_shape = list(vol.shape)
    debug_shape[-1] = 1
    debug_vol = np.zeros(debug_shape, np.float32)  # Adjusted for single channel predictions

    label_shape = np.copy(model_shape)
    label_shape[-1] = 1

    synapses = []
    for num_completed, (cube, dcoords) in enumerate(zip(cubes, coords)):
        if num_completed == 0:
            test_dict, sess = unet.main(
                train_input_shape=model_shape,  # [z for z in model_shape],
                train_label_shape=label_shape,  # [z for z in label_shape],
                test_input_shape=model_shape,  # [z for z in model_shape],
                test_label_shape=label_shape,  # [z for z in label_shape],
                checkpoint=ckpt_path,
                return_sess=True,
                return_restore_saver=False,
                force_return_model=True,
                evaluate=True,
                gpu_device=device)
        feed_dict = {
            test_dict['test_images']: cube[None],
        }
        it_test_dict = sess.run(
            test_dict,
            feed_dict=feed_dict)
        preds = it_test_dict['test_logits'].squeeze()
        if padded:
            preds = preds[padded: -padded, padded: -padded, padded: -padded]
        preds = preds[..., None]  # Adjust shape for specialist prediction case
        debug_vol[
            dcoords['d_s']: dcoords['d_e'],
            dcoords['h_s']: dcoords['h_e'],
            dcoords['w_s']: dcoords['w_e']] = preds
    debug_vol = debug_vol.squeeze()
    return debug_vol


def cube_data(vol, model_shape, divs, padded=False):
    """Chunk up data into cubes for processing separately."""
    # Reshape vol into 9 cubes and process each
    cubes = []
    assert model_shape[1] / divs[1] == np.round(model_shape[1] / divs[1])
    d_ind_start = np.arange(0, model_shape[0], model_shape[0] / divs[0]).astype(int)
    h_ind_start = np.arange(0, model_shape[1], model_shape[1] / divs[1]).astype(int)
    w_ind_start = np.arange(0, model_shape[2], model_shape[2] / divs[2]).astype(int)
    d_ind_end = d_ind_start + model_shape[0] / divs[0]
    h_ind_end = h_ind_start + model_shape[1] / divs[1]
    w_ind_end = w_ind_start + model_shape[2] / divs[2]
    d_ind_end = d_ind_end.astype(int)
    h_ind_end = h_ind_end.astype(int)
    w_ind_end = w_ind_end.astype(int)

    debug_coords = []
    for d_s, d_e in zip(d_ind_start, d_ind_end):
        for h_s, h_e in zip(h_ind_start, h_ind_end):
            for w_s, w_e in zip(w_ind_start, w_ind_end):
                it_cube = vol[d_s: d_e, h_s: h_e, w_s: w_e]
                if padded:
                    it_cube = np.pad(it_cube, ((padded, padded), (padded, padded), (padded, padded), (0, 0)), "reflect")
                cubes += [it_cube]
                debug_coords += [
                    {
                        'd_s': d_s,
                        'd_e': d_e,
                        'h_s': h_s,
                        'h_e': h_e,
                        'w_s': w_s,
                        'w_e': w_e
                    }
                ]
    return cubes, debug_coords


def get_segmentation(
        vol,
        ckpt_path,
        normalize=True,
        divs=[2, 4, 4],
        padded=8,  # (8, , ),  # Half-Pad size
        device="/gpu:0"):
    """Apply the FFN routines using fGRUs."""
    model_shape = list(vol.shape)

    # FOR SYNAPSE, NORMALIZE VOL
    if normalize:
        vol /= 255.

    cubes, coords = cube_data(vol=vol, model_shape=model_shape, divs=divs, padded=padded)
    ribbons = process_cubes(
            cubes=cubes,
            ckpt_path=ckpt_path,
            device=device,
            coords=coords,
            padded=padded)
    return ribbons


if __name__ == '__main__':
    conf = "configs/W-Q.yml"
    conf = OmegaConf.load(conf)
    x, y, z = 0, 0, 0
    vol = np.load("/media/data_cifs/projects/prj_connectomics/wong/mag1_membranes/x{}/y{}/z{}/110629_k0725_mag1_x{}_y{}_z{}.npy".format(x, y, z, x, y, z))
    ckpt_path = conf.ribbon_ckpt
    seg = get_segmentation(vol, ckpt_path)

