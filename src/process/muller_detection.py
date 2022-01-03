import os
import sys
import time
import argparse
import numpy as np
from membrane.models import l3_fgru_constr_adabn_synapse as unet
from tqdm import tqdm
from omegaconf import OmegaConf


def get_segmentation(
        vol,
        ckpt_path,
        normalize=True,
        stride=[60, 192, 192],  # Stride
        cube_size=[120, 384, 384],
        padded=4,  # (8, , ),  # Half-Pad size
        device="/gpu:0"):
    """Apply the FFN routines using fGRUs."""
    vol_shape = vol.shape
    model_shape = cube_size  # list(vol.shape)
    if padded:
        model_shape = [x + padded * 2 for x in model_shape]
    model_shape = model_shape + [2]
    label_shape = np.copy(model_shape)
    label_shape[-1] = 1

    # FOR SYNAPSE, NORMALIZE VOL
    if normalize:
        vol /= 255.

    # Loop through volume
    d_ind_start = np.arange(0, vol_shape[0], stride[0]).astype(int)[:-1]
    h_ind_start = np.arange(0, vol_shape[1], stride[1]).astype(int)[:-1]
    w_ind_start = np.arange(0, vol_shape[2], stride[2]).astype(int)[:-1]
    print(d_ind_start, h_ind_start, w_ind_start)
    ribbons = np.zeros(vol_shape[:-1], dtype=np.float32)
    num_completed = 0
    for d_start in tqdm(d_ind_start, desc="Depth coordinates", total=len(d_ind_start)):
        d_end = d_start + cube_size[0]
        for h_start in h_ind_start:
            h_end = h_start + cube_size[1]
            for w_start in w_ind_start:
                w_end = w_start + cube_size[2]
                cube = vol[d_start: d_end, h_start: h_end, w_start: w_end]
                if padded:
                    cube = np.pad(cube, ((padded, padded), (padded, padded), (padded, padded), (0, 0)), "reflect")
                if num_completed == 0:
                    test_dict, sess = unet.main_cell_type(
                        train_input_shape=model_shape,  # [z for z in model_shape],
                        train_label_shape=label_shape,  # [z for z in label_shape],
                        test_input_shape=model_shape,  # [z for z in model_shape],
                        test_label_shape=label_shape,  # [z for z in label_shape],
                        checkpoint=ckpt_path,
                        return_sess=True,
                        return_restore_saver=False,
                        force_return_model=True,
                        evaluate_cell_type=True,
                        gpu_device=device)
                feed_dict = {
                    test_dict['test_images']: cube[None] * 255.,
                }
                it_test_dict = sess.run(
                    test_dict,
                    feed_dict=feed_dict)
                preds = it_test_dict['test_logits'].squeeze()
                if padded:
                    preds = preds[padded: -padded, padded: -padded, padded: -padded]
                # preds = preds[..., None]  # Adjust shape for specialist prediction case
                ribbons[d_start: d_end, h_start: h_end, w_start: w_end] = np.maximum(
                    ribbons[d_start: d_end, h_start: h_end, w_start: w_end],
                    preds)
                num_completed += 1
    # from matplotlib import pyplot as plt;plt.imshow(ribbons[32]);plt.show() 
    print("Max value is {}".format(ribbons.max()))
    return ribbons


if __name__ == '__main__':
    conf = "configs/W-Q.yml"
    conf = OmegaConf.load(conf)
    x, y, z = 0, 0, 0
    vol = np.load("/media/data_cifs/projects/prj_connectomics/wong/mag1_membranes/x{}/y{}/z{}/110629_k0725_mag1_x{}_y{}_z{}.npy".format(x, y, z, x, y, z))
    ckpt_path = conf.inference.muller_ckpt
    seg = get_segmentation(vol, ckpt_path)
    path = conf.storage.muller_path_str.format(x, y, z, x, y, z)
    np.save(path, seg)

