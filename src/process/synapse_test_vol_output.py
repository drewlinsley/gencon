import os
import time
import logging
import argparse
import numpy as np
from db import db
from membrane.models import seung_unet3d_adabn_small as unet
from membrane.models import l3_fgru_constr_adabn_synapse as unet

from utils.hybrid_utils import pad_zeros, make_dir
from skimage.measure import label, regionprops
from skimage.feature import peak_local_max
from skimage.morphology import remove_small_objects, closing, erosion, ball, local_maxima
from skimage.segmentation import relabel_sequential
from utils.hybrid_utils import recursive_make_dir as rdirs
from tqdm import tqdm


def non_max_suppression(vol, window, verbose=False):
    """Perform non_max suppression on a 3d volume."""
    vs = vol.shape
    nms_vol = np.zeros_like(vol)
    z_ind = np.arange(0, vs[0], window[0])
    y_ind = np.arange(0, vs[1], window[1])
    x_ind = np.arange(0, vs[2], window[2])
    if verbose:
        for z in tqdm(z_ind, desc="NMS", total=len(z_ind)):
            for y in y_ind:
                for x in x_ind:
                    window_z = np.minimum(window[0], vs[0] - z)
                    window_y = np.minimum(window[1], vs[1] - y)
                    window_x = np.minimum(window[2], vs[2] - x)
                    sample = vol[z: z + window_z, y: y + window_y, x: x + window_x]
                    sample_max = np.max(sample)
                    if sample_max != 0:
                        sample_idx = np.where(sample == sample_max)
                        sample_idx = np.asarray((sample_idx[0][0], sample_idx[1][0], sample_idx[2][0])).astype(int)
                        nms_vol[sample_idx[0] + z, sample_idx[1] + y, sample_idx[2] + x] = sample_max
    else:
        for z in z_ind:
            for y in y_ind:
                for x in x_ind:
                    window_z = np.minimum(window[0], vs[0] - z)
                    window_y = np.minimum(window[1], vs[1] - y)
                    window_x = np.minimum(window[2], vs[2] - x)
                    sample = vol[z: z + window_z, y: y + window_y, x: x + window_x]
                    sample_max = np.max(sample)
                    if sample_max != 0:
                        sample_idx = np.where(sample == sample_max)
                        sample_idx = np.asarray((sample_idx[0][0], sample_idx[1][0], sample_idx[2][0])).astype(int)
                        nms_vol[sample_idx[0] + z, sample_idx[1] + y, sample_idx[2] + x] = sample_max
    return nms_vol


def process_cubes(cubes, debug_coords, debug, num_completed, seed, ribbons, amacrines, keep_processing, ckpt_path, device, config, vol, padded=False):
    """Get synapse preds for cubes."""
    if 1:
        model_shape = list(cubes[0].shape)
        if 1:  # debug:
            debug_shape = list(vol.shape)
            debug_shape[-1] = 1
            debug_vol = np.zeros(debug_shape, np.float32)  # Adjusted for single channel predictions
            # debug_vol = np.zeros(model_shape)
        else:
            debug_vol = None

        # Specialist models
        label_shape = np.copy(model_shape)
        label_shape[-1] = 1

        synapses = []
        for cube, dcoords in zip(cubes, debug_coords):
            if num_completed == 0:
                # preds, sess, test_dict = unet.main(
                #     test=cube[None],
                #     evaluate=True,
                #     adabn=True,
                #     return_sess=keep_processing,
                #     test_input_shape=model_shape,
                #     test_label_shape=label_shape,
                #     checkpoint=ckpt_path,
                #     gpu_device=device)
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
            num_completed += 1
    # debug_vol = debug_vol.squeeze().transpose(2, 1, 0)
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


def get_data(config, seed, pull_from_db, return_membrane=False, path_extent=[3, 9, 9]):
    vol = np.load(seed["path"])
    # vol /= 255.
    # Check vol/membrane scale
    # mem = vol[..., -1]
    # mem[np.isnan(mem)] = 0.
    # mem = np.stack((vol, mem), -1)
    # mem /= 255.
    # vol[..., -1] = mem
    return vol  # , None, seed


def process_preds(preds, config, offset, thresh=[0.51, 0.51], so_thresh=9, debug=False):
    """Extract likely synapse locations."""
    # Threshold and save results
    # Set threshold. Also potentially set
    # it to be lower for amacrine, with a WTA.
    # Using ribbon-predictions only right now
    # thresh_preds = np.clip(preds, thresh[0], 1.1)
    # thresh_preds[thresh_preds == thresh[0]] = 0.
    binary_preds = preds >= thresh[0]
    erosion_filter = ball(1)
    # binary_preds = erosion(binary_preds, erosion_filter)
    # binary_preds = closing(binary_preds, erosion_filter)
    thresh_pred_mask = remove_small_objects(binary_preds, so_thresh)
    # thresh_preds *= thresh_pred_mask
    preds *= thresh_pred_mask

    # Take max per 3d coordinate
    # peaks = peak_local_max(thresh_preds, min_distance=so_thresh)
    # label_img = label(thresh_preds > thresh[0], connectivity=thresh_preds.ndim)
    # label_img = label(preds > thresh[0])
    # props = regionprops(label_img)
    # peaks = np.asarray([np.asarray(x.centroid).round().astype(int) for x in props])
    peaks = local_maxima(preds, indices=False, connectivity=3)
    # peaks = np.asarray(peaks).T
    nms_peaks = non_max_suppression(vol=peaks * preds, window=(40, 40, 40))
    nms_peak_coords = np.asarray(np.where(nms_peaks != 0)).squeeze().T 

    if debug:
        debug_maxs = np.zeros_like(preds)
        for peak in nms_peak_coords:
            debug_maxs[peak[0], peak[1], peak[2]] = 1

        # from matplotlib import pyplot as plt
        # plt.subplot(131);plt.imshow(preds[50]);plt.subplot(132);plt.imshow(nms_peaks[45:55].sum(0) > 0);plt.subplot(133);plt.imshow(debug_maxs[45:55].sum(0)> 0);plt.show()

        import ipdb
        ipdb.set_trace()

    # # Add coords to the db
    synapses = []
    for s in nms_peak_coords:
        synapses += [
            {'x': s[0] + offset[0], 'y': s[1] + offset[1], 'z': s[2] + offset[2], 'size': 1, 'type': 'ribbon'}]  # noqa
    for s in nms_peak_coords:
        synapses += [
            {'x': s[0] + offset[0], 'y': s[1] + offset[1], 'z': s[2] + offset[2], 'size': 1, 'type': 'amacrine'}]  # noqa
    return synapses, len(synapses), len(synapses)  # len(ribbon_coords), len(amacrine_coords)


def process_preds_WTA(preds, config, offset, thresh=[0.6, 0.6], so_thresh=27):
    """Extract likely synapse locations."""
    # Threshold and save results
    # Set threshold. Also potentially set
    # it to be lower for amacrine, with a WTA.
    thresh_preds_r = np.clip(preds[..., 0], thresh[0], 1.1)
    thresh_preds_a = np.clip(preds[..., 1], thresh[1], 1.1)

    thresh_preds_r[thresh_preds_r <= thresh[0]] = 0.
    thresh_preds_a[thresh_preds_a <= thresh[1]] = 0.
    thresh_preds = np.stack((thresh_preds_r, thresh_preds_a), -1)
    thresh_pred_mask = remove_small_objects(thresh_preds > 0.5, so_thresh)
    thresh_preds *= thresh_pred_mask

    # Take max per 3d coordinate
    max_vals = np.max(thresh_preds, -1)
    argmax_vals = np.argmax(thresh_preds, -1)

    # Find peaks
    peaks = peak_local_max(max_vals, min_distance=28)
    # ids = relabel_sequential(thresh_pred_mask)[0]

    # Split into ribbon/amacrine
    ribbon_coords, amacrine_coords = [], []
    for p in peaks:
        ch = argmax_vals[p[0], p[1], p[2]]
        adj_p = offset + p
        # size = np.sum(ids[..., ch] == ids[p[0], p[1], p[2], ch])
        size = thresh_preds[p[0], p[1], p[2], ch]
        adj_p = np.concatenate((adj_p, [size]))
        if ch == 0:
            ribbon_coords += [adj_p]
        else:
            amacrine_coords += [adj_p]
    # # Take local max
    # ribbon_coords = peak_local_max(fixed_preds[..., 0], min_distance=32)
    # amacrine_coords = peak_local_max(fixed_preds[..., 1], min_distance=32)

    # # Add coords to the db
    # adjusted_ribbon_coordinates = [
    #     coord * config.shape for coord in ribbon_coords]
    # adjusted_amacrine_coordinates = [
    #     coord * config.shape for coord in amacrine_coords]
    synapses = []
    for s in ribbon_coords:
        synapses += [
            {'x': s[0], 'y': s[1], 'z': s[2], 'size': s[3], 'type': 'ribbon'}]  # noqa
    for s in amacrine_coords:
        synapses += [
            {'x': s[0], 'y': s[1], 'z': s[2], 'size': s[3], 'type': 'amacrine'}]  # noqa
    return synapses, len(ribbon_coords), len(amacrine_coords)


def test(
        ffn_transpose=(0, 1, 2),
        cube_size=128,
        output_dir='synapse_predictions_v0',
        # ckpt_path='new_synapse_checkpoints_new_dataloader_smaller_weight/30000/30000-30000.ckpt',  # noqa
        # ckpt_path='new_synapse_checkpoints_new_dataloader_smaller_weight/-65000.ckpt',  # noqa
        # ckpt_path='new_synapse_checkpoints_new_dataloader_smaller_weight/-85000.ckpt',  # noqa
        # ckpt_path='/cifs/data/tserre_lrs/projects/prj_connectomics/ffn_membrane_v2/synapse_fgru_ckpts/synapse_fgru_ckpts-165000',
        ckpt_path="/cifs/data/tserre/CLPS_Serre_Lab/projects/prj_connectomics/ffn_membrane_v2/synapse_fgru_ckpts/synapse_fgru_v2_ckpts-85000",
        paths='/media/data_cifs/connectomics/membrane_paths.npy',
        pull_from_db=False,
        keep_processing=0,
        path_extent=None,
        save_preds=False,
        divs=[2, 4, 4],
        debug=False,
        out_dir=None,
        segmentation_path=None,
        finish_membranes=False,
        seed=(6, 7, 7),
        padded=8,  # (8, , ),  # Half-Pad size
        device="/gpu:0",
        rotate=False):
    """Apply the FFN routines using fGRUs."""
    config = Config()
    path_extent = np.array([int(s) for s in path_extent.split(',')])
    out_path = os.path.join(config.write_project_directory, output_dir)
    make_dir(out_path)
    num_completed, fixed_membranes = 0, 0
    ribbons = 0
    amacrines = 0
    if segmentation_path is not None:
        seed = np.asarray([int(x) for x in segmentation_path.split(",")])
        seed = {"x": seed[0], "y": seed[1], "z": seed[2]}
        try:
            vol, error, seed = get_data(
                seed=seed, pull_from_db=False, config=config)
        except Exception as e:
            print(e)
            sess, feed_dict, test_dict = get_data_or_process(
                seed=seed, pull_from_db=False, config=config)
        model_shape = list(vol.shape)
        cubes, debug_coords = cube_data(vol=vol, model_shape=model_shape, divs=divs)
        synapse_preds = process_cubes(cubes, debug_coords, debug, num_completed, seed, ribbons, amacrines,  keep_processing, ckpt_path, device, config, vol)
        return synapse_preds, vol
    elif keep_processing and pull_from_db:
        count = 0
        while count < keep_processing:
            seed = db.get_next_synapse_path()

            # seed = {"path": "/gpfs/data/tserre/carney_data/berson_serre_connectomics/mag1_membranes/x0013/y0004/z0029/110629_k0725_mag1_x0013_y0004_z0029.raw.npy"}

            if seed is None:
                print('No more synapse coordinates to process. Finished!')
                os._exit(1)
            # CHECK THIS -- MAKE SURE DATA REFLECTS HYBRID_... IT IS EFFED RIGHT NOW
            # Compare the ding vol to the constituent niis
            try:
                vol = get_data(
                    seed=seed, pull_from_db=False, config=config)
            except Exception as e:
                print(e)
            model_shape = list(vol.shape)

            # FOR SYNAPSE, NORMALIZE VOL
            vol /= 255.

            cubes, debug_coords = cube_data(vol=vol, model_shape=model_shape, divs=divs, padded=padded)
            synapse_preds = process_cubes(cubes, debug_coords, debug, num_completed, seed, ribbons, amacrines,  keep_processing, ckpt_path, device, config, vol, padded=padded)
            # import pdb;pdb.set_trace()
            # from matplotlib import pyplot as plt
            # plt.subplot(131);plt.imshow(vol[128, ..., 0]);plt.subplot(132);plt.imshow(vol[128, ..., 1]);plt.subplot(133);plt.imshow(synapse_preds[128]);
            # plt.show()

            # Save raw to file structure
            it_out = os.path.join(out_path, os.path.sep.join(seed["path"].split(os.path.sep)[-4:-1]), "ribbon")
            rdirs(it_out)
            # it_out = os.path.join(it_out, "ribbon_synapses")
            # from matplotlib import pyplot as plt;plt.imshow(synapse_preds[128]);plt.show()
            np.save(it_out, synapse_preds)
            count += 1
    else:
        raise NotImplementedError
        vol, error, seed = get_data(seed=seed, pull_from_db=pull_from_db, config=config)
        preds = unet.main(
            test=vol,
            evaluate=True,
            adabn=True,
            return_sess=keep_processing,
            test_input_shape=model_shape,
            test_label_shape=model_shape,
            checkpoint=ckpt_path,
            gpu_device=device)
        preds = preds[0].squeeze()
        synapses = process_preds(preds, config)

        # Add to DB
        db.add_synapses(synapses)
        if save_preds:
            # Save raw to file structure
            it_out = out_path.replace(
                os.path.join(config.write_project_directory, 'mag1'), out_path)
            it_out = os.path.sep.join(it_out.split(os.path.sep)[:-1])
            np.save(it_out, preds)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--path_extent',
        dest='path_extent',
        type=str,
        default='9,9,3',
        help='Provide extent of segmentation in 128^3 volumes.')
    parser.add_argument(
        '--segmentation_path',
        dest='segmentation_path',
        type=str,
        default=None,
        help='Path to existing segmentation file that you want to get synapses for.')
    parser.add_argument(
        '--device',
        dest='device',
        type=str,
        default="/gpu:0",
        help="String for the device to use.")
    parser.add_argument(
        '--keep_processing',
        dest='keep_processing',
        type=int,
        default=0,
        help='Get coords.')
    parser.add_argument(
        '--pull_from_db',
        dest='pull_from_db',
        action='store_true',
        help='Get coords.')
    parser.add_argument(
        '--debug',
        dest='debug',
        action='store_true',
        help='Debug preds.')
    parser.add_argument(
        '--finish_membranes',
        dest='finish_membranes',
        action='store_true',
        help='Finish membrane generation.')
    args = parser.parse_args()
    start = time.time()
    test(**vars(args))
    end = time.time()
    print(('Testing took {}'.format(end - start)))
