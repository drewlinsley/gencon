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


logger = logging.getLogger()
logger.setLevel(logging.INFO)
AUGS = ['uniform', 'pixel', 'rot270', 'rot90', 'rot180', 'flip_left', 'flip_right', 'depth_flip']  # ['pixel']  # 'depth_flip']  # , 'rot90', 'rot180', 'rot270']
AUGS = ['flip_left', 'flip_right', 'depth_flip']
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
        deltas='[15, 15, 3]',  # '[27, 27, 6]'
        seed_policy='PolicyMembrane',  # 'PolicyPeaks'
        membrane_slice=[64, 384, 384],  # 576
        membrane_overlap_factor=[0.5, 0.5, 0.5],  # [0.875, 2./3., 2./3.],
        debug_resize=False,
        debug_nii=False,
        quick=True,
        path_extent=None,  # [1, 1, 1],
        rotate=False):
    """Apply the FFN routines using fGRUs."""
    # TEST_TIME_AUGS = None
    config = Config()
    assert move_threshold is not None
    assert segment_threshold is not None
    path_extent = np.asarray([int(x) for x in path_extent.split(",")])
    model_shape = (config.shape * path_extent)
    model_shape = (280, 280, 280)
    mpath = "/localscratch/middle_cube_membranes_for_ffn_training.npy"

    if quick:
        config.ffn_ckpt = os.path.join(config.read_project_directory, 'ffn_ckpts/64_fov/ts_1/model.ckpt-1632105')  # nopep8
        config.ffn_ckpt = os.path.join(config.read_project_directory, 'ffn_ckpts/64_fov/ts_1/model.ckpt-1150159')
        config.membrane_ckpt = os.path.join(config.read_project_directory, 'checkpoints/l3_fgru_constr_berson_0_berson_0_2019_02_16_22_32_22_290193/fixed_model_137000.ckpt-137000')  # nopep8
        # config.membrane_ckpt = os.path.join(config.read_project_directory, 'checkpoints/l3_fgru_constr_berson_0_berson_0_2019_02_16_22_32_22_290193/fixed_model_1150159.ckpt-1150159')  # nopep8
    else:
        config.ffn_ckpt = os.path.join(config.read_project_directory, 'ffn_ckpts/64_fov/feedback_hgru_v5_3l_notemp_f_v4_berson4x_w_inf_memb_r0/model.ckpt-225915')  # nopep8
        config.ffn_model = 'feedback_hgru_v5_3l_notemp_f_v4'  # 2382.443995

    # 4. Start FFN
    ffn_out = '/localscratch/middle_cube_segmentation'
    seg_dir = ffn_out
    recursive_make_dir(seg_dir)

    # Ran into an error with the 0/0 folders not being made sometimes -- Why?
    t_seg_dir = os.path.join(seg_dir, '0', '0')
    recursive_make_dir(t_seg_dir)

    # PASS FLAG TO CHOOSE WHETHER OR NOT TO SAVE SEGMENTATIONS
    print('Saving segmentations to: %s' % seg_dir)
    if seg_vol is not None:
        ffn_config = '''image {hdf5: "%s"}
            image_mean: 128
            image_stddev: 33
            seed_policy: "%s"
            model_checkpoint_path: "%s"
            model_name: "%s.ConvStack3DFFNModel"
            model_args: "{\\"depth\\": 12, \\"fov_size\\": [64, 64, 16], \\"deltas\\": %s, \\"shifts\\": [%s, %s, %s]}"
            init_segmentation: {hdf5: "%s"}
            segmentation_output_dir: "%s"
            inference_options {
                init_activation: 0.95
                pad_value: 0.05
                move_threshold: %s
                min_boundary_dist { x: 1 y: 1 z: 1}
                segment_threshold: %s
                min_segment_size: 256
            }''' % (
            mpath,
            seed_policy,
            config.ffn_ckpt,
            config.ffn_model,
            deltas,
            shift_z, shift_y, shift_x,
            seg_vol,
            seg_dir,
            move_threshold,
            segment_threshold)
    else:
        ffn_config = '''image {hdf5: "%s"}
            image_mean: 128
            image_stddev: 33
            seed_policy: "%s"
            model_checkpoint_path: "%s"
            model_name: "%s.ConvStack3DFFNModel"
            model_args: "{\\"depth\\": 12, \\"fov_size\\": [64, 64, 16], \\"deltas\\": %s}"
            segmentation_output_dir: "%s"
            inference_options {
                init_activation: 0.95
                pad_value: 0.05
                move_threshold: %s
                min_boundary_dist { x: 1 y: 1 z: 1}
                segment_threshold: %s
                min_segment_size: 256
            }''' % (
            mpath,
            seed_policy,
            config.ffn_ckpt,
            config.ffn_model,
            deltas,
            seg_dir,
            move_threshold,
            segment_threshold)
    req = inference_pb2.InferenceRequest()
    _ = text_format.Parse(ffn_config, req)
    runner = inference.Runner()
    runner.start(req, tag='_inference')
    _, segments, probabilities = runner.run(
        (0, 0, 0),
        (model_shape[0], model_shape[1], model_shape[2]))
    import pdb;pdb.set_trace()
    res_shape = [1152, 1152, 384]
    vol = resize(segments.transpose(1, 2, 0), res_shape, anti_aliasing=True, preserve_range=True, order=3).transpose(2, 0, 1)


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
        '--seed_policy',
        dest='seed_policy',
        type=str,
        default='PolicyMembrane',
        help='Policy for finding FFN seeds.')
    parser.add_argument(
        '--path_extent',
        dest='path_extent',
        type=str,
        default='3,9,9',
        help='Provide extent of segmentation in 128^3 volumes.')
    parser.add_argument(
        '--move_threshold',
        dest='move_threshold',
        type=float,
        default=0.8,  # 0.8
        help='Movement threshold. Higher is more likely to move.')
    parser.add_argument(
        '--segment_threshold',
        dest='segment_threshold',
        type=float,
        default=0.4,  # 0.5
        help='Segment threshold..')
    parser.add_argument(
        '--membrane_slice',
        dest='membrane_slice',
        type=str,
        default=None,
        help='Membrane chunking along z axis.')
    parser.add_argument(
        '--validate',
        dest='validate',
        action='store_true',
        help='Force berson validation dataset.')
    parser.add_argument(
        '--rotate',
        dest='rotate',
        action='store_true',
        help='Rotate the input data.')
    parser.add_argument(
        '--membrane_only',
        dest='membrane_only',
        action='store_true',
        help='Only process membranes.')
    parser.add_argument(
        '--segment_only',
        dest='segment_only',
        action='store_true',
        help='Only process segments.')
    parser.add_argument(
        '--merge_segment_only',
        dest='merge_segment_only',
        action='store_true',
        help='Only process merge segments.')
    args = parser.parse_args()
    start = time.time()
    get_segmentation(**vars(args))
    end = time.time()
    print(('Segmentation took {}'.format(end - start)))

