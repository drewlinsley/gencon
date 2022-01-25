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
from skimage.transform import resize
import functools


def get_segmentation(
        vol,
        ffn_ckpt,
        ffn_model,
        mem_seed_thresh=0.5,
        move_threshold=0.8,
        segment_threshold=0.5,
        seed_policy="PolicyMembrane",
        deltas='[15, 15, 3]'):
    """Apply the FFN routines using fGRUs."""
    assert move_threshold is not None
    assert segment_threshold is not None
    model_shape = vol.shape[:-1]  # Ignore last dim which is image/membrane

    # Start FFN
    ffn_config = '''image_mean: 128
        image_stddev: 33
        seed_policy: "%s"
        model_checkpoint_path: "%s"
        model_name: "%s.ConvStack3DFFNModel"
        model_args: "{\\"depth\\": 12, \\"fov_size\\": [64, 64, 16], \\"deltas\\": %s}"
        inference_options {
            init_activation: 0.95
            pad_value: 0.05
            move_threshold: %s
            min_boundary_dist { x: 1 y: 1 z: 1}
            segment_threshold: %s
            min_segment_size: 256
        }
        alignment_options {
            save_raw: False
        }''' % (
        seed_policy,
        ffn_ckpt,
        ffn_model,
        deltas,
        move_threshold,
        segment_threshold)
    req = inference_pb2.InferenceRequest()
    _ = text_format.Parse(ffn_config, req)
    runner = inference.Runner()
    runner.start(req, vol, tag='_inference')
    _, segments, probabilities = runner.run(
        (0, 0, 0),
        model_shape,
        mem_seed_thresh=mem_seed_thresh)
    return segments

