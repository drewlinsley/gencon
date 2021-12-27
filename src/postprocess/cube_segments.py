import sys
import time
import os
import gzip
import fastremap
try:
    import cc3d
except:
    print("Failed to import cc3d")
import numpy as np
from glob import glob
# from ffn.inference import segmentation
from skimage.segmentation import relabel_sequential as rfo
from tqdm import tqdm
from config import Config
import nibabel as nib
from scipy.spatial import distance
from utils.hybrid_utils import rdirs
from utils.hybrid_utils import pad_zeros
from utils.hybrid_utils import recursive_make_dir
from skimage import measure, morphology
# from numba import njit, jit, prange
from joblib import Parallel, delayed, parallel_backend
from scipy import ndimage
from skimage.morphology import skeletonize_3d
import numba
from skimage import feature


try:
    from db import db
except:
    print("Failed to load db.")
try:
    import wkw
    NOWKW = False
except:
    print("Failed to load wkw.")
    NOWKW = True


SPLIT_IDS = []


def edgedetect(sl, sigma=3):
    return feature.canny(sl, sigma=sigma)


@numba.njit  # (fastmath=True)
def sobel(input, axis=-1, output=None, mode="reflect", cval=0.0):
    """Calculate a Sobel filter.
    Parameters
    ----------
    %(input)s
    %(axis)s
    %(output)s
    %(mode_multiple)s
    %(cval)s
    Examples
    --------
    >>> from scipy import ndimage, misc
    >>> import matplotlib.pyplot as plt
    >>> fig = plt.figure()
    >>> plt.gray()  # show the filtered result in grayscale
    >>> ax1 = fig.add_subplot(121)  # left side
    >>> ax2 = fig.add_subplot(122)  # right side
    >>> ascent = misc.ascent()
    >>> result = ndimage.sobel(ascent)
    >>> ax1.imshow(ascent)
    >>> ax2.imshow(result)
    >>> plt.show()
    """
    input = np.asarray(input)
    # axis = np.core.multiarray.normalize_axis_index(axis, input.ndim)
    output = ndimage._ni_support._get_output(output, input)
    modes = ndimage._ni_support._normalize_sequence(mode, input.ndim)
    ndimage.filters.correlate1d(input, [-1, 0, 1], axis, output, modes[axis], cval, 0)
    axes = [ii for ii in range(input.ndim) if ii != axis]
    for ii in axes:
        ndimage.filters.correlate1d(output, [1, 2, 1], ii, output, modes[ii], cval, 0)
    return output



# @jit(parallel=True, fastmath=True)
def npad_zeros(x, total):
    """Pad x with zeros to total digits."""
    if not isinstance(x, str):
        x = str(x)
    total = total - len(x)
    for idx in range(total):
        x = '0' + x
    return x


# @jit(parallel=True, fastmath=True)
def nrecursive_make_dir(path, s=3):
    """Recursively build output paths."""
    split_path = path.split(os.path.sep)
    for idx, p in enumerate(split_path):
        if idx > s:
            d = '/'.join(split_path[:idx])
            if not os.path.exists(d):
                os.makedirs(d)
                # print('Created: %s' % d)
            else:
                # print('Reusing: %s' % d)
                pass


def get_edges(chunk, threshold=10, dtype=np.uint32):
    e = ndimage.generic_gradient_magnitude(chunk, ndimage.sobel) > threshold
    # e = (e > threshold).astype(dtype)
    return e


def cube_data(zidx, z, unique_z, out_dir, coordinates, config, dataset, cifs_path, mins, parallel, max_vox, prop_miss, prop_segmented, cheap_merge,  use_parallel=False, threshold=512, muller_curate=False, num_mullers=50, mem_thresh=255 * 0.2, muller_dir="/"):
    """Cube data in z."""

    # This plane
    z_sel_coors = coordinates[coordinates[:, 2] == z]
    z_sel_coors = np.unique(z_sel_coors, axis=0)

    # Next plane information
    if zidx < len(unique_z) - 1:
        z_next = unique_z[zidx + 1]
        max_z = z_next - z
    else:
        max_z = path_extent[-1]
        z_next = None

    # Allow for fast loading for debugging
    if os.path.exists(os.path.join(out_dir, 'plane_z{}.npy'.format(z))):
        # Load the current slab
        main = np.load(os.path.join(out_dir, 'plane_z{}.npy'.format(z)))
        # Combine the two into a merged slab
        merge_to = (path_extent[-1] - max_z) * config.shape[-1]  # ((z + path_extent[-1]) - z_next) * config.shape[-1]
        if cheap_merge:
            if merge_to < (config.shape[-1] * 3) and z_next is not None:
                # Load the next slab
                main_b = np.load(os.path.join(out_dir, 'plane_z{}.npy'.format(z_next)))
                main_b = main_b[..., merge_to // 2 : merge_to]  # noqa case of 1-offset: 128:256
                main[..., -(merge_to // 2):] = main_b  # noqa case of 1-offset: 256: 
        else:
            if merge_to < (config.shape[-1] * 3) and z_next is not None:
                # Load the next slab
                main_b = np.load(os.path.join(out_dir, 'plane_z{}.npy'.format(z_next)))
                main_b = main_b[..., merge_to // 2 : merge_to]  # noqa case of 1-offset: 128:256
                main_mask = (main[..., -(merge_to // 2):] == 0).astype(main_b.dtype)
                main_b = (main_mask * max_vox) * main_b  # 0 out overlapping segs
                _, max_b = fastremap.minmax(main_b)
                max_vox = max_b + 1
                main_b = main_b + main[..., -(merge_to // 2):]  # Potentially need to add an offset to main_b
                main[..., -(merge_to // 2):] = main_b
                """
                main_comp_a = (main_comp_a > 0).astype(np.int32)
                diff = (main_comp_a - (main_b > 0).astype(np.int32)).astype(np.int32)  # noqa Where do you not have As and not Bs and vice versa
                diff = np.maximum(diff, 0).astype(np.uint32)
                # diff_b = np.maximum(-diff, 0)

                # # Store how much was missing previously in an array -- only for debuggin
                # prop_miss.append(float(diff.sum()) / float(main_b.size))
                # prop_segmented.append(float(main_comp_a.sum()) / float(main_comp_a.size))

                # Remove regions smaller than a threshold
                diff_mask = (diff != 0).astype(np.uint32)
                diff = measure.label(diff)
                diff = cc3d.connected_components(diff).astype(np.uint32)
                # diff = morphology.remove_small_objects(diff, min_size=threshold).astype(main_b.dtype)

                # Keep all main_b regions, propogate main_a regions but add an offset
                max_a = diff.max() + 1
                main_b += (diff + max_vox) * diff_mask
                max_vox += max_a

                # Now move main_b over
                main[..., -(merge_to // 2):] = main_b  # noqa case of 1-offset: 256: 
                """
        muller_sz = os.path.getsize(os.path.join(muller_dir, 'plane_z{}.npy'.format(z)))
        if muller_curate and os.path.exists(os.path.join(muller_dir, 'plane_z{}.npy'.format(z))) and muller_sz > 200:
            # Check if these muller planes are available
            # Then, target only the biggest segments (use fastremap.unique to get counts and ids)
            del main_b
            muller = np.load(os.path.join(muller_dir, 'plane_z{}.npy'.format(z)))
            # Weird issue where muller might be padded beyond main
            muller = muller[:main.shape[0], :main.shape[1], :main.shape[2]]

            muller = (muller > 128).astype(main.dtype)  # Threshold
            to_proc = main * muller
            to_proc = cc3d.connected_components(to_proc, connectivity=6).astype(dtype)
            import pdb;pdb.set_trace()
            to_proc, _ = fastremap.renumber(to_proc, in_place=True, preserve_zero=True)  # rfo(to_proc)
            _, it_max = fastremap.minmax(to_proc)
            muller_mask = to_proc > 0
            to_proc = to_proc + ((muller_mask).astype(dtype) * max_vox)
            main[muller_mask] = 0.
            main += to_proc
            max_vox = max_vox + 1 + it_max

            # main = cc3d.connected_components(main > 0).astype(dtype)

            # ids, counts = fastremap.unique(main, return_counts=True)
            # count_sort = np.argsort(counts)[::-1]
            # ids_sort = ids[count_sort]
            # mask = fastremap.mask_except(main, ids_sort[:num_mullers].tolist())
            # mask = (max_vox * mullers) * mask
            # # Now split each part of the mask into a different volume with label
            # mask = measure.label(mask, connectivity=1, background=0)
            # main = (main + mask).astype(dtype)
            # _, max_b = fastremap.minmax(main)
            # max_vox = 1 + max_b
            # # Finally, label the volume so there's nothing disconnected

        if zidx < len(unique_z) - 1:
            min_z = 0  # Unless we are on the 0th iteration
        else:
            min_z = 0

        # Package data in parallel
        if use_parallel:
            parallel(delayed(convert_save_cubes)(dataset, main, sel_coor, cifs_path, mins, min_z, max_z, config, xoff, yoff) for sel_coor in z_sel_coors)
        else:
            for sel_coor in z_sel_coors:
                convert_save_cubes(dataset, main, sel_coor, cifs_path, mins, min_z, max_z, config, xoff, yoff)
    return max_vox, prop_miss, prop_segmented


# @jit(parallel=True, fastmath=True)
# def convert_save_cubes(coords, data, cifs_path, mins, max_z, config, dataset, corner):
def convert_save_cubes(dataset, main, sel_coor, cifs_path, mins, min_z, max_z, config, xoff, yoff):
    """All coords come from the same z-slice. Save these as npys to cifs."""
    # for x in range(path_extent[0]):
    #     for y in range(path_extent[1]):
    #          for z in range(max_z):
    # from skimage.segmentation import relabel_sequential as rs;from matplotlib import pyplot as plt;
    # plt.imshow(rs(seg.astype(np.uint32)[64])[0]);plt.savefig("512_512_0.png")
    coords = sel_coor
    corner = sel_coor[:-1]
    adj_coor = (sel_coor[:-1] - mins) * config.shape
    data = main[
        adj_coor[0]: adj_coor[0] + xoff,
        adj_coor[1]: adj_coor[1] + yoff]
    for x in range(path_extent[0]):  # max_z):
        for y in range(path_extent[1]):
            for z in range(min_z, max_z):
            # for z in range(min_z, max_z):
                seg = data[
                    x * config.shape[0]: x * config.shape[0] + config.shape[0],
                    y * config.shape[1]: y * config.shape[1] + config.shape[1],
                    z * config.shape[2]: z * config.shape[2] + config.shape[2]]
                it_corner = np.asarray([(x + corner[0]) * config.shape[0], (y + corner[1]) * config.shape[1], (z + corner[2]) * config.shape[2]])
                # it_corner = np.asarray([(z + corner[0]) * config.shape[0], (y + corner[1]) * config.shape[1], (x + corner[2]) * config.shape[2]])
                # print(x, y, z)
                if np.all(np.asarray(seg.shape) > 0):
                    dataset.write(it_corner, seg)
                else:
                    print("Failed {}, {}, {}".format(x, y, z))


path_extent = [9, 9, 3]
glob_debug = False  # True
save_cubes = False
merge_debug = False
remap_labels = False
in_place = False
z_max = 384
bu_margin = 2
config = Config()

# Get list of coordinates
og_coordinates = db.pull_main_seg_coors()
if glob_debug:
    raise NotImplementedError("Conflict with the new files.")
    new_og_coordinates = []
    for r in og_coordinates:
        sel_coor = [r['x'], r['y'], r['z']]
        check = glob(os.path.join('/media/data_cifs/connectomics/ding_segmentations/x{}/y{}/z{}/v0/0/0/seg-0_0_0.npz'.format(pad_zeros(sel_coor[0], 4), pad_zeros(sel_coor[1], 4), pad_zeros(sel_coor[2], 4))))
        if len(check):
            new_og_coordinates.append(sel_coor)
    og_coordinates = np.array(new_og_coordinates)
else:
    og_coordinates = np.array([[r['x'], r['y'], r['z']] for r in og_coordinates if r['processed_segmentation']])
master_merge_paths = np.load("all_segmentation_paths.npy")
merges = [[int(x.split(os.path.sep)[-7][1:]), int(x.split(os.path.sep)[-6][1:]), int(x.split(os.path.sep)[-5][1:])] for x in master_merge_paths]
if glob_debug:
    new_merges = []
    for r in merges:
        sel_coor = [r['x'], r['y'], r['z']]
        check = glob(os.path.join('/media/data_cifs/connectomics/ding_segmentations_merge/x{}/y{}/z{}/v0/0/0/seg-0_0_0.npz'.format(pad_zeros(sel_coor[0], 4), pad_zeros(sel_coor[1], 4), pad_zeros(sel_coor[2], 4))))
        if len(check):
            new_merges.append(sel_coor)
        else:
            print(sel_coor)
    merges = np.array(new_merges)
else:
    # merges = np.array([[r['x'], r['y'], r['z']] for r in merges if r['processed_segmentation']])
    merges = np.asarray(merges)

# Loop over coordinates
coordinates = np.concatenate((og_coordinates, np.zeros_like(og_coordinates)[:, 0][:, None]), 1)
merges = np.concatenate((merges, np.ones_like(merges)[:, 0][:, None]), 1)
coordinates = np.concatenate((coordinates, merges))
unique_z = np.unique(coordinates[:, -2])
print(unique_z)
np.save('unique_zs_for_merge', unique_z)

# Compute extents
path_extent = np.array(path_extent)
mins = np.min(coordinates[:, :-1], axis=0)
maxs = np.max(coordinates[:, :-1], axis=0)  # Add extent to this
print("Bounding-box TL corner is: {}".format(mins * 128))

mins_vs = mins * config.shape  # (config.shape * np.array(path_extent))
maxs_vs = maxs * config.shape  # (config.shape * np.array(path_extent))
maxs_vs += config.shape * path_extent  # np.array(path_extent)
diffs = (maxs_vs - mins_vs)
xoff, yoff, zoff = path_extent * config.shape  # [:2]

# Loop through x-axis
use_parallel = True
dtype = np.uint32
max_vox = 30000000  # Max seg is ~15M. Give a little buffer.
prop_miss, prop_segmented = [], []
count, prev = 0, None
cheap_merge = True
muller_curate = True  # False
membrane_curate = True  # True
slice_shape = np.concatenate((diffs[:-1], [z_max]))
dataset = wkw.Dataset.open(
    # "/media/data_cifs/connectomics/cubed_mag1/merge_data_wkw/1",
    # "/media/data_cifs/connectomics/merge_data_wkw/1",
    # "/media/data_cifs_lrs/projects/prj_connectomics/connectomics_data/merge_data_wkw/merge_data_wkw/1",
    # "/gpfs/data/tserre/data/wkcube/merge_data_wkw/1",
    "/gpfs/data/tserre/data/wkcube/merge_data_wkw_class/1",  # THIS SHOULD BE A VARIABLE EVENTUALLY
    wkw.Header(dtype))
# cifs_stem = '/media/data_cifs/connectomics/merge_data_nii_raw_v2/'
cifs_path = None  # '{}/1/x%s/y%s/z%s/110629_k0725_mag1_x%s_y%s_z%s.raw'.format(cifs_stem)
out_dir = "/gpfs/data/tserre/data/final_merge/"  # /localscratch/merge/'
# muller_dir = "/localscratch/muller/"
# muller_dir = "/cifs/data/tserre/CLPS_Serre_Lab/projects/prj_connectomics/connectomics_data_scratch_v1/muller_slices"
muller_dir = "/gpfs/data/tserre/data/final_muller"
# unique_z = [unique_z[:1]]
# unique_z = unique_z[:2]
n_jobs = 1  # 96
# unique_z = unique_z[33:]
with Parallel(n_jobs=n_jobs, backend='threading') as parallel:
    for zidx, z in tqdm(enumerate(unique_z), total=len(unique_z), desc="Z-slice main clock"):
        max_vox, prop_miss, prop_segmented = cube_data(zidx, z, unique_z, out_dir, coordinates, config, dataset, cifs_path, mins, parallel, max_vox, prop_miss, prop_segmented, cheap_merge=cheap_merge, use_parallel=use_parallel, muller_curate=muller_curate, muller_dir=muller_dir)
print("Max voxel is ".format(max_vox))

np.save("cubed_max_vox", max_vox)
np.save("cubed_prop_miss", prop_miss)

