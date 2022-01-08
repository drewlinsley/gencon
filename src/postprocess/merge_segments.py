import gc
import sys
import time
import os
import fastremap
import numpy as np
from glob import glob
from tqdm import tqdm
from scipy.spatial import distance
from utils.hybrid_utils import make_dir
from joblib import Parallel, delayed
from scipy.spatial.distance import pdist, squareform
from scipy.optimize import minimize
from skimage.transform import resize
from omegaconf import OmegaConf
# from memory_profiler import profile


def read_headers(fl):
    """Takes a path to an .npz file, which is a Zip archive of .npy files.
    Generates a sequence of (name, shape, np.dtype).
    """
    with open(fl, 'rb') as f:
        major, minor = np.lib.format.read_magic(f)
        shape, fortran, dtype = np.lib.format.read_array_header_1_0(f)
    return shape


def minimum_spanning_tree(X, copy_X=False, thresh=0.0015):
    """X are edge weights of fully connected graph"""
    if copy_X:
        X = X.copy()

    if X.shape[0] != X.shape[1]:
        raise ValueError("X needs to be square matrix of edge weights")
    n_vertices = X.shape[0]
    spanning_edges = []
    
    # initialize with node 0:                                                                                         
    visited_vertices = [0]                                                                                            
    num_visited = 1
    # exclude self connections:
    diag_indices = np.arange(n_vertices)
    X[diag_indices, diag_indices] = np.inf
    cumdist = 0
    while num_visited != n_vertices:
        visited = X[visited_vertices]
        # visited[visited < min_dist] = max_dist
        new_edge = np.argmin(visited, axis=None)
        new_dist = np.min(visited, axis=None)  # Get the distance for this edge

        # 2d encoding of new_edge from flat, get correct indices                                                      
        new_edge = divmod(new_edge, n_vertices)
        cumdist += new_dist
        cumdist = 0
        new_edge = [visited_vertices[new_edge[0]], new_edge[1]]
        spanning_edges.append(new_edge)
        visited_vertices.append(new_edge[1])

        # remove all edges inside current tree
        X[visited_vertices, new_edge[1]] = np.inf
        X[new_edge[1], visited_vertices] = np.inf

        num_visited += 1
    return np.vstack(spanning_edges)


def area(a, b, l=9):
    """Compute overlap between rectangles."""
    dx = min(a[1] + l, b[1] + l) - max(a[1], b[1])
    dy = min(a[0] + l, b[0] + l) - max(a[0], b[0])
    if (dx>=0) and (dy>=0):
        return dx*dy
    else:
        return 0


def prune_coords(P, thresh=0.005, plot=False):  # 0.015
    X = []
    for h in range(len(P)):
        x = []
        for w in range(len(P)):
            areai = area(P[h, :2], P[w, :2])
            x.append(areai)
        X.append(x)
    X = np.asarray(X).astype(float)
    X = 1. / (X + 1e-2)
    # thresh = X[X < X.max()].mean()
    # print(X)
    edge_list = minimum_spanning_tree(X, thresh=thresh, copy_X=True)

    cumweight = 0
    # thresh = 0.015
    keep_edges = []
    for edge in edge_list:
        weight = X[edge[0], edge[1]]
        cumweight += weight
        if cumweight > thresh:
            keep_edges.append(edge)
            cumweight = 0
    keep_edges = np.stack(keep_edges, 0)

    if plot:

        fig, ax = plt.subplots()
        all_edges = keep_edges[:, 1]
        plt.scatter(P[:, 0], P[:, 1], color="red")
        print(len(P))
        P = P[all_edges]
        print(len(P))
        for coor in P:
            r = rect(coor[:2], 9, 9, fc="blue", edgecolor="r", alpha=0.3)
            ax.add_patch(r)
        plt.axis("square")
        plt.show()
    return keep_edges


def get_main_merge_splits(P, thresh=0.016, max_coords=220):  # 0.015
    all_edges = prune_coords(P, thresh=thresh, plot=False)
    # plt.close("all")
    all_edges = all_edges[:, 1]
    # if len(all_edges) > max_coords:
    #     pruned_coords = P[all_edges]
    # else:
    #     pruned_coords = P

    # Then split priuned coords into mains/merges
    pruned_coords = P
    dm = squareform(pdist(pruned_coords[:, :2], "cityblock"))
    dm = dm + np.eye(dm.shape[0]) * 1000
    keeps = []
    for rid, rw in enumerate(dm):
        check = rw == 1000
        if len(rw) - check.sum() > 0:
            keeps.append(rid)
            mask = rw < 7
            dm[mask] = 1000
            dm[:, mask] = 1000
            dm[rid] = 1000
            dm[:, rid] = 1000
    z_sel_coors_main = pruned_coords[np.asarray(keeps)]
    z_sel_coors_merge = []
    for zm in pruned_coords:
        if not np.any((zm == z_sel_coors_main).sum(-1) == 4):
            z_sel_coors_merge.append(zm)
    z_sel_coors_merge = np.asarray(z_sel_coors_merge)
    # z_sel_coors_main[:, -1] = 0
    # z_sel_coors_merge[:, -1] = 1

    if len(z_sel_coors_merge) > max_coords:
        merge_coors = []
        count = 0
        while count < len(z_sel_coors_merge):
            merge_coors += [z_sel_coors_merge[count:count + max_coords]]
            count += max_coords
        z_sel_coors_merge = merge_coors
    else:
        z_sel_coors_merge = [z_sel_coors_merge]
    return z_sel_coors_main, z_sel_coors_merge


def remap_loop(um, tc, main_margin, merge_margin):
    masked_plane = main_margin == um
    overlap = merge_margin[masked_plane]
    overlap = overlap[overlap != 0]
    overlap_check = len(overlap)  # overlap.sum()
    update = False
    transfer = False
    remap = []
    if not overlap_check:
        # In this case, there's a segment in main that overlaps with empty space in merge. Let's propagate main.
        transfer = um
        update = True
    else:
        # In this case, there's overlap between the merge and the main. Let's pass the main label to all merges that are touched (This used to be an argmax).
        uni_over, counts = fastremap.unique(overlap, return_counts=True)
        # uni_over = uni_over[uni_over > 0]
        for ui, uc in zip(uni_over, counts):
            remap.append([ui, um, uc])  # Append merge ids for the overlap
    return remap, transfer, update


# @profile
def get_remapping(main_margin, merge_margin, parallel, use_numba=False, merge_wiggle=0.8):
    """Determine where to merge.

    Potential problem is that this assumes both main/merge are fully segmented. If this is not the
    case, and we have a non-zero main segmentation but a zero merge segmentation, the zeros
    will overwrite the mains segmentation.

    Shitty but not sure what to do yet or if it's a real issue.
    """
    # Loop through the margin in main, to find per-segment overlaps with merge
    if not len(main_margin):
        return None
    if not len(merge_margin):
        return None
    unique_main, unique_main_counts = fastremap.unique(main_margin, return_counts=True)
    unique_main_mask = unique_main > 0
    unique_main = unique_main[unique_main_mask]
    unique_main_counts = unique_main_counts[unique_main_mask]
    if not len(unique_main):
        return [], merge_margin, [], False
    remap = []
    transfers = []
    update = False
    # For each segment in main, find the corresponding seg in margin. Transfer the id over, or transfer the bigger segment over (second needs to be experimental).
    # Use parallel on this loop

    # info = parallel(delayed(remap_loop)(um, tc, main_margin, merge_margin) for um, tc in zip(unique_main, unique_main_counts))
    info = []
    for um, tc in zip(unique_main, unique_main_counts):
        info.append(remap_loop(um, tc, main_margin, merge_margin))

    updates, transfers, remaps = [], [], []
    for r in info:
        updates.append(r[2])
        transfers.append(r[1])
        remaps.append(r[0])

    remaps = [x for x in remaps if len(x)]
    try:
        if len(remaps):
            remaps = np.concatenate(remaps, 0)
    except:
        import pdb;pdb.set_trace()
    updates = np.max(updates)
    return remaps, merge_margin, transfers, updates


# @profile
def remap(vol, min_vol_size, in_place, max_vox, connectivity=6, disable_max_vox=True):  # Moved max_vox to main loop
    """Run a remapping and iterate by max_vox."""
    vol, _ = fastremap.renumber(vol, in_place=in_place, preserve_zero=True)  # .astype(np.uint32)
    vol = vol.astype(dtype)
    if not disable_max_vox:
        it_max_vox = vol.max()
        zeros = (vol > 0).astype(dtype)
        vol += max_vox  # Update to max_vox
        vol *= zeros
        max_vox = max_vox + it_max_vox + 1  # Increment
    return vol, max_vox


def get_margins(margin_idx, margin_end, margin_offset, main_margin_offset, edge_offset, candidate_diffs, div, adj_coor, idx):
    margin_start = np.abs(div)
    if np.any(margin_idx):
        # There is a candidate!!
        sel_idx = candidate_diffs[margin_idx][0]
        # directed_sel_idx = sel_idx[idx]
        margin_end = margin_start + margin_offset
        edge_start = adj_coor[idx] + margin_start - main_margin_offset
        merge = True
    else:
        edge_start = adj_coor[idx] + margin_start - 1
        merge = False
    edge_start = np.maximum(edge_start, 0)
    edge_end = edge_start + edge_offset
    return merge, edge_start, edge_end, margin_start, margin_end


# @profile
def get_merge_coords(
        candidate_diffs,
        main_margin_offset,
        edge_offset,
        margin_start,
        margin_end,
        margin_offset,
        vs,
        adj_coor,
        main,
        vol,
        parallel,
        div=10):
    """Find overlapping coordinates in top/bottom/left/right planes."""
    midpoints = vs // 2

    ## Top
    top_margin_idx = candidate_diffs[:, 0] < midpoints[0]
    bottom_margin_idx = candidate_diffs[:, 0] > midpoints[0]
    left_margin_idx = candidate_diffs[:, 1] < midpoints[1]
    right_margin_idx = candidate_diffs[:, 1] > midpoints[1]

    top_merge, top_edge_start, top_edge_end, top_margin_start, top_margin_end = get_margins(
        top_margin_idx, margin_end, margin_offset, main_margin_offset, edge_offset, candidate_diffs, div, adj_coor, idx=0)
    bottom_merge, bottom_edge_start, bottom_edge_end, bottom_margin_start, bottom_margin_end = get_margins(
        bottom_margin_idx, margin_end, margin_offset, main_margin_offset, edge_offset, candidate_diffs, div, adj_coor, idx=0)
    left_merge, left_edge_start, left_edge_end, left_margin_start, left_margin_end = get_margins(
        left_margin_idx, margin_end, margin_offset, main_margin_offset, edge_offset, candidate_diffs, div, adj_coor, idx=1)
    right_merge, right_edge_start, right_edge_end, right_margin_start, right_margin_end = get_margins(
        right_margin_idx, margin_end, margin_offset, main_margin_offset, edge_offset, candidate_diffs, div, adj_coor, idx=1)

    if top_merge:
        main_top_face = main[  # Should this be expanded to a matrix instead of a vector?? so noisy
            top_edge_start: top_edge_end,
            adj_coor[1]: adj_coor[1] + vs[1],
            :]
    if bottom_merge:
        main_bottom_face = main[
            bottom_edge_start: bottom_edge_end,
            adj_coor[1]: adj_coor[1] + vs[1],
            :]

    if left_merge:
        main_left_face = main[
            adj_coor[0]: adj_coor[0] + vs[0],
            left_edge_start: left_edge_end,
            :]
    if right_merge:
        main_right_face = main[
            adj_coor[0]: adj_coor[0] + vs[0],
            right_edge_start: right_edge_end,
            :]
    remap_top, remap_left, remap_right, remap_bottom = [], [], [], []
    bottom_trans, top_trans, right_trans, left_trans = [], [], [], []

    # Get remapping for each face
    if top_merge:
        merge_top_face = vol[top_margin_start: top_margin_end, :, :]
        remap_top, merge_top_face, top_trans, update = get_remapping(
            main_margin=main_top_face,
            merge_margin=merge_top_face, parallel=parallel)
    if left_merge:
        merge_left_face = vol[:, left_margin_start: left_margin_end, :]
        remap_left, merge_left_face, left_trans, update = get_remapping(
            main_margin=main_left_face,
            merge_margin=merge_left_face, parallel=parallel)
    if right_merge:
        merge_right_face = vol[:, -right_margin_end + 1: -right_margin_start + 1]
        remap_right, merge_right_face, right_trans, update = get_remapping(
            main_margin=main_right_face,
            merge_margin=merge_right_face, parallel=parallel)
    if bottom_merge:
        merge_bottom_face = vol[-bottom_margin_end + 1: -bottom_margin_start + 1]
        remap_bottom, merge_bottom_face, bottom_trans, update = get_remapping(
            main_margin=main_bottom_face,
            merge_margin=merge_bottom_face, parallel=parallel)
    return np.array(remap_top + remap_left + remap_right + remap_bottom), bottom_margin_start, top_margin_start, right_margin_start, left_margin_start, bottom_trans, top_trans, right_trans, left_trans


# @profile
def process_merge(
        main,
        vol,
        sel_coor,
        mins,
        vs,
        parallel,
        max_vox=None,
        margin_start=0,
        margin_end=1,
        test=0.50,
        prev=None,
        plane_coors=None,
        verbose=False,
        main_margin_offset=1,
        edge_offset=1,
        margin_offset=1,
        min_vol_size=256,
        margin=25):
    """
    Handle merge volumes.
    TODO: Just load section that could be used for merging two existing mains.
    # Merge function:
    # i. Take a local neighborhood of voxels around a merge
    # ii. reduce across orthogonal dimensions
    # iii. Compute correlations per segment
    # iv. if correlation > threshold, pass the label from main -> merge

    # # For horizontal
    # Insert all merges that will not colide with mains (add these to the plane_coors list as mains)
    # Find the collisions between merges + mains. Double check all 4 sides of the merge volume. Apply merge to the sides that have verified collisions.
    """
    if not len(vol):
        return main, max_vox

    # Resize vol if necessary
    vol_shape = vol.shape
    if not np.all(vs == vol_shape):
        vol = resize(
            vol,
            vs[:-1],
            anti_aliasing=False,
            preserve_range=True,
            order=0).astype(vol.dtype)
        vol_shape = vol.shape

    # Get centroid
    adj_coor = sel_coor - mins
    center = adj_coor + (vs // 2)
    center = center[:-1]
    if prev is None:  # direction == 'horizontal':
        # Signed distance between corners of adj_coor and other planes 
        adj_planes = plane_coors - mins

        # Drop T/B/L/R planes into main. See if there's anything there. Brute force.
        # adj_coor is TL corner of the merge we're looking at.
        # We also want these planes to be a bit offset from boundaries because of sparse segs there
        import pdb;pdb.set_trace()
        top_plane_hs = [adj_coor[0] + margin, adj_coor[0] + margin + 1]
        top_plane_ws = [adj_coor[1] + margin, adj_coor[1] + vs[1] - margin]
        bottom_plane_hs = [adj_coor[0] + vs[0] - margin - 1, adj_coor[0] + vs[0] - margin]
        bottom_plane_ws = [adj_coor[1] + margin, adj_coor[1] + vs[1] - margin]
        left_plane_hs = [adj_coor[0] + margin, adj_coor[0] + vs[0] - margin]
        left_plane_ws = [adj_coor[1] + margin, adj_coor[1] + margin + 1]
        right_plane_hs = [adj_coor[0] + margin, adj_coor[0] + vs[0] - margin]
        right_plane_ws = [adj_coor[1] + vs[1] - margin - 1, adj_coor[1] + vs[1] - margin]
        top_plane = main[top_plane_hs[0]: top_plane_hs[1], top_plane_ws[0]: top_plane_ws[1]]
        bottom_plane = main[bottom_plane_hs[0]: bottom_plane_hs[1], bottom_plane_ws[0]: bottom_plane_ws[1]]
        left_plane = main[left_plane_hs[0]: left_plane_hs[1], left_plane_ws[0]: left_plane_ws[1]]
        right_plane = main[right_plane_hs[0]: right_plane_hs[1], right_plane_ws[0]: right_plane_ws[1]]
        top_check, bottom_check, left_check, right_check = top_plane.sum(), bottom_plane.sum(), left_plane.sum(), right_plane.sum()
        trimmed = False  # Try leaving this on as default. The boundary stuff sucks.
        if np.any([top_check, bottom_check, left_check, right_check]):
            # Trim vol to match the margin-adjusted size
            trimmed = True
        vol = vol[margin: -margin, margin: -margin]  # Was previously a contingency in above if

        bottom_trans, top_trans, right_trans, left_trans = [], [], [], []
        all_remaps = []
        if top_check:  # Merge here
            merge_top_face = vol[0, :]
            remap_top, merge_top_face, top_trans, update = get_remapping(
                main_margin=top_plane.squeeze(),
                merge_margin=merge_top_face, parallel=parallel)
            all_remaps.append(remap_top)
        if left_check: 
            merge_left_face = vol[:, 0]
            remap_left, merge_left_face, left_trans, update = get_remapping(
                main_margin=left_plane.squeeze(),
                merge_margin=merge_left_face, parallel=parallel)
            all_remaps.append(remap_left)
        if right_check:
            merge_right_face = vol[:, -1]
            remap_right, merge_right_face, right_trans, update = get_remapping(
                main_margin=right_plane.squeeze(),
                merge_margin=merge_right_face, parallel=parallel)
            all_remaps.append(remap_right)
        if bottom_check:
            merge_bottom_face = vol[-1]
            remap_bottom, merge_bottom_face, bottom_trans, update = get_remapping(
                main_margin=bottom_plane.squeeze(),
                merge_margin=merge_bottom_face, parallel=parallel)
            all_remaps.append(remap_bottom)

        all_remaps = [x for x in all_remaps if len(x)]
        if len(all_remaps):
            all_remaps = np.concatenate(all_remaps, 0)

        # Get sizes and originals for every remap. Sort these for the final remap
        if len(all_remaps):
            remap_idx = np.argsort(all_remaps[:, -1])[::-1]  # Sort by sizes
            all_remaps = all_remaps[remap_idx]
            unique_remaps = fastremap.unique(all_remaps[:, 0], return_counts=False) 
            fixed_remaps = {}
            for ur in unique_remaps:  # , rc in zip(unique_remaps, remap_counts):
                mask = all_remaps[:, 0] == ur
                fixed_remaps[ur] = all_remaps[mask][0][1]  # Change all to the biggest
            vol = fastremap.remap(vol, fixed_remaps, preserve_missing_labels=True, in_place=True)
            if verbose:
                print('Finished: {}'.format(time.time() - elapsed))

            # Insert vol into main
            main[top_plane_hs[0]: bottom_plane_hs[1], left_plane_ws[0]: right_plane_ws[1]] = vol
        else:
            main[
                adj_coor[0] + margin: adj_coor[0] + xoff - margin,
                adj_coor[1] + margin: adj_coor[1] + yoff - margin,
                :] = vol  # rfo(vol)[0]
        return main, max_vox
    elif prev is not None:  #  == 'bottom-up':
        # Get distance in z
        fos = 32  # Fixed offset between the planes
        adj_dz = sel_coor[2] - plane_coors[:, 2][0] 
        # curr_bottom_face = main[..., fos]
        # prev_top_face = prev[..., fos + adj_dz]
        curr_bottom_face = main[
            adj_coor[0]: adj_coor[0] + vs[0],
            adj_coor[1]: adj_coor[1] + vs[1],
            fos]  # -1
        prev_vol = prev[
            adj_coor[0]: adj_coor[0] + vs[0],
            adj_coor[1]: adj_coor[1] + vs[1]]
        prev_top_face = prev_vol[..., fos + adj_dz] #  -adj_dz]
        # from matplotlib import pyplot as plt;plt.subplot(121);plt.imshow(rfo(curr_bottom_face)[0]);plt.subplot(122);plt.imshow(rfo(prev_top_face)[0]);plt.show()
        if not prev_top_face.sum():
            # Prev doesn't have any voxels, pass the original
            return main, {}
        if verbose:
            print('Running bottom-up remap')
            elapsed = time.time()
        all_remaps, _, transfers, update = get_remapping(
            main_margin=prev_top_face,
            merge_margin=curr_bottom_face,  # mapping from prev -> main
            parallel=parallel,
            use_numba=False)

        # Get sizes and originals for every remap. Sort these for the final remap
        all_remaps = np.array(all_remaps)
        fixed_remaps = {}
        if len(all_remaps):
            remap_idx = np.argsort(all_remaps[:, -1])[::-1]
            all_remaps = all_remaps[remap_idx]
            unique_remaps, remap_counts = fastremap.unique(all_remaps[:, 0], return_counts=True)
            for ur, rc in zip(unique_remaps, remap_counts):
                if ur != 0:
                    mask = all_remaps[:, 0] == ur
                    fixed_remaps[ur] = all_remaps[mask][0][1]  # Change all to the biggest
            if verbose:
                print('Finished: {}'.format(time.time() - elapsed))
        # del curr_bottom_face

        # Also overwrite the sparse bottom-facing edge in main with the segs in prev
        # main[..., :fos] = prev[..., adj_dz: fos + adj_dz]

        return main, fixed_remaps  # main, max_vox
    else:
        raise RuntimeError('Something fucked up.')


def add_to_main(
        vol,
        sel_coor,
        dtype,
        remap_labels,
        min_vol_size,
        in_place,
        max_vox,
        mincoor,
        res_shape,
        main):
    # Load mains in this plane
    if remap_labels:
        vol, it_max_vox = remap(
            vol=vol,
            min_vol_size=min_vol_size,
            in_place=in_place,
            max_vox=max_vox)
        max_vox = it_max_vox
    adj_coor = sel_coor - mincoor
    vol_shape = np.asarray(vol.shape)
    if not np.all(res_shape == vol_shape):
        vol = resize(
            vol,
            res_shape[:-1],
            anti_aliasing=False,
            preserve_range=True,
            order=0)
        vol_shape = vol.shape
    try:
        main[
            adj_coor[0]: adj_coor[0] + vol_shape[0],
            adj_coor[1]: adj_coor[1] + vol_shape[1]] = vol
    except:
        print(vol.sum())
        import pdb;pdb.set_trace()


def batch_load(sel_coor, sel_path):
    """Load a volume and return its coordinates."""
    vol = np.load(sel_path)
    return [vol, sel_coor]


def increment_max_vox(vol, max_vox, dtype):
    it_max_vox = vol.max()
    zeros = (vol > 0).astype(dtype)
    vol += max_vox  # Update to max_vox
    vol *= zeros
    max_vox = max_vox + it_max_vox + 1  # Increment
    return vol, max_vox


def main(
        conf, 
        load_processed=True,  # Cache intermediate results and reuse if there's a failure
        remap_labels=False,  # noqa Remap each volume's labels to start from 0. # Only needed if you have processed vols that have globally incremented voxel ids.
        in_place=True,  # Use in-place transforms when possible. This will save memory.
        bu_offset=2,
        magic_merge_number_max=7,  # 10
        magic_merge_number_min=5,  # 6
        dtype=np.uint32,  # Data type of merged segmentations
        n_jobs=25,
        h_margin=25,
        bu_margin=25,
        min_vol_size=256):
    """
    Propogate segment ids through overlapping volumes.
    First do this "horizontally" within a fixed range of z coordinates.
    Then propogate "bottom-up" over the z axis.

    conf: Path to the project's config file.
    load_processed: Cache intermediate results and reuse if there's a failure.
    remap_labels: Remap each volume's labels to start from 0.
                  Only needed if you have processed vols that have globally incremented voxel ids.
    in_place: Use in-place transforms when possible. This will save memory.
    bu_margin: ???
    magic_merge_number_max: ???
    magic_merge_number_min: ???
    dtype: Datatype of the merged segments.
    n_jobs: Number of processes for parallization. Reduce to 1 to disable multiproc.
    min_vol_size: Minimum small volume size in merged volume.
                  Only applied if remap_labels=True
    """
    conf = OmegaConf.load(conf)
    file_path = conf.storage.seg_path_root  # For globbing
    seg_path = conf.storage.seg_path_str  # For loading files
    out_dir = conf.storage.merge_seg_path
    res_shape = np.asarray(conf.ds.vol_shape[::-1])
    # version of the main volume. Account for this when merging.
    make_dir(out_dir)

    # Get list of coordinates via glob.
    # Could also pull from DB but this is safer albeit slow.
    paths = glob(
        os.path.join(
            file_path, "**", "**", "**", "*.npy"))  # noqa Assuming file_path + x/y/z + file
    coordinates = np.asarray(
        [[int(c[1:]) for c in p.split(os.path.sep)[-4:-1]] for p in paths])
    unique_z = np.unique(coordinates[:, -1])  # Get z coordinates for merging
    print("Merging over these z coordinates: {}".format(unique_z))

    # Quick load one vol and get shape info
    vol_shape = read_headers(paths[0])

    # Compute extents to preallocate sub-volumes
    mincoor = coordinates.min(0)
    maxcoor = coordinates.max(0)
    maxcoor += np.asarray(res_shape)
    slice_shape = maxcoor - mincoor  # We will process data in this sized
    slice_shape[-1] = vol_shape[-1]  # Use processing-z shape
    # volume partitions.

    # Merge loop
    max_vox, count, prev = 1, 0, None
    all_failed_skeletons = []
    pbar = tqdm(enumerate(unique_z), total=len(unique_z), desc="Z-axis progress")
    with Parallel(max_nbytes=None, n_jobs=n_jobs, require='sharedmem') as parallel:
        for zidx, z in pbar:
            # Allocate tensor
            main = np.zeros(slice_shape, dtype)

            # This plane
            z_sel_coors = coordinates[coordinates[:, 2] == z]
            z_sel_coors = np.unique(z_sel_coors, axis=0)

            # Allow for fast loading for debugging
            skip_processing = False
            if load_processed:
                check_curr = os.path.exists(os.path.join(out_dir, 'plane_z{}.npy'.format(z)))
                if zidx < (len(unique_z) - 1):  # Can't check next on the last slice
                    check_next = os.path.exists(os.path.join(out_dir, 'plane_z{}.npy'.format(unique_z[zidx + 1])))
                else:
                    check_next = False
                if check_curr and check_next:
                    print("This slice and next already processed. Skipping...")
                    continue
                elif check_curr and not check_next:
                    print("This slice is already proccessed. Loading to get max voxel id before processing next.")
                    max_vox = np.load(os.path.join(out_dir, 'plane_z{}.npy'.format(z))).max()
                    continue

            if not skip_processing:
                # Get main/merge splits
                z_sel_coors_main, z_sel_coors_merge = get_main_merge_splits(z_sel_coors)

                # Start processing
                start = time.time()
                vols_vox = None
                if len(z_sel_coors_main):
                    # Now lets parload the data, but sequentially iterate the maxvox, then parload into main
                    main_paths = ["{}.npy".format(seg_path.format(c[0], c[1], c[2], c[0], c[1], c[2])) for c in z_sel_coors_main]  # noqa
                    vols_vox = parallel(delayed(batch_load)(sel_coor, sel_path) for sel_coor, sel_path in tqdm(zip(z_sel_coors_main, main_paths), desc='Z (loading mains): {}'.format(z)))  # noqa

                    # Increment IDs so that they are strictly increasing
                    main_vols, main_sel_coors = [], []
                    for vm in vols_vox:
                        it_vol = vm[0]
                        it_vol, max_vox = increment_max_vox(vol=it_vol, max_vox=max_vox, dtype=dtype)
                        main_vols.append(it_vol), main_sel_coors.append(vm[1])

                # with Parallel(max_nbytes=None, n_jobs=-1, require='sharedmem') as sh_parallel:
                parallel(delayed(add_to_main)(
                    vol=vol,
                    sel_coor=sel_coor,
                    dtype=dtype,
                    remap_labels=remap_labels,
                    min_vol_size=min_vol_size,
                    in_place=in_place,
                    max_vox=max_vox,
                    mincoor=mincoor,
                    res_shape=res_shape,
                    main=main) for vol, sel_coor in tqdm(zip(main_vols, main_sel_coors), desc='Z (Adding mains to main): {}'.format(z)))
            else:
                raise RuntimeError("Should not have 0 mains.")
            del main_vols, it_vol, vols_vox
            gc.collect()
            end = time.time()
            print("Main parloop load time: {}".format(end-start))

            # Loop over merges in batches in case there's tons that exceed the limit
            for it_merge in z_sel_coors_merge:
                start = time.time()
                it_merge_paths = ["{}.npy".format(seg_path.format(c[0], c[1], c[2], c[0], c[1], c[2])) for c in it_merge]  # noqa
                vols_vox = parallel(delayed(batch_load)(sel_coor, sel_path) for sel_coor, sel_path in tqdm(zip(it_merge, it_merge_paths), desc='Z (loading merges): {}'.format(z)))  # noqa
                merge_vols, max_voxes, new_sel_coors = [], [], []
                for vm in vols_vox:
                    it_vol = vm[0]
                    it_vol, max_vox = increment_max_vox(vol=it_vol, max_vox=max_vox, dtype=dtype)
                    merge_vols.append(it_vol), new_sel_coors.append(vm[1])
                del vols_vox
                gc.collect()
                end = time.time()
                print("Merge parloop load time: {}".format(end-start))

                # Perform horizontal merge if there's admixed main/merge
                # for sel_coor in tqdm(z_sel_coors_merge, desc='H Merging: {}'.format(z)):
                for idx, sel_coor in tqdm(enumerate(new_sel_coors), desc='H Merging: {}'.format(z)):
                    main, max_vox = process_merge(
                        main=main,
                        sel_coor=sel_coor,
                        mins=mincoor,
                        vol=merge_vols[idx],
                        parallel=parallel,
                        max_vox=max_vox,
                        plane_coors=z_sel_coors_main,  # np.copy(z_sel_coors_main),
                        min_vol_size=min_vol_size,
                        margin=h_margin,
                        vs=res_shape)
                    pbar.set_description("Z-slice main clock (current max is {})".format(max_vox))
                    if max_vox == 1:
                        import pdb;pdb.set_trace()
                    z_sel_coors_main = np.concatenate((z_sel_coors_main, [sel_coor]), 0)
                del merge_vols
            print("Finished h-merge")

            # Perform bottom-up merge
            if zidx > 0:
                margin = res_shape[-1] * (unique_z[zidx] - unique_z[zidx - 1])
                if margin < res_shape[-1]:
                    all_remaps = {}
                    prev = np.load("merge_coordinates/{}".format(unique_z[zidx - 1]))
                    for sel_coor in tqdm(z_sel_coors_main, desc='BU Merging: {}'.format(z)):
                        main, remaps = process_merge(
                            vol=[1],  # Just pass a dummy for BU
                            main=main,
                            sel_coor=sel_coor,
                            margin_start=margin,
                            margin_end=margin + bu_offset,
                            parallel=parallel,
                            mins=mins,
                            plane_coors=prev_coords,
                            vs=res_shape,
                            min_vol_size=min_vol_size,
                            margin=bu_margin,
                            prev=prev)
                        if len(remaps):
                            all_remaps.update(remaps)
                    if len(all_remaps):
                        # Perform a single remapping
                        print('Performing BU remapping of {} ids'.format(len(all_remaps)))
                        main = fastremap.remap(main, all_remaps, preserve_missing_labels=True, in_place=True)

            print("Finished b-merge")

            # Save the current main and retain info for the next slice
            np.save(os.path.join(out_dir, 'plane_z{}'.format(z)), main)
            z_sel_coors_merge = np.concatenate(z_sel_coors_merge, 0)
        else:
            # Slice is already processed, skip this one.
            if not len(z_sel_coors_main):
                z_sel_coors_main = np.copy(z_sel_coors_merge)
            # max_vox += mv
            print('Skipping plane {}. Current max: {}'.format(z, max_vox))
        unique_mains = np.unique(np.concatenate((z_sel_coors_main[:, :-1], z_sel_coors_merge[:, :-1]), 0), axis=0)  # ADDED TO ENSURE BU-prop AT ALL LOCATIONS
        z_sel_coors_main = np.concatenate((unique_mains, np.zeros((len(unique_mains), 1))), 1)  # ADDED TO ENSURE BU-prop AT ALL LOCATIONS
        z_sel_coors_main = z_sel_coors_main.astype(int)
        prev_coords = z_sel_coors_main

        # Now save this layer's coordinates
        np.savez("merge_coordinates/{}".format(z), main=z_sel_coors_main, merge=z_sel_coors_merge)
        gc.collect()
    ###### ENDING EARLY
    if merge_skeletons:
        np.save("all_failed_skeletons", all_failed_skeletons)


if __name__ == '__main__':
    conf = sys.argv[1]
    main(conf)

