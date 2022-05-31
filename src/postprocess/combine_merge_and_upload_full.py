import os
import sys
from time import gmtime, strftime
import numpy as np
import webknossos as wk
from webknossos.dataset import COLOR_CATEGORY, SEGMENTATION_CATEGORY
from webknossos.dataset.properties import LayerViewConfiguration
from skimage.segmentation import relabel_sequential as rs
from skimage.transform import resize
from omegaconf import OmegaConf
from tqdm import tqdm
from glob import glob
from joblib import Parallel, delayed
import fastremap
from src.postprocess.merge_segments import get_remapping


def process_bu_merge(
        main,
        prev,
        margin,
        dz):
    """Merge ids from prev to main.
    Return the remaps to pass through fastremap later."""
    # Use all h/w coords
    curr_bottom_face = main[..., margin]
    prev_top_face = prev[..., margin + dz]
    if not prev_top_face.sum():
        # Prev doesn't have any voxels, pass the original
        return {}

    all_remaps, _, transfers, update = get_remapping(
        main_margin=prev_top_face,
        merge_margin=curr_bottom_face,  # mapping from prev -> main
        parallel=False,
        use_numba=False)

    return all_remaps


def write_cube(add_segs, data, coord, image_shape, zstart, zend, zwritestart):
    h, w = coord
    cube = data[h: h + image_shape[1], w: w + image_shape[2], zstart: zend]
    add_segs.write(cube, offset=(h, w, zwritestart))


def main(conf, n_jobs=1):
    """Upload a partially processed volume."""
    conf = OmegaConf.load(conf)
    name = conf.name
    ds_input = conf.ds.path
    image_layer = conf.ds.img_layer
    scale = tuple(conf.ds.scale)
    image_shape = conf.ds.vol_shape[::-1]
    merge_path = conf.storage.merge_seg_path
    res_shape = np.asarray(image_shape)
    extent = conf.ds.extent
    token = conf.token
    color_layer_name = conf.ds.img_layer

    seg_dtype = np.uint32
    mem_dtype = np.uint8
    img_dtype = np.uint8
    with wk.webknossos_context(url="https://webknossos.org", token=token):

        # Get native resolution
        ds = wk.Dataset(ds_input, scale=scale, exist_ok=True)
        try:
            ds.delete_layer("segmentation")
            print("Deleting prior segmentation layer.")
        except:
            print("No prior segmentation layer found.")

        # Compress and downsample images
        color_layer = ds.get_layer(image_layer)
        color_mag = color_layer.get_mag("1")
        color_layer = ds.layers[color_layer_name]
        color_mag = color_layer.get_mag("1")
        # color_layer.downsample(compress=True)

        # Add segmentations
        layer_segmentations = ds.add_layer(
            "segmentation",
            SEGMENTATION_CATEGORY,
            seg_dtype,
            num_channels=1,
            largest_segment_id=4294000000,  # In the future write this out during the merge proc
        )  # color? maxval? channels?
        add_segs = layer_segmentations.add_mag(1)
        
        # Write segmentations incrementally assuming they're too big for a one-shot
        seg_files = glob(os.path.join(merge_path, "*.npy"))
        seg_files = [x for x in seg_files if "max" not in x]
        height, width, _ = tuple(ds._properties.data_layers[0].bounding_box.size)
        seg_files = sorted(seg_files, key=os.path.getctime)
        zs = [int(f.split(os.path.sep)[-1].split("_")[1].split(".")[0][1:]) for f in seg_files]

        if extent:
            height = min(height, extent[0])
            width = min(width, extent[1])
        stride = image_shape[-1] - zs[1] - zs[0]
        write_extent = stride + int(stride * 0.5)
        offset = int((image_shape[-1] - write_extent) * 0.75)
        
        # In this loop we will fill in wk while avoiding edge artifacts
        start_z = 0
        with Parallel(n_jobs=n_jobs) as parallel:
            for idx, f in tqdm(enumerate(seg_files), desc="Processing merged segmentation files", total=len(seg_files)):
                z = int(f.split(os.path.sep)[-1].split("_")[1].split(".")[0][1:])

                if idx == 0:
                    # First slice, start from 0
                    zstart = 0
                    zend = write_extent
                    zwritestart = 0
                elif idx == len(seg_files) - 1:
                    # Last slice, go until the end
                    zstart = offset * 2
                    zend = image_shape[-1]
                    zwritestart = z + (offset * 2)
                else:
                    zstart = offset
                    zend = image_shape[-1] - offset
                    zwritestart = z + offset
                zwriteend = zwritestart + (zend - zstart)
                print("Writing from {} to {} in slice [{}, {}]".format(zwritestart, zwriteend, z, z + image_shape[-1]))
                coords = []
                for h in range(0, height, image_shape[1]):
                    for w in range(0, width, image_shape[2]):
                        coords.append((h, w))
                try:
                    merged_segs = np.load(f)
                except:
                    import pdb;pdb.set_trace()
                    print("Failed to load {}".format(f))

                # Trim
                if extent:
                    merged_segs = merged_segs[:height, :width, :]

                if idx > 0:
                    # BU merge
                    all_remaps = process_bu_merge(
                        main=merged_segs,
                        prev=prev,
                        margin=offset,
                        dz=stride)
                    if len(all_remaps):
                        # Postprocess remaps to assign ids to biggest neurite then perform a single remapping
                        fixed_remaps = {}
                        remap_idx = np.argsort(all_remaps[:, -1])[::-1]
                        all_remaps = all_remaps[remap_idx]
                        unique_remaps, remap_counts = fastremap.unique(all_remaps[:, 0], return_counts=True)
                        for ur, rc in zip(unique_remaps, remap_counts):
                            if ur != 0:
                                mask = all_remaps[:, 0] == ur
                                # fixed_remaps[ur] = all_remaps[mask][0][1]  # Change all to the biggest
                                biggest = all_remaps[mask][0][1]  # Get the biggest new id
                                fixed_remaps[ur] = biggest
                                rems = all_remaps[mask][1:]
                                for rem in rems:
                                    if rem[1] > 0:
                                        fixed_remaps[rem[1]] = biggest  # Change remainders in this selection to biggest
                        print('Performing BU remapping of {} ids'.format(len(fixed_remaps)))
                        merged_segs = fastremap.remap(merged_segs, fixed_remaps, preserve_missing_labels=True, in_place=True)

                parallel(delayed(write_cube)(add_segs, merged_segs, coord, image_shape, zstart, zend, zwritestart) for coord in coords)
                prev = merged_segs

        # Downsample
        layer_segmentations.downsample(compress=True)

        # # Upload
        # url = ds.upload()
        # print(f"Successfully uploaded {url}")


if __name__ == '__main__':
    conf = sys.argv[1]
    main(conf)

