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
    mem_path_str = conf.storage.mem_path_str
    seg_path_str = conf.storage.seg_path_str
    merge_path = conf.storage.merge_seg_path
    res_shape = np.asarray(image_shape)
    token = conf.token

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
        color_layer = ds.layers["color"]
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
        stride = image_shape[-1] - zs[1] - zs[0]
        write_extent = stride + int(stride * 0.5)
        offset = (image_shape[-1] - write_extent) // 2
        
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
                merged_segs = np.load(f)
                parallel(delayed(write_cube)(add_segs, merged_segs, coord, image_shape, zstart, zend, zwritestart) for coord in coords)
                # cube = merged_segs[h: h + image_shape[1], w: w + image_shape[2], zstart: zend]
                # add_segs.write(cube, offset=(h, w, zwritestart))

        # Downsample
        layer_segmentations.downsample(compress=True)

        # Upload
        url = ds.upload()
        print(f"Successfully uploaded {url}")


if __name__ == '__main__':
    conf = sys.argv[1]
    main(conf)

