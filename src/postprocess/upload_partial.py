from time import gmtime, strftime
import numpy as np
import webknossos as wk
from webknossos.dataset import COLOR_CATEGORY, SEGMENTATION_CATEGORY
from webknossos.dataset.properties import LayerViewConfiguration
from skimage.segmentation import relabel_sequential as rs
from skimage.transform import resize


# Create the dataset
x, y, z = 0, 0, 0
scale = (5, 5, 50)
seg_dtype = np.uint32
mem_dtype = np.uint8
ds_input = "/media/data_cifs/connectomics/cubed_mag1/pbtest/wong_1"
token = "UTUOQJvbbyRFbD_NSnTMig"
image_shape = [240, 4740, 4740]
with wk.webknossos_context(url="https://webknossos.org", token=token):
    img_mem = np.load("/media/data_cifs/projects/prj_connectomics/wong/mag1_membranes/x{}/y{}/z{}/110629_k0725_mag1_x{}_y{}_z{}.npy".format(x, y, z, x, y, z))
    # img = img_mem[..., 0]
    mem = img_mem[..., 1].astype(mem_dtype).transpose(1, 2, 0)
    seg = np.load("/media/data_cifs/projects/prj_connectomics/wong/mag1_segs/x{}/y{}/z{}/110629_k0725_mag1_x{}_y{}_z{}.npy".format(x, y, z, x, y, z))
    seg = rs(seg)[0].astype(seg_dtype)

    # Get native resolution
    ds_native = wk.Dataset(ds_input, scale=[5, 5, 50], exist_ok=True)
    layer = ds_native.get_layer("color")
    mag1 = layer.get_mag("1")
    print("Extracting images")
    img = mag1.read((x, y, z), image_shape[::-1])[0]

    # Resize images
    res_shape = np.asarray(image_shape)
    print("Resizing mems and segs")
    mem = resize(mem, res_shape[::-1][:-1], anti_aliasing=False, preserve_range=True, order=0)
    seg = resize(seg, res_shape[::-1][:-1], anti_aliasing=False, preserve_range=True, order=0)

    # Choose a name for our dataset
    time_str = strftime("%Y-%m-%d_%H-%M-%S", gmtime())
    name = "W-Q_x{}_y{}_z{}_{}".format(x, y, z, time_str)

    # Create the dataset
    ds = wk.Dataset(name, scale=scale)

    # Add images
    layer_images = ds.add_layer(
        "images",
        COLOR_CATEGORY,
        np.uint8,
    )
    layer_images.add_mag(1, compress=True).write(img)
    layer_images.downsample()
    del img, img_mem, mag1, ds_native, layer

    # Add membranes
    # mem_config = LayerViewConfiguration(color=[255, 0, 0])
    layer_membranes = ds.add_layer(
        "membranes",
        COLOR_CATEGORY,  # SEGMENTATION_CATEGORY,
        mem_dtype,
        num_channels=1,
        # default_view_configuration=mem_config,
        largest_segment_id=1,  # np.iinfo(np.uint8).max
    )
    layer_membranes.add_mag(1, compress=True).write(mem)
    layer_membranes.downsample()
    del mem

    # Add segmentations
    layer_segmentations = ds.add_layer(
        "segmentations",
        SEGMENTATION_CATEGORY,
        seg_dtype,
        num_channels=1,
        # compressed=True,
        largest_segment_id=seg.max(),
    )  # color? maxval? channels?
    layer_segmentations.add_mag(1, compress=True).write(seg)
    layer_segmentations.downsample()

    # Upload
    url = ds.upload()
    print(f"Successfully uploaded {url}")

