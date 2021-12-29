from time import gmtime, strftime
import numpy as np
import webknossos as wk
from webknossos.dataset import COLOR_CATEGORY, SEGMENTATION_CATEGORY
from webknossos.dataset.properties import LayerViewConfiguration
from config import Config
from skimage.segmentation import relabel_sequential as rs


# Create the dataset
config = Config()
x, y, z = 0, 0, 0
scale = (5, 5, 50)
seg_dtype = np.int32
with wk.webknossos_context(url="http://localhost:9000", token=config.token):
    img_mem = np.load("/cifs/data/tserre_lrs/projects/prj_connectomics/wong/mag1_membranes/x{}/y{}/z{}/110629_k0725_mag1_x{}_y{}_z{}.npy".format(x, y, z, x, y, z))
    img = img_mem[..., 0]
    mem = img_mem[..., 1]
    seg = np.load("/cifs/data/tserre_lrs/projects/prj_connectomics/wong/mag1_segs/x{}/y{}/z{}/110629_k0725_mag1_x{}_y{}_z{}.npy".format(x, y, z, x, y, z))
    seg = rs(seg)[0].astype(seg_dtype).transpose(2, 0, 1)

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

    # Add membranes
    mem_config = LayerViewConfiguration(color=[255, 0, 0])
    layer_membranes = ds.add_layer(
        "membranes",
        SEGMENTATION_CATEGORY,
        np.uint8,
        num_channels=1,
        default_view_configuration=mem_config,
        largest_segment_id=np.iinfo(np.uint8).max
    )
    layer_membranes.add_mag(1, compress=True).write(mem)
    layer_membranes.downsample()

    # Add segmentations
    layer_segmentations = ds.add_layer(
        "segmentations",
        SEGMENTATION_CATEGORY,
        seg_dtype,
        num_channels=1,
        # compressed=True,
        largest_segment_id=8000,  # seg.max()
    )  # color? maxval? channels?
    layer_segmentations.add_mag(1, compress=True).write(seg)
    layer_segmentations.downsample()

    # Upload
    url = ds.upload()
    print(f"Successfully uploaded {url}")

