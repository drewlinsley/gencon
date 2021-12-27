import os
import numpy as np
from skimage import io
from glob import glob
from joblib import Parallel, delayed, parallel_backend
import wkw
from tqdm import tqdm



def cube_data(data, dataset, h_inds, w_inds, d_inds, parallel, cube_size):
    """Figure out w/d inds then process in parallel."""
    inds = []
    for h in h_inds:
        for w in w_inds:
            for d in d_inds:
                inds.append([h, w, d])
    parallel(delayed(convert_save_cubes)(data, dataset, ind, cube_size) for ind in inds)


def convert_save_cubes(data, dataset, ind, cube_size):
    """All coords come from the same z-slice. Save these as npys to cifs."""
    h, w, d = ind
    seg = data[
        h: h + cube_size[0],
        w: w + cube_size[1],
        d: d + cube_size[2]]
    if np.all(np.asarray(seg.shape) > 0):
        dataset.write(np.asarray(ind), seg)
    else:
        print("Failed {}, {}, {}".format(h, w, d))


ims = glob(os.path.join(config.wong_raw_image_path))
zid = [int(x.split("_")[0].split(os.path.sep)[1]) for x in ims]
zarg = np.argsort(zid)
sort_ims = np.asarray(ims)[zarg]
cube_size = [128, 128, 128]

vol_z = sort_ims
cube = []
for f in vol_z:
  cube.append(io.imread(f))
cube = np.asarray(cube).astype(np.uint8)
print("Pretranspose shape: {}".format(cube.shape))
cube = cube.transpose(2, 1, 0)
print("Posttranspose shape: {}".format(cube.shape))

cube_shape = cube.shape
h_inds = np.arange(0, cube_shape[0], cube_size[0], dtype=int)
w_inds = np.arange(0, cube_shape[1], cube_size[1], dtype=int)
d_inds = np.arange(0, cube_shape[2], cube_size[2], dtype=int)

# Write the dataset to wkw
dtype = np.uint8
dataset = wkw.Dataset.open(
    config.wong_wkw_image_path,  
    wkw.Header(dtype))
with Parallel(n_jobs=-1, backend='threading') as parallel:
    cube_data(cube, dataset=dataset, cube_size=cube_size, parallel=parallel, h_inds=h_inds, w_inds=w_inds, d_inds=d_inds)

# Now split the dataset up into coordinates, adjusted for segmentation size. These
# are the coords we will read with our reconstruction workers. 

