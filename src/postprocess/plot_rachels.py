import numpy as np
from matplotlib import pyplot as plt
from skimage import measure


me = np.load("/localscratch/middle_cube_membranes_for_ffn_training.npy")
d = np.load("/localscratch/middle_cube_segmentation/0/0/seg-0_0_0.npz")
seg = d["segmentation"]
imn=32
plt.subplot(131)
plt.imshow(measure.label(seg[imn]))
plt.subplot(132)
plt.imshow(me[imn, ..., 1])
plt.subplot(133)
plt.imshow(me[imn, ..., 0], cmap="Greys_r")
plt.show()
