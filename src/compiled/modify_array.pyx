from numpy cimport ndarray as array
import numpy as np
cimport cython
# from cpython cimport array


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef modarr(array parent, array child, array pcoor, array ccoor):
    cdef int i, n=len(pcoor)  # , h=len(xy[0]), w=len(xy[0][0]), d=len(xy[0][0][0])
    # cdef ar[int, ndim=4] new = np.empty((n, h, w, d), dtype=np.int32)  # np.int32
    for i in xrange(n):
        parent[pcoor[i][0], pcoor[i][1], pcoor[i][2]] = child[ccoor[i][0], ccoor[i][1], ccoor[i][2]]
    return parent

