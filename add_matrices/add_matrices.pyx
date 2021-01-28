
# Function that adds together two 2D numpy matrices
import numpy as np
DTYPE = np.float64


def add(double[:, :] A, double[:, :] B):
   
    cdef size_t A_w = A.shape[0]
    cdef size_t A_h = A.shape[1]
    cdef size_t B_w = B.shape[0]
    cdef size_t B_h = B.shape[1]

    assert A_w == B_w and A_h == B_h

    result = np.zeros((A_w, A_h), dtype=DTYPE)
    cdef double[:, :] result_view = result

    cdef int i = 0
    cdef int j = 0
    cdef tmp

    for i in range(0, A_w):
        for j in range(0, A_h):
            tmp = A[i, j] + B[i, j]
            result_view[i, j] = tmp

    return result

