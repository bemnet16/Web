import numpy as np


def show_dimension(arr):
    np_arr = np.asarray(arr)
    return np_arr.ndim


arr_1d = [1, 2, 3, 4, 5]
print(show_dimension(arr_1d))
arr_2d = [[1, 2, 3], [4, 5, 6]]
print(show_dimension(arr_2d))

arr_3d = [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]
print(show_dimension(arr_3d))
