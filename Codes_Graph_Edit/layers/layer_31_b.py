# -*- coding: utf-8 -*-
"""
Weights and bias of layer 31

@author: Ghafoor
"""

import numpy as np
from scipy.sparse import csr_matrix, hstack, vstack


import numpy as np
from scipy.sparse import csr_matrix, hstack, vstack




import numpy as np
from scipy.sparse import csr_matrix, hstack, vstack

import numpy as np
from scipy.sparse import csr_matrix, hstack, vstack, identity

def build_W31(n, d, C):
    """
    Memory-efficient construction of W31.

    Structure (columns): [T3 (n_2d) | H1 (n_2d*n_2d) | H2 (n*d) | H3 (2*d)]
    Blocks:
      block1: rows = n_2d
      block2: rows = n_2d * n_2d
      block3: rows = d
      block4: rows = d
    """
    n_2d = n + 2 * d

    # Column widths
    cols_T3 = n_2d
    cols_H1_H4_H7 = n_2d * n_2d
    cols_H2_H5_H8 = n * d
    cols_H3_H6_H9 = 2 * d

    # -----------------------
    # BLOCK 1: U2 as identity map (rows1 = n_2d)
    # -----------------------
    rows1 = n_2d
    T3_block1 = identity(rows1, format='csr', dtype=np.float64)
    H1_block1 = csr_matrix((rows1, cols_H1_H4_H7))
    H2_block1 = csr_matrix((rows1, cols_H2_H5_H8))
    H3_block1 = csr_matrix((rows1, cols_H3_H6_H9))

    block1 = hstack([T3_block1, H1_block1, H2_block1, H3_block1], format='csr')

    # -----------------------
    # BLOCK 2: tau2 as identity map (rows2 = n_2d * n_2d)
    # -----------------------
    rows2 = n_2d * n_2d
    T3_block2 = csr_matrix((rows2, cols_T3))
    # H1 is identity on tau2 section
    H1_block2 = identity(rows2, format='csr', dtype=np.float64)
    H2_block2 = csr_matrix((rows2, cols_H2_H5_H8))
    H3_block2 = csr_matrix((rows2, cols_H3_H6_H9))

    block2 = hstack([T3_block2, H1_block2, H2_block2, H3_block2], format='csr')

    # -----------------------
    # BLOCK 3: x' nodes (rows3 = d)
    # -----------------------
    rows3 = d
    # T3 zeros
    T3_block3 = csr_matrix((rows3, cols_T3))
    # H4 zeros (tau2 section)
    H4_block3 = csr_matrix((rows3, cols_H1_H4_H7))

    # H5: for each k in 0..d-1 (row k) set columns (i*d + k) for i=0..n-1 to value C
    # number of nonzeros = n * d
    H5_rows = np.repeat(np.arange(d, dtype=np.int64), n)     # each k repeated n times
    # for each k, columns = i*d + k for i=0..n-1
    i_indices = np.tile(np.arange(n, dtype=np.int64), d)
    H5_cols = i_indices * d + np.repeat(np.arange(d, dtype=np.int64), n)
    H5_data = np.full_like(H5_cols, fill_value=float(C), dtype=np.float64)

    H5_block3 = csr_matrix((H5_data, (H5_rows, H5_cols)),
                          shape=(rows3, cols_H2_H5_H8))

    # H6: identity (d x (2*d) section) placed in first d columns of H3 block
    # rows: 0..d-1, cols: 0..d-1
    H6_rows = np.arange(d, dtype=np.int64)
    H6_cols = np.arange(d, dtype=np.int64)
    H6_data = np.ones(d, dtype=np.float64)
    H6_block3 = csr_matrix((H6_data, (H6_rows, H6_cols)), shape=(rows3, cols_H3_H6_H9))

    block3 = hstack([T3_block3, H4_block3, H5_block3, H6_block3], format='csr')

    # -----------------------
    # BLOCK 4: x_{j+d} nodes (rows4 = d)
    # -----------------------
    rows4 = d
    T3_block4 = csr_matrix((rows4, cols_T3))
    H7_block4 = csr_matrix((rows4, cols_H1_H4_H7))
    H8_block4 = csr_matrix((rows4, cols_H2_H5_H8))

    # H9: identity offset by d (put identity in columns d..2d-1 of H3 block)
    H9_rows = np.arange(d, dtype=np.int64)
    H9_cols = np.arange(d, d + d, dtype=np.int64)  # d .. 2d-1
    H9_data = np.ones(d, dtype=np.float64)
    H9_block4 = csr_matrix((H9_data, (H9_rows, H9_cols)), shape=(rows4, cols_H3_H6_H9))

    block4 = hstack([T3_block4, H7_block4, H8_block4, H9_block4], format='csr')

    # -----------------------
    # Assemble W31
    # -----------------------
    W31 = vstack([block1, block2, block3, block4], format='csr')

    # Optional: quick verification (comment out if not desired)
    # expected_rows = n_2d + (n_2d * n_2d) + d + d
    # expected_cols = cols_T3 + cols_H1_H4_H7 + cols_H2_H5_H8 + cols_H3_H6_H9
    # assert W31.shape == (expected_rows, expected_cols)

    return W31


def build_B31(n, d, C):
    n_2d = n + 2*d

    # Section sizes (based on your loops)
    rows_U2   = n_2d
    rows_tau2 = n_2d * n_2d
    rows_x    = d                 # first d entries = -C
    rows_xd   = d                 # last d entries = 0

    total_rows = rows_U2 + rows_tau2 + rows_x + rows_xd

    # Only the first `rows_x` of the last 2d-block contain non-zero values (-C)
    non_zero_count = rows_x

    data = np.full(non_zero_count, -C, dtype=np.float64)

    # positions of non-zero entries
    base_index = rows_U2 + rows_tau2  # skip initial zero blocks
    rows = np.arange(base_index, base_index + rows_x, dtype=np.int32)

    # Sparse column vector
    B31 = csr_matrix((data, (rows, np.zeros(non_zero_count, dtype=np.int32))),
                     shape=(total_rows, 1))

    return B31
