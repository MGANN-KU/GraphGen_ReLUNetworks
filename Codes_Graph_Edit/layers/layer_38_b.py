# -*- coding: utf-8 -*-
"""
Weights and bias of layer 38

@author: Ghafoor
"""
import numpy as np
from scipy.sparse import coo_matrix
import sys

from scipy.sparse import eye, csr_matrix, hstack, vstack, lil_matrix
import numpy as np

import numpy as np
from scipy.sparse import csr_matrix, hstack, vstack

import numpy as np
from scipy.sparse import csr_matrix, hstack, vstack

import numpy as np
from scipy.sparse import csr_matrix, hstack, vstack

import numpy as np
from scipy.sparse import csr_matrix, hstack, vstack

def build_W38(n, d, C):
    # sizes (zero-based index math used)
    n_d = n + d
    n_2d = n + 2 * d
    d2 = d + 1

    # ---------- BLOCK 1 ----------
    rows1 = n_d * n_2d * d2

    # row index decomposition for block1:
    # loop order in original: for i in 1..n_d, for l in 1..n_2d, for j in 1..d2
    i_idx = np.repeat(np.arange(n_d), n_2d * d2)               # length rows1
    l_idx = np.tile(np.repeat(np.arange(n_2d), d2), n_d)
    j_idx = np.tile(np.arange(d2), n_d * n_2d)

    # --- G1: one 1.0 when k = i + j - 1 in [1..n_2d] and p == l  ---
    k_idx = i_idx + j_idx            # zero-based k: corresponds to (i+j-1)-1
    mask_valid = (k_idx >= 0) & (k_idx < n_2d)

    # columns for G1: col = k * n_2d + l
    G1_row = np.nonzero(mask_valid)[0]
    if G1_row.size:
        G1_col = (k_idx[mask_valid] * n_2d + l_idx[mask_valid]).astype(np.int64)
        G1_data = np.ones_like(G1_col, dtype=np.float64)
    else:
        G1_col = np.array([], dtype=np.int64)
        G1_data = np.array([], dtype=np.float64)

    G1 = csr_matrix((G1_data, (G1_row, G1_col)),
                    shape=(rows1, n_2d * n_2d))

    # --- G2 & G3: two entries per row (q=0 -> +C, q=1 -> -C) when
    # columns formula: (p-1)*n_2d*d2*2 + (r-1)*d2*2 + (k-1)*2 + q
    # here p=i, r=l, k=j
    # build both q=0 and q=1 in vectorized form
    base_col = (i_idx.astype(np.int64) * (n_2d * d2 * 2)
                + l_idx.astype(np.int64) * (d2 * 2)
                + j_idx.astype(np.int64) * 2)
    # repeat each row index twice (for q=0 and q=1)
    G23_rows = np.repeat(np.arange(rows1, dtype=np.int64), 2)
    G23_cols = np.empty(rows1 * 2, dtype=np.int64)
    G23_cols[0::2] = base_col + 0   # q = 0
    G23_cols[1::2] = base_col + 1   # q = 1
    G23_data = np.empty(rows1 * 2, dtype=np.float64)
    G23_data[0::2] = C
    G23_data[1::2] = -C

    # Both G2 and G3 have same entries in original code
    G2 = csr_matrix((G23_data, (G23_rows, G23_cols)),
                    shape=(rows1, n_d * n_2d * d2 * 2))
    G3 = csr_matrix((G23_data, (G23_rows, G23_cols)),
                    shape=(rows1, n_d * n_2d * d2 * 2))

    # T10, G4..G7 are zero blocks with known shapes
    T10 = csr_matrix((rows1, n_2d))
    G4 = csr_matrix((rows1, n_d * d2 * 2))
    G5 = csr_matrix((rows1, n_d * d2 * 2))
    G6 = csr_matrix((rows1, n_d * n_d * d2 * 2))
    G7 = csr_matrix((rows1, n_d * n_d * d2 * 2))

    block1 = hstack([T10, G1, G2, G3, G4, G5, G6, G7], format='csr')

    # ---------- BLOCK 2 ----------
    rows2 = n_d * d2
    # T10 for block2: rows indexed by (i,j)
    i2_idx = np.repeat(np.arange(n_d), d2)
    j2_idx = np.tile(np.arange(d2), n_d)
    k2_idx = i2_idx + j2_idx
    mask2 = (k2_idx >= 0) & (k2_idx < n_2d)

    T10_rows = np.nonzero(mask2)[0]
    if T10_rows.size:
        T10_cols = k2_idx[mask2].astype(np.int64)
        T10_data = np.ones_like(T10_cols, dtype=np.float64)
    else:
        T10_cols = np.array([], dtype=np.int64)
        T10_data = np.array([], dtype=np.float64)

    T10_b2 = csr_matrix((T10_data, (T10_rows, T10_cols)), shape=(rows2, n_2d))

    # G8, G9, G10 are zero blocks
    G8 = csr_matrix((rows2, n_2d * n_2d))
    G9 = csr_matrix((rows2, n_d * n_2d * d2 * 2))
    G10 = csr_matrix((rows2, n_d * n_2d * d2 * 2))

    # G11: two entries per (i,j) row mapping into col_block
    col_block = n_d * d2 * 2
    base_col_b2 = (i2_idx.astype(np.int64) * (d2 * 2)
                   + j2_idx.astype(np.int64) * 2)
    G11_rows = np.repeat(np.arange(rows2, dtype=np.int64), 2)
    G11_cols = np.empty(rows2 * 2, dtype=np.int64)
    G11_cols[0::2] = base_col_b2 + 0
    G11_cols[1::2] = base_col_b2 + 1
    G11_data = np.empty(rows2 * 2, dtype=np.float64)
    G11_data[0::2] = C
    G11_data[1::2] = -C
    G11 = csr_matrix((G11_data, (G11_rows, G11_cols)), shape=(rows2, col_block))

    # G12 is essentially same structure as G11 in your original loops
    G12 = csr_matrix((G11_data, (G11_rows, G11_cols)), shape=(rows2, col_block))

    G13 = csr_matrix((rows2, n_d * n_d * d2 * 2))
    G14 = csr_matrix((rows2, n_d * n_d * d2 * 2))

    block2 = hstack([T10_b2, G8, G9, G10, G11, G12, G13, G14], format='csr')

    # ---------- BLOCK 3 ----------
    rows3 = n_d * n_d * d2
    # row ordering: for i in 1..n_d, for l in 1..n_d, for j in 1..d2
    i3_idx = np.repeat(np.arange(n_d), n_d * d2)
    l3_idx = np.tile(np.repeat(np.arange(n_d), d2), n_d)
    j3_idx = np.tile(np.arange(d2), n_d * n_d)

    T10_b3 = csr_matrix((rows3, n_2d))
    G15 = csr_matrix((rows3, n_2d * n_2d))
    G16 = csr_matrix((rows3, n_d * n_2d * d2 * 2))
    G17 = csr_matrix((rows3, n_d * n_2d * d2 * 2))
    G18 = csr_matrix((rows3, n_d * d2 * 2))
    G19 = csr_matrix((rows3, n_d * d2 * 2))

    # G20: two entries per row, with values 1 and -1
    col_block3 = n_d * n_d * d2 * 2
    base_col_b3 = ((i3_idx.astype(np.int64) * n_d + l3_idx.astype(np.int64)) * d2 * 2
                   + j3_idx.astype(np.int64) * 2)
    G20_rows = np.repeat(np.arange(rows3, dtype=np.int64), 2)
    G20_cols = np.empty(rows3 * 2, dtype=np.int64)
    G20_cols[0::2] = base_col_b3 + 0
    G20_cols[1::2] = base_col_b3 + 1
    G20_data = np.empty(rows3 * 2, dtype=np.float64)
    G20_data[0::2] = 1.0
    G20_data[1::2] = -1.0

    G20 = csr_matrix((G20_data, (G20_rows, G20_cols)), shape=(rows3, col_block3))
    G21 = G20.copy()

    block3 = hstack([T10_b3, G15, G16, G17, G18, G19, G20, G21], format='csr')

    # final assembly
    W38 = vstack([block1, block2, block3], format='csr')
    return W38


import numpy as np

def build_B38(n, d, C):
    n_d  = n + d
    n_2d = n + 2*d
    d2   = d + 1

    rows_block1 = n_d * n_2d * d2
    rows_block2 = n_d * d2
    rows_block3 = n_d * n_d * d2

    # Build each block directly with final values
    block1 = np.full(rows_block1, -2.0 * C, dtype=np.float32)
    block2 = np.full(rows_block2, -2.0 * C, dtype=np.float32)
    block3 = np.full(rows_block3, -1.0, dtype=np.float32)

    # Concatenate once (efficient)
    return np.concatenate((block1, block2, block3))
