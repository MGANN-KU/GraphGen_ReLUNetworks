# -*- coding: utf-8 -*-
"""
Weights and bias of layer 32

@author: Ghafoor
"""

import numpy as np
from scipy.sparse import csr_matrix, hstack, vstack


import numpy as np
from scipy.sparse import csr_matrix, hstack, vstack




import numpy as np
from scipy.sparse import csr_matrix, hstack, vstack

import numpy as np
from scipy.sparse import csr_matrix, hstack, vstack

import numpy as np
from scipy.sparse import csr_matrix, hstack, vstack, identity

def build_W32(n, d, eps):
    """
    Memory-efficient construction of W32 with fixed duplicate entries issue.
    """
    n_2d = n + 2 * d
    one_over_eps = 1.0 / eps

    # Column sizes
    cols_T4 = n_2d
    cols_H10_etc = n_2d * n_2d
    cols_H11_etc = d
    cols_H12_etc = d
    total_cols = cols_T4 + cols_H10_etc + cols_H11_etc + cols_H12_etc

    blocks = []

    # -----------------------
    # BLOCK 1: U2 as identity map
    # -----------------------
    rows1 = n_2d
    T4_block1 = identity(rows1, format='csr', dtype=np.float64)
    block1 = hstack([
        T4_block1,
        csr_matrix((rows1, cols_H10_etc)),
        csr_matrix((rows1, cols_H11_etc)),
        csr_matrix((rows1, cols_H12_etc))
    ], format='csr')
    blocks.append(block1)

    # -----------------------
    # BLOCK 2: tau2 as identity map
    # -----------------------
    rows2 = n_2d * n_2d
    H10_block2 = identity(rows2, format='csr', dtype=np.float64)
    block2 = hstack([
        csr_matrix((rows2, cols_T4)),
        H10_block2,
        csr_matrix((rows2, cols_H11_etc)),
        csr_matrix((rows2, cols_H12_etc))
    ], format='csr')
    blocks.append(block2)

    # -----------------------
    # BLOCKS 3 & 4 & 5 & 6: patterns with rows = n_2d * d * 2
    # -----------------------
    rows_3_6 = n_2d * d * 2
    r = np.arange(rows_3_6, dtype=np.int64)
    
    # Compute indices
    # Original loops: for i in range(1, n+2*d+1):
    #                  for j in range(1, d+1):
    #                    for q in range(2):
    div = d * 2  # (d * 2) elements for each i
    i_vals = r // div            # i index (0-based)
    rem = r % div
    j_vals = rem // 2            # j index (0-based)
    q_vals = rem % 2             # q in {0,1}

    # BLOCK 3: alpha3/alpha4 -> H14 (1/eps where j==k)
    # Each row has ONE nonzero at column j_vals
    rows_block3 = r
    cols_block3 = j_vals  # Column in H11 section
    data_block3 = one_over_eps * np.ones(rows_3_6, dtype=np.float64)
    
    H14_block3 = csr_matrix((data_block3, (rows_block3, cols_block3)), 
                           shape=(rows_3_6, cols_H11_etc))

    block3 = hstack([
        csr_matrix((rows_3_6, cols_T4)),
        csr_matrix((rows_3_6, cols_H10_etc)),
        H14_block3,
        csr_matrix((rows_3_6, cols_H12_etc))
    ], format='csr')
    blocks.append(block3)

    # BLOCK 4: beta3/beta4 -> H17 (-1/eps where j==k)
    data_block4 = -one_over_eps * np.ones(rows_3_6, dtype=np.float64)
    H17_block4 = csr_matrix((data_block4, (rows_block3, cols_block3)), 
                           shape=(rows_3_6, cols_H11_etc))

    block4 = hstack([
        csr_matrix((rows_3_6, cols_T4)),
        csr_matrix((rows_3_6, cols_H10_etc)),
        H17_block4,
        csr_matrix((rows_3_6, cols_H12_etc))
    ], format='csr')
    blocks.append(block4)

    # BLOCK 5: alpha5/alpha6 -> H20 (1/eps where j == i)
    # Note: This block has different indexing in original (k,j,q instead of i,j,q)
    # Let's recalculate indices for this specific pattern
    # Original: for k in range(1, n+2*d+1):
    #            for j in range(1, d+1):
    #              for q in range(2):
    # So rows are still n_2d * d * 2, but organized differently
    # For our purposes, the pattern is the same: each row gets a value at column j
    
    # Actually, looking at the original H20 code:
    # if j==i: w=1/eps
    # So for block 5, each row corresponds to (k,j,q) and gets a value at column j when j==i
    # But in this block, 'i' refers to the column index in H20 (which goes from 1 to d+1)
    # So essentially same as blocks 3-4: each row gets value at column j_vals
    
    data_block5 = one_over_eps * np.ones(rows_3_6, dtype=np.float64)
    H20_block5 = csr_matrix((data_block5, (r, j_vals)), 
                           shape=(rows_3_6, cols_H11_etc))

    block5 = hstack([
        csr_matrix((rows_3_6, cols_T4)),
        csr_matrix((rows_3_6, cols_H10_etc)),
        H20_block5,
        csr_matrix((rows_3_6, cols_H12_etc))
    ], format='csr')
    blocks.append(block5)

    # BLOCK 6: beta5/beta6 -> H23 (-1/eps where j == k)
    data_block6 = -one_over_eps * np.ones(rows_3_6, dtype=np.float64)
    H23_block6 = csr_matrix((data_block6, (r, j_vals)), 
                           shape=(rows_3_6, cols_H11_etc))

    block6 = hstack([
        csr_matrix((rows_3_6, cols_T4)),
        csr_matrix((rows_3_6, cols_H10_etc)),
        H23_block6,
        csr_matrix((rows_3_6, cols_H12_etc))
    ], format='csr')
    blocks.append(block6)

    # -----------------------
    # BLOCKS 7 & 8: small d*2 blocks (rows = 2*d)
    # -----------------------
    rows7 = d * 2
    r7 = np.arange(rows7, dtype=np.int64)
    # j pattern: j = r7 // 2 (0-based)
    j_vals7 = r7 // 2
    
    # BLOCK 7: H26 (1/eps where j==k) in H11, H27 (-1/eps where j+d==k) in H12
    # H26: column j_vals7, value +1/eps
    # H27: column j_vals7, value -1/eps
    
    # For H26 (in H11 section)
    data_h26 = one_over_eps * np.ones(rows7, dtype=np.float64)
    H26_block7 = csr_matrix((data_h26, (r7, j_vals7)), shape=(rows7, cols_H11_etc))
    
    # For H27 (in H12 section)
    data_h27 = -one_over_eps * np.ones(rows7, dtype=np.float64)
    H27_block7 = csr_matrix((data_h27, (r7, j_vals7)), shape=(rows7, cols_H12_etc))

    block7 = hstack([
        csr_matrix((rows7, cols_T4)),
        csr_matrix((rows7, cols_H10_etc)),
        H26_block7,
        H27_block7
    ], format='csr')
    blocks.append(block7)

    # BLOCK 8: H29 (-1/eps where j==k) in H11, H30 (+1/eps where j+d==k) in H12
    data_h29 = -one_over_eps * np.ones(rows7, dtype=np.float64)
    H29_block8 = csr_matrix((data_h29, (r7, j_vals7)), shape=(rows7, cols_H11_etc))
    
    data_h30 = one_over_eps * np.ones(rows7, dtype=np.float64)
    H30_block8 = csr_matrix((data_h30, (r7, j_vals7)), shape=(rows7, cols_H12_etc))

    block8 = hstack([
        csr_matrix((rows7, cols_T4)),
        csr_matrix((rows7, cols_H10_etc)),
        H29_block8,
        H30_block8
    ], format='csr')
    blocks.append(block8)

    # -----------------------
    # Assemble W32
    # -----------------------
    W32 = vstack(blocks, format='csr')
    
    # Check for duplicates and sum them if needed
    # Convert to COO to sum duplicates, then back to CSR
    W32_coo = W32.tocoo()
    
    # Sum duplicates
    from scipy.sparse import coo_matrix
    unique_coords = {}
    for i, j, v in zip(W32_coo.row, W32_coo.col, W32_coo.data):
        if (i, j) in unique_coords:
            unique_coords[(i, j)] += v
        else:
            unique_coords[(i, j)] = v
    
    # Create new arrays
    rows, cols, data = [], [], []
    for (i, j), v in unique_coords.items():
        rows.append(i)
        cols.append(j)
        data.append(v)
    
    W32_clean = coo_matrix((data, (rows, cols)), shape=W32.shape).tocsr()
    
    return W32_clean
import numpy as np
from scipy.sparse import csr_matrix

def build_B32(n, d, eps):
    n_2d = n + 2*d

    # ----- SECTION SIZES (based on original loops) -----
    rows_U2     = n_2d
    rows_tau2   = n_2d * n_2d

    # alpha3/alpha4 and beta3/beta4 blocks:  i in 1..n+2d, j in 1..d, q in {0,1}
    rows_alpha34 = n_2d * d * 2
    rows_beta34  = n_2d * d * 2

    # alpha5/alpha6 and beta5/beta6 blocks:  k in 1..n+2d, j in 1..d, q in {0,1}
    rows_alpha56 = n_2d * d * 2
    rows_beta56  = n_2d * d * 2

    # alpha7/alpha8 and beta7/beta8: j in 1..d, q in {0,1}
    rows_alpha78 = d * 2
    rows_beta78  = d * 2

    total_rows = (rows_U2 + rows_tau2 + rows_alpha34 + rows_beta34 +
                  rows_alpha56 + rows_beta56 + rows_alpha78 + rows_beta78)

    # ----- COUNT NON-ZERO VALUES -----
    # Alpha/Beta 3/4 produce 2*(i/eps)+bias
    non_zero_alpha34 = rows_alpha34
    non_zero_beta34  = rows_beta34
    non_zero_alpha56 = rows_alpha56
    non_zero_beta56  = rows_beta56
    non_zero_alpha78 = d       # only q==0 → value = 1
    non_zero_beta78  = d       # same

    non_zero_count = (non_zero_alpha34 + non_zero_beta34 +
                      non_zero_alpha56 + non_zero_beta56 +
                      non_zero_alpha78 + non_zero_beta78)

    data = np.zeros(non_zero_count, dtype=np.float64)
    rows = np.zeros(non_zero_count, dtype=np.int32)

    idx = rows_U2 + rows_tau2  # skip zero blocks
    p = 0  # pointer into data/rows arrays

    # ------- α3/α4 block: (-i/eps)+1 and (-i/eps) -------
    for i in range(1, n_2d+1):
        val1 = (-i/eps) + 1
        val2 = (-i/eps)
        for j in range(d):
            data[p] = val1; rows[p] = idx; p += 1; idx += 1
            data[p] = val2; rows[p] = idx; p += 1; idx += 1

    # ------- β3/β4 block: (i/eps)+1 and (i/eps) -------
    for i in range(1, n_2d+1):
        val1 = (i/eps) + 1
        val2 = (i/eps)
        for j in range(d):
            data[p] = val1; rows[p] = idx; p += 1; idx += 1
            data[p] = val2; rows[p] = idx; p += 1; idx += 1

    # ------- α5/α6 block: (-k/eps)+1 and (-k/eps) -------
    for k in range(1, n_2d+1):
        val1 = (-k/eps) + 1
        val2 = (-k/eps)
        for j in range(d):
            data[p] = val1; rows[p] = idx; p += 1; idx += 1
            data[p] = val2; rows[p] = idx; p += 1; idx += 1

    # ------- β5/β6 block: (k/eps)+1 and (k/eps) -------
    for k in range(1, n_2d+1):
        val1 = (k/eps) + 1
        val2 = (k/eps)
        for j in range(d):
            data[p] = val1; rows[p] = idx; p += 1; idx += 1
            data[p] = val2; rows[p] = idx; p += 1; idx += 1

    # ------- α7/α8 block: only q==0 → value = 1 -------
    for j in range(d):
        data[p] = 1; rows[p] = idx; p += 1
        idx += 2  # q=0 row filled, q=1 row skipped (0)

    # ------- β7/β8 block: only q==0 → value = 1 -------
    for j in range(d):
        data[p] = 1; rows[p] = idx; p += 1
        idx += 2

    # ---- Sparse vector result ----
    B32 = csr_matrix((data, (rows, np.zeros(non_zero_count, dtype=np.int32))),
                     shape=(total_rows, 1))

    return B32
