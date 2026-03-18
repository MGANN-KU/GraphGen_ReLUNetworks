# -*- coding: utf-8 -*-
"""
Weights and bias of layer 30_b

@author: Ghafoor
"""

import numpy as np
from scipy.sparse import csr_matrix, hstack, vstack


import numpy as np
from scipy.sparse import csr_matrix, hstack, vstack

import numpy as np
from scipy.sparse import csr_matrix

def build_W30(n, d):
    """Completely loop-free vectorized W30 implementation"""
    n_2d = n + 2*d
    
    # Row counts
    rows1 = n_2d
    rows2 = n_2d * n_2d
    rows3 = n * d
    rows4 = 2 * d
    total_rows = rows1 + rows2 + rows3 + rows4
    
    # Column sections
    cols_T2 = n_2d
    cols_G1 = n_2d * n_2d
    cols_G2G3 = n * 2
    cols_G4 = n * d
    cols_G5 = 2 * d
    total_cols = cols_T2 + cols_G1 + 2*cols_G2G3 + cols_G4 + cols_G5
    
    # Pre-compute exact nnz
    nnz = rows1 + rows2 + 4*rows3 + rows3 + rows4
    
    # Pre-allocate
    data = np.empty(nnz, dtype=np.float64)
    rows = np.empty(nnz, dtype=np.int32)
    cols = np.empty(nnz, dtype=np.int32)
    
    # BLOCK 1: T2 identity
    block1_rows = np.arange(rows1)
    data[:rows1] = 1.0
    rows[:rows1] = block1_rows
    cols[:rows1] = block1_rows
    pos = rows1
    
    # BLOCK 2: G1 identity
    block2_start_row = rows1
    block2_rows = np.arange(rows2) + block2_start_row
    data[pos:pos+rows2] = 1.0
    rows[pos:pos+rows2] = block2_rows
    cols[pos:pos+rows2] = block2_rows - block2_start_row + cols_T2
    pos += rows2
    
    # BLOCK 3: G7 (±1 in G2 section)
    block3_start_row = rows1 + rows2
    i_vals = np.repeat(np.arange(n), d)
    g2_start = cols_T2 + cols_G1
    
    # G7 q=0 (+1)
    data[pos:pos+rows3] = 1.0
    rows[pos:pos+rows3] = np.arange(rows3) + block3_start_row
    cols[pos:pos+rows3] = g2_start + i_vals * 2
    pos += rows3
    
    # G7 q=1 (-1)
    data[pos:pos+rows3] = -1.0
    rows[pos:pos+rows3] = np.arange(rows3) + block3_start_row
    cols[pos:pos+rows3] = g2_start + i_vals * 2 + 1
    pos += rows3
    
    # BLOCK 3: G8 (±1 in G3 section)
    g3_start = g2_start + cols_G2G3
    
    # G8 q=0 (+1)
    data[pos:pos+rows3] = 1.0
    rows[pos:pos+rows3] = np.arange(rows3) + block3_start_row
    cols[pos:pos+rows3] = g3_start + i_vals * 2
    pos += rows3
    
    # G8 q=1 (-1)
    data[pos:pos+rows3] = -1.0
    rows[pos:pos+rows3] = np.arange(rows3) + block3_start_row
    cols[pos:pos+rows3] = g3_start + i_vals * 2 + 1
    pos += rows3
    
    # BLOCK 3: G9 identity in G4 section
    g4_start = g3_start + cols_G2G3
    data[pos:pos+rows3] = 1.0
    rows[pos:pos+rows3] = np.arange(rows3) + block3_start_row
    cols[pos:pos+rows3] = g4_start + np.arange(rows3)
    pos += rows3
    
    # BLOCK 4: G15 identity
    block4_start_row = rows1 + rows2 + rows3
    g5_start = g4_start + cols_G4
    data[pos:pos+rows4] = 1.0
    rows[pos:pos+rows4] = np.arange(rows4) + block4_start_row
    cols[pos:pos+rows4] = g5_start + np.arange(rows4)
    
    return csr_matrix((data, (rows, cols)), shape=(total_rows, total_cols))
    

def build_B30(n, d):
    n_2d = n + 2*d

    # Section sizes (same as original logic)
    rows_U2   = n_2d
    rows_tau2 = n_2d * n_2d
    rows_tau3 = n * d
    rows_xj   = 2 * d

    total_rows = rows_U2 + rows_tau2 + rows_tau3 + rows_xj

    # Only tau3 block is non-zero
    non_zero_count = rows_tau3

    data = np.full(non_zero_count, -2, dtype=np.float64)
    rows = np.arange(rows_U2 + rows_tau2, rows_U2 + rows_tau2 + rows_tau3, dtype=np.int32)

    # Sparse column vector
    B30 = csr_matrix((data, (rows, np.zeros(non_zero_count, dtype=np.int32))),
                     shape=(total_rows, 1))

    return B30
