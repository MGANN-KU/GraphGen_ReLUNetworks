# -*- coding: utf-8 -*-
"""
Weights and bias of layer 29

@author: Ghafoor
"""

import numpy as np
from scipy.sparse import csr_matrix, hstack, vstack

def build_W29(n, d, eps):
    """Complete vectorized implementation of W29 matrix with NO loops"""
    n_2d = n + 2*d
    eps_inv = 1.0 / eps
    
    W29_blocks = []
    
    # Common column sizes
    cols_T1 = n_2d  # U2 nodes
    cols_F1 = n_2d * n_2d  # tau2 nodes
    cols_F2 = n  # t'' nodes
    cols_F3 = n * d  # delta nodes
    cols_F4 = 2 * d  # xj nodes
    
    # ===== BLOCK 1: U2 nodes as identity map =====
    rows1 = n_2d
    
    # T1: Identity matrix (n_2d x n_2d)
    T1_b1 = csr_matrix((np.ones(rows1), (np.arange(rows1), np.arange(rows1))),
                       shape=(rows1, cols_T1))
    
    # F1: All zeros (n_2d x (n_2d * n_2d))
    F1_b1 = csr_matrix((rows1, cols_F1))
    
    # F2: All zeros (n_2d x n)
    F2_b1 = csr_matrix((rows1, cols_F2))
    
    # F3: All zeros (n_2d x (n*d))
    F3_b1 = csr_matrix((rows1, cols_F3))
    
    # F4: All zeros (n_2d x (2*d))
    F4_b1 = csr_matrix((rows1, cols_F4))
    
    block1 = hstack([T1_b1, F1_b1, F2_b1, F3_b1, F4_b1], format='csr')
    W29_blocks.append(block1)
    
    # ===== BLOCK 2: tau2 nodes as identity map =====
    rows2 = n_2d * n_2d
    
    # T1: All zeros (n_2d² x n_2d)
    T1_b2 = csr_matrix((rows2, cols_T1))
    
    # F1: Identity at positions where k==i and l==j
    F1_data = np.ones(rows2, dtype=np.float64)
    F1_b2 = csr_matrix((F1_data, (np.arange(rows2), np.arange(rows2))),
                       shape=(rows2, cols_F1))
    
    # F2: All zeros (n_2d² x n)
    F2_b2 = csr_matrix((rows2, cols_F2))
    
    # F3: All zeros (n_2d² x (n*d))
    F3_b2 = csr_matrix((rows2, cols_F3))
    
    # F4: All zeros (n_2d² x (2*d))
    F4_b2 = csr_matrix((rows2, cols_F4))
    
    block2 = hstack([T1_b2, F1_b2, F2_b2, F3_b2, F4_b2], format='csr')
    W29_blocks.append(block2)
    
    # ===== BLOCKS 3 & 4: alpha'/beta' nodes for delta(t''_i,0) =====
    # Both blocks have rows = n * 2
    rows_3_4 = n * 2
    
    if rows_3_4 > 0:
        # Common for blocks 3 and 4
        # T1: All zeros (n*2 x n_2d)
        T1_b3_4 = csr_matrix((rows_3_4, cols_T1))
        
        # F5/F9: All zeros (n*2 x (n_2d*n_2d))
        F5_b3 = csr_matrix((rows_3_4, cols_F1))
        F9_b4 = csr_matrix((rows_3_4, cols_F1))
        
        # F6: 1/eps at i==k positions for alpha' nodes
        i_indices_b3 = np.repeat(np.arange(n), 2)
        row_indices_b3 = np.arange(rows_3_4)
        
        F6_data = np.full(rows_3_4, eps_inv, dtype=np.float64)
        F6_b3 = csr_matrix((F6_data, (row_indices_b3, i_indices_b3)),
                          shape=(rows_3_4, cols_F2))
        
        # F10: -1/eps at i==k positions for beta' nodes
        F10_data = np.full(rows_3_4, -eps_inv, dtype=np.float64)
        F10_b4 = csr_matrix((F10_data, (row_indices_b3, i_indices_b3)),
                           shape=(rows_3_4, cols_F2))
        
        # F7/F11: All zeros (n*2 x (n*d))
        F7_b3 = csr_matrix((rows_3_4, cols_F3))
        F11_b4 = csr_matrix((rows_3_4, cols_F3))
        
        # F8/F12: All zeros (n*2 x (2*d))
        F8_b3 = csr_matrix((rows_3_4, cols_F4))
        F12_b4 = csr_matrix((rows_3_4, cols_F4))
        
        # Block 3: alpha'1, alpha'2 nodes
        block3 = hstack([T1_b3_4, F5_b3, F6_b3, F7_b3, F8_b3], format='csr')
        W29_blocks.append(block3)
        
        # Block 4: beta'1, beta'2 nodes
        block4 = hstack([T1_b3_4, F9_b4, F10_b4, F11_b4, F12_b4], format='csr')
        W29_blocks.append(block4)
    else:
        # Create empty blocks for n=0
        total_cols = cols_T1 + cols_F1 + cols_F2 + cols_F3 + cols_F4
        empty_block = csr_matrix((0, total_cols))
        W29_blocks.append(empty_block)  # Block 3
        W29_blocks.append(empty_block)  # Block 4
    
    # ===== BLOCK 5: delta(xj,i) nodes =====
    rows5 = n * d
    
    if rows5 > 0:
        # T1: All zeros (n*d x n_2d)
        T1_b5 = csr_matrix((rows5, cols_T1))
        
        # F13: All zeros (n*d x (n_2d*n_2d))
        F13_b5 = csr_matrix((rows5, cols_F1))
        
        # F14: All zeros (n*d x n)
        F14_b5 = csr_matrix((rows5, cols_F2))
        
        # F15: Identity at positions where j==i and l==k
        F15_data = np.ones(rows5, dtype=np.float64)
        F15_b5 = csr_matrix((F15_data, (np.arange(rows5), np.arange(rows5))),
                           shape=(rows5, cols_F3))
        
        # F16: All zeros (n*d x (2*d))
        F16_b5 = csr_matrix((rows5, cols_F4))
        
        block5 = hstack([T1_b5, F13_b5, F14_b5, F15_b5, F16_b5], format='csr')
        W29_blocks.append(block5)
    else:
        # Create empty block for n*d=0
        total_cols = cols_T1 + cols_F1 + cols_F2 + cols_F3 + cols_F4
        empty_block = csr_matrix((0, total_cols))
        W29_blocks.append(empty_block)
    
    # ===== BLOCK 6: xj nodes as identity map =====
    rows6 = 2 * d
    
    # T1: All zeros (2d x n_2d)
    T1_b6 = csr_matrix((rows6, cols_T1))
    
    # F17: All zeros (2d x (n_2d*n_2d))
    F17_b6 = csr_matrix((rows6, cols_F1))
    
    # F18: All zeros (2d x n)
    F18_b6 = csr_matrix((rows6, cols_F2))
    
    # F19: All zeros (2d x (n*d))
    F19_b6 = csr_matrix((rows6, cols_F3))
    
    # F20: Identity matrix (2d x 2d)
    if rows6 > 0:
        F20_data = np.ones(rows6, dtype=np.float64)
        F20_b6 = csr_matrix((F20_data, (np.arange(rows6), np.arange(rows6))),
                           shape=(rows6, cols_F4))
    else:
        F20_b6 = csr_matrix((rows6, cols_F4))
    
    block6 = hstack([T1_b6, F17_b6, F18_b6, F19_b6, F20_b6], format='csr')
    W29_blocks.append(block6)
    
    # Final assembly
    W29 = vstack(W29_blocks, format='csr')
    
    return W29
    


import numpy as np
from scipy.sparse import csr_matrix

def build_B29(n, d):

    n2d = n + 2*d

    # ---- Block sizes (same order as original code) ----
    size_U2      = n2d
    size_tau2    = n2d * n2d
    size_alpha   = n * 2     # (q = 0,1)
    size_beta    = n * 2     # same structure as alpha
    size_delta   = n * d
    size_xj      = 2*d

    total_rows = size_U2 + size_tau2 + size_alpha + size_beta + size_delta + size_xj

    # -----------------------------------------------------
    # Only nonzeros occur in alpha and beta (q==0 → value 1)
    # Each alpha/beta block has one "1" for every pair of entries → n each.
    # -----------------------------------------------------
    nonzero_count = n + n  # alpha + beta

    data = np.ones(nonzero_count, dtype=np.float32)
    rows = np.zeros(nonzero_count, dtype=np.int32)

    write_pos = 0
    cursor = 0


    # 1) U2 nodes → all zero
    cursor += size_U2

    # 2) tau2 nodes → all zero
    cursor += size_tau2

    # 3) alpha' block → pattern: [1,0,1,0,...]
    alpha_indices = cursor + np.arange(0, size_alpha, 2, dtype=np.int32)
    rows[write_pos:write_pos + n] = alpha_indices
    write_pos += n
    cursor += size_alpha

    # 4) beta' block → same pattern
    beta_indices = cursor + np.arange(0, size_beta, 2, dtype=np.int32)
    rows[write_pos:write_pos + n] = beta_indices
    write_pos += n
    cursor += size_beta

    # 5) delta(xj,i) block → all zero
    cursor += size_delta

    # 6) xj block → all zero
    cursor += size_xj

    # Create sparse column vector
    col_idx = np.zeros(nonzero_count, dtype=np.int32)

    return csr_matrix((data, (rows, col_idx)), shape=(total_rows, 1))
