# -*- coding: utf-8 -*-
"""
Weights and bias of layer 24

@author: Ghafoor
"""
import numpy as np
from scipy.sparse import coo_matrix
import sys
from scipy.sparse import csr_matrix, hstack, vstack



import numpy as np
from scipy.sparse import csr_matrix, hstack, vstack

import numpy as np
from scipy.sparse import csr_matrix, hstack, vstack

import numpy as np
from scipy.sparse import csr_matrix, hstack, vstack

import numpy as np
from scipy.sparse import csr_matrix, hstack, vstack

import numpy as np
from scipy.sparse import csr_matrix, hstack, vstack

def build_W24(n, d, C, eps):
    """Complete vectorized implementation of W24 matrix"""
    n_2d = n + 2*d
    
    W24_blocks = []
    
    # ===== BLOCK 1: subs output =====
    rows1 = n_2d
    # M23: Identity matrix
    M23 = csr_matrix((np.ones(rows1), (np.arange(rows1), np.arange(rows1))), 
                     shape=(rows1, n_2d))
    
    zeros_n2d2_1 = csr_matrix((rows1, n_2d * n_2d))
    zeros_d_1 = csr_matrix((rows1, d))
    zeros_n2d_d_1 = csr_matrix((rows1, n_2d * d))
    zeros_n2d_1b = csr_matrix((rows1, n_2d))
    zeros_2d_1 = csr_matrix((rows1, 2*d))
    
    block1 = hstack([M23, zeros_n2d2_1, zeros_d_1, zeros_n2d_d_1, 
                     zeros_n2d_1b, zeros_n2d_1b, zeros_n2d2_1, zeros_2d_1], format='csr')
    W24_blocks.append(block1)
    
    # ===== BLOCK 2: tau8 for h_ij =====
    rows2 = n_2d * d
    zeros_n2d_2 = csr_matrix((rows2, n_2d))
    zeros_n2d2_2 = csr_matrix((rows2, n_2d * n_2d))
    zeros_n2d_2b = csr_matrix((rows2, n_2d))
    zeros_2d_2 = csr_matrix((rows2, 2*d))
    
    if rows2 > 0:
        # M25: Identity at j==k positions
        row_indices2 = np.arange(rows2)
        j_indices = np.tile(np.arange(d), n_2d)  # j for each row
        M25_data = np.ones(rows2, dtype=np.float64)
        M25 = csr_matrix((M25_data, (row_indices2, j_indices)), shape=(rows2, d))
        
        # M26: C at (i==l and j==k) positions
        i_indices = np.repeat(np.arange(n_2d), d)
        col_indices26 = (i_indices * d) + j_indices
        M26_data = np.full(rows2, C, dtype=np.float64)
        M26 = csr_matrix((M26_data, (row_indices2, col_indices26)), 
                         shape=(rows2, n_2d * d))
    else:
        M25 = csr_matrix((rows2, d))
        M26 = csr_matrix((rows2, n_2d * d))
    
    block2 = hstack([zeros_n2d_2, zeros_n2d2_2, M25, M26, 
                     zeros_n2d_2b, zeros_n2d_2b, zeros_n2d2_2, zeros_2d_2], format='csr')
    W24_blocks.append(block2)
    
    # ===== BLOCK 3: zeta2 as identity map =====
    rows3 = n_2d * n_2d
    # M29: Identity at (i==k and j==l) positions
    M29 = csr_matrix((np.ones(rows3), (np.arange(rows3), np.arange(rows3))),
                     shape=(rows3, n_2d * n_2d))
    
    zeros_n2d_3 = csr_matrix((rows3, n_2d))
    zeros_n2d2_3 = csr_matrix((rows3, n_2d * n_2d))
    zeros_d_3 = csr_matrix((rows3, d))
    zeros_n2d_d_3 = csr_matrix((rows3, n_2d * d))
    zeros_n2d_3b = csr_matrix((rows3, n_2d))
    zeros_2d_3 = csr_matrix((rows3, 2*d))
    
    block3 = hstack([zeros_n2d_3, zeros_n2d2_3, zeros_d_3, zeros_n2d_d_3,
                     zeros_n2d_3b, zeros_n2d_3b, M29, zeros_2d_3], format='csr')
    W24_blocks.append(block3)
    
    # ===== BLOCK 4: zeta4 for max(r_ik - C·s'_ik, 0) =====
    rows4 = n_2d * n_2d
    # M24: Identity at (i==k and j==l) positions
    M24 = csr_matrix((np.ones(rows4), (np.arange(rows4), np.arange(rows4))),
                     shape=(rows4, n_2d * n_2d))
    
    # M29: -C at (i==k and j==l) positions
    M29_data = np.full(rows4, -C, dtype=np.float64)
    M29 = csr_matrix((M29_data, (np.arange(rows4), np.arange(rows4))),
                     shape=(rows4, n_2d * n_2d))
    
    zeros_n2d_4 = csr_matrix((rows4, n_2d))
    zeros_d_4 = csr_matrix((rows4, d))
    zeros_n2d_d_4 = csr_matrix((rows4, n_2d * d))
    zeros_n2d_4b = csr_matrix((rows4, n_2d))
    zeros_2d_4 = csr_matrix((rows4, 2*d))
    
    block4 = hstack([zeros_n2d_4, M24, zeros_d_4, zeros_n2d_d_4,
                     zeros_n2d_4b, zeros_n2d_4b, M29, zeros_2d_4], format='csr')
    W24_blocks.append(block4)
    
    # ===== BLOCK 5: eta22 =====
    rows5 = n_2d
    # M27: Identity matrix
    M27 = csr_matrix((np.ones(rows5), (np.arange(rows5), np.arange(rows5))),
                     shape=(rows5, n_2d))
    
    # Define all zero matrices for block 5
    zeros_n2d_5 = csr_matrix((rows5, n_2d))
    zeros_n2d2_5 = csr_matrix((rows5, n_2d * n_2d))
    zeros_d_5 = csr_matrix((rows5, d))
    zeros_n2d_d_5 = csr_matrix((rows5, n_2d * d))
    zeros_n2d_5b = csr_matrix((rows5, n_2d))
    zeros_2d_5 = csr_matrix((rows5, 2*d))
    
    block5 = hstack([zeros_n2d_5, zeros_n2d2_5, zeros_d_5, zeros_n2d_d_5,
                     M27, zeros_n2d_5b, zeros_n2d2_5, zeros_2d_5], format='csr')
    W24_blocks.append(block5)
    
    # ===== BLOCK 6: eta23 =====
    rows6 = n_2d
    # M28: Identity matrix
    M28 = csr_matrix((np.ones(rows6), (np.arange(rows6), np.arange(rows6))),
                     shape=(rows6, n_2d))
    
    # Define all zero matrices for block 6
    zeros_n2d_6 = csr_matrix((rows6, n_2d))
    zeros_n2d2_6 = csr_matrix((rows6, n_2d * n_2d))
    zeros_d_6 = csr_matrix((rows6, d))
    zeros_n2d_d_6 = csr_matrix((rows6, n_2d * d))
    zeros_n2d_6b = csr_matrix((rows6, n_2d))
    zeros_2d_6 = csr_matrix((rows6, 2*d))
    
    block6 = hstack([zeros_n2d_6, zeros_n2d2_6, zeros_d_6, zeros_n2d_d_6,
                     zeros_n2d_6b, M28, zeros_n2d2_6, zeros_2d_6], format='csr')
    W24_blocks.append(block6)
    
    # ===== BLOCK 7: eta1, eta2 to solve delta(x_j, x_{j+d}) =====
    rows7 = d * 2
    
    # Create zero matrices for this block
    zeros_n2d_7 = csr_matrix((rows7, n_2d))
    zeros_n2d2_7 = csr_matrix((rows7, n_2d * n_2d))
    zeros_d_7 = csr_matrix((rows7, d))
    zeros_n2d_d_7 = csr_matrix((rows7, n_2d * d))
    zeros_n2d_7b = csr_matrix((rows7, n_2d))
    
    if rows7 > 0:
        # M30: 1/eps at j==k, -1/eps at (j+d)==k
        eps_inv = 1.0 / eps
        j_indices7 = np.repeat(np.arange(d), 2)
        row_indices7 = np.arange(rows7)
        
        # First entries: j==k
        rows_30_1 = row_indices7
        cols_30_1 = j_indices7
        data_30_1 = np.full(rows7, eps_inv, dtype=np.float64)
        
        # Second entries: (j+d)==k
        rows_30_2 = row_indices7
        cols_30_2 = j_indices7 + d
        data_30_2 = np.full(rows7, -eps_inv, dtype=np.float64)
        
        # Combine both sets
        M30_rows = np.concatenate([rows_30_1, rows_30_2])
        M30_cols = np.concatenate([cols_30_1, cols_30_2])
        M30_data = np.concatenate([data_30_1, data_30_2])
        
        M30 = csr_matrix((M30_data, (M30_rows, M30_cols)), shape=(rows7, 2*d))
    else:
        M30 = csr_matrix((rows7, 2*d))
    
    block7 = hstack([zeros_n2d_7, zeros_n2d2_7, zeros_d_7, zeros_n2d_d_7,
                     zeros_n2d_7b, zeros_n2d_7b, zeros_n2d2_7, M30], format='csr')
    W24_blocks.append(block7)
    
    # ===== BLOCK 8: eta3, eta4 to solve delta(x_j, x_{j+d}) =====
    rows8 = d * 2
    
    # Create zero matrices for this block
    zeros_n2d_8 = csr_matrix((rows8, n_2d))
    zeros_n2d2_8 = csr_matrix((rows8, n_2d * n_2d))
    zeros_d_8 = csr_matrix((rows8, d))
    zeros_n2d_d_8 = csr_matrix((rows8, n_2d * d))
    zeros_n2d_8b = csr_matrix((rows8, n_2d))
    
    if rows8 > 0:
        # M30: -1/eps at j==k, 1/eps at (j+d)==k
        rows_30_3 = row_indices7
        cols_30_3 = j_indices7
        data_30_3 = np.full(rows8, -eps_inv, dtype=np.float64)
        
        rows_30_4 = row_indices7
        cols_30_4 = j_indices7 + d
        data_30_4 = np.full(rows8, eps_inv, dtype=np.float64)
        
        M30_rows = np.concatenate([rows_30_3, rows_30_4])
        M30_cols = np.concatenate([cols_30_3, cols_30_4])
        M30_data = np.concatenate([data_30_3, data_30_4])
        
        M30 = csr_matrix((M30_data, (M30_rows, M30_cols)), shape=(rows8, 2*d))
    else:
        M30 = csr_matrix((rows8, 2*d))
    
    block8 = hstack([zeros_n2d_8, zeros_n2d2_8, zeros_d_8, zeros_n2d_d_8,
                     zeros_n2d_8b, zeros_n2d_8b, zeros_n2d2_8, M30], format='csr')
    W24_blocks.append(block8)
    
    # ===== BLOCKS 9-16: alpha/beta nodes with similar patterns =====
    # All have rows = n_2d * d * 2
    rows_9_16 = n_2d * d * 2
    
    if rows_9_16 > 0:
        # Common indices for all blocks 9-16
        i_indices_9_16 = np.repeat(np.arange(n_2d), d * 2)
        j_indices_9_16 = np.tile(np.repeat(np.arange(d), 2), n_2d)
        row_indices_9_16 = np.arange(rows_9_16)
        eps_inv = 1.0 / eps
        
        # Create zero matrices for blocks 9-16
        zeros_n2d_9_16 = csr_matrix((rows_9_16, n_2d))
        zeros_n2d2_9_16 = csr_matrix((rows_9_16, n_2d * n_2d))
        zeros_d_9_16 = csr_matrix((rows_9_16, d))
        zeros_n2d_d_9_16 = csr_matrix((rows_9_16, n_2d * d))
        zeros_n2d_9_16b = csr_matrix((rows_9_16, n_2d))
        
        # BLOCK 9: alpha1, alpha2 - 1/eps at j==k
        M30_data_9 = np.full(rows_9_16, eps_inv, dtype=np.float64)
        M30_9 = csr_matrix((M30_data_9, (row_indices_9_16, j_indices_9_16)), 
                           shape=(rows_9_16, 2*d))
        block9 = hstack([zeros_n2d_9_16, zeros_n2d2_9_16, zeros_d_9_16, zeros_n2d_d_9_16,
                         zeros_n2d_9_16b, zeros_n2d_9_16b, zeros_n2d2_9_16, M30_9], format='csr')
        W24_blocks.append(block9)
        
        # BLOCK 10: beta1, beta2 - -1/eps at j==k
        M30_data_10 = np.full(rows_9_16, -eps_inv, dtype=np.float64)
        M30_10 = csr_matrix((M30_data_10, (row_indices_9_16, j_indices_9_16)), 
                            shape=(rows_9_16, 2*d))
        block10 = hstack([zeros_n2d_9_16, zeros_n2d2_9_16, zeros_d_9_16, zeros_n2d_d_9_16,
                          zeros_n2d_9_16b, zeros_n2d_9_16b, zeros_n2d2_9_16, M30_10], format='csr')
        W24_blocks.append(block10)
        
        # BLOCK 11: alpha3, alpha4 - 1/eps at (j+d)==k
        cols_11 = j_indices_9_16 + d
        M30_data_11 = np.full(rows_9_16, eps_inv, dtype=np.float64)
        M30_11 = csr_matrix((M30_data_11, (row_indices_9_16, cols_11)), 
                            shape=(rows_9_16, 2*d))
        block11 = hstack([zeros_n2d_9_16, zeros_n2d2_9_16, zeros_d_9_16, zeros_n2d_d_9_16,
                          zeros_n2d_9_16b, zeros_n2d_9_16b, zeros_n2d2_9_16, M30_11], format='csr')
        W24_blocks.append(block11)
        
        # BLOCK 12: beta3, beta4 - -1/eps at (j+d)==k
        M30_data_12 = np.full(rows_9_16, -eps_inv, dtype=np.float64)
        M30_12 = csr_matrix((M30_data_12, (row_indices_9_16, cols_11)), 
                            shape=(rows_9_16, 2*d))
        block12 = hstack([zeros_n2d_9_16, zeros_n2d2_9_16, zeros_d_9_16, zeros_n2d_d_9_16,
                          zeros_n2d_9_16b, zeros_n2d_9_16b, zeros_n2d2_9_16, M30_12], format='csr')
        W24_blocks.append(block12)
        
        # BLOCK 13: alpha5, alpha6 - 1/eps at j==k
        M30_data_13 = np.full(rows_9_16, eps_inv, dtype=np.float64)
        M30_13 = csr_matrix((M30_data_13, (row_indices_9_16, j_indices_9_16)), 
                            shape=(rows_9_16, 2*d))
        block13 = hstack([zeros_n2d_9_16, zeros_n2d2_9_16, zeros_d_9_16, zeros_n2d_d_9_16,
                          zeros_n2d_9_16b, zeros_n2d_9_16b, zeros_n2d2_9_16, M30_13], format='csr')
        W24_blocks.append(block13)
        
        # BLOCK 14: beta5, beta6 - -1/eps at j==k
        M30_data_14 = np.full(rows_9_16, -eps_inv, dtype=np.float64)
        M30_14 = csr_matrix((M30_data_14, (row_indices_9_16, j_indices_9_16)), 
                            shape=(rows_9_16, 2*d))
        block14 = hstack([zeros_n2d_9_16, zeros_n2d2_9_16, zeros_d_9_16, zeros_n2d_d_9_16,
                          zeros_n2d_9_16b, zeros_n2d_9_16b, zeros_n2d2_9_16, M30_14], format='csr')
        W24_blocks.append(block14)
        
        # BLOCK 15: alpha7, alpha8 - 1/eps at (j+d)==k
        M30_data_15 = np.full(rows_9_16, eps_inv, dtype=np.float64)
        M30_15 = csr_matrix((M30_data_15, (row_indices_9_16, cols_11)), 
                            shape=(rows_9_16, 2*d))
        block15 = hstack([zeros_n2d_9_16, zeros_n2d2_9_16, zeros_d_9_16, zeros_n2d_d_9_16,
                          zeros_n2d_9_16b, zeros_n2d_9_16b, zeros_n2d2_9_16, M30_15], format='csr')
        W24_blocks.append(block15)
        
        # BLOCK 16: beta7, beta8 - -1/eps at (j+d)==k
        M30_data_16 = np.full(rows_9_16, -eps_inv, dtype=np.float64)
        M30_16 = csr_matrix((M30_data_16, (row_indices_9_16, cols_11)), 
                            shape=(rows_9_16, 2*d))
        block16 = hstack([zeros_n2d_9_16, zeros_n2d2_9_16, zeros_d_9_16, zeros_n2d_d_9_16,
                          zeros_n2d_9_16b, zeros_n2d_9_16b, zeros_n2d2_9_16, M30_16], format='csr')
        W24_blocks.append(block16)
    else:
        # Create empty blocks for d=0
        total_cols = n_2d + n_2d*n_2d + d + n_2d*d + n_2d + n_2d + n_2d*n_2d + 2*d
        empty_block = csr_matrix((0, total_cols))
        for _ in range(8):  # Blocks 9-16
            W24_blocks.append(empty_block)
    
    # ===== BLOCK 17: copy x_j =====
    rows17 = 2*d
    
    # Create zero matrices for this block
    zeros_n2d_17 = csr_matrix((rows17, n_2d))
    zeros_n2d2_17 = csr_matrix((rows17, n_2d * n_2d))
    zeros_d_17 = csr_matrix((rows17, d))
    zeros_n2d_d_17 = csr_matrix((rows17, n_2d * d))
    zeros_n2d_17b = csr_matrix((rows17, n_2d))
    
    # M30: Identity matrix
    if rows17 > 0:
        M30_17 = csr_matrix((np.ones(rows17), (np.arange(rows17), np.arange(rows17))),
                            shape=(rows17, 2*d))
    else:
        M30_17 = csr_matrix((rows17, 2*d))
    
    block17 = hstack([zeros_n2d_17, zeros_n2d2_17, zeros_d_17, zeros_n2d_d_17,
                      zeros_n2d_17b, zeros_n2d_17b, zeros_n2d2_17, M30_17], format='csr')
    W24_blocks.append(block17)
    
    # Final assembly
    W24 = vstack(W24_blocks, format='csr')
    
    return W24

def build_B24(n, d, C, eps):
    """Build B24 as sparse vector in same style as B25."""
    from scipy.sparse import csr_matrix
    import numpy as np
    
    n_2d = n + 2*d
    
    # Calculate section sizes (same order as your code)
    size_subs = n_2d  # zeros
    size_tau8 = n_2d * d  # non-zero: -C
    size_zeta2 = n_2d * n_2d  # zeros
    size_zeta4 = n_2d * n_2d  # zeros
    size_eta22 = n_2d  # zeros
    size_eta23 = n_2d  # zeros
    size_eta1_2 = 2 * d  # non-zero (q=0 only: 1)
    size_eta3_4 = 2 * d  # non-zero (q=0 only: 1)
    size_alpha1_2 = 2 * n_2d * d  # non-zero
    size_beta1_2 = 2 * n_2d * d  # non-zero
    size_alpha3_4 = 2 * n_2d * d  # non-zero
    size_beta3_4 = 2 * n_2d * d  # non-zero
    size_alpha5_6 = 2 * n_2d * d  # non-zero
    size_beta5_6 = 2 * n_2d * d  # non-zero
    size_alpha7_8 = 2 * n_2d * d  # non-zero
    size_beta7_8 = 2 * n_2d * d  # non-zero
    size_xj = 2 * d  # zeros
    
    total_size = (size_subs + size_tau8 + size_zeta2 + size_zeta4 + 
                  size_eta22 + size_eta23 + size_eta1_2 + size_eta3_4 + 
                  size_alpha1_2 + size_beta1_2 + size_alpha3_4 + 
                  size_beta3_4 + size_alpha5_6 + size_beta5_6 + 
                  size_alpha7_8 + size_beta7_8 + size_xj)
    
    # Non-zero sections
    non_zero_count = (size_tau8 + size_eta1_2 + size_eta3_4 + 
                      size_alpha1_2 + size_beta1_2 + size_alpha3_4 + 
                      size_beta3_4 + size_alpha5_6 + size_beta5_6 + 
                      size_alpha7_8 + size_beta7_8)
    
    data = np.zeros(non_zero_count, dtype=np.float64)
    rows = np.zeros(non_zero_count, dtype=np.int32)
    
    idx = 0  # Current position in full vector
    data_idx = 0  # Current position in data array
    
    # 1. Skip subs section (all zeros)
    idx += size_subs
    
    # 2. tau8 section (-C)
    for i in range(size_tau8):
        data[data_idx] = -C
        rows[data_idx] = idx
        data_idx += 1
        idx += 1
    
    # 3. Skip zeta2 section (all zeros)
    idx += size_zeta2
    
    # 4. Skip zeta4 section (all zeros)
    idx += size_zeta4
    
    # 5. Skip eta22 section (all zeros)
    idx += size_eta22
    
    # 6. Skip eta23 section (all zeros)
    idx += size_eta23
    
    # 7. eta1_2 section (q=0: 1, q=1: 0)
    for j in range(1, d+1):
        for q in range(2):
            if q == 0:
                data[data_idx] = 1.0
            else:
                data[data_idx] = 0.0
            rows[data_idx] = idx
            data_idx += 1
            idx += 1
    
    # 8. eta3_4 section (q=0: 1, q=1: 0)
    for j in range(1, d+1):
        for q in range(2):
            if q == 0:
                data[data_idx] = 1.0
            else:
                data[data_idx] = 0.0
            rows[data_idx] = idx
            data_idx += 1
            idx += 1
    
    # 9. alpha1_2 section
    for i in range(1, n_2d+1):
        for j in range(1, d+1):
            for q in range(2):
                if q == 0:
                    data[data_idx] = (-i/eps) + 1
                else:
                    data[data_idx] = -i/eps
                rows[data_idx] = idx
                data_idx += 1
                idx += 1
    
    # 10. beta1_2 section
    for i in range(1, n_2d+1):
        for j in range(1, d+1):
            for q in range(2):
                if q == 0:
                    data[data_idx] = (i/eps) + 1
                else:
                    data[data_idx] = i/eps
                rows[data_idx] = idx
                data_idx += 1
                idx += 1
    
    # 11. alpha3_4 section (l instead of i)
    for l in range(1, n_2d+1):
        for j in range(1, d+1):
            for q in range(2):
                if q == 0:
                    data[data_idx] = (-l/eps) + 1
                else:
                    data[data_idx] = -l/eps
                rows[data_idx] = idx
                data_idx += 1
                idx += 1
    
    # 12. beta3_4 section (l instead of i)
    for l in range(1, n_2d+1):
        for j in range(1, d+1):
            for q in range(2):
                if q == 0:
                    data[data_idx] = (l/eps) + 1
                else:
                    data[data_idx] = l/eps
                rows[data_idx] = idx
                data_idx += 1
                idx += 1
    
    # 13. alpha5_6 section (l instead of i)
    for l in range(1, n_2d+1):
        for j in range(1, d+1):
            for q in range(2):
                if q == 0:
                    data[data_idx] = (-l/eps) + 1
                else:
                    data[data_idx] = -l/eps
                rows[data_idx] = idx
                data_idx += 1
                idx += 1
    
    # 14. beta5_6 section (l instead of i)
    for l in range(1, n_2d+1):
        for j in range(1, d+1):
            for q in range(2):
                if q == 0:
                    data[data_idx] = (l/eps) + 1
                else:
                    data[data_idx] = l/eps
                rows[data_idx] = idx
                data_idx += 1
                idx += 1
    
    # 15. alpha7_8 section
    for i in range(1, n_2d+1):
        for j in range(1, d+1):
            for q in range(2):
                if q == 0:
                    data[data_idx] = (-i/eps) + 1
                else:
                    data[data_idx] = -i/eps
                rows[data_idx] = idx
                data_idx += 1
                idx += 1
    
    # 16. beta7_8 section
    for i in range(1, n_2d+1):
        for j in range(1, d+1):
            for q in range(2):
                if q == 0:
                    data[data_idx] = (i/eps) + 1
                else:
                    data[data_idx] = i/eps
                rows[data_idx] = idx
                data_idx += 1
                idx += 1
    
    # 17. Skip xj section (all zeros)
    idx += size_xj
    
    # Create sparse matrix
    B24 = csr_matrix((data, (rows, np.zeros(len(data), dtype=np.int32))), 
                     shape=(total_size, 1))
    
    # print(f"B24 shape: {B24.shape}, Non-zeros: {B24.nnz}")
    return B24