# -*- coding: utf-8 -*-
"""
Weights and bias of layer 21

@author: Ghafoor
"""
import numpy as np
from scipy.sparse import coo_matrix
import sys
from scipy.sparse import csr_matrix, hstack, vstack


import numpy as np
from scipy.sparse import csr_matrix, hstack, vstack

import numpy as np
from scipy.sparse import csr_matrix, eye, hstack, vstack

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

import numpy as np
from scipy.sparse import csr_matrix, hstack, vstack

import numpy as np
from scipy.sparse import csr_matrix, hstack, vstack

import numpy as np
from scipy.sparse import csr_matrix, hstack, vstack

import numpy as np
from scipy.sparse import csr_matrix, hstack, vstack

def build_W20(n, d, C):  # Note: C parameter is actually NOT needed for most blocks
    """Corrected vectorized implementation matching dense code"""
    n_2d = n + 2*d
    
    W20_blocks = []
    
    # ===== BLOCK 1: subs nodes =====
    rows1 = n_2d
    J11 = csr_matrix((np.ones(rows1), (np.arange(rows1), np.arange(rows1))), 
                     shape=(rows1, n_2d))
    
    zeros_d1 = csr_matrix((rows1, d))
    zeros_dd2_1 = csr_matrix((rows1, d * d * 2))
    zeros_n2d_1 = csr_matrix((rows1, n_2d))
    zeros_n2d2_1 = csr_matrix((rows1, n_2d * n_2d))
    zeros_2d_1 = csr_matrix((rows1, 2*d))
    
    block1 = hstack([J11, zeros_d1, zeros_dd2_1, zeros_dd2_1, zeros_d1, zeros_d1,
                     zeros_n2d_1, zeros_n2d_1, zeros_n2d2_1, zeros_n2d2_1, 
                     zeros_2d_1, zeros_2d_1], format='csr')
    W20_blocks.append(block1)
    
    # ===== BLOCK 2: tau4(gj) nodes =====
    rows2 = d
    J12 = csr_matrix((np.ones(rows2), (np.arange(rows2), np.arange(rows2))),
                     shape=(rows2, d))
    
    zeros_n2d_2 = csr_matrix((rows2, n_2d))
    zeros_dd2_2 = csr_matrix((rows2, d * d * 2))
    zeros_d_2 = csr_matrix((rows2, d))
    zeros_n2d2_2 = csr_matrix((rows2, n_2d * n_2d))
    zeros_2d_2 = csr_matrix((rows2, 2*d))
    
    block2 = hstack([zeros_n2d_2, J12, zeros_dd2_2, zeros_dd2_2, zeros_d_2, zeros_d_2,
                     zeros_n2d_2, zeros_n2d_2, zeros_n2d2_2, zeros_n2d2_2,
                     zeros_2d_2, zeros_2d_2], format='csr')
    W20_blocks.append(block2)
    
    # ===== BLOCK 3: tau5_j(g'_j) nodes =====
    rows3 = d
    if d > 0:
        # FIXED: Use 1.0 and -1.0 instead of C and -C
        i_vals = np.arange(d)
        j_vals = np.arange(d)
        k_vals = np.arange(d)
        
        i_grid, j_grid, k_grid = np.meshgrid(i_vals, j_vals, k_vals, indexing='ij')
        
        # J13: Pattern with (1,-1) at i==j and k<=(i-1)
        mask_j13 = (i_grid == j_grid) & (k_grid <= (i_grid - 1))
        i_j13 = i_grid[mask_j13]
        j_j13 = j_grid[mask_j13]
        k_j13 = k_grid[mask_j13]
        
        rows_j13 = np.repeat(i_j13, 2)
        base_cols_j13 = (j_j13 * d * 2) + (k_j13 * 2)
        cols_j13 = np.tile([0, 1], len(i_j13)) + np.repeat(base_cols_j13, 2)
        data_j13 = np.tile([1.0, -1.0], len(i_j13))  # FIXED: 1.0, -1.0 NOT C, -C
        
        J13 = csr_matrix((data_j13, (rows_j13, cols_j13)), shape=(rows3, d * d * 2))
        
        # J14: Pattern with (-1,1) at i==j and k>=i
        mask_j14 = (i_grid == j_grid) & (k_grid >= i_grid)
        i_j14 = i_grid[mask_j14]
        j_j14 = j_grid[mask_j14]
        k_j14 = k_grid[mask_j14]
        
        rows_j14 = np.repeat(i_j14, 2)
        base_cols_j14 = (j_j14 * d * 2) + (k_j14 * 2)
        cols_j14 = np.tile([0, 1], len(i_j14)) + np.repeat(base_cols_j14, 2)
        data_j14 = np.tile([-1.0, 1.0], len(i_j14))  # FIXED: -1.0, 1.0 NOT -C, C
        
        J14 = csr_matrix((data_j14, (rows_j14, cols_j14)), shape=(rows3, d * d * 2))
    else:
        J13 = csr_matrix((rows3, 0))
        J14 = csr_matrix((rows3, 0))
    
    zeros_n2d_3 = csr_matrix((rows3, n_2d))
    zeros_d_3 = csr_matrix((rows3, d))
    zeros_n2d2_3 = csr_matrix((rows3, n_2d * n_2d))
    zeros_2d_3 = csr_matrix((rows3, 2*d))
    
    block3 = hstack([zeros_n2d_3, zeros_d_3, J13, J14, zeros_d_3, zeros_d_3,
                     zeros_n2d_3, zeros_n2d_3, zeros_n2d2_3, zeros_n2d2_3,
                     zeros_2d_3, zeros_2d_3], format='csr')
    W20_blocks.append(block3)
    
    # ===== BLOCK 4: x1_j nodes =====
    rows4 = d
    if d > 0:
        # FIXED: Use -C as in dense code (not -1)
        J15_data = np.full(rows4, -C, dtype=np.float64)  # This should be -C as in dense
        J15 = csr_matrix((J15_data, (np.arange(rows4), np.arange(rows4))),
                         shape=(rows4, d))
        
        # J22: 1 at i==l positions (identity)
        J22_data = np.ones(rows4, dtype=np.float64)
        J22 = csr_matrix((J22_data, (np.arange(rows4), np.arange(rows4))),
                         shape=(rows4, 2*d))
    else:
        J15 = csr_matrix((rows4, 0))
        J22 = csr_matrix((rows4, 0))
    
    zeros_n2d_4 = csr_matrix((rows4, n_2d))
    zeros_d_4 = csr_matrix((rows4, d))
    zeros_dd2_4 = csr_matrix((rows4, d * d * 2))
    zeros_n2d_4b = csr_matrix((rows4, n_2d))
    zeros_n2d2_4 = csr_matrix((rows4, n_2d * n_2d))
    zeros_2d_4 = csr_matrix((rows4, 2*d))
    
    block4 = hstack([zeros_n2d_4, zeros_d_4, zeros_dd2_4, zeros_dd2_4, 
                     J15, zeros_d_4, zeros_n2d_4b, zeros_n2d_4b, 
                     zeros_n2d2_4, zeros_n2d2_4, J22, zeros_2d_4], format='csr')
    W20_blocks.append(block4)
    
    # ===== BLOCK 5: x2_j nodes =====
    rows5 = d
    if d > 0:
        # FIXED: Use -C as in dense code (not -1)
        J16_data = np.full(rows5, -C, dtype=np.float64)  # This should be -C as in dense
        J16 = csr_matrix((J16_data, (np.arange(rows5), np.arange(rows5))),
                         shape=(rows5, d))
        
        # J22: 1 at (i+d)==l positions
        J22_data_b5 = np.ones(rows5, dtype=np.float64)
        J22_b5 = csr_matrix((J22_data_b5, (np.arange(rows5), np.arange(rows5) + d)),
                            shape=(rows5, 2*d))
    else:
        J16 = csr_matrix((rows5, 0))
        J22_b5 = csr_matrix((rows5, 0))
    
    zeros_n2d_5 = csr_matrix((rows5, n_2d))
    zeros_d_5 = csr_matrix((rows5, d))
    zeros_dd2_5 = csr_matrix((rows5, d * d * 2))
    zeros_n2d_5b = csr_matrix((rows5, n_2d))
    zeros_n2d2_5 = csr_matrix((rows5, n_2d * n_2d))
    zeros_2d_5 = csr_matrix((rows5, 2*d))
    
    block5 = hstack([zeros_n2d_5, zeros_d_5, zeros_dd2_5, zeros_dd2_5,
                     zeros_d_5, J16, zeros_n2d_5b, zeros_n2d_5b,
                     zeros_n2d2_5, zeros_n2d2_5, J22_b5, zeros_2d_5], format='csr')
    W20_blocks.append(block5)
    
    # ===== BLOCK 6: psi3 nodes =====
    rows6 = n_2d * n_2d
    if rows6 > 0:
        # FIXED: Use 1.0 instead of C (dense has w=C, but C=1? Actually dense has w=C)
        # Looking at dense: if i==j and k==l: w=C (where C is a parameter)
        # So this should be C, not 1.0
        row_indices = np.arange(rows6)
        col_indices = np.arange(rows6)
        J19_data = np.full(rows6, C, dtype=np.float64)  # Keep as C - dense has w=C
        J19 = csr_matrix((J19_data, (row_indices, col_indices)),
                         shape=(rows6, n_2d * n_2d))
    else:
        J19 = csr_matrix((rows6, n_2d * n_2d))
    
    zeros_n2d_6 = csr_matrix((rows6, n_2d))
    zeros_d_6 = csr_matrix((rows6, d))
    zeros_dd2_6 = csr_matrix((rows6, d * d * 2))
    zeros_n2d_6b = csr_matrix((rows6, n_2d))
    zeros_n2d2_6 = csr_matrix((rows6, n_2d * n_2d))
    zeros_2d_6 = csr_matrix((rows6, 2*d))
    
    block6 = hstack([zeros_n2d_6, zeros_d_6, zeros_dd2_6, zeros_dd2_6,
                     zeros_d_6, zeros_d_6, zeros_n2d_6b, zeros_n2d_6b,
                     J19, zeros_n2d2_6, zeros_2d_6, zeros_2d_6], format='csr')
    W20_blocks.append(block6)
    
    # ===== BLOCK 7: psi4 nodes =====
    rows7 = n_2d * n_2d
    if rows7 > 0:
        # FIXED: Use C as in dense code
        J20_data = np.full(rows7, C, dtype=np.float64)  # Keep as C - dense has w=C
        J20 = csr_matrix((J20_data, (row_indices, col_indices)),
                         shape=(rows7, n_2d * n_2d))
    else:
        J20 = csr_matrix((rows7, n_2d * n_2d))
    
    zeros_n2d_7 = csr_matrix((rows7, n_2d))
    zeros_d_7 = csr_matrix((rows7, d))
    zeros_dd2_7 = csr_matrix((rows7, d * d * 2))
    zeros_n2d_7b = csr_matrix((rows7, n_2d))
    zeros_n2d2_7 = csr_matrix((rows7, n_2d * n_2d))
    zeros_2d_7 = csr_matrix((rows7, 2*d))
    
    block7 = hstack([zeros_n2d_7, zeros_d_7, zeros_dd2_7, zeros_dd2_7,
                     zeros_d_7, zeros_d_7, zeros_n2d_7b, zeros_n2d_7b,
                     zeros_n2d2_7, J20, zeros_2d_7, zeros_2d_7], format='csr')
    W20_blocks.append(block7)
    
    # ===== BLOCK 8: eta22 nodes =====
    rows8 = n_2d
    J17 = csr_matrix((np.ones(rows8), (np.arange(rows8), np.arange(rows8))),
                     shape=(rows8, n_2d))
    
    zeros_n2d_8 = csr_matrix((rows8, n_2d))
    zeros_d_8 = csr_matrix((rows8, d))
    zeros_dd2_8 = csr_matrix((rows8, d * d * 2))
    zeros_n2d2_8 = csr_matrix((rows8, n_2d * n_2d))
    zeros_2d_8 = csr_matrix((rows8, 2*d))
    
    block8 = hstack([zeros_n2d_8, zeros_d_8, zeros_dd2_8, zeros_dd2_8,
                     zeros_d_8, zeros_d_8, J17, zeros_n2d_8,
                     zeros_n2d2_8, zeros_n2d2_8, zeros_2d_8, zeros_2d_8], format='csr')
    W20_blocks.append(block8)
    
    # ===== BLOCK 9: eta23 nodes =====
    rows9 = n_2d
    J18 = csr_matrix((np.ones(rows9), (np.arange(rows9), np.arange(rows9))),
                     shape=(rows9, n_2d))
    
    zeros_n2d_9 = csr_matrix((rows9, n_2d))
    zeros_d_9 = csr_matrix((rows9, d))
    zeros_dd2_9 = csr_matrix((rows9, d * d * 2))
    zeros_n2d2_9 = csr_matrix((rows9, n_2d * n_2d))
    zeros_2d_9 = csr_matrix((rows9, 2*d))
    
    block9 = hstack([zeros_n2d_9, zeros_d_9, zeros_dd2_9, zeros_dd2_9,
                     zeros_d_9, zeros_d_9, zeros_n2d_9, J18,
                     zeros_n2d2_9, zeros_n2d2_9, zeros_2d_9, zeros_2d_9], format='csr')
    W20_blocks.append(block9)
    
    # ===== BLOCK 10: deletion nodes =====
    rows10 = 2*d
    if rows10 > 0:
        J23 = csr_matrix((np.ones(rows10), (np.arange(rows10), np.arange(rows10))),
                         shape=(rows10, 2*d))
    else:
        J23 = csr_matrix((rows10, 2*d))
    
    zeros_n2d_10 = csr_matrix((rows10, n_2d))
    zeros_d_10 = csr_matrix((rows10, d))
    zeros_dd2_10 = csr_matrix((rows10, d * d * 2))
    zeros_n2d_10b = csr_matrix((rows10, n_2d))
    zeros_n2d2_10 = csr_matrix((rows10, n_2d * n_2d))
    zeros_2d_10 = csr_matrix((rows10, 2*d))
    
    block10 = hstack([zeros_n2d_10, zeros_d_10, zeros_dd2_10, zeros_dd2_10,
                      zeros_d_10, zeros_d_10, zeros_n2d_10b, zeros_n2d_10b,
                      zeros_n2d2_10, zeros_n2d2_10, zeros_2d_10, J23], format='csr')
    W20_blocks.append(block10)
    
    # Final assembly
    W20 = vstack(W20_blocks, format='csr')
    
    return W20


 
    
def build_B20(n, d, B, C):
    """Build B20 as sparse vector in same style as B25."""
    from scipy.sparse import csr_matrix
    import numpy as np
    
    n_2d = n + 2*d
    
    # Calculate section sizes (same order as your code)
    size_subs = n_2d  # zeros
    size_gj = d  # zeros
    size_gj_prime = d  # non-zero: d-i+1
    size_x1 = d  # zeros
    size_x2 = d  # zeros
    size_phi3 = n_2d * n_2d  # non-zero: B-C
    size_phi4 = n_2d * n_2d  # non-zero: B-C
    size_eta22 = n_2d  # zeros
    size_eta23 = n_2d  # zeros
    size_deletion = 2 * d  # zeros
    
    total_size = (size_subs + size_gj + size_gj_prime + size_x1 + 
                  size_x2 + size_phi3 + size_phi4 + size_eta22 + 
                  size_eta23 + size_deletion)
    
    # Non-zero sections: gj_prime, phi3, phi4
    non_zero_count = size_gj_prime + size_phi3 + size_phi4
    
    data = np.zeros(non_zero_count, dtype=np.float64)
    rows = np.zeros(non_zero_count, dtype=np.int32)
    
    idx = 0  # Current position in full vector
    data_idx = 0  # Current position in data array
    
    # 1. Skip subs section (all zeros)
    idx += size_subs
    
    # 2. Skip gj section (all zeros)
    idx += size_gj
    
    # 3. gj_prime section (d-i+1)
    for i in range(1, d+1):
        data[data_idx] = d - i + 1
        rows[data_idx] = idx
        data_idx += 1
        idx += 1
    
    # 4. Skip x1 section (all zeros)
    idx += size_x1
    
    # 5. Skip x2 section (all zeros)
    idx += size_x2
    
    # 6. phi3 section (B-C)
    for i in range(size_phi3):
        data[data_idx] = B - C
        rows[data_idx] = idx
        data_idx += 1
        idx += 1
    
    # 7. phi4 section (B-C)
    for i in range(size_phi4):
        data[data_idx] = B - C
        rows[data_idx] = idx
        data_idx += 1
        idx += 1
    
    # 8. Skip eta22 section (all zeros)
    idx += size_eta22
    
    # 9. Skip eta23 section (all zeros)
    idx += size_eta23
    
    # 10. Skip deletion section (all zeros)
    idx += size_deletion
    
    # Create sparse matrix
    B20 = csr_matrix((data, (rows, np.zeros(len(data), dtype=np.int32))), 
                     shape=(total_size, 1))
    
    # print(f"B20 shape: {B20.shape}, Non-zeros: {B20.nnz}")
    return B20