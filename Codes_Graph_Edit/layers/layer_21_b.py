# -*- coding: utf-8 -*-
"""
Weights and bias of layer 21
fixed an error happened to be in the original dense code

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

def build_W21(n, d, eps):
    """Complete vectorized implementation of W21 matrix"""
    if eps == 0:
        raise ValueError("eps must be nonzero")
    
    n_2d = n + 2*d
    eps_inv = 1.0 / eps
    
    W21_blocks = []
    
    # ===== BLOCK 1: Subs output =====
    rows1 = n_2d
    # M1: Identity
    M1 = csr_matrix((np.ones(rows1), (np.arange(rows1), np.arange(rows1))), 
                    shape=(rows1, n_2d))
    
    zeros_d1 = csr_matrix((rows1, d))
    zeros_n2d2_1 = csr_matrix((rows1, n_2d * n_2d))
    zeros_n2d_1 = csr_matrix((rows1, n_2d))
    zeros_2d_1 = csr_matrix((rows1, 2*d))
    
    block1 = hstack([M1, zeros_d1, zeros_d1, zeros_d1, zeros_d1, 
                    zeros_n2d2_1, zeros_n2d2_1, zeros_n2d_1, zeros_n2d_1, zeros_2d_1], 
                    format='csr')
    W21_blocks.append(block1)
    
    # ===== BLOCK 2: Insertion mu1 =====
    rows2 = n_2d * n_2d
    # M6: -Identity
    M6 = csr_matrix((-np.ones(rows2), (np.arange(rows2), np.arange(rows2))),
                    shape=(rows2, n_2d * n_2d))
    
    zeros_n2d_b2 = csr_matrix((rows2, n_2d))
    zeros_d_b2 = csr_matrix((rows2, d))
    zeros_2d_b2 = csr_matrix((rows2, 2*d))
    
    block2 = hstack([zeros_n2d_b2, zeros_d_b2, zeros_d_b2, zeros_d_b2, zeros_d_b2,
                    M6, M6, zeros_n2d_b2, zeros_n2d_b2, zeros_2d_b2], 
                    format='csr')
    W21_blocks.append(block2)
    
    # ===== BLOCK 3: tau4 - gj identity =====
    rows3 = d
    # M2: Identity
    M2 = csr_matrix((np.ones(rows3), (np.arange(rows3), np.arange(rows3))),
                    shape=(rows3, d))
    
    zeros_n2d_b3 = csr_matrix((rows3, n_2d))
    zeros_d_b3 = csr_matrix((rows3, d))
    zeros_n2d2_b3 = csr_matrix((rows3, n_2d * n_2d))
    zeros_2d_b3 = csr_matrix((rows3, 2*d))
    
    block3 = hstack([zeros_n2d_b3, M2, zeros_d_b3, zeros_d_b3, zeros_d_b3,
                    zeros_n2d2_b3, zeros_n2d2_b3, zeros_n2d_b3, zeros_n2d_b3, zeros_2d_b3],
                    format='csr')
    W21_blocks.append(block3)
    
    # ===== BLOCK 4: alpha17, alpha18 =====
    rows4 = d * d * 2
    if rows4 > 0:
        # Create indices for j, i, q
        j_idx = np.repeat(np.arange(d), d * 2)  # j varies slowest
        i_idx = np.tile(np.repeat(np.arange(d), 2), d)  # i for each j,q
        row_idx = np.arange(rows4)
        
        # M3: -eps_inv at column i (when k == i)
        M3_data = np.full(rows4, -eps_inv, dtype=np.float64)
        M3 = csr_matrix((M3_data, (row_idx, i_idx)), shape=(rows4, d))
        
        zeros_n2d_b4 = csr_matrix((rows4, n_2d))
        zeros_d_b4 = csr_matrix((rows4, d))
        zeros_n2d2_b4 = csr_matrix((rows4, n_2d * n_2d))
        zeros_2d_b4 = csr_matrix((rows4, 2*d))
        
        block4 = hstack([zeros_n2d_b4, zeros_d_b4, M3, zeros_d_b4, zeros_d_b4,
                        zeros_n2d2_b4, zeros_n2d2_b4, zeros_n2d_b4, zeros_n2d_b4, zeros_2d_b4],
                        format='csr')
        W21_blocks.append(block4)
    
    # ===== BLOCK 5: beta17, beta18 =====
    rows5 = d * d * 2
    if rows5 > 0:
        # M3: +eps_inv at column i (when k == i)
        M3_data = np.full(rows5, eps_inv, dtype=np.float64)
        M3 = csr_matrix((M3_data, (row_idx, i_idx)), shape=(rows5, d))
        
        zeros_n2d_b5 = csr_matrix((rows5, n_2d))
        zeros_d_b5 = csr_matrix((rows5, d))
        zeros_n2d2_b5 = csr_matrix((rows5, n_2d * n_2d))
        zeros_2d_b5 = csr_matrix((rows5, 2*d))
        
        block5 = hstack([zeros_n2d_b5, zeros_d_b5, M3, zeros_d_b5, zeros_d_b5,
                        zeros_n2d2_b5, zeros_n2d2_b5, zeros_n2d_b5, zeros_n2d_b5, zeros_2d_b5],
                        format='csr')
        W21_blocks.append(block5)
    
    # ===== BLOCK 6: eta22 identity =====
    rows6 = n_2d
    # M8: Identity
    M8 = csr_matrix((np.ones(rows6), (np.arange(rows6), np.arange(rows6))),
                    shape=(rows6, n_2d))
    
    zeros_n2d_b6 = csr_matrix((rows6, n_2d))
    zeros_d_b6 = csr_matrix((rows6, d))
    zeros_n2d2_b6 = csr_matrix((rows6, n_2d * n_2d))
    zeros_2d_b6 = csr_matrix((rows6, 2*d))
    
    block6 = hstack([zeros_n2d_b6, zeros_d_b6, zeros_d_b6, zeros_d_b6, zeros_d_b6,
                    zeros_n2d2_b6, zeros_n2d2_b6, M8, zeros_n2d_b6, zeros_2d_b6],
                    format='csr')
    W21_blocks.append(block6)
    
    # ===== BLOCK 7: eta23 identity =====
    rows7 = n_2d
    # M9: Identity
    M9 = csr_matrix((np.ones(rows7), (np.arange(rows7), np.arange(rows7))),
                    shape=(rows7, n_2d))
    
    block7 = hstack([zeros_n2d_b6, zeros_d_b6, zeros_d_b6, zeros_d_b6, zeros_d_b6,
                    zeros_n2d2_b6, zeros_n2d2_b6, zeros_n2d_b6, M9, zeros_2d_b6],
                    format='csr')
    W21_blocks.append(block7)
    
       # ===== BLOCK 8: alpha7, alpha8 =====
    rows8 = d * 2
    if rows8 > 0:
        # CORRECTED: Based on errors, should be: M4: +eps_inv, M5: -eps_inv
        row_idx8 = np.arange(rows8)
        j_idx8 = np.repeat(np.arange(d), 2)
        
        # FIXED SIGNS:
        M4_data = np.full(rows8, eps_inv, dtype=np.float64)      # Was: -eps_inv
        M5_data = np.full(rows8, -eps_inv, dtype=np.float64)    # Was: +eps_inv
        
        M4 = csr_matrix((M4_data, (row_idx8, j_idx8)), shape=(rows8, d))
        M5 = csr_matrix((M5_data, (row_idx8, j_idx8)), shape=(rows8, d))
        
        zeros_n2d_b8 = csr_matrix((rows8, n_2d))
        zeros_d_b8 = csr_matrix((rows8, d))
        zeros_n2d2_b8 = csr_matrix((rows8, n_2d * n_2d))
        zeros_2d_b8 = csr_matrix((rows8, 2*d))
        
        block8 = hstack([zeros_n2d_b8, zeros_d_b8, zeros_d_b8, M4, M5,
                        zeros_n2d2_b8, zeros_n2d2_b8, zeros_n2d_b8, zeros_n2d_b8, zeros_2d_b8],
                        format='csr')
        W21_blocks.append(block8)
    
    # ===== BLOCK 9: beta7, beta8 =====
    rows9 = d * 2
    if rows9 > 0:
        # CORRECTED: Based on errors, should be: M4: -eps_inv, M5: +eps_inv
        # FIXED SIGNS:
        M4_data = np.full(rows9, -eps_inv, dtype=np.float64)    # Was: +eps_inv
        M5_data = np.full(rows9, eps_inv, dtype=np.float64)     # Was: -eps_inv
        
        M4 = csr_matrix((M4_data, (row_idx8, j_idx8)), shape=(rows9, d))
        M5 = csr_matrix((M5_data, (row_idx8, j_idx8)), shape=(rows9, d))
        
        zeros_n2d_b9 = csr_matrix((rows9, n_2d))
        zeros_d_b9 = csr_matrix((rows9, d))
        zeros_n2d2_b9 = csr_matrix((rows9, n_2d * n_2d))
        zeros_2d_b9 = csr_matrix((rows9, 2*d))
        
        block9 = hstack([zeros_n2d_b9, zeros_d_b9, zeros_d_b9, M4, M5,
                        zeros_n2d2_b9, zeros_n2d2_b9, zeros_n2d_b9, zeros_n2d_b9, zeros_2d_b9],
                        format='csr')
        W21_blocks.append(block9)
    
    # ===== BLOCK 10: alpha9, alpha10 =====
    rows10 = n_2d * d * 2
    if rows10 > 0:
        # M4: +eps_inv at column j (for x1)
        i_idx10 = np.repeat(np.arange(n_2d), d * 2)
        j_idx10 = np.tile(np.repeat(np.arange(d), 2), n_2d)
        row_idx10 = np.arange(rows10)
        
        M4_data = np.full(rows10, eps_inv, dtype=np.float64)
        M4 = csr_matrix((M4_data, (row_idx10, j_idx10)), shape=(rows10, d))
        
        zeros_n2d_b10 = csr_matrix((rows10, n_2d))
        zeros_d_b10 = csr_matrix((rows10, d))
        zeros_n2d2_b10 = csr_matrix((rows10, n_2d * n_2d))
        zeros_2d_b10 = csr_matrix((rows10, 2*d))
        
        block10 = hstack([zeros_n2d_b10, zeros_d_b10, zeros_d_b10, M4, zeros_d_b10,
                         zeros_n2d2_b10, zeros_n2d2_b10, zeros_n2d_b10, zeros_n2d_b10, zeros_2d_b10],
                         format='csr')
        W21_blocks.append(block10)
    
    # ===== BLOCK 11: beta9, beta10 =====
    rows11 = n_2d * d * 2
    if rows11 > 0:
        # M4: -eps_inv at column j (for x1)
        M4_data = np.full(rows11, -eps_inv, dtype=np.float64)
        M4 = csr_matrix((M4_data, (row_idx10, j_idx10)), shape=(rows11, d))
        
        zeros_n2d_b11 = csr_matrix((rows11, n_2d))
        zeros_d_b11 = csr_matrix((rows11, d))
        zeros_n2d2_b11 = csr_matrix((rows11, n_2d * n_2d))
        zeros_2d_b11 = csr_matrix((rows11, 2*d))
        
        block11 = hstack([zeros_n2d_b11, zeros_d_b11, zeros_d_b11, M4, zeros_d_b11,
                         zeros_n2d2_b11, zeros_n2d2_b11, zeros_n2d_b11, zeros_n2d_b11, zeros_2d_b11],
                         format='csr')
        W21_blocks.append(block11)
    
    # ===== BLOCK 12: alpha11, alpha12 =====
    rows12 = n_2d * d * 2
    if rows12 > 0:
        # M4: +eps_inv at column j (for x1)
        # Note: This uses k index instead of i
        M4_data = np.full(rows12, eps_inv, dtype=np.float64)
        M4 = csr_matrix((M4_data, (row_idx10, j_idx10)), shape=(rows12, d))
        
        zeros_n2d_b12 = csr_matrix((rows12, n_2d))
        zeros_d_b12 = csr_matrix((rows12, d))
        zeros_n2d2_b12 = csr_matrix((rows12, n_2d * n_2d))
        zeros_2d_b12 = csr_matrix((rows12, 2*d))
        
        block12 = hstack([zeros_n2d_b12, zeros_d_b12, zeros_d_b12, M4, zeros_d_b12,
                         zeros_n2d2_b12, zeros_n2d2_b12, zeros_n2d_b12, zeros_n2d_b12, zeros_2d_b12],
                         format='csr')
        W21_blocks.append(block12)
    
    # ===== BLOCK 13: beta11, beta12 =====
    rows13 = n_2d * d * 2
    if rows13 > 0:
        # M4: -eps_inv at column j (for x1)
        M4_data = np.full(rows13, -eps_inv, dtype=np.float64)
        M4 = csr_matrix((M4_data, (row_idx10, j_idx10)), shape=(rows13, d))
        
        zeros_n2d_b13 = csr_matrix((rows13, n_2d))
        zeros_d_b13 = csr_matrix((rows13, d))
        zeros_n2d2_b13 = csr_matrix((rows13, n_2d * n_2d))
        zeros_2d_b13 = csr_matrix((rows13, 2*d))
        
        block13 = hstack([zeros_n2d_b13, zeros_d_b13, zeros_d_b13, M4, zeros_d_b13,
                         zeros_n2d2_b13, zeros_n2d2_b13, zeros_n2d_b13, zeros_n2d_b13, zeros_2d_b13],
                         format='csr')
        W21_blocks.append(block13)
    
    # ===== BLOCK 14: alpha13, alpha14 =====
    rows14 = n_2d * d * 2
    if rows14 > 0:
        # M5: +eps_inv at column j (for x2)
        M5_data = np.full(rows14, eps_inv, dtype=np.float64)
        M5 = csr_matrix((M5_data, (row_idx10, j_idx10)), shape=(rows14, d))
        
        zeros_n2d_b14 = csr_matrix((rows14, n_2d))
        zeros_d_b14 = csr_matrix((rows14, d))
        zeros_n2d2_b14 = csr_matrix((rows14, n_2d * n_2d))
        zeros_2d_b14 = csr_matrix((rows14, 2*d))
        
        block14 = hstack([zeros_n2d_b14, zeros_d_b14, zeros_d_b14, zeros_d_b14, M5,
                         zeros_n2d2_b14, zeros_n2d2_b14, zeros_n2d_b14, zeros_n2d_b14, zeros_2d_b14],
                         format='csr')
        W21_blocks.append(block14)
    
    # ===== BLOCK 15: beta13, beta14 =====
    rows15 = n_2d * d * 2
    if rows15 > 0:
        # M4: -eps_inv at column j (for x1)
        # M5: -eps_inv at column j (for x2)
        #M4_data = np.full(rows15, -eps_inv, dtype=np.float64)
        
        M5_data = np.full(rows15, -eps_inv, dtype=np.float64)
        
        #M4 = csr_matrix((M4_data, (row_idx10, j_idx10)), shape=(rows15, d))
        M4 = csr_matrix((rows15, d))
        M5 = csr_matrix((M5_data, (row_idx10, j_idx10)), shape=(rows15, d))
        
        zeros_n2d_b15 = csr_matrix((rows15, n_2d))
        zeros_d_b15 = csr_matrix((rows15, d))
        zeros_n2d2_b15 = csr_matrix((rows15, n_2d * n_2d))
        zeros_2d_b15 = csr_matrix((rows15, 2*d))
        
        block15 = hstack([zeros_n2d_b15, zeros_d_b15, zeros_d_b15, M4, M5,
                         zeros_n2d2_b15, zeros_n2d2_b15, zeros_n2d_b15, zeros_n2d_b15, zeros_2d_b15],
                         format='csr')
        W21_blocks.append(block15)
    
    # ===== BLOCK 16: alpha15, alpha16 =====
    rows16 = n_2d * d * 2
    if rows16 > 0:
        # M5: +eps_inv at column j (for x2) - uses i index pattern
        M5_data = np.full(rows16, eps_inv, dtype=np.float64)
        M5 = csr_matrix((M5_data, (row_idx10, j_idx10)), shape=(rows16, d))
        
        zeros_n2d_b16 = csr_matrix((rows16, n_2d))
        zeros_d_b16 = csr_matrix((rows16, d))
        zeros_n2d2_b16 = csr_matrix((rows16, n_2d * n_2d))
        zeros_2d_b16 = csr_matrix((rows16, 2*d))
        
        block16 = hstack([zeros_n2d_b16, zeros_d_b16, zeros_d_b16, zeros_d_b16, M5,
                         zeros_n2d2_b16, zeros_n2d2_b16, zeros_n2d_b16, zeros_n2d_b16, zeros_2d_b16],
                         format='csr')
        W21_blocks.append(block16)
    
    # ===== BLOCK 17: beta15, beta16 =====
    rows17 = n_2d * d * 2
    if rows17 > 0:
        # M5: -eps_inv at column j (for x2) - uses k index pattern
        M5_data = np.full(rows17, -eps_inv, dtype=np.float64)
        M5 = csr_matrix((M5_data, (row_idx10, j_idx10)), shape=(rows17, d))
        
        zeros_n2d_b17 = csr_matrix((rows17, n_2d))
        zeros_d_b17 = csr_matrix((rows17, d))
        zeros_n2d2_b17 = csr_matrix((rows17, n_2d * n_2d))
        zeros_2d_b17 = csr_matrix((rows17, 2*d))
        
        block17 = hstack([zeros_n2d_b17, zeros_d_b17, zeros_d_b17, zeros_d_b17, M5,
                         zeros_n2d2_b17, zeros_n2d2_b17, zeros_n2d_b17, zeros_n2d_b17, zeros_2d_b17],
                         format='csr')
        W21_blocks.append(block17)
    
    # ===== BLOCK 18: alpha19, alpha20 =====
    rows18 = n_2d * d * 2
    if rows18 > 0:
        zeros_n2d_b18 = csr_matrix((rows18, n_2d))
        zeros_d_b18 = csr_matrix((rows18, d))
        zeros_n2d2_b18 = csr_matrix((rows18, n_2d * n_2d))
        zeros_2d_b18 = csr_matrix((rows18, 2*d))
        
        block18 = hstack([zeros_n2d_b18, zeros_d_b18, zeros_d_b18, zeros_d_b18, zeros_d_b18,
                         zeros_n2d2_b18, zeros_n2d2_b18, zeros_n2d_b18, zeros_n2d_b18, zeros_2d_b18],
                         format='csr')
        W21_blocks.append(block18)
    
    # ===== BLOCK 19: beta19, beta20 =====
    rows19 = n_2d * d * 2
    if rows19 > 0:
        zeros_n2d_b19 = csr_matrix((rows19, n_2d))
        zeros_d_b19 = csr_matrix((rows19, d))
        zeros_n2d2_b19 = csr_matrix((rows19, n_2d * n_2d))
        zeros_2d_b19 = csr_matrix((rows19, 2*d))
        
        block19 = hstack([zeros_n2d_b19, zeros_d_b19, zeros_d_b19, zeros_d_b19, zeros_d_b19,
                         zeros_n2d2_b19, zeros_n2d2_b19, zeros_n2d_b19, zeros_n2d_b19, zeros_2d_b19],
                         format='csr')
        W21_blocks.append(block19)
    
    # ===== BLOCK 20: deletion identity =====
    rows20 = 2*d
    # M10: Identity
    M10 = csr_matrix((np.ones(rows20), (np.arange(rows20), np.arange(rows20))),
                     shape=(rows20, 2*d))
    
    zeros_n2d_b20 = csr_matrix((rows20, n_2d))
    zeros_d_b20 = csr_matrix((rows20, d))
    zeros_n2d2_b20 = csr_matrix((rows20, n_2d * n_2d))
    zeros_n2d_b20_2 = csr_matrix((rows20, n_2d))
    
    block20 = hstack([zeros_n2d_b20, zeros_d_b20, zeros_d_b20, zeros_d_b20, zeros_d_b20,
                     zeros_n2d2_b20, zeros_n2d2_b20, zeros_n2d_b20_2, zeros_n2d_b20_2, M10],
                     format='csr')
    W21_blocks.append(block20)
    
    # Final assembly
    W21 = vstack(W21_blocks, format='csr')
    
    return W21










def build_B21(n, d, V1, eps, C):
    """Build B21 as sparse vector in same style as B25."""
    from scipy.sparse import csr_matrix
    import numpy as np
    
    n_2d = n + 2*d
    
    # Calculate section sizes (same order as your code)
    size_subs = n_2d  # zeros
    size_mu1 = n_2d * n_2d  # non-zero from V1
    size_gj = d  # zeros
    size_alpha17_18 = 2 * d * d  # non-zero
    size_beta17_18 = 2 * d * d  # non-zero
    size_eta22 = n_2d  # zeros
    size_eta23 = n_2d  # zeros
    size_alpha7_8 = 2 * d  # non-zero (q=0 only)
    size_beta7_8 = 2 * d  # non-zero (q=0 only)
    size_alpha9_10 = 2 * n_2d * d  # non-zero
    size_beta9_10 = 2 * n_2d * d  # non-zero
    size_alpha11_12 = 2 * n_2d * d  # non-zero
    size_beta11_12 = 2 * n_2d * d  # non-zero
    size_alpha13_14 = 2 * n_2d * d  # non-zero
    size_beta13_14 = 2 * n_2d * d  # non-zero
    size_alpha15_16 = 2 * n_2d * d  # non-zero
    size_beta15_16 = 2 * n_2d * d  # non-zero
    size_alpha19_20 = 2 * n_2d * d  # non-zero
    size_beta19_20 = 2 * n_2d * d  # non-zero
    size_xj = 2 * d  # zeros
    
    total_size = (size_subs + size_mu1 + size_gj + size_alpha17_18 + 
                  size_beta17_18 + size_eta22 + size_eta23 + size_alpha7_8 + 
                  size_beta7_8 + size_alpha9_10 + size_beta9_10 + 
                  size_alpha11_12 + size_beta11_12 + size_alpha13_14 + 
                  size_beta13_14 + size_alpha15_16 + size_beta15_16 + 
                  size_alpha19_20 + size_beta19_20 + size_xj)
    
    # All sections have non-zero values except: subs, gj, eta22, eta23, xj
    non_zero_count = (size_mu1 + size_alpha17_18 + size_beta17_18 + 
                      size_alpha7_8 + size_beta7_8 + size_alpha9_10 + 
                      size_beta9_10 + size_alpha11_12 + size_beta11_12 + 
                      size_alpha13_14 + size_beta13_14 + size_alpha15_16 + 
                      size_beta15_16 + size_alpha19_20 + size_beta19_20)
    
    data = np.zeros(non_zero_count, dtype=np.float64)
    rows = np.zeros(non_zero_count, dtype=np.int32)
    
    idx = 0  # Current position in full vector
    data_idx = 0  # Current position in data array
    
    # 1. Skip subs section (all zeros)
    idx += size_subs
    
    # 2. mu1 section (from V1)
    for i in range(size_mu1):
        data[data_idx] = V1[i]
        rows[data_idx] = idx
        data_idx += 1
        idx += 1
    
    # 3. Skip gj section (all zeros)
    idx += size_gj
    
    # 4. alpha17_18 section
    for j in range(1, d+1):
        for i in range(1, d+1):
            for q in range(2):
                if q == 0:
                    data[data_idx] = 1.0 + ((j-1)/eps)
                else:
                    data[data_idx] = (j-1)/eps
                rows[data_idx] = idx
                data_idx += 1
                idx += 1
    
    # 5. beta17_18 section
    for j in range(1, d+1):
        for i in range(1, d+1):
            for q in range(2):
                if q == 0:
                    data[data_idx] = 1.0 + ((-j+1)/eps)
                else:
                    data[data_idx] = (-j+1)/eps
                rows[data_idx] = idx
                data_idx += 1
                idx += 1
    
    # 6. Skip eta22 section (all zeros)
    idx += size_eta22
    
    # 7. Skip eta23 section (all zeros)
    idx += size_eta23
    
    # 8. alpha7_8 section
    for j in range(1, d+1):
        for q in range(2):
            if q == 0:
                data[data_idx] = 1.0
            else:
                data[data_idx] = 0.0
            rows[data_idx] = idx
            data_idx += 1
            idx += 1
    
    # 9. beta7_8 section
    for j in range(1, d+1):
        for q in range(2):
            if q == 0:
                data[data_idx] = 1.0
            else:
                data[data_idx] = 0.0
            rows[data_idx] = idx
            data_idx += 1
            idx += 1
    
    # 10. alpha9_10 section
    for i in range(1, n_2d+1):
        for j in range(1, d+1):
            for q in range(2):
                if q == 0:
                    data[data_idx] = 1.0 + (-i/eps)
                else:
                    data[data_idx] = -i/eps
                rows[data_idx] = idx
                data_idx += 1
                idx += 1
    
    # 11. beta9_10 section
    for i in range(1, n_2d+1):
        for j in range(1, d+1):
            for q in range(2):
                if q == 0:
                    data[data_idx] = 1.0 + (i/eps)
                else:
                    data[data_idx] = i/eps
                rows[data_idx] = idx
                data_idx += 1
                idx += 1
    
    # 12. alpha11_12 section
    for i in range(1, n_2d+1):
        for j in range(1, d+1):
            for q in range(2):
                if q == 0:
                    data[data_idx] = 1.0 + (-i/eps)
                else:
                    data[data_idx] = -i/eps
                rows[data_idx] = idx
                data_idx += 1
                idx += 1
    
    # 13. beta11_12 section
    for i in range(1, n_2d+1):
        for j in range(1, d+1):
            for q in range(2):
                if q == 0:
                    data[data_idx] = 1.0 + (i/eps)
                else:
                    data[data_idx] = i/eps
                rows[data_idx] = idx
                data_idx += 1
                idx += 1
    
    # 14. alpha13_14 section
    for k in range(1, n_2d+1):
        for j in range(1, d+1):
            for q in range(2):
                if q == 0:
                    data[data_idx] = 1.0 + (-k/eps)
                else:
                    data[data_idx] = -k/eps
                rows[data_idx] = idx
                data_idx += 1
                idx += 1
    
    # 15. beta13_14 section
    for k in range(1, n_2d+1):
        for j in range(1, d+1):
            for q in range(2):
                if q == 0:
                    data[data_idx] = 1.0 + (k/eps)
                else:
                    data[data_idx] = k/eps
                rows[data_idx] = idx
                data_idx += 1
                idx += 1
    
    # 16. alpha15_16 section
    for i in range(1, n_2d+1):
        for j in range(1, d+1):
            for q in range(2):
                if q == 0:
                    data[data_idx] = 1.0 + (-i/eps)
                else:
                    data[data_idx] = -i/eps
                rows[data_idx] = idx
                data_idx += 1
                idx += 1
    
    # 17. beta15_16 section
    for i in range(1, n_2d+1):
        for j in range(1, d+1):
            for q in range(2):
                if q == 0:
                    data[data_idx] = 1.0 + (i/eps)
                else:
                    data[data_idx] = i/eps
                rows[data_idx] = idx
                data_idx += 1
                idx += 1
    
    # 18. alpha19_20 section
    for i in range(1, n_2d+1):
        for j in range(1, d+1):
            for q in range(2):
                if q == 0:
                    data[data_idx] = 1.0 + ((i-j-n)/eps)
                else:
                    data[data_idx] = (i-j-n)/eps
                rows[data_idx] = idx
                data_idx += 1
                idx += 1
    
    # 19. beta19_20 section
    for i in range(1, n_2d+1):
        for j in range(1, d+1):
            for q in range(2):
                if q == 0:
                    data[data_idx] = 1.0 + ((j-i+n)/eps)
                else:
                    data[data_idx] = (j-i+n)/eps
                rows[data_idx] = idx
                data_idx += 1
                idx += 1
    
    # 20. Skip xj section (all zeros)
    idx += size_xj
    
    # Create sparse matrix
    B21 = csr_matrix((data, (rows, np.zeros(len(data), dtype=np.int32))), 
                     shape=(total_size, 1))
    
    # print(f"B21 shape: {B21.shape}, Non-zeros: {B21.nnz}")
    return B21