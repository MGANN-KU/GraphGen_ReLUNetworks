# -*- coding: utf-8 -*-
"""
Weights and bias of layer 27

@author: Ghafoor
"""
import numpy as np
from scipy.sparse import csr_matrix, hstack, vstack

def build_W27(n, d):
    """Complete vectorized implementation of W27 matrix"""
    n_2d = n + 2*d
    
    W27_blocks = []
    
    # ===== BLOCK 1: U2 nodes (label matrix after insertion) =====
    rows1 = n_2d
    
    # N30: Identity matrix (n_2d x n_2d)
    N30 = csr_matrix((np.ones(rows1), (np.arange(rows1), np.arange(rows1))),
                     shape=(rows1, n_2d))
    
    # N31: Identity matrix (n_2d x n_2d) - same as N30
    N31 = N30.copy()
    
    # N32: All zeros (n_2d x (n_2d * n_2d))
    N32_cols = n_2d * n_2d
    N32 = csr_matrix((rows1, N32_cols))
    
    # N33: All zeros (n_2d x (n_2d * n_2d))
    N33 = csr_matrix((rows1, N32_cols))
    
    # N34: All zeros (n_2d x (n * d))
    N34_cols = n * d
    N34 = csr_matrix((rows1, N34_cols))
    
    # N35: All zeros (n_2d x (2*d))
    N35_cols = 2 * d
    N35 = csr_matrix((rows1, N35_cols))
    
    block1 = hstack([N30, N31, N32, N33, N34, N35], format='csr')
    W27_blocks.append(block1)
    
    # ===== BLOCK 2: tau2 nodes for t'_ik =====
    rows2 = n_2d * n_2d
    
    # N30: All zeros (rows2 x n_2d)
    N30_b2 = csr_matrix((rows2, n_2d))
    
    # N31: All zeros (rows2 x n_2d)
    N31_b2 = csr_matrix((rows2, n_2d))
    
    # N32: Identity mapping at positions where l==i and k==j
    # This creates a diagonal matrix: row (l,k) has 1 at column (l,k)
    N32_data = np.ones(rows2, dtype=np.float64)
    N32_b2 = csr_matrix((N32_data, (np.arange(rows2), np.arange(rows2))),
                        shape=(rows2, N32_cols))
    
    # N33: -1 at positions where l==i and k==j
    N33_data = np.full(rows2, -1.0, dtype=np.float64)
    N33_b2 = csr_matrix((N33_data, (np.arange(rows2), np.arange(rows2))),
                        shape=(rows2, N32_cols))
    
    # N34: All zeros (rows2 x (n * d))
    N34_b2 = csr_matrix((rows2, N34_cols))
    
    # N35: All zeros (rows2 x (2*d))
    N35_b2 = csr_matrix((rows2, N35_cols))
    
    block2 = hstack([N30_b2, N31_b2, N32_b2, N33_b2, N34_b2, N35_b2], format='csr')
    W27_blocks.append(block2)
    
    # ===== BLOCK 3: delta(x_j, i) as identity map =====
    rows3 = n * d
    
    # N30: All zeros (n*d x n_2d)
    N30_b3 = csr_matrix((rows3, n_2d))
    
    # N31: All zeros (n*d x n_2d)
    N31_b3 = csr_matrix((rows3, n_2d))
    
    # N32: All zeros (n*d x (n_2d * n_2d))
    N32_b3 = csr_matrix((rows3, N32_cols))
    
    # N33: All zeros (n*d x (n_2d * n_2d))
    N33_b3 = csr_matrix((rows3, N32_cols))
    
    # N34: Identity at positions where k==i and l==p
    # This maps each (i,l) pair to column (i,l) where i in [1,n], l in [1,d]
    N34_data = np.ones(rows3, dtype=np.float64)
    N34_b3 = csr_matrix((N34_data, (np.arange(rows3), np.arange(rows3))),
                        shape=(rows3, N34_cols))
    
    # N35: All zeros (n*d x (2*d))
    N35_b3 = csr_matrix((rows3, N35_cols))
    
    block3 = hstack([N30_b3, N31_b3, N32_b3, N33_b3, N34_b3, N35_b3], format='csr')
    W27_blocks.append(block3)
    
    # ===== BLOCK 4: xj as identity map =====
    rows4 = 2 * d
    
    # N30: All zeros (2d x n_2d)
    N30_b4 = csr_matrix((rows4, n_2d))
    
    # N31: All zeros (2d x n_2d)
    N31_b4 = csr_matrix((rows4, n_2d))
    
    # N32: All zeros (2d x (n_2d * n_2d))
    N32_b4 = csr_matrix((rows4, N32_cols))
    
    # N33: All zeros (2d x (n_2d * n_2d))
    N33_b4 = csr_matrix((rows4, N32_cols))
    
    # N34: All zeros (2d x (n * d))
    N34_b4 = csr_matrix((rows4, N34_cols))
    
    # N35: Identity matrix (2d x 2d)
    N35_data = np.ones(rows4, dtype=np.float64)
    N35_b4 = csr_matrix((N35_data, (np.arange(rows4), np.arange(rows4))),
                        shape=(rows4, N35_cols))
    
    block4 = hstack([N30_b4, N31_b4, N32_b4, N33_b4, N34_b4, N35_b4], format='csr')
    W27_blocks.append(block4)
    
    # Final assembly
    W27 = vstack(W27_blocks, format='csr')
    
    return W27
