# -*- coding: utf-8 -*-
"""
Weights and bias of layer 28

@author: Ghafoor
"""
import numpy as np
from scipy.sparse import csr_matrix, hstack, vstack

import numpy as np
from scipy.sparse import csr_matrix, hstack, vstack

def build_W28(n, d):
    """Complete vectorized implementation of W28 matrix with NO loops"""
    n_2d = n + 2*d
    
    W28_blocks = []
    
    # Common column sizes
    cols_N30 = n_2d  # U2 nodes
    cols_N31 = n_2d * n_2d  # tau2 nodes
    cols_N32 = n * d  # delta(x_j, i) nodes
    cols_N33 = 2 * d  # x_j nodes
    
    # ===== BLOCK 1: label matrix as identity map =====
    rows1 = n_2d
    
    # N30: Identity matrix (n_2d x n_2d)
    N30_b1 = csr_matrix((np.ones(rows1), (np.arange(rows1), np.arange(rows1))),
                       shape=(rows1, cols_N30))
    
    # N31: All zeros (n_2d x (n_2d * n_2d))
    N31_b1 = csr_matrix((rows1, cols_N31))
    
    # N32: All zeros (n_2d x (n * d))
    N32_b1 = csr_matrix((rows1, cols_N32))
    
    # N33: All zeros (n_2d x (2*d))
    N33_b1 = csr_matrix((rows1, cols_N33))
    
    block1 = hstack([N30_b1, N31_b1, N32_b1, N33_b1], format='csr')
    W28_blocks.append(block1)
    
    # ===== BLOCK 2: tau2 as identity map =====
    rows2 = n_2d * n_2d
    
    # N30: All zeros (n_2d² x n_2d)
    N30_b2 = csr_matrix((rows2, cols_N30))
    
    # N31: Identity at positions where j==i and l==k
    # This is a diagonal matrix: row (i,l) has 1 at column (i,l) in flattened n_2d² space
    N31_data = np.ones(rows2, dtype=np.float64)
    N31_b2 = csr_matrix((N31_data, (np.arange(rows2), np.arange(rows2))),
                       shape=(rows2, cols_N31))
    
    # N32: All zeros (n_2d² x (n * d))
    N32_b2 = csr_matrix((rows2, cols_N32))
    
    # N33: All zeros (n_2d² x (2*d))
    N33_b2 = csr_matrix((rows2, cols_N33))
    
    block2 = hstack([N30_b2, N31_b2, N32_b2, N33_b2], format='csr')
    W28_blocks.append(block2)
    
    # ===== BLOCK 3: t'' to check degree of nodes are zero or not =====
    rows3 = n
    
    # N30: All zeros (n x n_2d)
    N30_b3 = csr_matrix((rows3, cols_N30))
    
    # N31: 1 at positions where i==l and k<=n
    # VECTORIZED: Create all i,k combinations where i in [0,n-1], k in [0,n-1]
    i_vals = np.arange(n)
    k_vals = np.arange(n)
    
    # Create meshgrid for all combinations
    i_grid, k_grid = np.meshgrid(i_vals, k_vals, indexing='ij')
    
    # Flatten the grids
    i_flat = i_grid.ravel()
    k_flat = k_grid.ravel()
    
    # Row indices are i values (0 to n-1)
    N31_rows = i_flat
    
    # Column indices: (i * n_2d) + k  (since l = i)
    N31_cols = (i_flat * n_2d) + k_flat
    
    # Data: all ones
    N31_data = np.ones(len(i_flat), dtype=np.float64)
    
    # Build sparse matrix
    N31_b3 = csr_matrix((N31_data, (N31_rows, N31_cols)), 
                       shape=(rows3, cols_N31))
    
    # N32: All zeros (n x (n * d))
    N32_b3 = csr_matrix((rows3, cols_N32))
    
    # N33: All zeros (n x (2*d))
    N33_b3 = csr_matrix((rows3, cols_N33))
    
    block3 = hstack([N30_b3, N31_b3, N32_b3, N33_b3], format='csr')
    W28_blocks.append(block3)
    
    # ===== BLOCK 4: delta(x_j, i) as identity map =====
    rows4 = n * d
    
    # N30: All zeros (n*d x n_2d)
    N30_b4 = csr_matrix((rows4, cols_N30))
    
    # N31: All zeros (n*d x (n_2d * n_2d))
    N31_b4 = csr_matrix((rows4, cols_N31))
    
    # N32: Identity at positions where i==k and l==j
    # This maps each (i,l) pair to column (i,l)
    N32_data = np.ones(rows4, dtype=np.float64)
    N32_b4 = csr_matrix((N32_data, (np.arange(rows4), np.arange(rows4))),
                       shape=(rows4, cols_N32))
    
    # N33: All zeros (n*d x (2*d))
    N33_b4 = csr_matrix((rows4, cols_N33))
    
    block4 = hstack([N30_b4, N31_b4, N32_b4, N33_b4], format='csr')
    W28_blocks.append(block4)
    
    # ===== BLOCK 5: xj as identity map =====
    rows5 = 2 * d
    
    # N30: All zeros (2d x n_2d)
    N30_b5 = csr_matrix((rows5, cols_N30))
    
    # N31: All zeros (2d x (n_2d * n_2d))
    N31_b5 = csr_matrix((rows5, cols_N31))
    
    # N32: All zeros (2d x (n * d))
    N32_b5 = csr_matrix((rows5, cols_N32))
    
    # N33: Identity matrix (2d x 2d)
    N33_data = np.ones(rows5, dtype=np.float64)
    N33_b5 = csr_matrix((N33_data, (np.arange(rows5), np.arange(rows5))),
                       shape=(rows5, cols_N33))
    
    block5 = hstack([N30_b5, N31_b5, N32_b5, N33_b5], format='csr')
    W28_blocks.append(block5)
    
    # Final assembly
    W28 = vstack(W28_blocks, format='csr')
    
    return W28