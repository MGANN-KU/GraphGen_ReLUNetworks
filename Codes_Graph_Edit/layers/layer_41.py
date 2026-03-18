# -*- coding: utf-8 -*-
"""
Weights and bias of layer 41

@author: Ghafoor
"""

import numpy as np
from scipy.sparse import csr_matrix, hstack, vstack


import numpy as np
from scipy.sparse import csr_matrix, hstack, vstack

import numpy as np
from scipy.sparse import csr_matrix, hstack, vstack

def build_W41(n, d):
    """Completely vectorized, loop-free W41 implementation"""
    n_d = n + d
    d1 = d + 1  # d+1, not d+2!
    
    # Column dimensions
    cols_H16 = n_d                     # (n+d) columns
    cols_H17 = n_d * n_d * d1          # (n+d)*(n+d)*(d+1) columns
    total_cols = cols_H16 + cols_H17
    
    # ===== BLOCK 1: zeta4 nodes =====
    rows1 = n_d  # (n+d) rows
    
    # H16: Identity matrix
    H16_block1 = csr_matrix(np.eye(rows1, dtype=np.float64))
    
    # H17: All zeros
    H17_block1 = csr_matrix((rows1, cols_H17))
    
    block1 = hstack([H16_block1, H17_block1], format='csr')
    
    # ===== BLOCK 2: zeta6 nodes =====
    rows2 = n_d * n_d  # (n+d)*(n+d) rows
    
    # H18: All zeros
    H18_block2 = csr_matrix((rows2, cols_H16))
    
    # H19: Pattern where for each (i,l), we have d1 ones
    # Create i and l indices for all (i,l) pairs
    i_vals_base = np.repeat(np.arange(n_d), n_d)      # length n_d²
    l_vals_base = np.tile(np.arange(n_d), n_d)        # length n_d²
    
    # Repeat each (i,l) pair d1 times
    i_vals = np.repeat(i_vals_base, d1)  # Length: n_d² * d1
    l_vals = np.repeat(l_vals_base, d1)  # Length: n_d² * d1
    
    # Create j values: [0,1,...,d1-1] repeated for each (i,l) pair
    j_vals = np.tile(np.arange(d1), n_d * n_d)  # Length: n_d² * d1
    
    # Row indices: each row (i,l pair) repeated d1 times
    row_indices = np.repeat(np.arange(rows2), d1)
    
    # Column indices: position = (i * n_d + l) * d1 + j
    col_indices = (i_vals * n_d + l_vals) * d1 + j_vals
    
    # Data: all ones
    data = np.ones(len(row_indices), dtype=np.float64)
    
    H19_block2 = csr_matrix((data, (row_indices, col_indices)), 
                           shape=(rows2, cols_H17))
    
    block2 = hstack([H18_block2, H19_block2], format='csr')
    
    # ===== ASSEMBLE =====
    W41 = vstack([block1, block2], format='csr')
    
    return W41

