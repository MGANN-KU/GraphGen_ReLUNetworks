# -*- coding: utf-8 -*-
"""
Weights and bias of layer 34

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

from scipy.sparse import csr_matrix, hstack, vstack

import numpy as np
from scipy.sparse import csr_matrix, hstack
import numpy as np
from scipy.sparse import csr_matrix, hstack, vstack, lil_matrix


import numpy as np
from scipy.sparse import csr_matrix, hstack, vstack

def build_W34_block1_sparse(n, d):
    rows = n + 2*d
    n_2d = rows
    
    T6_data = np.ones(rows, dtype=np.float64)
    T6_row = np.arange(rows)
    T6_col = np.arange(rows)
    T6 = csr_matrix((T6_data, (T6_row, T6_col)), shape=(rows, n_2d))
    
    I1 = csr_matrix((rows, n_2d * n_2d))
    I2 = csr_matrix((rows, n_2d * n_2d * d))
    I3 = csr_matrix((rows, n_2d * d))
    I4 = csr_matrix((rows, (n + d) * n_2d * d))
    
    block1 = hstack([T6, I1, I2, I3, I4], format='csr')
    return block1

def build_W34_block2_sparse(n, d):
    rows = (n + 2*d) * (n + 2*d)
    n_2d = n + 2*d
    
    T6 = csr_matrix((rows, n_2d))
    
    nnz = rows
    I1_data = np.ones(nnz, dtype=np.float64)
    I1_row = np.arange(rows)
    I1_col = np.arange(rows)
    I1 = csr_matrix((I1_data, (I1_row, I1_col)), shape=(rows, n_2d * n_2d))
    
    I2 = csr_matrix((rows, n_2d * n_2d * d))
    I3 = csr_matrix((rows, n_2d * d))
    I4 = csr_matrix((rows, (n + d) * n_2d * d))
    
    block2 = hstack([T6, I1, I2, I3, I4], format='csr')
    return block2

def build_W34_block3_sparse(n, d):
    rows = (n + 2*d) * (n + 2*d)
    n_2d = n + 2*d
    
    T6 = csr_matrix((rows, n_2d))
    I5 = csr_matrix((rows, n_2d * n_2d))
    
    nnz = rows * d
    I6_data = np.full(nnz, -1.0, dtype=np.float64)
    I6_row = np.repeat(np.arange(rows), d)
    I6_col = np.empty(nnz, dtype=int)
    
    for idx in range(rows):
        for j in range(d):
            I6_col[idx*d + j] = idx * d + j
    
    I6 = csr_matrix((I6_data, (I6_row, I6_col)), shape=(rows, n_2d * n_2d * d))
    
    I7 = csr_matrix((rows, n_2d * d))
    I8 = csr_matrix((rows, (n + d) * n_2d * d))
    
    block3 = hstack([T6, I5, I6, I7, I8], format='csr')
    return block3

def build_W34_block4_sparse(n, d):
    rows = n + 2*d
    n_2d = rows
    
    T6 = csr_matrix((rows, n_2d))
    I9 = csr_matrix((rows, n_2d * n_2d))
    I10 = csr_matrix((rows, n_2d * n_2d * d))
    
    nnz = rows * d
    I11_data = np.full(nnz, -1.0, dtype=np.float64)
    I11_row = np.repeat(np.arange(rows), d)
    I11_col = np.empty(nnz, dtype=int)
    
    for idx in range(rows):
        for j in range(d):
            I11_col[idx*d + j] = idx * d + j
    
    I11 = csr_matrix((I11_data, (I11_row, I11_col)), shape=(rows, n_2d * d))
    
    I12 = csr_matrix((rows, (n + d) * n_2d * d))
    
    block4 = hstack([T6, I9, I10, I11, I12], format='csr')
    return block4

def build_W34_block5_sparse(n, d):
    rows = (n + d) * (n + 2*d)
    n_2d = n + 2*d
    n_d = n + d
    
    T6 = csr_matrix((rows, n_2d))
    I13 = csr_matrix((rows, n_2d * n_2d))
    I14 = csr_matrix((rows, n_2d * n_2d * d))
    I15 = csr_matrix((rows, n_2d * d))
    
    nnz = rows * d
    I16_data = np.full(nnz, -1.0, dtype=np.float64)
    I16_row = np.repeat(np.arange(rows), d)
    I16_col = np.empty(nnz, dtype=int)
    
    for idx in range(rows):
        for j in range(d):
            I16_col[idx*d + j] = idx * d + j
    
    I16 = csr_matrix((I16_data, (I16_row, I16_col)), shape=(rows, n_d * n_2d * d))
    
    block5 = hstack([T6, I13, I14, I15, I16], format='csr')
    return block5

def build_W34(n, d):
    W34_blocks = []
    
    block1 = build_W34_block1_sparse(n, d)
    W34_blocks.append(block1)
    #print(f"Block 1: {block1.shape}")
    
    block2 = build_W34_block2_sparse(n, d)
    W34_blocks.append(block2)
    #print(f"Block 2: {block2.shape}")
    
    block3 = build_W34_block3_sparse(n, d)
    W34_blocks.append(block3)
    #print(f"Block 3: {block3.shape}")
    
    block4 = build_W34_block4_sparse(n, d)
    W34_blocks.append(block4)
    #print(f"Block 4: {block4.shape}")
    
    block5 = build_W34_block5_sparse(n, d)
    W34_blocks.append(block5)
    #print(f"Block 5: {block5.shape}")
    
    W34 = vstack(W34_blocks, format='csr')
    
    #print(f"\nTotal W34 shape: {W34.shape}")
    #print(f"Number of blocks: {len(W34_blocks)}")
    #print(f"Total non-zero entries: {W34.nnz}")
    
    return W34



def build_B34(n, d):
    """Build B34 as sparse vector."""
    from scipy.sparse import csr_matrix
    import numpy as np
    
    n_2d = n + 2*d
    
    # Calculate section sizes
    # 1. U2 nodes: n_2d entries (all zeros)
    size_u2 = n_2d
    
    # 2. tau2 nodes: n_2d * n_2d entries (all zeros)
    size_tau2 = n_2d * n_2d
    
    # 3. gamma2 nodes (e_ik): n_2d * n_2d entries (all = 1)
    size_gamma2 = n_2d * n_2d
    
    # 4. gamma3 nodes (e'_i): n_2d entries (all = 1)
    size_gamma3 = n_2d
    
    # 5. gamma4 nodes (p_ik): (n+d) * n_2d entries (all = 1)
    size_gamma4 = (n + d) * n_2d
    
    total_size = size_u2 + size_tau2 + size_gamma2 + size_gamma3 + size_gamma4
    
    #print(f"B34 total size calculated: {total_size}")
    #print(f"Sections: u2={size_u2}, tau2={size_tau2}, gamma2={size_gamma2}, "
          #f"gamma3={size_gamma3}, gamma4={size_gamma4}")
    
    # Non-zero sections: gamma2, gamma3, gamma4 (all = 1)
    non_zero_count = size_gamma2 + size_gamma3 + size_gamma4
    
    data = np.zeros(non_zero_count, dtype=np.float64)
    rows = np.zeros(non_zero_count, dtype=np.int32)
    
    idx = 0  # Current position in the full vector
    data_idx = 0  # Current position in the data/rows arrays
    
    # Skip zeros sections: u2 and tau2
    idx += size_u2 + size_tau2
    
    # gamma2 section (1)
    for i in range(size_gamma2):
        data[data_idx] = 1.0
        rows[data_idx] = idx
        data_idx += 1
        idx += 1
    
    # gamma3 section (1)
    for i in range(size_gamma3):
        data[data_idx] = 1.0
        rows[data_idx] = idx
        data_idx += 1
        idx += 1
    
    # gamma4 section (1)
    for i in range(size_gamma4):
        data[data_idx] = 1.0
        rows[data_idx] = idx
        data_idx += 1
        idx += 1
    
    # Create sparse column vector
    B34 = csr_matrix((data, (rows, np.zeros(len(data), dtype=np.int32))), 
                     shape=(total_size, 1))
    
    #print(f"B34 shape: {B34.shape}, Non-zeros: {B34.nnz}")
    return B34