# -*- coding: utf-8 -*-
"""
Weights and bias of layer 36

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

def build_W36_block1_sparse(n, d):
    rows = n + 2*d
    n_2d = rows
    
    T8_data = np.ones(rows, dtype=np.float64)
    T8_row = np.arange(rows)
    T8_col = np.arange(rows)
    T8 = csr_matrix((T8_data, (T8_row, T8_col)), shape=(rows, n_2d))
    
    I29 = csr_matrix((rows, n_2d * n_2d))
    I30 = csr_matrix((rows, n_2d * n_2d))
    I31 = csr_matrix((rows, n_2d))
    I32 = csr_matrix((rows, n_2d * n_2d * 2))
    I33 = csr_matrix((rows, n_2d * n_2d * 2))
    I34 = csr_matrix((rows, n_2d * 2))
    I35 = csr_matrix((rows, n_2d * 2))
    I36 = csr_matrix((rows, (n + d) * n_2d))
    I37 = csr_matrix((rows, (n + d) * n_2d * 2))
    I38 = csr_matrix((rows, (n + d) * n_2d * 2))
    
    block1 = hstack([T8, I29, I30, I31, I32, I33, I34, I35, I36, I37, I38], format='csr')
    return block1

def build_W36_block2_sparse(n, d):
    rows = (n + 2*d) * (n + 2*d)
    n_2d = n + 2*d
    
    T8 = csr_matrix((rows, n_2d))
    
    nnz = rows
    I29_data = np.ones(nnz, dtype=np.float64)
    I29_row = np.arange(rows)
    I29_col = np.arange(rows)
    I29 = csr_matrix((I29_data, (I29_row, I29_col)), shape=(rows, n_2d * n_2d))
    
    I30 = csr_matrix((rows, n_2d * n_2d))
    I31 = csr_matrix((rows, n_2d))
    I32 = csr_matrix((rows, n_2d * n_2d * 2))
    I33 = csr_matrix((rows, n_2d * n_2d * 2))
    I34 = csr_matrix((rows, n_2d * 2))
    I35 = csr_matrix((rows, n_2d * 2))
    I36 = csr_matrix((rows, (n + d) * n_2d))
    I37 = csr_matrix((rows, (n + d) * n_2d * 2))
    I38 = csr_matrix((rows, (n + d) * n_2d * 2))
    
    block2 = hstack([T8, I29, I30, I31, I32, I33, I34, I35, I36, I37, I38], format='csr')
    return block2

def build_W36_block3_sparse(n, d, B, C):
    rows = (n + 2*d) * (n + 2*d)
    n_2d = n + 2*d
    
    T8 = csr_matrix((rows, n_2d))
    E11 = csr_matrix((rows, n_2d * n_2d))
    
    nnz = rows
    E12_data = np.full(nnz, B, dtype=np.float64)
    E12_row = np.arange(rows)
    E12_col = np.arange(rows)
    E12 = csr_matrix((E12_data, (E12_row, E12_col)), shape=(rows, n_2d * n_2d))
    
    E13 = csr_matrix((rows, n_2d))
    
    nnz2 = rows * 2
    E14_data = np.empty(nnz2, dtype=np.float64)
    E14_row = np.repeat(np.arange(rows), 2)
    E14_col = np.empty(nnz2, dtype=int)
    
    for idx in range(rows):
        base_col = idx * 2
        E14_col[idx*2] = base_col
        E14_col[idx*2 + 1] = base_col + 1
        E14_data[idx*2] = -C
        E14_data[idx*2 + 1] = C
    
    E14 = csr_matrix((E14_data, (E14_row, E14_col)), shape=(rows, n_2d * n_2d * 2))
    
    E15 = csr_matrix((E14_data, (E14_row, E14_col)), shape=(rows, n_2d * n_2d * 2))
    
    E16 = csr_matrix((rows, n_2d * 2))
    E17 = csr_matrix((rows, n_2d * 2))
    E18 = csr_matrix((rows, (n + d) * n_2d))
    E19 = csr_matrix((rows, (n + d) * n_2d * 2))
    E20 = csr_matrix((rows, (n + d) * n_2d * 2))
    
    block3 = hstack([T8, E11, E12, E13, E14, E15, E16, E17, E18, E19, E20], format='csr')
    return block3

def build_W36_block4_sparse(n, d, B, C):
    rows = n + 2*d
    n_2d = rows
    
    T8 = csr_matrix((rows, n_2d))
    E21 = csr_matrix((rows, n_2d * n_2d))
    E22 = csr_matrix((rows, n_2d * n_2d))
    
    nnz = rows
    E23_data = np.full(nnz, B, dtype=np.float64)
    E23_row = np.arange(rows)
    E23_col = np.arange(rows)
    E23 = csr_matrix((E23_data, (E23_row, E23_col)), shape=(rows, n_2d))
    
    E24 = csr_matrix((rows, n_2d * n_2d * 2))
    E25 = csr_matrix((rows, n_2d * n_2d * 2))
    
    nnz2 = rows * 2
    E26_data = np.empty(nnz2, dtype=np.float64)
    E26_row = np.repeat(np.arange(rows), 2)
    E26_col = np.empty(nnz2, dtype=int)
    
    for idx in range(rows):
        base_col = idx * 2
        E26_col[idx*2] = base_col
        E26_col[idx*2 + 1] = base_col + 1
        E26_data[idx*2] = -C
        E26_data[idx*2 + 1] = C
    
    E26 = csr_matrix((E26_data, (E26_row, E26_col)), shape=(rows, n_2d * 2))
    
    E27 = csr_matrix((E26_data, (E26_row, E26_col)), shape=(rows, n_2d * 2))
    
    E28 = csr_matrix((rows, (n + d) * n_2d))
    E29 = csr_matrix((rows, (n + d) * n_2d * 2))
    E30 = csr_matrix((rows, (n + d) * n_2d * 2))
    
    block4 = hstack([T8, E21, E22, E23, E24, E25, E26, E27, E28, E29, E30], format='csr')
    return block4

def build_W36_block5_sparse(n, d, B, C):
    rows = (n + d) * (n + 2*d)
    n_2d = n + 2*d
    n_d = n + d
    
    T8 = csr_matrix((rows, n_2d))
    E31 = csr_matrix((rows, n_2d * n_2d))
    E32 = csr_matrix((rows, n_2d * n_2d))
    E33 = csr_matrix((rows, n_2d))
    E34 = csr_matrix((rows, n_2d * n_2d * 2))
    E35 = csr_matrix((rows, n_2d * n_2d * 2))
    E36 = csr_matrix((rows, n_2d * 2))
    E37 = csr_matrix((rows, n_2d * 2))
    
    nnz = rows
    E38_data = np.full(nnz, B, dtype=np.float64)
    E38_row = np.arange(rows)
    E38_col = np.arange(rows)
    E38 = csr_matrix((E38_data, (E38_row, E38_col)), shape=(rows, n_d * n_2d))
    
    nnz2 = rows * 2
    E39_data = np.empty(nnz2, dtype=np.float64)
    E39_row = np.repeat(np.arange(rows), 2)
    E39_col = np.empty(nnz2, dtype=int)
    
    for idx in range(rows):
        base_col = idx * 2
        E39_col[idx*2] = base_col
        E39_col[idx*2 + 1] = base_col + 1
        E39_data[idx*2] = -C
        E39_data[idx*2 + 1] = C
    
    E39 = csr_matrix((E39_data, (E39_row, E39_col)), shape=(rows, n_d * n_2d * 2))
    
    E40 = csr_matrix((E39_data, (E39_row, E39_col)), shape=(rows, n_d * n_2d * 2))
    
    block5 = hstack([T8, E31, E32, E33, E34, E35, E36, E37, E38, E39, E40], format='csr')
    return block5

def build_W36(n, d, B, C):
    W36_blocks = []
    
    block1 = build_W36_block1_sparse(n, d)
    W36_blocks.append(block1)
    
    block2 = build_W36_block2_sparse(n, d)
    W36_blocks.append(block2)
    
    block3 = build_W36_block3_sparse(n, d, B, C)
    W36_blocks.append(block3)
    
    block4 = build_W36_block4_sparse(n, d, B, C)
    W36_blocks.append(block4)
    
    block5 = build_W36_block5_sparse(n, d, B, C)
    W36_blocks.append(block5)
    
    W36 = vstack(W36_blocks, format='csr')
    
    return W36
def build_B36(n, d, C):
    """Build B36 as sparse vector."""
    from scipy.sparse import csr_matrix
    import numpy as np
    
    n_2d = n + 2*d
    
    # Calculate section sizes
    # 1. U2 nodes: n_2d entries (all zeros)
    size_u2 = n_2d
    
    # 2. tau2 nodes: n_2d * n_2d entries (all zeros)
    size_tau2 = n_2d * n_2d
    
    # 3. gamma8 nodes (f_ik): n_2d * n_2d entries (all = C)
    size_gamma8 = n_2d * n_2d
    
    # 4. gamma9 nodes (f_i): n_2d entries (all = C)
    size_gamma9 = n_2d
    
    # 5. gamma10 nodes (q_il): (n+d) * n_2d entries (all = C)
    # Note: i from 1 to n+d, l from 1 to n+2d
    size_gamma10 = (n + d) * n_2d
    
    total_size = size_u2 + size_tau2 + size_gamma8 + size_gamma9 + size_gamma10
    
    #print(f"B36 total size calculated: {total_size}")
    #print(f"Sections: u2={size_u2}, tau2={size_tau2}, gamma8={size_gamma8}, "
          #f"gamma9={size_gamma9}, gamma10={size_gamma10}")
    
    # Non-zero sections: gamma8, gamma9, gamma10 (all = C)
    non_zero_count = size_gamma8 + size_gamma9 + size_gamma10
    
    data = np.zeros(non_zero_count, dtype=np.float64)
    rows = np.zeros(non_zero_count, dtype=np.int32)
    
    idx = 0  # Current position in the full vector
    data_idx = 0  # Current position in the data/rows arrays
    
    # Skip zeros sections: u2 and tau2
    idx += size_u2 + size_tau2
    
    # gamma8 section (C)
    for i in range(size_gamma8):
        data[data_idx] = C
        rows[data_idx] = idx
        data_idx += 1
        idx += 1
    
    # gamma9 section (C)
    for i in range(size_gamma9):
        data[data_idx] = C
        rows[data_idx] = idx
        data_idx += 1
        idx += 1
    
    # gamma10 section (C)
    for i in range(size_gamma10):
        data[data_idx] = C
        rows[data_idx] = idx
        data_idx += 1
        idx += 1
    
    # Create sparse column vector
    B36 = csr_matrix((data, (rows, np.zeros(len(data), dtype=np.int32))), 
                     shape=(total_size, 1))
    
    #print(f"B36 shape: {B36.shape}, Non-zeros: {B36.nnz}")
    return B36