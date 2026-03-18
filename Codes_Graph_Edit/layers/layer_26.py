# -*- coding: utf-8 -*-
"""
Weights and bias of layer 26

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

import numpy as np
from scipy.sparse import csr_matrix, hstack, vstack


import numpy as np
from scipy.sparse import csr_matrix, hstack, vstack

import numpy as np
from scipy.sparse import csr_matrix, hstack, vstack

import numpy as np
from scipy.sparse import csr_matrix, hstack, vstack

def build_W26_block1_sparse(n, d, C):
    rows = n + 2*d
    n_2d = n + 2*d
    
    N18_data = np.ones(rows, dtype=np.float64)
    N18_row = np.arange(rows)
    N18_col = np.arange(rows)
    N18 = csr_matrix((N18_data, (N18_row, N18_col)), shape=(rows, n_2d))
    
    N19 = csr_matrix((rows, n_2d))
    N20 = csr_matrix((rows, n_2d * n_2d))
    
    N21_data = np.full(rows, C, dtype=np.float64)
    N21_row = np.arange(rows)
    N21_col = np.arange(rows)
    N21 = csr_matrix((N21_data, (N21_row, N21_col)), shape=(rows, n_2d))
    
    N22 = csr_matrix((rows, n_2d))
    N23 = csr_matrix((rows, n_2d * n_2d * d))
    N24 = csr_matrix((rows, n_2d * n_2d * d))
    N25 = csr_matrix((rows, n * d))
    N26 = csr_matrix((rows, 2*d))
    
    block1 = hstack([N18, N19, N20, N21, N22, N23, N24, N25, N26], format='csr')
    return block1

def build_W26_block2_sparse(n, d, C):
    rows = n + 2*d
    n_2d = n + 2*d
    
    N18 = csr_matrix((rows, n_2d))
    
    N19_data = np.ones(rows, dtype=np.float64)
    N19_row = np.arange(rows)
    N19_col = np.arange(rows)
    N19 = csr_matrix((N19_data, (N19_row, N19_col)), shape=(rows, n_2d))
    
    N20 = csr_matrix((rows, n_2d * n_2d))
    N21 = csr_matrix((rows, n_2d))
    
    N22_data = np.full(rows, C, dtype=np.float64)
    N22_row = np.arange(rows)
    N22_col = np.arange(rows)
    N22 = csr_matrix((N22_data, (N22_row, N22_col)), shape=(rows, n_2d))
    
    N23 = csr_matrix((rows, n_2d * n_2d * d))
    N24 = csr_matrix((rows, n_2d * n_2d * d))
    N25 = csr_matrix((rows, n * d))
    N26 = csr_matrix((rows, 2*d))
    
    block2 = hstack([N18, N19, N20, N21, N22, N23, N24, N25, N26], format='csr')
    return block2

def build_W26_block3_sparse(n, d):
    """nu1 as identity map - FIXED with proper pattern"""
    n_2d = n + 2*d
    rows = n_2d * n_2d
    
    # N18: All zeros
    N18_data = []
    N18_row = []
    N18_col = []
    
    # N19: All zeros
    N19_data = []
    N19_row = []
    N19_col = []
    
    # N20: Identity pattern for nu1
    N20_data = []
    N20_row = []
    N20_col = []
    
    # N21: All zeros
    N21_data = []
    N21_row = []
    N21_col = []
    
    # N22: All zeros
    N22_data = []
    N22_row = []
    N22_col = []
    
    # N23: All zeros
    N23_data = []
    N23_row = []
    N23_col = []
    
    # N24: All zeros
    N24_data = []
    N24_row = []
    N24_col = []
    
    # N25: All zeros
    N25_data = []
    N25_row = []
    N25_col = []
    
    # N26: All zeros
    N26_data = []
    N26_row = []
    N26_col = []
    
    for idx in range(rows):
        i = idx // n_2d  # 0 to n_2d-1
        p = idx % n_2d   # 0 to n_2d-1
        
        # N20: if i == k and p==j: w = 1
        # k corresponds to column index in N20
        # N20 has size n_2d * n_2d
        col = i * n_2d + p  # Maps to position (i,p) in the flattened N20
        N20_data.append(1.0)
        N20_row.append(idx)
        N20_col.append(col)
    
    # Create sparse matrices
    N18 = csr_matrix((N18_data, (N18_row, N18_col)), shape=(rows, n_2d))
    N19 = csr_matrix((N19_data, (N19_row, N19_col)), shape=(rows, n_2d))
    N20 = csr_matrix((N20_data, (N20_row, N20_col)), shape=(rows, n_2d * n_2d))
    N21 = csr_matrix((N21_data, (N21_row, N21_col)), shape=(rows, n_2d))
    N22 = csr_matrix((N22_data, (N22_row, N22_col)), shape=(rows, n_2d))
    N23 = csr_matrix((N23_data, (N23_row, N23_col)), shape=(rows, n_2d * n_2d * d))
    N24 = csr_matrix((N24_data, (N24_row, N24_col)), shape=(rows, n_2d * n_2d * d))
    N25 = csr_matrix((N25_data, (N25_row, N25_col)), shape=(rows, n * d))
    N26 = csr_matrix((N26_data, (N26_row, N26_col)), shape=(rows, 2*d))
    
    block3 = hstack([N18, N19, N20, N21, N22, N23, N24, N25, N26], format='csr')
    return block3

def build_W26_block4_sparse(n, d):
    """gamma1 nodes for t_ik - FIXED with proper pattern"""
    n_2d = n + 2*d
    rows = n_2d * n_2d
    
    # N18: All zeros
    N18_data = []
    N18_row = []
    N18_col = []
    
    # N19: All zeros
    N19_data = []
    N19_row = []
    N19_col = []
    
    # N20: All zeros
    N20_data = []
    N20_row = []
    N20_col = []
    
    # N21: All zeros
    N21_data = []
    N21_row = []
    N21_col = []
    
    # N22: All zeros
    N22_data = []
    N22_row = []
    N22_col = []
    
    # N23: Identity pattern for eta5
    N23_data = []
    N23_row = []
    N23_col = []
    
    # N24: Identity pattern for eta'5
    N24_data = []
    N24_row = []
    N24_col = []
    
    # N25: All zeros
    N25_data = []
    N25_row = []
    N25_col = []
    
    # N26: All zeros
    N26_data = []
    N26_row = []
    N26_col = []
    
    for idx in range(rows):
        i = idx // n_2d  # 0 to n_2d-1
        p = idx % n_2d   # 0 to n_2d-1
        
        # N23: if i == k and p==l: w = 1 for each j in 1..d
        # N23 has size n_2d * n_2d * d
        # Need to place 1 at positions corresponding to (i,p) for each j
        for j in range(d):
            col = (i * n_2d + p) * d + j
            N23_data.append(1.0)
            N23_row.append(idx)
            N23_col.append(col)
            
            # N24: same pattern
            N24_data.append(1.0)
            N24_row.append(idx)
            N24_col.append(col)
    
    # Create sparse matrices
    N18 = csr_matrix((N18_data, (N18_row, N18_col)), shape=(rows, n_2d))
    N19 = csr_matrix((N19_data, (N19_row, N19_col)), shape=(rows, n_2d))
    N20 = csr_matrix((N20_data, (N20_row, N20_col)), shape=(rows, n_2d * n_2d))
    N21 = csr_matrix((N21_data, (N21_row, N21_col)), shape=(rows, n_2d))
    N22 = csr_matrix((N22_data, (N22_row, N22_col)), shape=(rows, n_2d))
    N23 = csr_matrix((N23_data, (N23_row, N23_col)), shape=(rows, n_2d * n_2d * d))
    N24 = csr_matrix((N24_data, (N24_row, N24_col)), shape=(rows, n_2d * n_2d * d))
    N25 = csr_matrix((N25_data, (N25_row, N25_col)), shape=(rows, n * d))
    N26 = csr_matrix((N26_data, (N26_row, N26_col)), shape=(rows, 2*d))
    
    block4 = hstack([N18, N19, N20, N21, N22, N23, N24, N25, N26], format='csr')
    return block4

def build_W26_block5_sparse(n, d):
    """delta(x_j,i) as identity map"""
    rows = n * d
    n_2d = n + 2*d
    
    N18 = csr_matrix((rows, n_2d))
    N19 = csr_matrix((rows, n_2d))
    N20 = csr_matrix((rows, n_2d * n_2d))
    N21 = csr_matrix((rows, n_2d))
    N22 = csr_matrix((rows, n_2d))
    N23 = csr_matrix((rows, n_2d * n_2d * d))
    N24 = csr_matrix((rows, n_2d * n_2d * d))
    
    N25_data = np.ones(rows, dtype=np.float64)
    N25_row = np.arange(rows)
    N25_col = np.arange(rows)
    N25 = csr_matrix((N25_data, (N25_row, N25_col)), shape=(rows, n * d))
    
    N26 = csr_matrix((rows, 2*d))
    
    block5 = hstack([N18, N19, N20, N21, N22, N23, N24, N25, N26], format='csr')
    return block5

def build_W26_block6_sparse(n, d):
    """xj as identity map"""
    rows = 2*d
    n_2d = n + 2*d
    
    N18 = csr_matrix((rows, n_2d))
    N19 = csr_matrix((rows, n_2d))
    N20 = csr_matrix((rows, n_2d * n_2d))
    N21 = csr_matrix((rows, n_2d))
    N22 = csr_matrix((rows, n_2d))
    N23 = csr_matrix((rows, n_2d * n_2d * d))
    N24 = csr_matrix((rows, n_2d * n_2d * d))
    N25 = csr_matrix((rows, n * d))
    
    N26_data = np.ones(rows, dtype=np.float64)
    N26_row = np.arange(rows)
    N26_col = np.arange(rows)
    N26 = csr_matrix((N26_data, (N26_row, N26_col)), shape=(rows, 2*d))
    
    block6 = hstack([N18, N19, N20, N21, N22, N23, N24, N25, N26], format='csr')
    return block6

def build_W26(n, d, C):
    W26_blocks = []
    
    block1 = build_W26_block1_sparse(n, d, C)
    W26_blocks.append(block1)
    #print(f"W26 Block 1: {block1.shape}")
    
    block2 = build_W26_block2_sparse(n, d, C)
    W26_blocks.append(block2)
    #print(f"W26 Block 2: {block2.shape}")
    
    block3 = build_W26_block3_sparse(n, d)
    W26_blocks.append(block3)
    #print(f"W26 Block 3: {block3.shape}")
    
    block4 = build_W26_block4_sparse(n, d)
    W26_blocks.append(block4)
    #print(f"W26 Block 4: {block4.shape}")
    
    block5 = build_W26_block5_sparse(n, d)
    W26_blocks.append(block5)
    #print(f"W26 Block 5: {block5.shape}")
    
    block6 = build_W26_block6_sparse(n, d)
    W26_blocks.append(block6)
    #print(f"W26 Block 6: {block6.shape}")
    
    W26 = vstack(W26_blocks, format='csr')
    
    #print(f"\nTotal W26 shape: {W26.shape}")
    #print(f"W26 non-zero entries: {W26.nnz}")
    
    return W26
def build_B26(n, d, C):
    """Build B26 as sparse vector."""
    from scipy.sparse import csr_matrix
    import numpy as np
    
    n_2d = n + 2*d
    
    # Calculate section sizes
    # 1. tau10 nodes: n_2d entries (all = -C)
    size_tau10 = n_2d
    
    # 2. tau11 nodes: n_2d entries (all = -C)
    size_tau11 = n_2d
    
    # 3. nu1 nodes: n_2d * n_2d entries (all zeros)
    size_nu1 = n_2d * n_2d
    
    # 4. gamma1 nodes (t_ik): n_2d * n_2d entries (all zeros)
    size_gamma1 = n_2d * n_2d
    
    # 5. Next section: n * d entries (all zeros)
    # Note: i from 1 to n, p from 1 to d
    size_section5 = n * d
    
    # 6. Last section: 2*d entries (all zeros)
    size_section6 = 2*d
    
    total_size = size_tau10 + size_tau11 + size_nu1 + size_gamma1 + size_section5 + size_section6
    
    #print(f"B26 total size calculated: {total_size}")
    #print(f"Sections: tau10={size_tau10}, tau11={size_tau11}, nu1={size_nu1}, "
          #f"gamma1={size_gamma1}, section5={size_section5}, section6={size_section6}")
    
    # Non-zero sections: tau10 and tau11 (both = -C)
    non_zero_count = size_tau10 + size_tau11
    
    data = np.zeros(non_zero_count, dtype=np.float64)
    rows = np.zeros(non_zero_count, dtype=np.int32)
    
    idx = 0  # Current position in the full vector
    data_idx = 0  # Current position in the data/rows arrays
    
    # tau10 section (-C)
    for i in range(size_tau10):
        data[data_idx] = -C
        rows[data_idx] = idx
        data_idx += 1
        idx += 1
    
    # tau11 section (-C)
    for i in range(size_tau11):
        data[data_idx] = -C
        rows[data_idx] = idx
        data_idx += 1
        idx += 1
    
    # Skip zeros sections: nu1, gamma1, section5, section6
    # (idx already incremented to correct position)
    
    # Create sparse column vector
    B26 = csr_matrix((data, (rows, np.zeros(len(data), dtype=np.int32))), 
                     shape=(total_size, 1))
    
    #print(f"B26 shape: {B26.shape}, Non-zeros: {B26.nnz}")
    return B26