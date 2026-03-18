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


import numpy as np
from scipy.sparse import csr_matrix, hstack, vstack

import numpy as np
from scipy.sparse import csr_matrix, hstack, vstack

def build_W23_block1_sparse(n, d):
    rows = n + 2*d
    n_2d = n + 2*d
    
    M15_data = np.ones(rows, dtype=np.float64)
    M15_row = np.arange(rows)
    M15_col = np.arange(rows)
    M15 = csr_matrix((M15_data, (M15_row, M15_col)), shape=(rows, n_2d))
    
    M16 = csr_matrix((rows, n_2d * n_2d))
    M17 = csr_matrix((rows, d * d))
    M18 = csr_matrix((rows, n_2d))
    M19 = csr_matrix((rows, n_2d))
    M20 = csr_matrix((rows, n_2d * n_2d * d))
    M21 = csr_matrix((rows, n_2d * n_2d * d))
    M22 = csr_matrix((rows, n_2d * d))
    M23 = csr_matrix((rows, 2*d))
    
    block1 = hstack([M15, M16, M17, M18, M19, M20, M21, M22, M23], format='csr')
    return block1

def build_W23_block2_sparse(n, d):
    rows = (n + 2*d) * (n + 2*d)
    n_2d = n + 2*d
    
    M15 = csr_matrix((rows, n_2d))
    
    M16_data = np.ones(rows, dtype=np.float64)
    M16_row = np.arange(rows)
    M16_col = np.arange(rows)
    M16 = csr_matrix((M16_data, (M16_row, M16_col)), shape=(rows, n_2d * n_2d))
    
    M17 = csr_matrix((rows, d * d))
    M18 = csr_matrix((rows, n_2d))
    M19 = csr_matrix((rows, n_2d))
    M20 = csr_matrix((rows, n_2d * n_2d * d))
    M21 = csr_matrix((rows, n_2d * n_2d * d))
    M22 = csr_matrix((rows, n_2d * d))
    M23 = csr_matrix((rows, 2*d))
    
    block2 = hstack([M15, M16, M17, M18, M19, M20, M21, M22, M23], format='csr')
    return block2

def build_W23_block3_sparse(n, d):
    rows = d
    n_2d = n + 2*d
    
    M15 = csr_matrix((rows, n_2d))
    M16 = csr_matrix((rows, n_2d * n_2d))
    
    M17_data = []
    M17_row = []
    M17_col = []
    
    for k in range(d):
        for j in range(d):
            for i in range(d):
                w = 1.0 if j == k else 0.0
                col = (j * d + i)
                M17_data.append(w)
                M17_row.append(k)
                M17_col.append(col)
    
    M17 = csr_matrix((M17_data, (M17_row, M17_col)), shape=(rows, d * d))
    
    M18 = csr_matrix((rows, n_2d))
    M19 = csr_matrix((rows, n_2d))
    M20 = csr_matrix((rows, n_2d * n_2d * d))
    M21 = csr_matrix((rows, n_2d * n_2d * d))
    M22 = csr_matrix((rows, n_2d * d))
    M23 = csr_matrix((rows, 2*d))
    
    block3 = hstack([M15, M16, M17, M18, M19, M20, M21, M22, M23], format='csr')
    return block3

def build_W23_block4_sparse(n, d):
    rows = (n + 2*d) * d
    n_2d = n + 2*d
    
    M15 = csr_matrix((rows, n_2d))
    M16 = csr_matrix((rows, n_2d * n_2d))
    M17 = csr_matrix((rows, d * d))
    M18 = csr_matrix((rows, n_2d))
    M19 = csr_matrix((rows, n_2d))
    M20 = csr_matrix((rows, n_2d * n_2d * d))
    M21 = csr_matrix((rows, n_2d * n_2d * d))
    
    M22_data = np.ones(rows, dtype=np.float64)
    M22_row = np.arange(rows)
    M22_col = np.arange(rows)
    M22 = csr_matrix((M22_data, (M22_row, M22_col)), shape=(rows, n_2d * d))
    
    M23 = csr_matrix((rows, 2*d))
    
    block4 = hstack([M15, M16, M17, M18, M19, M20, M21, M22, M23], format='csr')
    return block4

def build_W23_block5_sparse(n, d):
    rows = n + 2*d
    n_2d = n + 2*d
    
    M15 = csr_matrix((rows, n_2d))
    M16 = csr_matrix((rows, n_2d * n_2d))
    M17 = csr_matrix((rows, d * d))
    
    M18_data = np.ones(rows, dtype=np.float64)
    M18_row = np.arange(rows)
    M18_col = np.arange(rows)
    M18 = csr_matrix((M18_data, (M18_row, M18_col)), shape=(rows, n_2d))
    
    M19 = csr_matrix((rows, n_2d))
    M20 = csr_matrix((rows, n_2d * n_2d * d))
    M21 = csr_matrix((rows, n_2d * n_2d * d))
    M22 = csr_matrix((rows, n_2d * d))
    M23 = csr_matrix((rows, 2*d))
    
    block5 = hstack([M15, M16, M17, M18, M19, M20, M21, M22, M23], format='csr')
    return block5

def build_W23_block6_sparse(n, d):
    rows = n + 2*d
    n_2d = n + 2*d
    
    M15 = csr_matrix((rows, n_2d))
    M16 = csr_matrix((rows, n_2d * n_2d))
    M17 = csr_matrix((rows, d * d))
    M18 = csr_matrix((rows, n_2d))
    
    M19_data = np.ones(rows, dtype=np.float64)
    M19_row = np.arange(rows)
    M19_col = np.arange(rows)
    M19 = csr_matrix((M19_data, (M19_row, M19_col)), shape=(rows, n_2d))
    
    M20 = csr_matrix((rows, n_2d * n_2d * d))
    M21 = csr_matrix((rows, n_2d * n_2d * d))
    M22 = csr_matrix((rows, n_2d * d))
    M23 = csr_matrix((rows, 2*d))
    
    block6 = hstack([M15, M16, M17, M18, M19, M20, M21, M22, M23], format='csr')
    return block6

def build_W23_block7_sparse(n, d):
    rows = (n + 2*d) * (n + 2*d)
    n_2d = n + 2*d
    
    M15 = csr_matrix((rows, n_2d))
    M16 = csr_matrix((rows, n_2d * n_2d))
    M17 = csr_matrix((rows, d * d))
    M18 = csr_matrix((rows, n_2d))
    M19 = csr_matrix((rows, n_2d))
    
    M20_data = np.ones(rows * d, dtype=np.float64)
    M20_row = []
    M20_col = []
    
    M21_data = np.ones(rows * d, dtype=np.float64)
    M21_row = []
    M21_col = []
    
    for idx in range(rows):
        i = idx // n_2d
        j = idx % n_2d
        
        for p in range(d):
            col = (i * n_2d + j) * d + p
            M20_row.append(idx)
            M20_col.append(col)
            M21_row.append(idx)
            M21_col.append(col)
    
    M20 = csr_matrix((M20_data, (M20_row, M20_col)), shape=(rows, n_2d * n_2d * d))
    M21 = csr_matrix((M21_data, (M21_row, M21_col)), shape=(rows, n_2d * n_2d * d))
    
    M22 = csr_matrix((rows, n_2d * d))
    M23 = csr_matrix((rows, 2*d))
    
    block7 = hstack([M15, M16, M17, M18, M19, M20, M21, M22, M23], format='csr')
    return block7

def build_W23_block8_sparse(n, d):
    rows = 2*d
    n_2d = n + 2*d
    
    M15 = csr_matrix((rows, n_2d))
    M16 = csr_matrix((rows, n_2d * n_2d))
    M17 = csr_matrix((rows, d * d))
    M18 = csr_matrix((rows, n_2d))
    M19 = csr_matrix((rows, n_2d))
    M20 = csr_matrix((rows, n_2d * n_2d * d))
    M21 = csr_matrix((rows, n_2d * n_2d * d))
    M22 = csr_matrix((rows, n_2d * d))
    
    M23_data = np.ones(rows, dtype=np.float64)
    M23_row = np.arange(rows)
    M23_col = np.arange(rows)
    M23 = csr_matrix((M23_data, (M23_row, M23_col)), shape=(rows, 2*d))
    
    block8 = hstack([M15, M16, M17, M18, M19, M20, M21, M22, M23], format='csr')
    return block8

def build_W23(n, d):
    W23_blocks = []
    
    block1 = build_W23_block1_sparse(n, d)
    W23_blocks.append(block1)
    #print(f"W23 Block 1: {block1.shape}")
    
    block2 = build_W23_block2_sparse(n, d)
    W23_blocks.append(block2)
    #print(f"W23 Block 2: {block2.shape}")
    
    block3 = build_W23_block3_sparse(n, d)
    W23_blocks.append(block3)
    #print(f"W23 Block 3: {block3.shape}")
    
    block4 = build_W23_block4_sparse(n, d)
    W23_blocks.append(block4)
    #print(f"W23 Block 4: {block4.shape}")
    
    block5 = build_W23_block5_sparse(n, d)
    W23_blocks.append(block5)
    #print(f"W23 Block 5: {block5.shape}")
    
    block6 = build_W23_block6_sparse(n, d)
    W23_blocks.append(block6)
    #print(f"W23 Block 6: {block6.shape}")
    
    block7 = build_W23_block7_sparse(n, d)
    W23_blocks.append(block7)
    #print(f"W23 Block 7: {block7.shape}")
    
    block8 = build_W23_block8_sparse(n, d)
    W23_blocks.append(block8)
    #print(f"W23 Block 8: {block8.shape}")
    
    W23 = vstack(W23_blocks, format='csr')
    
    #print(f"\nTotal W23 shape: {W23.shape}")
    #print(f"W23 non-zero entries: {W23.nnz}")
    
    return W23
def build_B23(n, d):
    """Build B23 as sparse all-zeros vector."""
    from scipy.sparse import csr_matrix
    import numpy as np
    
    # Create empty sparse matrix with correct size
    # The size calculation is the same as above
    n_2d = n + 2*d
    total_size = (n_2d + n_2d * n_2d + d + n_2d * d + 
                  n_2d + n_2d + n_2d * n_2d + 2*d)
    
    return csr_matrix(np.zeros((total_size, 1)))