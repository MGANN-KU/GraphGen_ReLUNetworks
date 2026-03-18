# -*- coding: utf-8 -*-
"""
Weights and bias of layer 40

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


# ============================================================
# BLOCK 1 (already optimal — kept as is)
# ============================================================

def build_W40_block1_sparse(n, d):
    rows = n + d
    n_d = rows
    n_2d = n + 2 * d

    H10 = csr_matrix((rows, n_d * n_2d))

    H11_data = np.ones(rows, dtype=np.float64)
    H11_row = np.arange(rows)
    H11_col = np.arange(rows)
    H11 = csr_matrix((H11_data, (H11_row, H11_col)), shape=(rows, n_d))

    H12 = csr_matrix((rows, n_d * n_d * (d + 1)))

    return hstack([H10, H11, H12], format="csr")


# ============================================================
# BLOCK 2 (FIXED)
# ============================================================

def build_W40_block2_sparse(n, d, C):
    n_d  = n + d
    n_2d = n + 2 * d
    d2   = d + 1

    rows = n_d * n_d * d2

    # ------------------------------
    # H13 (single 1 per row)
    # ------------------------------

    idx = np.arange(rows)

    i = idx // (n_d * d2)
    rem = idx % (n_d * d2)
    l = rem // d2
    j = rem % d2

    p = l + j
    valid = p < n_2d

    H13_row = idx[valid]
    H13_col = i[valid] * n_2d + p[valid]
    H13_data = np.ones(len(H13_row), dtype=np.float64)

    H13 = csr_matrix(
        (H13_data, (H13_row, H13_col)),
        shape=(rows, n_d * n_2d)
    )

    # ------------------------------
    # H14 (all zeros)
    # ------------------------------

    H14 = csr_matrix((rows, n_d))

    # ------------------------------
    # H15 (diagonal with C)
    # ------------------------------

    H15 = csr_matrix(
        (np.full(rows, C, dtype=np.float64), (idx, idx)),
        shape=(rows, n_d * n_d * d2)
    )

    return hstack([H13, H14, H15], format="csr")


# ============================================================
# FINAL W40
# ============================================================

def build_W40(n, d, C):
    block1 = build_W40_block1_sparse(n, d)
    block2 = build_W40_block2_sparse(n, d, C)

    return vstack([block1, block2], format="csr")
  
def build_B40(n, d, C):
    """Build B40 as sparse vector."""
    from scipy.sparse import csr_matrix
    import numpy as np
    
    n_d = n + d
    
    # Calculate section sizes
    # 1. zeta4 nodes: n+d entries (all zeros)
    size_zeta4 = n_d
    
    # 2. zeta5 nodes: (n+d) * (n+d) * (d+1) entries (all = -C)
    # Note: j from 1 to d+1 (inclusive), so that's d+1 entries
    size_zeta5 = n_d * n_d * (d + 1)
    
    total_size = size_zeta4 + size_zeta5
    
    #print(f"B40 total size calculated: {total_size}")
    #print(f"Sections: zeta4={size_zeta4}, zeta5={size_zeta5}")
    
    # Non-zero sections: only zeta5 has non-zero values (-C)
    non_zero_count = size_zeta5
    
    data = np.zeros(non_zero_count, dtype=np.float64)
    rows = np.zeros(non_zero_count, dtype=np.int32)
    
    idx = 0  # Current position in the full vector
    data_idx = 0  # Current position in the data/rows arrays
    
    # Skip zeros section: zeta4
    idx += size_zeta4
    
    # zeta5 section (-C)
    for i in range(size_zeta5):
        data[data_idx] = -C
        rows[data_idx] = idx
        data_idx += 1
        idx += 1
    
    # Create sparse column vector
    B40 = csr_matrix((data, (rows, np.zeros(len(data), dtype=np.int32))), 
                     shape=(total_size, 1))
    
    #print(f"B40 shape: {B40.shape}, Non-zeros: {B40.nnz}")
    return B40