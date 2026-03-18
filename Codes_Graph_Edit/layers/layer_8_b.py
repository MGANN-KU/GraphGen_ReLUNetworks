# -*- coding: utf-8 -*-
"""
Weights and bias of layer 8

@author: Ghafoor
"""
import numpy as np
from scipy.sparse import lil_matrix


import numpy as np
from scipy.sparse import csr_matrix

def build_W8(d):
    """
    Correct vectorized implementation matching dense dimensions
    For d=2: Returns (20, 54) sparse matrix
    """
    total_rows = 10 * d
    total_cols = 27 * d
    
    # Initialize COO arrays
    data = []
    rows = []
    cols = []
    
    # COLUMN OFFSETS (in terms of scalar columns, not q-pairs)
    # Each E block contributes d columns (except E27=3d) in dense representation
    # But in sparse, we need to account for q dimension in some blocks
    
    # For blocks WITH q loop (E20, E21, E23-E26, E28-E31):
    # In dense: d columns, but each column position has 2 entries (q=0,1)
    # In sparse: we need 2d actual columns
    
    # For blocks WITHOUT q loop (E22, E27, E32):
    # In dense: 2d columns for E22/E32, 3d for E27
    # In sparse: same number of columns
    
    # Calculate column bases correctly:
    # E20: 0-2d-1 (2d columns for q dimension)
    # E21: 2d-4d-1 (2d columns)
    # E22: 4d-6d-1 (2d columns, no q dimension)
    # E23: 6d-8d-1 (2d columns for q dimension)
    # E24: 8d-10d-1 (2d columns)
    # E25: 10d-12d-1 (2d columns)
    # E26: 12d-14d-1 (2d columns)
    # E27: 14d-17d-1 (3d columns, no q dimension)
    # E28: 17d-19d-1 (2d columns for q dimension)
    # E29: 19d-21d-1 (2d columns)
    # E30: 21d-23d-1 (2d columns)
    # E31: 23d-25d-1 (2d columns)
    # E32: 25d-27d-1 (2d columns, no q dimension)
    
    # This matches total_cols = 27d
    
    # BLOCK 1: zeta1 nodes (d rows) - E20, E21
    row_block1 = np.arange(d)
    
    # E20: columns 0 to 2d-1, pattern at positions 2j and 2j+1
    data.extend([-1] * d + [1] * d)
    rows.extend(np.concatenate([row_block1, row_block1]))
    cols.extend(np.concatenate([2*row_block1, 2*row_block1 + 1]))
    
    # E21: columns 2d to 4d-1, same pattern
    data.extend([-1] * d + [1] * d)
    rows.extend(np.concatenate([row_block1, row_block1]))
    cols.extend(np.concatenate([2*d + 2*row_block1, 2*d + 2*row_block1 + 1]))
    
    # BLOCK 2: xj identity for E22 (2d rows)
    row_block2 = np.arange(d, 3*d)
    col_base_e22 = 4 * d
    col_e22 = col_base_e22 + (row_block2 - d)
    
    data.extend([1] * (2*d))
    rows.extend(row_block2)
    cols.extend(col_e22)
    
    # BLOCK 3: zeta2 nodes (d rows) - E23, E24, E25, E26
    row_block3 = np.arange(3*d, 4*d)
    row_block3_local = np.arange(d)
    
    # E23: columns 6d to 8d-1
    col_base_e23 = 6 * d
    data.extend([-1] * d + [1] * d)
    rows.extend(np.concatenate([row_block3, row_block3]))
    cols.extend(np.concatenate([col_base_e23 + 2*row_block3_local, 
                               col_base_e23 + 2*row_block3_local + 1]))
    
    # E24: columns 8d to 10d-1
    col_base_e24 = 8 * d
    data.extend([-1] * d + [1] * d)
    rows.extend(np.concatenate([row_block3, row_block3]))
    cols.extend(np.concatenate([col_base_e24 + 2*row_block3_local,
                               col_base_e24 + 2*row_block3_local + 1]))
    
    # E25: columns 10d to 12d-1
    col_base_e25 = 10 * d
    data.extend([-1] * d + [1] * d)
    rows.extend(np.concatenate([row_block3, row_block3]))
    cols.extend(np.concatenate([col_base_e25 + 2*row_block3_local,
                               col_base_e25 + 2*row_block3_local + 1]))
    
    # E26: columns 12d to 14d-1
    col_base_e26 = 12 * d
    data.extend([-1] * d + [1] * d)
    rows.extend(np.concatenate([row_block3, row_block3]))
    cols.extend(np.concatenate([col_base_e26 + 2*row_block3_local,
                               col_base_e26 + 2*row_block3_local + 1]))
    
    # BLOCK 4: xj identity for E27 (3d rows)
    row_block4 = np.arange(4*d, 7*d)
    col_base_e27 = 14 * d
    col_e27 = col_base_e27 + (row_block4 - 4*d)
    
    data.extend([1] * (3*d))
    rows.extend(row_block4)
    cols.extend(col_e27)
    
    # BLOCK 5: zeta3 nodes (d rows) - E28, E29, E30, E31
    row_block5 = np.arange(7*d, 8*d)
    row_block5_local = np.arange(d)
    
    # E28: columns 17d to 19d-1
    col_base_e28 = 17 * d
    data.extend([-1] * d + [1] * d)
    rows.extend(np.concatenate([row_block5, row_block5]))
    cols.extend(np.concatenate([col_base_e28 + 2*row_block5_local,
                               col_base_e28 + 2*row_block5_local + 1]))
    
    # E29: columns 19d to 21d-1
    col_base_e29 = 19 * d
    data.extend([-1] * d + [1] * d)
    rows.extend(np.concatenate([row_block5, row_block5]))
    cols.extend(np.concatenate([col_base_e29 + 2*row_block5_local,
                               col_base_e29 + 2*row_block5_local + 1]))
    
    # E30: columns 21d to 23d-1
    col_base_e30 = 21 * d
    data.extend([-1] * d + [1] * d)
    rows.extend(np.concatenate([row_block5, row_block5]))
    cols.extend(np.concatenate([col_base_e30 + 2*row_block5_local,
                               col_base_e30 + 2*row_block5_local + 1]))
    
    # E31: columns 23d to 25d-1
    col_base_e31 = 23 * d
    data.extend([-1] * d + [1] * d)
    rows.extend(np.concatenate([row_block5, row_block5]))
    cols.extend(np.concatenate([col_base_e31 + 2*row_block5_local,
                               col_base_e31 + 2*row_block5_local + 1]))
    
    # BLOCK 6: xj identity for E32 (2d rows)
    row_block6 = np.arange(8*d, 10*d)
    col_base_e32 = 25 * d
    col_e32 = col_base_e32 + (row_block6 - 8*d)
    
    data.extend([1] * (2*d))
    rows.extend(row_block6)
    cols.extend(col_e32)
    
    # Convert to arrays for efficiency
    data = np.array(data, dtype=np.float64)
    rows = np.array(rows, dtype=np.int32)
    cols = np.array(cols, dtype=np.int32)
    
    # Create sparse matrix
    W8 = csr_matrix((data, (rows, cols)), shape=(total_rows, total_cols))
    
    return W8





def build_B8(d):
    rows_total = 10 * d
    B8 = np.zeros(rows_total)
    
    idx = 0
    # First section: d rows with value 2
    B8[idx:idx+d] = 2
    idx += d
    
    # Second section: 2d rows with value 0 (skip)
    idx += 2*d
    
    # Third section: d rows with value 3  
    B8[idx:idx+d] = 3
    idx += d
    
    # Fourth section: 3d rows with value 0 (skip)
    idx += 3*d
    
    # Fifth section: d rows with value 3
    B8[idx:idx+d] = 3
    idx += d
    
    # Sixth section: 2d rows with value 0 (skip)
    idx += 2*d
    
    return B8