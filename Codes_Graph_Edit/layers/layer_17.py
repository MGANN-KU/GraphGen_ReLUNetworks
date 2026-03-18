# -*- coding: utf-8 -*-
"""
Weights and bias of layer 17

@author: Ghafoor
"""

import numpy as np
from scipy import sparse

def build_W17(n, d, C):
    """
    Build W17 weight matrix efficiently using sparse matrices
    """
    # Calculate dimensions correctly
    identity_dim = n + 2 * d
    i16_dim = 2 * d  # eta1, eta2 nodes (2 per d)
    i7_dim = 2 * d   # eta3, eta4 nodes (2 per d)
    i8_dim = 5 * d   # I8 dimension
    
    total_cols = identity_dim + i16_dim + i7_dim + i8_dim
    total_rows = identity_dim + d + d + d + 2*d + 2*d + 1  # 6 sections
    
    # Use COO format for efficient construction
    data = []
    rows = []
    cols = []
    
    current_row = 0
    
    # Part 1: Identity mapping (first n+2*d rows)
    for i in range(identity_dim):
        data.append(1.0)
        rows.append(current_row + i)
        cols.append(i)
    
    current_row += identity_dim
    
    # Define block offsets
    i16_offset = identity_dim
    i7_offset = i16_offset + i16_dim
    i8_offset = i7_offset + i7_dim
    
    # Part 2: tau2 for g1_j (d rows)
    for j in range(d):
        # I16 part: ±C on diagonal
        data.extend([C, -C])
        rows.extend([current_row + j, current_row + j])
        cols.extend([i16_offset + 2*j, i16_offset + 2*j + 1])
        
        # I7 part: ±C on diagonal  
        data.extend([C, -C])
        rows.extend([current_row + j, current_row + j])
        cols.extend([i7_offset + 2*j, i7_offset + 2*j + 1])
        
        # I8 part: single 1 at position j+2d
        data.append(1.0)
        rows.append(current_row + j)
        cols.append(i8_offset + j + 2*d)
    
    current_row += d
    
    # Part 3: tau3 for g2_j (d rows)
    for j in range(d):
        # I16 part: ∓C on diagonal (reversed signs)
        data.extend([-C, C])
        rows.extend([current_row + j, current_row + j])
        cols.extend([i16_offset + 2*j, i16_offset + 2*j + 1])
        
        # I7 part: ∓C on diagonal
        data.extend([-C, C])
        rows.extend([current_row + j, current_row + j])
        cols.extend([i7_offset + 2*j, i7_offset + 2*j + 1])
    
    current_row += d
    
    # Part 4: eta5 for delta(xj, x{j+d}) (d rows)
    for j in range(d):
        # I16 part: ±1 on diagonal
        data.extend([1.0, -1.0])
        rows.extend([current_row + j, current_row + j])
        cols.extend([i16_offset + 2*j, i16_offset + 2*j + 1])
        
        # I7 part: ±1 on diagonal
        data.extend([1.0, -1.0])
        rows.extend([current_row + j, current_row + j])
        cols.extend([i7_offset + 2*j, i7_offset + 2*j + 1])
    
    current_row += d
    
    # Part 5: xj for insertion (2*d rows)
    for i in range(2*d):
        data.append(1.0)
        rows.append(current_row + i)
        cols.append(i8_offset + i)
    
    current_row += 2*d
    
    # Part 6: xj for deletion (2*d rows)
    for i in range(2*d):
        data.append(1.0)
        rows.append(current_row + i)
        cols.append(i8_offset + i + 3*d)
    
    current_row += 2*d
    
    # Part 7: d' = summation delta(xj, x{j+d}) (1 row)
    for k in range(d):
        data.extend([1.0, -1.0])
        rows.extend([current_row, current_row])
        cols.extend([i16_offset + 2*k, i16_offset + 2*k + 1])
        
        data.extend([1.0, -1.0])
        rows.extend([current_row, current_row])
        cols.extend([i7_offset + 2*k, i7_offset + 2*k + 1])
    
    # Verify dimensions before creating matrix
    max_row = max(rows) if rows else 0
    max_col = max(cols) if cols else 0
    
    if max_row >= total_rows or max_col >= total_cols:
        print(f"Dimension mismatch detected:")
        print(f"Max row index: {max_row}, Total rows: {total_rows}")
        print(f"Max col index: {max_col}, Total cols: {total_cols}")
        print(f"Adjusting dimensions...")
        total_rows = max_row + 1
        total_cols = max_col + 1
    
    # Create sparse matrix
    W17 = sparse.coo_matrix((data, (rows, cols)), shape=(total_rows, total_cols)).tocsr()
    
    return W17




def build_B17(n, d, C, B):
    """
    Build B17 bias vector efficiently
    """
    # Calculate total length
    total_length = (n + 2*d) + d + d + d + 2*d + 2*d + 1
    
    # Initialize bias vector with zeros
    B17 = np.zeros(total_length)
    
    current_idx = 0
    
    # Part 1: subs nodes (n+2*d rows) - all zeros
    current_idx += n + 2*d
    
    # Part 2: tau2 for g1_j (d rows) - bias = -2*C
    B17[current_idx:current_idx + d] = -2 * C
    current_idx += d
    
    # Part 3: tau3 for g2_j (d rows) - bias = B + C
    B17[current_idx:current_idx + d] = B + C
    current_idx += d
    
    # Part 4: eta5 for delta(xj, x{j+d}) (d rows) - bias = -1
    B17[current_idx:current_idx + d] = -1
    current_idx += d
    
    # Part 5: insertion (2*d rows) - all zeros
    current_idx += 2*d
    
    # Part 6: deletion (2*d rows) - all zeros
    current_idx += 2*d
    
    # Part 7: d' (1 row) - bias = -d
    B17[current_idx] = -d
    
    return B17

