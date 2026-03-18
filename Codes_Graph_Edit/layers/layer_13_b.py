# -*- coding: utf-8 -*-
"""
Weights and bias of layer 13
modified to 13_b
@author: Ghafoor
"""
import numpy as np
from scipy.sparse import coo_matrix

import numpy as np
from scipy.sparse import csr_matrix

import numpy as np
from scipy.sparse import csr_matrix

def build_W13(n, d, eps):
    """
    Vectorized implementation of W13
    For n=4, d=2: Returns (84, 14) sparse matrix
    """
    m = n + 2 * d
    total_cols = 7 * d  # E15(2d) + E16(d) + E18(2d) + E19(2d) = 7d
    
    # Calculate total rows
    # 1. alpha nodes: m × d × 2
    # 2. beta nodes: m × d × 2
    # 3. bottom: d
    # 4. zeta51: d × 2
    # 5. zeta61: d × 2  
    # 6. x_j identity (2d+1..3d): d
    # 7. x_j+2d identity (3d+1..5d): 2d
    # 8. deletion nodes (5d+1..7d): 2d
    
    total_rows = (2 * m * d * 2) + d + (2 * d * 2) + d + 2*d + 2*d
    # For n=4, d=2, m=8: 
    # 2*8*2*2=64 + 2 + 8 + 2 + 4 + 4 = 84 ✓
    
    # Initialize arrays
    data = []
    rows = []
    cols = []
    
    row_idx = 0
    
    # SECTION 1: alpha nodes (substitution)
    # For each l=1..m, k=1..d, q=0,1: set 1/eps at column k
    for l in range(m):
        for k in range(d):
            for q in range(2):
                # In dense: column = k (1-based indexing)
                # In sparse: column = k (0-based indexing)
                data.append(1.0/eps)
                rows.append(row_idx)
                cols.append(k)  # Just column k, NOT 2*k or similar
                row_idx += 1
    
    # SECTION 2: beta nodes (substitution)
    # Same pattern but -1/eps
    for l in range(m):
        for k in range(d):
            for q in range(2):
                data.append(-1.0/eps)
                rows.append(row_idx)
                cols.append(k)
                row_idx += 1
    
    # SECTION 3: bottom nodes (k=d+1..2d)
    # Set 1 at column k
    for k in range(d, 2*d):  # k from d to 2d-1
        data.append(1.0)
        rows.append(row_idx)
        cols.append(k)
        row_idx += 1
    
    # SECTION 4: zeta51, zeta52 nodes
    # Important: This section uses E15, E16, E18, E19 concatenated
    # Total columns: E15(2d) + E16(d) + E18(2d) + E19(2d) = 7d
    
    # E16: 1/eps at position (j + 2d) where j=1..d
    # Since E15 has 2d columns, E16 starts at column 2d
    for j in range(d):  # j from 0 to d-1
        for q in range(2):
            # Column position in concatenated row:
            # E15 occupies cols 0..2d-1 (all zeros)
            # E16 occupies cols 2d..3d-1
            # So for j: column = 2d + j
            data.append(1.0/eps)
            rows.append(row_idx)
            cols.append(2*d + j)
            row_idx += 1
    
    # SECTION 5: zeta61, zeta62 nodes
    # Same as section 4 but -1/eps
    for j in range(d):
        for q in range(2):
            data.append(-1.0/eps)
            rows.append(row_idx)
            cols.append(2*d + j)
            row_idx += 1
    
    # SECTION 6: x_j identity (2d+1..3d)
    # E16: identity at position j for j=2d+1..3d
    # Column = j (since E15 has 2d zeros before it)
    for j in range(2*d, 3*d):  # j from 2d to 3d-1
        data.append(1.0)
        rows.append(row_idx)
        cols.append(j)  # Column j in the full concatenated row
        row_idx += 1
    
    # SECTION 7: x_j+2d identity (3d+1..5d)
    # E18: identity at position j for j=3d+1..5d
    # E15(2d) + E16(d) = 3d columns before E18
    for j in range(3*d, 5*d):  # j from 3d to 5d-1
        data.append(1.0)
        rows.append(row_idx)
        cols.append(j)  # Column j in the full concatenated row
        row_idx += 1
    
    # SECTION 8: deletion nodes (5d+1..7d)
    # G90: identity at position k for k=5d+1..7d
    # G87(2d) + G88(d) + G89(2d) = 5d columns before G90
    for k in range(5*d, 7*d):  # k from 5d to 7d-1
        data.append(1.0)
        rows.append(row_idx)
        cols.append(k)  # Column k in the full concatenated row
        row_idx += 1
    
    # Convert to numpy arrays
    data = np.array(data, dtype=np.float64)
    rows = np.array(rows, dtype=np.int32)
    cols = np.array(cols, dtype=np.int32)
    
    # Create sparse matrix
    W13 = csr_matrix((data, (rows, cols)), shape=(total_rows, total_cols))
    
    return W13



import numpy as np

def build_B13(n, d, eps):
    """
    Vectorized implementation of B13 (bias vector)
    For n=4, d=2: Returns (84,) 1D array
    """
    m = n + 2 * d
    
    # Calculate total rows (same as W13)
    # 1. alpha nodes: m × d × 2
    # 2. beta nodes: m × d × 2
    # 3. bottom: d
    # 4. zeta51: d × 2
    # 5. zeta61: d × 2  
    # 6. x_j identity (2d+1..3d): d
    # 7. x_j+2d identity (3d+1..5d): 2d
    # 8. deletion nodes (5d+1..7d): 2d
    
    total_rows = (2 * m * d * 2) + d + (2 * d * 2) + d + 2*d + 2*d
    # For n=4, d=2, m=8: total_rows = 84
    
    # Initialize bias vector
    B13 = np.zeros(total_rows)
    
    row_idx = 0
    
    # SECTION 1: alpha nodes
    # For l=1..m, k=1..d, q=0,1:
    # if q==0: b = -l/eps + 1
    # if q==1: b = -l/eps
    for l in range(1, m+1):  # 1-based l
        for k in range(d):
            for q in range(2):
                if q == 0:
                    B13[row_idx] = -l/eps + 1
                else:  # q == 1
                    B13[row_idx] = -l/eps
                row_idx += 1
    
    # SECTION 2: beta nodes
    # For l=1..m, k=1..d, q=0,1:
    # if q==0: b = l/eps + 1
    # if q==1: b = l/eps
    for l in range(1, m+1):
        for k in range(d):
            for q in range(2):
                if q == 0:
                    B13[row_idx] = l/eps + 1
                else:  # q == 1
                    B13[row_idx] = l/eps
                row_idx += 1
    
    # SECTION 3: xj nodes (d+1..2d) - all zeros
    # Skipping zeros since array is already initialized to zeros
    row_idx += d
    
    # SECTION 4: insertion zeta51 nodes
    # For j=1..d, q=0,1:
    # if q==0: b = 1
    # if q==1: b = 0
    for j in range(d):
        for q in range(2):
            if q == 0:
                B13[row_idx] = 1
            # else: q==1 is already 0
            row_idx += 1
    
    # SECTION 5: insertion zeta61 nodes (same as section 4)
    for j in range(d):
        for q in range(2):
            if q == 0:
                B13[row_idx] = 1
            row_idx += 1
    
    # SECTIONS 6-8: all zeros
    # x_j identity (2d+1..3d): d rows
    # x_j+2d identity (3d+1..5d): 2d rows
    # deletion nodes (5d+1..7d): 2d rows
    # Total: d + 2d + 2d = 5d rows of zeros
    row_idx += 5*d
    
    return B13  # Return as 1D array (not column vector)


