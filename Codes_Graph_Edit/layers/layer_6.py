# -*- coding: utf-8 -*-
"""
Weights and bias of layer 6

@author: Ghafoor
"""


import numpy as np
from scipy.sparse import lil_matrix, csr_matrix

def build_W6(d, C):
    # Calculate dimensions correctly
    cols_total = (2*d) + (d*d) + (3*d) + (d*d) + (2*d)  # 22 for d=2
    rows_total = (2*d) + (d) + (2*d) + (d) + (d)        # 14 for d=2
    
    # print(f"W6 dimensions: {rows_total} × {cols_total}")
    
    W6 = lil_matrix((rows_total, cols_total))
    row_idx = 0
    
    
    # Substitution nodes (2d rows)
    for l in range(1, 2*d+1):
        # D1: identity mapping
        for j in range(1, 2*d+1):
            if l == j:
                W6[row_idx, j-1] = 1
        
        # D3: zeros (d*d columns)
        # D4: zeros (3*d columns)
        # D6: zeros (d*d columns) 
        # D7: zeros (2*d columns)
        
        row_idx += 1
    
    # Insertion nodes - tau1 (d rows)
    for l in range(1, d+1):
        # D8: zeros (2*d columns)
        
        # D10: psi1 weights
        base1 = 2*d
        for j in range(1, d+1):
            for k in range(1, d+1):
                if j == l and k < j:
                    col_idx = base1 + (j-1) * d + (k-1)
                    W6[row_idx, col_idx] = -C
        
        # D11: xj identity
        base2 = 2*d + d*d
        for j in range(1, 3*d+1):
            if l == j:
                W6[row_idx, base2 + (j-1)] = 1
        
        # D13: zeros (d*d columns)
        # D14: zeros (2*d columns)
        
        row_idx += 1
    
    # x_{j+d} and x_{j+2d} (2d rows)
    for k in range(d+1, 3*d+1):
        # D29: zeros (2*d columns)
        
        # D31: zeros (d*d columns)
        
        # D32: xj identity
        base3 = 2*d + d*d
        for j in range(1, 3*d+1):
            if j == k:
                W6[row_idx, base3 + (j-1)] = 1
        
        # D34: zeros (d*d columns)
        # D35: zeros (2*d columns)
        
        row_idx += 1
    
    # Deletion portion - s_j (d rows)
    for l in range(1, d+1):
        # D29: zeros (2*d columns)
        
        # D31: zeros (d*d columns)
        
        # D32: zeros (3*d columns)
        
        # D34: deletion weights
        base4 = 2*d + d*d + 3*d
        for j in range(1, d+1):
            for k in range(1, d+1):
                if j == l and k < l:
                    col_idx = base4 + (j-1) * d + (k-1)
                    W6[row_idx, col_idx] = -C
        
        # D35: identity mapping
        base5 = 2*d + d*d + 3*d + d*d
        for j in range(1, 2*d+1):
            if j == l:
                W6[row_idx, base5 + (j-1)] = 1
        
        row_idx += 1
    
    # Deletion portion - x_j+d (d rows)
    for l in range(1, d+1):
        # D29: zeros (2*d columns)
        
        # D31: zeros (d*d columns)
        
        # D32: zeros (3*d columns)
        
        # D34: zeros (d*d columns)
        
        # D35: identity mapping for x_j+d
        base6 = 2*d + d*d + 3*d + d*d
        for j in range(1, 2*d+1):
            if j == l + d:
                W6[row_idx, base6 + (j-1)] = 1
        
        row_idx += 1
    
    return W6.tocsr()

def build_B6(d):
    # Calculate total length
    total_length = (2*d) + (d) + (2*d) + (2*d)
    
    B6 = np.zeros(total_length)
    idx = 0
    
    # First 2d rows - zeros
    idx += 2*d
    
    # Next d rows (tau1) - zeros  
    idx += d
    
    # Next 2d rows (x_j+d) - zeros
    idx += 2*d
    
    # Last 2d rows (deletion) - zeros
    idx += 2*d
    
    return B6