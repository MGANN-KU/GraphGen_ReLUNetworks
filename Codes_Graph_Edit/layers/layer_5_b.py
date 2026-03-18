# -*- coding: utf-8 -*-
"""
Weights and bias of layer 5

@author: Ghafoor
"""


import numpy as np
from scipy.sparse import lil_matrix, csr_matrix

def build_W5(d, C):
    # Calculate dimensions properly
    cols_total = (d * d * 2) + (d * d * 2) + (d) + (d) + (d * 2) + (d * 2) + (d * d * 2) + (d * d * 2) + (d * d * 2) + (d * d * 2) + (3 * d) + (d * d * 2) + (d * d * 2) + (d * d * 2) + (d * d * 2) + (2 * d)
    rows_total = (2 * d) + (d * d) + (3 * d) + (d * d) + (2 * d)
    
    # print(f"W5 dimensions: {rows_total} × {cols_total}")
    
    W5 = lil_matrix((rows_total, cols_total))
    row_idx = 0
    
    # Calculate all base offsets
    base_A1 = 0
    base_A2 = d * d * 2
    base_A3 = base_A2 + d * d * 2
    base_A4 = base_A3 + d
    base_A5 = base_A4 + d
    base_A6 = base_A5 + d * 2
    base_A7 = base_A6 + d * 2
    base_A8 = base_A7 + d * d * 2
    base_A9 = base_A8 + d * d * 2
    base_A10 = base_A9 + d * d * 2
    base_A11 = base_A10 + d * d * 2
    base_A12 = base_A11 + 3 * d
    base_A13 = base_A12 + d * d * 2
    base_A14 = base_A13 + d * d * 2
    base_A15 = base_A14 + d * d * 2
    base_A16 = base_A15 + d * d * 2
    
    # First 2d rows (A1-A16 and A17-A32)
    for section in range(2):
        for j in range(1, d+1):
            # A1/A17: psi nodes
            for k in range(1, d+1):
                for l in range(1, d+1):
                    for q in range(2):
                        if section == 0 and l == j and k < l:
                            col_idx = base_A1 + (k-1) * (d * 2) + (l-1) * 2 + q
                            if col_idx < cols_total:
                                if q == 0:
                                    W5[row_idx, col_idx] = -C
                                else:
                                    W5[row_idx, col_idx] = C
            
            # A2/A18: rho nodes
            for k in range(1, d+1):
                for l in range(1, d+1):
                    for q in range(2):
                        if section == 0 and l == j and k < l:
                            col_idx = base_A2 + (k-1) * (d * 2) + (l-1) * 2 + q
                            if col_idx < cols_total:
                                if q == 0:
                                    W5[row_idx, col_idx] = -C
                                else:
                                    W5[row_idx, col_idx] = C
            
            # A3/A19: x_j nodes
            for k in range(1, d+1):
                if section == 0 and k == j:
                    col_idx = base_A3 + (k-1)
                    if col_idx < cols_total:
                        W5[row_idx, col_idx] = 1
            
            # A4/A20: x_j nodes  
            for k in range(1, d+1):
                if section == 1 and j == k:
                    col_idx = base_A4 + (k-1)
                    if col_idx < cols_total:
                        W5[row_idx, col_idx] = 1
            
            row_idx += 1
    
    # Next d*d rows (A49-A64: psi1 nodes)
    for i in range(1, d+1):
        for j in range(1, d+1):
            # FIXED: A53: eta1, eta2 nodes - condition should be if l == i (outer loop)
            for l in range(1, d+1):
                for q in range(2):
                    if l == i:  # CHANGED: l == i, not l == j
                        # For eta1,eta2 (A53)
                        col_idx = base_A5 + (l-1) * 2 + q
                        if col_idx < cols_total:
                            if q == 0:
                                W5[row_idx, col_idx] = -1
                            else:
                                W5[row_idx, col_idx] = 1
                        
                        # For eta3,eta4 (A54)
                        col_idx = base_A6 + (l-1) * 2 + q
                        if col_idx < cols_total:
                            if q == 0:
                                W5[row_idx, col_idx] = -1
                            else:
                                W5[row_idx, col_idx] = 1
            
        
            # A55: alpha1, alpha2 nodes
            for p in range(1, d+1):
                for k in range(1, d+1):
                    for q in range(2):
                        if i == p and j == k:
                            col_idx = base_A7 + (p-1) * (d * 2) + (k-1) * 2 + q
                            if col_idx < cols_total:
                                if q == 0:
                                    W5[row_idx, col_idx] = 1
                                else:
                                    W5[row_idx, col_idx] = -1
            
            # A56: beta1, beta2 nodes
            for p in range(1, d+1):
                for k in range(1, d+1):
                    for q in range(2):
                        if i == p and j == k:
                            col_idx = base_A8 + (p-1) * (d * 2) + (k-1) * 2 + q
                            if col_idx < cols_total:
                                if q == 0:
                                    W5[row_idx, col_idx] = 1
                                else:
                                    W5[row_idx, col_idx] = -1
            
            # A57: alpha'1, alpha'2 nodes
            for p in range(d+1, 2*d+1):
                for k in range(d+1, 2*d+1):
                    for q in range(2):
                        if i+d == p and j+d == k:
                            col_idx = base_A9 + ((p-d-1) * (d * 2) + (k-d-1) * 2 + q)
                            if col_idx < cols_total:
                                if q == 0:
                                    W5[row_idx, col_idx] = 1
                                else:
                                    W5[row_idx, col_idx] = -1
            
            # A58: beta'1, beta'2 nodes
            for p in range(d+1, 2*d+1):
                for k in range(d+1, 2*d+1):
                    for q in range(2):
                        if i+d == p and j+d == k:
                            col_idx = base_A10 + ((p-d-1) * (d * 2) + (k-d-1) * 2 + q)
                            if col_idx < cols_total:
                                if q == 0:
                                    W5[row_idx, col_idx] = 1
                                else:
                                    W5[row_idx, col_idx] = -1
            
            row_idx += 1
    
    # Next 3*d rows (A65-A80: identity map)
    for l in range(1, 3*d+1):
        # A75: xj identity
        col_idx = base_A11 + (l-1)
        if col_idx < cols_total:
            W5[row_idx, col_idx] = 1
        row_idx += 1
    
    # Next d*d rows (A97-A112: psi'1 nodes)
    for i in range(1, d+1):
        for j in range(1, d+1):
            # A108: deletion nodes
            for l_val in range(1, d+1):
                for k in range(1, d+1):
                    for q in range(2):
                        if i == l_val and j == k:
                            col_idx = base_A12 + (l_val-1) * (d * 2) + (k-1) * 2 + q
                            if col_idx < cols_total:
                                if q == 0:
                                    W5[row_idx, col_idx] = 1
                                else:
                                    W5[row_idx, col_idx] = -1
            
            # A109: beta' nodes
            for l_val in range(1, d+1):
                for k in range(1, d+1):
                    for q in range(2):
                        if i == l_val and j == k:
                            col_idx = base_A13 + (l_val-1) * (d * 2) + (k-1) * 2 + q
                            if col_idx < cols_total:
                                if q == 0:
                                    W5[row_idx, col_idx] = 1
                                else:
                                    W5[row_idx, col_idx] = -1
            
            # A110: alpha'' nodes (first set)
            for l_val in range(d+1, 2*d+1):
                for k in range(d+1, 2*d+1):
                    for q in range(2):
                        if i+d == l_val and j+d == k:
                            col_idx = base_A14 + ((l_val-d-1) * (d * 2) + (k-d-1) * 2 + q)
                            if col_idx < cols_total:
                                if q == 0:
                                    W5[row_idx, col_idx] = 1
                                else:
                                    W5[row_idx, col_idx] = -1
            
            # A111: alpha'' nodes (second set) - FIXED: Need separate loop for A111
            for l_val in range(d+1, 2*d+1):
                for k in range(d+1, 2*d+1):
                    for q in range(2):
                        if i+d == l_val and j+d == k:
                            col_idx = base_A15 + ((l_val-d-1) * (d * 2) + (k-d-1) * 2 + q)
                            if col_idx < cols_total:
                                if q == 0:
                                    W5[row_idx, col_idx] = 1
                                else:
                                    W5[row_idx, col_idx] = -1
            
            row_idx += 1
    
    # Last 2*d rows (A113-A128: x_j identity map)
    for i in range(1, 2*d+1):
        # A128: x'_j identity
        col_idx = base_A16 + (i-1)
        if col_idx < cols_total:
            W5[row_idx, col_idx] = 1
        row_idx += 1
    
    return W5.tocsr()
    
    
    
def build_B5(d, C):
    total_length = d + d + (d * d) + (3 * d) + (d * d) + (2 * d)
    
    B5 = np.zeros(total_length)
    idx = 0
    
    # bias of x' nodes
    for j in range(1, d+1):
        B5[idx] = C * (j - 1)
        idx += 1
    
    # bias of x_j+d nodes (zeros)
    idx += d
    
    # bias for psi1 nodes
    for l in range(1, d+1):
        for j in range(1, d+1):
            B5[idx] = -2
            idx += 1
    
    # bias for xj nodes (zeros)
    idx += 3 * d
    
    # bias for psi'1 nodes
    for l in range(1, d+1):
        for j in range(1, d+1):
            B5[idx] = -3
            idx += 1
    
    # bias for xj nodes (zeros)
    idx += 2 * d
    
    return B5