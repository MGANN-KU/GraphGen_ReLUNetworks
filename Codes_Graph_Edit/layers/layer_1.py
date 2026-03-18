# -*- coding: utf-8 -*-
"""
Weights and bias of layer 1

@author: Ghafoor
"""

import numpy as np
from scipy import sparse




def build_W1(n, d, m, eps):
    """
    Build W1 weight matrix efficiently using sparse matrices
    """
    input_dim = 7 * d
    
    # Use COO format
    data = []
    rows = []
    cols = []
    
    current_row = 0
    
    def add_weight(j, value):
        data.append(value)
        rows.append(current_row)
        cols.append(j-1)
    
    # === SECTIONS 1-4: Gamma 11,12,21,22,31,32,41,42 ===
    # Each: 3 ranges × (n+1) × 2
    # Range 1: d × (n+1) × 2
    # Range 2: 4d × (n+1) × 2  
    # Range 3: 2d × (n+1) × 2
    # Total per section: (d + 4d + 2d) × (n+1) × 2 = 7d × (n+1) × 2
    sections_1_4 = [1/eps, -1/eps, 1/eps, -1/eps]
    
    for section_value in sections_1_4:
        # Range 1: 1<=j<=d
        for j in range(1, d+1):
            for i in range(n + 1):
                for q in range(2):
                    add_weight(j, section_value)
                    current_row += 1
        
        # Range 2: d+1<=j<=5d (all zeros)
        for j in range(d+1, 5*d+1):
            for i in range(n + 1):
                for q in range(2):
                    current_row += 1
        
        # Range 3: 5d+1<=j<=7d
        for j in range(5*d+1, 7*d+1):
            for i in range(n + 1):
                for q in range(2):
                    add_weight(j, section_value)
                    current_row += 1
    
    # print(f"After sections 1-4: {current_row} rows")
    
    # === SECTIONS 5-8: Gamma' 11,12,21,22,31,32,41,42 ===
    # Each: 3 ranges × (n+d) × 2
    # Range 1: 2d × (n+d) × 2
    # Range 2: 2d × (n+d) × 2
    # Range 3: 3d × (n+d) × 2
    # Total per section: (2d + 2d + 3d) × (n+d) × 2 = 7d × (n+d) × 2
    sections_5_8 = [1/eps, -1/eps, 1/eps, -1/eps]
    
    for section_value in sections_5_8:
        # Range 1: 1<=j<=2d (all zeros)
        for j in range(1, 2*d+1):
            for i in range(n + d):
                for q in range(2):
                    current_row += 1
        
        # Range 2: 2d+1<=j<=4d
        for j in range(2*d+1, 4*d+1):
            for i in range(n + d):
                for q in range(2):
                    add_weight(j, section_value)
                    current_row += 1
        
        # Range 3: 4d+1<=j<=7d (all zeros)
        for j in range(4*d+1, 7*d+1):
            for i in range(n + d):
                for q in range(2):
                    current_row += 1
    
    # print(f"After sections 5-8: {current_row} rows")
    
    # === SECTIONS 9-10: Gamma 51,52,61,62 ===
    # Each: 5 ranges × m × 2
    # Range 1: d × m × 2
    # Range 2: d × m × 2  
    # Range 3: 2d × m × 2
    # Range 4: d × m × 2
    # Range 5: 2d × m × 2
    # Total per section: (d + d + 2d + d + 2d) × m × 2 = 7d × m × 2
    sections_9_10 = [1/eps, -1/eps]
    
    for section_value in sections_9_10:
        # Range 1: 1<=j<=d (all zeros)
        for j in range(1, d+1):
            for l in range(1, m+1):
                for q in range(2):
                    current_row += 1
        
        # Range 2: d+1<=j<=2d
        for j in range(d+1, 2*d+1):
            for l in range(1, m+1):
                for q in range(2):
                    add_weight(j, section_value)
                    current_row += 1
        
        # Range 3: 2d+1<=j<=4d (all zeros)
        for j in range(2*d+1, 4*d+1):
            for l in range(1, m+1):
                for q in range(2):
                    current_row += 1
        
        # Range 4: 4d+1<=j<=5d
        for j in range(4*d+1, 5*d+1):
            for l in range(1, m+1):
                for q in range(2):
                    add_weight(j, section_value)
                    current_row += 1
        
        # Range 5: 5d+1<=j<=7d (all zeros)
        for j in range(5*d+1, 7*d+1):
            for l in range(1, m+1):
                for q in range(2):
                    current_row += 1
    
    # print(f"After sections 9-10: {current_row} rows")
    
    # === SECTIONS 11-12: Gamma 71,72,81,82 ===
    # Each: 5 ranges × m × 2
    # Same structure as sections 9-10
    sections_11_12 = [1/eps, -1/eps]
    
    for section_value in sections_11_12:
        # Range 1: 1<=j<=d (all zeros)
        for j in range(1, d+1):
            for l in range(1, m+1):
                for q in range(2):
                    current_row += 1
        
        # Range 2: d+1<=j<=2d (only when l != 1)
        for j in range(d+1, 2*d+1):
            for l in range(1, m+1):
                for q in range(2):
                    if l != 1:
                        add_weight(j, section_value)
                    current_row += 1
        
        # Range 3: 2d+1<=j<=4d (all zeros)
        for j in range(2*d+1, 4*d+1):
            for l in range(1, m+1):
                for q in range(2):
                    current_row += 1
        
        # Range 4: 4d+1<=j<=5d (only when l != 1)
        for j in range(4*d+1, 5*d+1):
            for l in range(1, m+1):
                for q in range(2):
                    if l != 1:
                        add_weight(j, section_value)
                    current_row += 1
        
        # Range 5: 5d+1<=j<=7d (all zeros)
        for j in range(5*d+1, 7*d+1):
            for l in range(1, m+1):
                for q in range(2):
                    current_row += 1
    
    # print(f"Final: {current_row} rows")
    
    # Calculate expected total
    expected_total = (4 * 7*d * (n+1) * 2 +      # Sections 1-4
                     4 * 7*d * (n+d) * 2 +       # Sections 5-8  
                     2 * 7*d * m * 2 +           # Sections 9-10
                     2 * 7*d * m * 2)            # Sections 11-12
    
    # print(f"Expected: {expected_total} rows")
    
    W1 = sparse.coo_matrix((data, (rows, cols)), shape=(current_row, input_dim))
    return W1.tocsr()






def build_B1(n, d, m, eps):
    """
    Build B1 bias vector efficiently - CORRECTED VERSION
    """
    # Correct total length calculation (matches W1)
    total_length = (4 * 7*d * (n+1) * 2 +      # Sections 1-4
                   4 * 7*d * (n+d) * 2 +       # Sections 5-8  
                   2 * 7*d * m * 2 +           # Sections 9-10
                   2 * 7*d * m * 2)            # Sections 11-12
    
    # print(f"B1 total length: {total_length}")
    
    # Initialize bias vector with zeros
    B1 = np.zeros(total_length)
    
    current_idx = 0
    
    # Helper functions for bias calculations
    def calc_bias(q, i_val, denom):
        if q == 0:
            return (-(i_val - 1) / (eps * denom)) + 1
        else:
            return -(i_val - 1) / (eps * denom)
    
    def calc_bias_reverse(q, i_val, denom):
        if q == 0:
            return (i_val / (eps * denom)) + 1
        else:
            return i_val / (eps * denom)
    
    # === SECTIONS 1-4: Gamma 11,12,21,22,31,32,41,42 ===
    bias_funcs_1_4 = [calc_bias, calc_bias_reverse, calc_bias, 
                      lambda q, i, denom: calc_bias_reverse(q, i-1, denom)]
    denoms_1_4 = [n, n, n, n]
    
    for bias_func, denom in zip(bias_funcs_1_4, denoms_1_4):
        # Range 1: 1<=j<=d
        for j in range(1, d+1):
            for i in range(n + 1):
                for q in range(2):
                    B1[current_idx] = bias_func(q, i, denom)
                    current_idx += 1
        
        # Range 2: d+1<=j<=5d (all zeros - already initialized to 0)
        for j in range(d+1, 5*d+1):
            for i in range(n + 1):
                for q in range(2):
                    current_idx += 1  # Just advance index
        
        # Range 3: 5d+1<=j<=7d
        for j in range(5*d+1, 7*d+1):
            for i in range(n + 1):
                for q in range(2):
                    B1[current_idx] = bias_func(q, i, denom)
                    current_idx += 1
    
    # print(f"After sections 1-4: {current_idx} biases set")
    
    # === SECTIONS 5-8: Gamma' 11,12,21,22,31,32,41,42 ===
    bias_funcs_5_8 = [calc_bias, calc_bias_reverse, calc_bias,
                      lambda q, i, denom: calc_bias_reverse(q, i-1, denom)]
    denoms_5_8 = [n+d-1, n+d-1, n+d-1, n+d-1]
    
    for bias_func, denom in zip(bias_funcs_5_8, denoms_5_8):
        # Range 1: 1<=j<=2d (all zeros)
        for j in range(1, 2*d+1):
            for i in range(n + d):
                for q in range(2):
                    current_idx += 1
        
        # Range 2: 2d+1<=j<=4d
        for j in range(2*d+1, 4*d+1):
            for i in range(n + d):
                for q in range(2):
                    B1[current_idx] = bias_func(q, i, denom)
                    current_idx += 1
        
        # Range 3: 4d+1<=j<=7d (all zeros)
        for j in range(4*d+1, 7*d+1):
            for i in range(n + d):
                for q in range(2):
                    current_idx += 1
    
    # print(f"After sections 5-8: {current_idx} biases set")
    
    # === SECTIONS 9-10: Gamma 51,52,61,62 ===
    bias_funcs_9_10 = [calc_bias, calc_bias_reverse]
    denoms_9_10 = [m, m]
    
    for bias_func, denom in zip(bias_funcs_9_10, denoms_9_10):
        # Range 1: 1<=j<=d (all zeros)
        for j in range(1, d+1):
            for l in range(1, m+1):
                for q in range(2):
                    current_idx += 1
        
        # Range 2: d+1<=j<=2d
        for j in range(d+1, 2*d+1):
            for l in range(1, m+1):
                for q in range(2):
                    B1[current_idx] = bias_func(q, l, denom)
                    current_idx += 1
        
        # Range 3: 2d+1<=j<=4d (all zeros)
        for j in range(2*d+1, 4*d+1):
            for l in range(1, m+1):
                for q in range(2):
                    current_idx += 1
        
        # Range 4: 4d+1<=j<=5d
        for j in range(4*d+1, 5*d+1):
            for l in range(1, m+1):
                for q in range(2):
                    B1[current_idx] = bias_func(q, l, denom)
                    current_idx += 1
        
        # Range 5: 5d+1<=j<=7d (all zeros)
        for j in range(5*d+1, 7*d+1):
            for l in range(1, m+1):
                for q in range(2):
                    current_idx += 1
    
    # print(f"After sections 9-10: {current_idx} biases set")
    
    # === SECTIONS 11-12: Gamma 71,72,81,82 ===
    bias_funcs_11_12 = [calc_bias, 
                       lambda q, i, denom: calc_bias_reverse(q, i-1, denom)]
    denoms_11_12 = [m, m]
    
    for bias_func, denom in zip(bias_funcs_11_12, denoms_11_12):
        # Range 1: 1<=j<=d (all zeros)
        for j in range(1, d+1):
            for l in range(1, m+1):
                for q in range(2):
                    current_idx += 1
        
        # Range 2: d+1<=j<=2d (only when l != 1)
        for j in range(d+1, 2*d+1):
            for l in range(1, m+1):
                for q in range(2):
                    if l != 1:
                        B1[current_idx] = bias_func(q, l, denom)
                    # If l == 1, bias remains 0 (already initialized)
                    current_idx += 1
        
        # Range 3: 2d+1<=j<=4d (all zeros)
        for j in range(2*d+1, 4*d+1):
            for l in range(1, m+1):
                for q in range(2):
                    current_idx += 1
        
        # Range 4: 4d+1<=j<=5d (only when l != 1)
        for j in range(4*d+1, 5*d+1):
            for l in range(1, m+1):
                for q in range(2):
                    if l != 1:
                        B1[current_idx] = bias_func(q, l, denom)
                    current_idx += 1
        
        # Range 5: 5d+1<=j<=7d (all zeros)
        for j in range(5*d+1, 7*d+1):
            for l in range(1, m+1):
                for q in range(2):
                    current_idx += 1
    
    # print(f"Final: {current_idx} biases set")
    
    if current_idx != total_length:
        print(f" Warning: Set {current_idx} biases, expected {total_length}")
    
    return B1

