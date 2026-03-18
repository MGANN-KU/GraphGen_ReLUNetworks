# -*- coding: utf-8 -*-
"""
Weights and bias of layer 7

@author: Ghafoor
"""


import numpy as np
from scipy.sparse import lil_matrix, csr_matrix

def build_W7(d, eps):
    
    cols_total = (2*d) + (2*d) + (d) + (2*d)
    rows_total = (d*2)*10 + (2*d)*3 + d
    
    W7 = lil_matrix((rows_total, cols_total))
    row_idx = 0
    
    base_E15 = 0
    base_E16 = 2*d
    base_E18 = 2*d + 2*d
    base_E19 = 2*d + 2*d + d
    
    # zeta11, zeta12 nodes
    for j in range(1, d+1):
        for q in range(2):
            for k in range(1, 2*d+1):
                if j == k:
                    W7[row_idx, base_E15 + (k-1)] = 1/eps
            row_idx += 1
    
    # zeta21, zeta22 nodes  
    for j in range(1, d+1):
        for q in range(2):
            for k in range(1, 2*d+1):
                if j == k:
                    W7[row_idx, base_E15 + (k-1)] = -1/eps
            row_idx += 1
    
    # x_j+d
    for j in range(1, 2*d+1):
        for k in range(1, 2*d+1):
            if j == k:
                W7[row_idx, base_E15 + (k-1)] = 1
        row_idx += 1
    
    # zeta31, zeta32 nodes
    for j in range(1, d+1):
        for q in range(2):
            for k in range(1, 2*d+1):
                if j == k:
                    W7[row_idx, base_E16 + (k-1)] = 1/eps
            row_idx += 1
    
    # zeta41, zeta42 nodes
    for j in range(1, d+1):
        for q in range(2):
            for k in range(1, 2*d+1):
                if j == k:
                    W7[row_idx, base_E16 + (k-1)] = -1/eps
            row_idx += 1
    
    # zeta51, zeta52 nodes
    for j in range(1, d+1):
        for q in range(2):
            for k in range(1, 2*d+1):
                if j + d == k:
                    W7[row_idx, base_E16 + (k-1)] = 1/eps
            row_idx += 1
    
    # zeta61, zeta62 nodes
    for j in range(1, d+1):
        for q in range(2):
            for k in range(1, 2*d+1):
                if j + d == k:
                    W7[row_idx, base_E16 + (k-1)] = -1/eps
            row_idx += 1
    
    # x_j as identity map
    for j in range(1, 2*d+1):
        for k in range(1, 2*d+1):
            if j == k:
                W7[row_idx, base_E16 + (k-1)] = 1
        row_idx += 1
    
    # x_j+2d as identity map
    for j in range(2*d+1, 3*d+1):
        for k in range(2*d+1, 3*d+1):
            if j == k:
                W7[row_idx, base_E18 + (k-2*d-1)] = 1
        row_idx += 1
    
    # zeta71, zeta72 nodes
    for j in range(1, d+1):
        for q in range(2):
            for k in range(1, 2*d+1):
                if j == k:
                    W7[row_idx, base_E19 + (k-1)] = 1/eps
            row_idx += 1
    
    # zeta81, zeta82 nodes
    for j in range(1, d+1):
        for q in range(2):
            for k in range(1, 2*d+1):
                if j == k:
                    W7[row_idx, base_E19 + (k-1)] = -1/eps
            row_idx += 1
    
    # zeta91, zeta92 nodes
    for j in range(1, d+1):
        for q in range(2):
            for k in range(1, 2*d+1):
                if j + d == k:
                    W7[row_idx, base_E19 + (k-1)] = 1/eps
            row_idx += 1
    
    # zeta'91, zeta'92 nodes
    for j in range(1, d+1):
        for q in range(2):
            for k in range(1, 2*d+1):
                if j + d == k:
                    W7[row_idx, base_E19 + (k-1)] = -1/eps
            row_idx += 1
    
    # x_j as identity map
    for j in range(1, 2*d+1):
        for k in range(1, 2*d+1):
            if j == k:
                W7[row_idx, base_E19 + (k-1)] = 1
        row_idx += 1
    
    return W7.tocsr()

def build_B7(d):
    # Calculate total length
    total_length = (d*2) + (d*2) + (2*d) + (d*2) + (d*2) + (d*2) + (d*2) + (2*d) + (d) + (d*2) + (d*2) + (d*2) + (d*2) + (2*d)
    
    B7 = np.zeros(total_length)
    idx = 0
    
    # First section: zeta11, zeta12 (d*2 rows)
    for j in range(1, d+1):
        for q in range(2):
            if q == 0:
                B7[idx] = 1
            idx += 1
    
    # Second section: zeta21, zeta22 (d*2 rows)
    for j in range(1, d+1):
        for q in range(2):
            if q == 0:
                B7[idx] = 1
            idx += 1
    
    # Third section: x_j+d (2*d rows) - all zeros
    idx += 2*d
    
    # Fourth section: zeta31, zeta32 (d*2 rows)
    for j in range(1, d+1):
        for q in range(2):
            if q == 0:
                B7[idx] = 1
            idx += 1
    
    # Fifth section: zeta41, zeta42 (d*2 rows)
    for j in range(1, d+1):
        for q in range(2):
            if q == 0:
                B7[idx] = 1
            idx += 1
    
    # Sixth section: zeta51, zeta52 (d*2 rows)
    for j in range(1, d+1):
        for q in range(2):
            if q == 0:
                B7[idx] = 1
            idx += 1
    
    # Seventh section: zeta61, zeta62 (d*2 rows)
    for j in range(1, d+1):
        for q in range(2):
            if q == 0:
                B7[idx] = 1
            idx += 1
    
    # Eighth section: x_j identity (2*d rows) - all zeros
    idx += 2*d
    
    # Ninth section: x_j+2d identity (d rows) - all zeros
    idx += d
    
    # Tenth section: zeta71, zeta72 (d*2 rows)
    for j in range(1, d+1):
        for q in range(2):
            if q == 0:
                B7[idx] = 1
            idx += 1
    
    # Eleventh section: zeta81, zeta82 (d*2 rows)
    for j in range(1, d+1):
        for q in range(2):
            if q == 0:
                B7[idx] = 1
            idx += 1
    
    # Twelfth section: zeta91, zeta92 (d*2 rows)
    for j in range(1, d+1):
        for q in range(2):
            if q == 0:
                B7[idx] = 1
            idx += 1
    
    # Thirteenth section: zeta'91, zeta'92 (d*2 rows)
    for j in range(1, d+1):
        for q in range(2):
            if q == 0:
                B7[idx] = 1
            idx += 1
    
    # Fourteenth section: x_j identity (2*d rows) - all zeros
    idx += 2*d
    
    return B7