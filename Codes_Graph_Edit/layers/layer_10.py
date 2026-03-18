# -*- coding: utf-8 -*-
"""
Weights and bias of layer 10

@author: Ghafoor
"""

import numpy as np
from scipy.sparse import lil_matrix

def build_W10(d, eps):
    cols_total = 21 * d
    rows_total = 35 * d
    W10 = lil_matrix((rows_total, cols_total))
    row_idx = 0
    eps_inv = 1 / eps
    
    for j in range(7*d):
        for q in range(2):
            for k in range(j + 1):
                W10[row_idx, k] = eps_inv
            row_idx += 1
    
    for j in range(7*d):
        W10[row_idx, 7*d + j] = 1
        row_idx += 1
    
    for j in range(7*d):
        if j < 3*d:
            W10[row_idx, 14*d + j] = 1
        if 4*d <= j < 7*d:
            W10[row_idx, 18*d + (j - 4*d)] = 1
        row_idx += 1
    
    for j in range(7*d):
        if 3*d <= j < 4*d:
            W10[row_idx, 17*d + (j - 3*d)] = 1
        row_idx += 1
    
    return W10.tocsr()

def build_B10(d, eps):
    rows_total = 35 * d
    B10 = np.zeros(rows_total)
    idx = 0
    eps_inv = (d + 1) / eps
    
    for j in range(7*d):
        for q in range(2):
            if q == 0:
                B10[idx] = -eps_inv + 1
            else:
                B10[idx] = -eps_inv
            idx += 1
    
    return B10