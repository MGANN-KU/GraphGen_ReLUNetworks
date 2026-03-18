# -*- coding: utf-8 -*-
"""
Weights and bias of layer 11

@author: Ghafoor
"""

import numpy as np
from scipy.sparse import lil_matrix

def build_W11(d, C):
    cols_total = 35 * d
    rows_total = 21 * d
    W11 = lil_matrix((rows_total, cols_total))
    row_idx = 0
    
    for j in range(7*d):
        W11[row_idx, 2*j] = -C
        W11[row_idx, 2*j + 1] = C
        W11[row_idx, 14*d + j] = 1
        row_idx += 1
    
    for j in range(7*d):
        W11[row_idx, 21*d + j] = 1
        row_idx += 1
    
    for j in range(7*d):
        W11[row_idx, 28*d + j] = 1
        row_idx += 1
    
    return W11.tocsr()
    
def build_B11(d):
    return np.zeros(21 * d)