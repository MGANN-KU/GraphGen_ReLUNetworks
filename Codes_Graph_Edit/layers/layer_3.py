# -*- coding: utf-8 -*-
"""
Weights and bias of layer 3

@author: Ghafoor
"""




import numpy as np
from scipy.sparse import lil_matrix, csr_matrix

def build_W3(d):
    rows_total = 7 * d
    cols_total = 21 * d
    
    W3 = lil_matrix((rows_total, cols_total))
    
    for j in range(1, 7*d + 1):
        row_idx = j - 1
        W3[row_idx, j-1] = 1
        W3[row_idx, (7*d) + (j-1)] = 1
        W3[row_idx, (14*d) + (j-1)] = 1
    
    return W3.tocsr()