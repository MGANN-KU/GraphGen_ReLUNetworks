# -*- coding: utf-8 -*-
"""
Weights and bias of layer 12

@author: Ghafoor
"""

import numpy as np
from scipy.sparse import lil_matrix

def build_W12(d):
    cols_total = 21 * d
    rows_total = 7 * d
    W12 = lil_matrix((rows_total, cols_total))
    
    for j in range(7*d):
        W12[j, j] = 1
        W12[j, 7*d + j] = 1
        W12[j, 14*d + j] = 1
    
    return W12.tocsr()