# -*- coding: utf-8 -*-
"""
Weights and bias of layer 9

@author: Ghafoor
"""

import numpy as np
from scipy.sparse import lil_matrix

def build_W9(d):
    # Total columns: E33(d) + E34(2d) + E35(d) + E36(3d) + E37(d) + E38(2d) = 10d
    cols_total = 10 * d
    rows_total = 21 * d  # 7d + 7d + 7d
    
    W9 = lil_matrix((rows_total, cols_total))
    row_idx = 0
    
    # Section 1: zeta4 nodes (7d rows) - E33, E35, E37 only
    for j in range(7*d):
        # E33: w=1 if k == j where k from 1 to d
        if 0 <= j < d:
            W9[row_idx, j] = 1
        
        # E35: w=1 if k+2*d == j where k from 1 to d
        if 2*d <= j < 3*d:
            W9[row_idx, 3*d + (j - 2*d)] = 1
        
        # E37: w=1 if k+5*d == j where k from 1 to d
        if 5*d <= j < 6*d:
            W9[row_idx, 7*d + (j - 5*d)] = 1
        
        row_idx += 1
    
    # Section 2: x''j nodes for j in {1,...,d, 2d+1,...,3d, 5d+1,...,6d} (7d rows)
    for j in range(7*d):
        # E34: w=1 if j == k where k from 1 to d
        if 0 <= j < d:
            W9[row_idx, d + j] = 1
        
        # E37: w=1 if j == k+2*d where k from 1 to d
        if 2*d <= j < 3*d:
            W9[row_idx, 4*d + (j - 2*d)] = 1
        
        # E40: w=1 if j == k+5*d where k from 1 to d
        if 5*d <= j < 6*d:
            W9[row_idx, 8*d + (j - 5*d)] = 1
        
        row_idx += 1
    
    # Section 3: x''j nodes for j in {d+1,...,2d, 3d+1,...,5d, 6d+1,...,7d} (7d rows)
    for j in range(7*d):
        # E35: w=1 if j == k where k from d+1 to 2d
        if d <= j < 2*d:
            W9[row_idx, 2*d + (j - d)] = 1
        
        # E38: w=1 if j == k+2*d where k from d+1 to 3d
        if 3*d <= j < 5*d:
            W9[row_idx, 5*d + (j - 3*d)] = 1
        
        # E41: w=1 if j == k+5*d where k from d+1 to 2d
        if 6*d <= j < 7*d:
            W9[row_idx, 9*d + (j - 6*d)] = 1
        
        row_idx += 1
    
    return W9.tocsr()