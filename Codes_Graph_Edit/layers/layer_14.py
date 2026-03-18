# -*- coding: utf-8 -*-
"""
Weights and bias of layer 13
modified to 13_b
@author: Ghafoor
"""
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix

import numpy as np
from scipy.sparse import coo_matrix

import numpy as np
from scipy.sparse import csr_matrix

import numpy as np
from scipy.sparse import csr_matrix

import numpy as np
from scipy.sparse import csr_matrix, hstack, vstack

import numpy as np
from scipy.sparse import csr_matrix

import numpy as np
from scipy.sparse import csr_matrix

import numpy as np
from scipy.sparse import csr_matrix

import numpy as np
from scipy.sparse import csr_matrix

import numpy as np
from scipy.sparse import csr_matrix

def build_W14(n, d, C):
    """
    Efficient sparse implementation matching original exactly.
    """
    N = n + 2 * d
    
    # ====== DIMENSIONS ======
    H1_size = 2 * d * N
    H2_size = H1_size
    H3_size = d
    H45_size = 2 * d
    H6_size = 5 * d
    
    total_cols = H1_size + H2_size + H3_size + H45_size + H45_size + H6_size
    
    rows_section1 = N
    rows_section2 = N * d
    rows_section3 = d
    rows_section4 = d
    rows_section5 = 4 * d
    total_rows = rows_section1 + rows_section2 + rows_section3 + rows_section4 + rows_section5
    
    # ====== BUILD LISTS ======
    data = []
    rows = []
    cols = []
    
    # ====== SECTION 1 ======
    for i in range(rows_section1):
        # H1 block
        for l in range(N):
            for k in range(d):
                if i == l:
                    # q=0
                    data.append(-C)
                    rows.append(i)
                    cols.append(l * 2 * d + 2 * k)
                    
                    # q=1
                    data.append(C)
                    rows.append(i)
                    cols.append(l * 2 * d + 2 * k + 1)
        
        # H2 block
        H2_start = H1_size
        for l in range(N):
            for k in range(d):
                if i == l:
                    # q=0
                    data.append(-C)
                    rows.append(i)
                    cols.append(H2_start + l * 2 * d + 2 * k)
                    
                    # q=1
                    data.append(C)
                    rows.append(i)
                    cols.append(H2_start + l * 2 * d + 2 * k + 1)
    
    # ====== SECTION 2 ======
    current_row = rows_section1
    
    for i in range(N):
        for j in range(d):
            row_idx = current_row + i * d + j
            
            # H1_mu
            data.append(C)
            rows.append(row_idx)
            cols.append(i * 2 * d + j * 2)
            
            data.append(-C)
            rows.append(row_idx)
            cols.append(i * 2 * d + j * 2 + 1)
            
            # H2_mu
            H2_start = H1_size
            data.append(C)
            rows.append(row_idx)
            cols.append(H2_start + i * 2 * d + j * 2)
            
            data.append(-C)
            rows.append(row_idx)
            cols.append(H2_start + i * 2 * d + j * 2 + 1)
            
            # H3_mu
            H3_start = H1_size + H2_size
            data.append(1.0)
            rows.append(row_idx)
            cols.append(H3_start + j)
    
    # ====== SECTION 3 ======
    current_row = rows_section1 + rows_section2
    
    for i in range(rows_section3):
        row_idx = current_row + i
        
        # H4_w1
        H4_start = H1_size + H2_size + H3_size
        data.append(-C)
        rows.append(row_idx)
        cols.append(H4_start + 2 * i)
        
        data.append(C)
        rows.append(row_idx)
        cols.append(H4_start + 2 * i + 1)
        
        # H5_w1
        H5_start = H1_size + H2_size + H3_size + H45_size
        data.append(-C)
        rows.append(row_idx)
        cols.append(H5_start + 2 * i)
        
        data.append(C)
        rows.append(row_idx)
        cols.append(H5_start + 2 * i + 1)
        
        # H6_w1
        H6_start = H1_size + H2_size + H3_size + H45_size + H45_size
        data.append(1.0)
        rows.append(row_idx)
        cols.append(H6_start + i)
    
    # ====== SECTION 4 ======
    current_row = rows_section1 + rows_section2 + rows_section3
    
    for i in range(rows_section4):
        row_idx = current_row + i
        
        # H4_w2
        H4_start = H1_size + H2_size + H3_size
        data.append(C)
        rows.append(row_idx)
        cols.append(H4_start + 2 * i)
        
        data.append(-C)
        rows.append(row_idx)
        cols.append(H4_start + 2 * i + 1)
        
        # H5_w2
        H5_start = H1_size + H2_size + H3_size + H45_size
        data.append(C)
        rows.append(row_idx)
        cols.append(H5_start + 2 * i)
        
        data.append(-C)
        rows.append(row_idx)
        cols.append(H5_start + 2 * i + 1)
    
    # ====== SECTION 5 ======
    current_row = rows_section1 + rows_section2 + rows_section3 + rows_section4
    
    H6_start = H1_size + H2_size + H3_size + H45_size + H45_size
    
    for idx in range(rows_section5):
        row_idx = current_row + idx
        
        # i in original: 3*d+1 to 7*d
        i_1based = 3 * d + 1 + idx
        
        # Column in H6: j - (2*d+1) where j = i
        col_in_H6 = i_1based - (2 * d + 1)
        
        data.append(1.0)
        rows.append(row_idx)
        cols.append(H6_start + col_in_H6)
    
    # ====== CREATE SPARSE MATRIX ======
    # Verify array lengths match
    if not (len(data) == len(rows) == len(cols)):
        # Find the mismatch and fix it
        min_len = min(len(data), len(rows), len(cols))
        data = data[:min_len]
        rows = rows[:min_len]
        cols = cols[:min_len]
    
    W14 = csr_matrix((np.array(data, dtype=np.float64), 
                      (np.array(rows, dtype=np.int32), 
                       np.array(cols, dtype=np.int32))),
                     shape=(total_rows, total_cols))
    
    return W14
    


def build_B14(n, d, C, B, U):
    """
    Optimized vectorized version for B14 construction.
    
    Parameters:
    n, d, C, B: Same as original
    U: List or array of length (n+2*d) containing U[i-1] values
    
    Returns: Numpy array B14 of shape (total_rows, 1)
    """
    N = n + 2 * d
    
    # Convert U to numpy array if it's a list
    if not isinstance(U, np.ndarray):
        U = np.array(U, dtype=np.float64)
    
    # ====== CALCULATE TOTAL ROWS ======
    rows_sec1 = N                     # First N rows
    rows_sec2 = N * d                 # Next N*d rows
    rows_sec3 = d                     # w'1 rows
    rows_sec4 = d                     # w'2 rows
    rows_sec5 = 4 * d                 # xj nodes rows
    total_rows = rows_sec1 + rows_sec2 + rows_sec3 + rows_sec4 + rows_sec5
    
    # ====== BUILD VECTORIZED ======
    B14 = np.zeros((total_rows, 1), dtype=np.float64)
    
    # Section 1: First N rows - b = U[i-1] + d*C
    B14[:rows_sec1, 0] = U[:N] + d * C
    
    # Section 2: Next N*d rows - b = -2*C
    B14[rows_sec1:rows_sec1+rows_sec2, 0] = -2.0 * C
    
    # Section 3: w'1 rows - b = C
    start = rows_sec1 + rows_sec2
    B14[start:start+rows_sec3, 0] = C
    
    # Section 4: w'2 rows - b = B - 2*C
    start += rows_sec3
    B14[start:start+rows_sec4, 0] = B - 2.0 * C
    
    # Section 5: xj nodes rows - b = 0 (already zeros)
    
    return B14
