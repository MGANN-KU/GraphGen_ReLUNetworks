# -*- coding: utf-8 -*-
"""
Weights and bias of layer 33_b

@author: Ghafoor
"""

import numpy as np
from scipy.sparse import csr_matrix, hstack, vstack


import numpy as np
from scipy.sparse import csr_matrix, hstack, vstack

import numpy as np
from scipy.sparse import csr_matrix, hstack, vstack, identity

def build_W33(n, d):
    """Memory-efficient construction of W33 using direct index computation.

    Avoids huge np.repeat/np.tile expansions by computing per-row indices
    (only the actual nonzeros) and building sparse blocks from those.
    """
    n_2d = n + 2 * d
    n_d = n + d

    # Column dimensions
    cols_T5 = n_2d
    cols_H31_etc = n_2d * n_2d
    cols_H32_etc = n_2d * 2 * d
    cols_H33_etc = n_2d * 2 * d
    cols_H34_etc = n_2d * 2 * d
    cols_H35_etc = n_2d * 2 * d
    cols_H36_etc = d * 2
    cols_H37_etc = d * 2

    # -------------------------
    # BLOCK 1: U2 as identity map
    # -------------------------
    rows1 = n_2d
    T5_block1 = identity(rows1, format='csr', dtype=np.float64)
    zeros1 = csr_matrix((rows1, cols_H31_etc + cols_H32_etc + cols_H33_etc +
                        cols_H34_etc + cols_H35_etc + cols_H36_etc + cols_H37_etc))
    block1 = hstack([
        T5_block1,
        csr_matrix((rows1, cols_H31_etc)),
        csr_matrix((rows1, cols_H32_etc)),
        csr_matrix((rows1, cols_H33_etc)),
        csr_matrix((rows1, cols_H34_etc)),
        csr_matrix((rows1, cols_H35_etc)),
        csr_matrix((rows1, cols_H36_etc)),
        csr_matrix((rows1, cols_H37_etc))
    ], format='csr')

    # -------------------------
    # BLOCK 2: tau2 as identity map
    # -------------------------
    rows2 = n_2d * n_2d
    H31_block2 = identity(rows2, format='csr', dtype=np.float64)
    block2 = hstack([
        csr_matrix((rows2, cols_T5)),
        H31_block2,
        csr_matrix((rows2, cols_H32_etc)),
        csr_matrix((rows2, cols_H33_etc)),
        csr_matrix((rows2, cols_H34_etc)),
        csr_matrix((rows2, cols_H35_etc)),
        csr_matrix((rows2, cols_H36_etc)),
        csr_matrix((rows2, cols_H37_etc))
    ], format='csr')

    # -------------------------
    # BLOCK 3: eta6 (rows3 = n_2d * n_2d * d)
    # Build H39/H40 and H43/H44 by direct per-row index computation
    # -------------------------
    rows3 = n_2d * n_2d * d

    # Row indexing scheme: row -> (i, l, j)
    row_idx = np.arange(rows3, dtype=np.int64)
    div = n_2d * d
    i_vals = row_idx // div
    rem = row_idx % div
    # l_vals = rem // d   # not needed for these blocks, but kept for clarity
    j_vals = rem % d

    # H39/H40: for each row, two entries at columns (i * d + j)*2 + q with q=0 => +1, q=1 => -1
    base_cols_h32 = (i_vals * d + j_vals) * 2  # length rows3
    rows_repeat = np.repeat(row_idx, 2)         # each row has two nonzeros
    cols_h39 = np.empty(rows3 * 2, dtype=np.int64)
    cols_h39[0::2] = base_cols_h32
    cols_h39[1::2] = base_cols_h32 + 1
    data_h39 = np.empty(rows3 * 2, dtype=np.float64)
    data_h39[0::2] = 1.0
    data_h39[1::2] = -1.0

    H39_block3 = csr_matrix((data_h39, (rows_repeat, cols_h39)),
                            shape=(rows3, cols_H32_etc))
    H40_block3 = H39_block3.copy()

    # H43/H44: if j==k -> columns k*2+q where k == j_vals
    base_cols_h36 = j_vals * 2
    cols_h43 = np.empty(rows3 * 2, dtype=np.int64)
    cols_h43[0::2] = base_cols_h36
    cols_h43[1::2] = base_cols_h36 + 1
    data_h43 = np.empty(rows3 * 2, dtype=np.float64)
    data_h43[0::2] = 1.0
    data_h43[1::2] = -1.0

    H43_block3 = csr_matrix((data_h43, (rows_repeat, cols_h43)),
                            shape=(rows3, cols_H36_etc))
    H44_block3 = H43_block3.copy()

    block3 = hstack([
        csr_matrix((rows3, cols_T5)),
        csr_matrix((rows3, cols_H31_etc)),
        H39_block3,
        H40_block3,
        csr_matrix((rows3, cols_H34_etc)),
        csr_matrix((rows3, cols_H35_etc)),
        H43_block3,
        H44_block3
    ], format='csr')

    # -------------------------
    # BLOCK 4: eta7 (rows4 = n_2d * d)
    # -------------------------
    rows4 = n_2d * d
    row_idx4 = np.arange(rows4, dtype=np.int64)
    div4 = d
    i_vals4 = row_idx4 // div4
    j_vals4 = row_idx4 % div4

    # H46/H47: same pattern as H39/H40 but with i in 0..n_2d-1 and j in 0..d-1 and fewer rows
    base_cols_h32_4 = (i_vals4 * d + j_vals4) * 2
    rows_repeat4 = np.repeat(row_idx4, 2)
    cols_h46 = np.empty(rows4 * 2, dtype=np.int64)
    cols_h46[0::2] = base_cols_h32_4
    cols_h46[1::2] = base_cols_h32_4 + 1
    data_h46 = np.empty(rows4 * 2, dtype=np.float64)
    data_h46[0::2] = 1.0
    data_h46[1::2] = -1.0

    H46_block4 = csr_matrix((data_h46, (rows_repeat4, cols_h46)),
                            shape=(rows4, cols_H32_etc))
    H47_block4 = H46_block4.copy()

    # H50/H51: j==k pattern -> columns j*2 and j*2+1 (k range 0..d-1)
    base_cols_h36_4 = j_vals4 * 2
    cols_h50 = np.empty(rows4 * 2, dtype=np.int64)
    cols_h50[0::2] = base_cols_h36_4
    cols_h50[1::2] = base_cols_h36_4 + 1
    data_h50 = np.empty(rows4 * 2, dtype=np.float64)
    data_h50[0::2] = 1.0
    data_h50[1::2] = -1.0

    H50_block4 = csr_matrix((data_h50, (rows_repeat4, cols_h50)),
                            shape=(rows4, cols_H36_etc))
    H51_block4 = H50_block4.copy()

    block4 = hstack([
        csr_matrix((rows4, cols_T5)),
        csr_matrix((rows4, cols_H31_etc)),
        H46_block4,
        H47_block4,
        csr_matrix((rows4, cols_H34_etc)),
        csr_matrix((rows4, cols_H35_etc)),
        H50_block4,
        H51_block4
    ], format='csr')

    # -------------------------
    # BLOCK 5: eta8 (rows5 = n_d * n_2d * d)
    # -------------------------
    rows5 = n_d * n_2d * d
    row_idx5 = np.arange(rows5, dtype=np.int64)
    div5 = n_2d * d
    i_vals5 = row_idx5 // div5
    rem5 = row_idx5 % div5
    l_vals5 = rem5 // d
    j_vals5 = rem5 % d

    # H55/H56: condition k==l and j==r => columns (l * d + j)*2 + q
    base_cols_h34 = (l_vals5 * d + j_vals5) * 2
    rows_repeat5 = np.repeat(row_idx5, 2)
    cols_h55 = np.empty(rows5 * 2, dtype=np.int64)
    cols_h55[0::2] = base_cols_h34
    cols_h55[1::2] = base_cols_h34 + 1
    data_h55 = np.empty(rows5 * 2, dtype=np.float64)
    data_h55[0::2] = 1.0
    data_h55[1::2] = -1.0

    H55_block5 = csr_matrix((data_h55, (rows_repeat5, cols_h55)),
                            shape=(rows5, cols_H34_etc))
    H56_block5 = H55_block5.copy()

    # H57/H58: j==k -> columns j*2 and j*2+1 (k range 0..d-1)
    base_cols_h36_5 = j_vals5 * 2
    cols_h57 = np.empty(rows5 * 2, dtype=np.int64)
    cols_h57[0::2] = base_cols_h36_5
    cols_h57[1::2] = base_cols_h36_5 + 1
    data_h57 = np.empty(rows5 * 2, dtype=np.float64)
    data_h57[0::2] = 1.0
    data_h57[1::2] = -1.0

    H57_block5 = csr_matrix((data_h57, (rows_repeat5, cols_h57)),
                            shape=(rows5, cols_H36_etc))
    H58_block5 = H57_block5.copy()

    block5 = hstack([
        csr_matrix((rows5, cols_T5)),
        csr_matrix((rows5, cols_H31_etc)),
        csr_matrix((rows5, cols_H32_etc)),
        csr_matrix((rows5, cols_H33_etc)),
        H55_block5,
        H56_block5,
        H57_block5,
        H58_block5
    ], format='csr')

    # -------------------------
    # Assemble full W33
    # -------------------------
    W33 = vstack([block1, block2, block3, block4, block5], format='csr')
    return W33






# def build_W33(n, d):
    # """Completely vectorized, loop-free W33 implementation"""
    # n_2d = n + 2*d
    # n_d = n + d
    
    # # Column dimensions (ALL BLOCKS MUST HAVE SAME TOTAL)
    # cols_T5 = n_2d                     # 10
    # cols_H31_etc = n_2d * n_2d        # 100
    # cols_H32_etc = n_2d * 2 * d       # 10×2×2 = 40
    # cols_H33_etc = n_2d * 2 * d       # 40
    # cols_H34_etc = n_2d * 2 * d       # 40
    # cols_H35_etc = n_2d * 2 * d       # 40
    # cols_H36_etc = d * 2              # 2×2 = 4
    # cols_H37_etc = d * 2              # 4
    
    # total_cols = (cols_T5 + cols_H31_etc + cols_H32_etc + cols_H33_etc + 
                  # cols_H34_etc + cols_H35_etc + cols_H36_etc + cols_H37_etc)
    
    # # ===== BLOCK 1: U2 as identity map =====
    # rows1 = n_2d
    
    # # T5: Identity
    # T5_block1 = csr_matrix(np.eye(rows1, dtype=np.float64))
    
    # # All zeros for H31-H37
    # block1 = hstack([
        # T5_block1,
        # csr_matrix((rows1, cols_H31_etc)),
        # csr_matrix((rows1, cols_H32_etc)),
        # csr_matrix((rows1, cols_H33_etc)),
        # csr_matrix((rows1, cols_H34_etc)),
        # csr_matrix((rows1, cols_H35_etc)),
        # csr_matrix((rows1, cols_H36_etc)),
        # csr_matrix((rows1, cols_H37_etc))
    # ], format='csr')
    
    # # ===== BLOCK 2: tau2 as identity map =====
    # rows2 = n_2d * n_2d
    
    # # H31: Identity
    # H31_block2 = csr_matrix(np.eye(rows2, dtype=np.float64))
    
    # block2 = hstack([
        # csr_matrix((rows2, cols_T5)),
        # H31_block2,
        # csr_matrix((rows2, cols_H32_etc)),
        # csr_matrix((rows2, cols_H33_etc)),
        # csr_matrix((rows2, cols_H34_etc)),
        # csr_matrix((rows2, cols_H35_etc)),
        # csr_matrix((rows2, cols_H36_etc)),
        # csr_matrix((rows2, cols_H37_etc))
    # ], format='csr')
    
    # # ===== BLOCK 3: eta6 that corresponding to e_ik (VECTORIZED) =====
    # rows3 = n_2d * n_2d * d
    
    # # Create all i, l, j combinations without loops
    # # i: 0..n_2d-1, each repeated n_2d*d times
    # i_vals = np.repeat(np.arange(n_2d), n_2d * d)
    # # l: pattern [0,0,1,1,...] repeated for each i
    # l_vals = np.tile(np.repeat(np.arange(n_2d), d), n_2d)
    # # j: pattern [0,1,0,1,...] for d=2
    # j_vals = np.tile(np.arange(d), n_2d * n_2d)
    
    # # Row indices: 0 to rows3-1
    # row_indices = np.arange(rows3)
    
    # # H39 and H40: if i==p and j==k: w=±1 for q=0,1
    # # Create all p, k, q combinations
    # total_patterns = rows3 * n_2d * d * 2
    
    # # Expand i, l, j to match pattern size
    # i_expanded = np.repeat(i_vals, n_2d * d * 2)
    # j_expanded = np.repeat(j_vals, n_2d * d * 2)
    
    # # Create p, k, q values
    # p_vals = np.tile(np.repeat(np.arange(n_2d), d * 2), rows3)
    # k_vals = np.tile(np.repeat(np.arange(d), 2), rows3 * n_2d)
    # q_vals = np.tile([0, 1], rows3 * n_2d * d)
    
    # # Row indices expanded
    # rows_expanded = np.repeat(row_indices, n_2d * d * 2)
    
    # # Condition: i==p and j==k
    # condition = (i_expanded == p_vals) & (j_expanded == k_vals)
    
    # # Apply condition
    # valid_mask = condition
    
    # # Column indices: (p * d + k) * 2 + q
    # cols_H39 = (p_vals * d + k_vals) * 2 + q_vals
    
    # # Data: +1 for q=0, -1 for q=1
    # data_H39 = np.where(q_vals == 0, 1.0, -1.0)
    
    # # Filter valid entries
    # valid_rows = rows_expanded[valid_mask]
    # valid_cols = cols_H39[valid_mask]
    # valid_data = data_H39[valid_mask]
    
    # H39_block3 = csr_matrix((valid_data, (valid_rows, valid_cols)), 
                           # shape=(rows3, cols_H32_etc))
    # H40_block3 = H39_block3.copy()
    
    # # H43 and H44: if j==k: w=±1 for q=0,1
    # # Create k, q combinations for each row
    # total_patterns2 = rows3 * d * 2
    
    # # Expand j values
    # j_expanded2 = np.repeat(j_vals, d * 2)
    # rows_expanded2 = np.repeat(row_indices, d * 2)
    # k_vals2 = np.tile(np.repeat(np.arange(d), 2), rows3)
    # q_vals2 = np.tile([0, 1], rows3 * d)
    
    # # Condition: j==k
    # condition2 = (j_expanded2 == k_vals2)
    # valid_mask2 = condition2
    
    # # Column indices: k * 2 + q
    # cols_H43 = k_vals2 * 2 + q_vals2
    
    # # Data: +1 for q=0, -1 for q=1
    # data_H43 = np.where(q_vals2 == 0, 1.0, -1.0)
    
    # # Filter valid entries
    # valid_rows2 = rows_expanded2[valid_mask2]
    # valid_cols2 = cols_H43[valid_mask2]
    # valid_data2 = data_H43[valid_mask2]
    
    # H43_block3 = csr_matrix((valid_data2, (valid_rows2, valid_cols2)), 
                           # shape=(rows3, cols_H36_etc))
    # H44_block3 = H43_block3.copy()
    
    # block3 = hstack([
        # csr_matrix((rows3, cols_T5)),
        # csr_matrix((rows3, cols_H31_etc)),  # H38
        # H39_block3,
        # H40_block3,
        # csr_matrix((rows3, cols_H34_etc)),  # H41
        # csr_matrix((rows3, cols_H35_etc)),  # H42
        # H43_block3,
        # H44_block3
    # ], format='csr')
    
    # # ===== BLOCK 4: eta7 that corresponding to e'_i (VECTORIZED) =====
    # rows4 = n_2d * d
    
    # # Create i, j combinations
    # i_vals4 = np.repeat(np.arange(n_2d), d)
    # j_vals4 = np.tile(np.arange(d), n_2d)
    # row_indices4 = np.arange(rows4)
    
    # # H46 and H47: if i==p and j==k: w=±1 for q=0,1
    # total_patterns4 = rows4 * n_2d * d * 2
    
    # # Expand i, j
    # i_expanded4 = np.repeat(i_vals4, n_2d * d * 2)
    # j_expanded4 = np.repeat(j_vals4, n_2d * d * 2)
    
    # # Create p, k, q
    # p_vals4 = np.tile(np.repeat(np.arange(n_2d), d * 2), rows4)
    # k_vals4 = np.tile(np.repeat(np.arange(d), 2), rows4 * n_2d)
    # q_vals4 = np.tile([0, 1], rows4 * n_2d * d)
    # rows_expanded4 = np.repeat(row_indices4, n_2d * d * 2)
    
    # # Condition: i==p and j==k
    # condition4 = (i_expanded4 == p_vals4) & (j_expanded4 == k_vals4)
    # valid_mask4 = condition4
    
    # # Column indices and data
    # cols_H46 = (p_vals4 * d + k_vals4) * 2 + q_vals4
    # data_H46 = np.where(q_vals4 == 0, 1.0, -1.0)
    
    # valid_rows4 = rows_expanded4[valid_mask4]
    # valid_cols4 = cols_H46[valid_mask4]
    # valid_data4 = data_H46[valid_mask4]
    
    # H46_block4 = csr_matrix((valid_data4, (valid_rows4, valid_cols4)), 
                           # shape=(rows4, cols_H32_etc))
    # H47_block4 = H46_block4.copy()
    
    # # H50 and H51: if j==k: w=±1 for q=0,1
    # total_patterns5 = rows4 * d * 2
    
    # j_expanded5 = np.repeat(j_vals4, d * 2)
    # rows_expanded5 = np.repeat(row_indices4, d * 2)
    # k_vals5 = np.tile(np.repeat(np.arange(d), 2), rows4)
    # q_vals5 = np.tile([0, 1], rows4 * d)
    
    # condition5 = (j_expanded5 == k_vals5)
    # valid_mask5 = condition5
    
    # cols_H50 = k_vals5 * 2 + q_vals5
    # data_H50 = np.where(q_vals5 == 0, 1.0, -1.0)
    
    # valid_rows5 = rows_expanded5[valid_mask5]
    # valid_cols5 = cols_H50[valid_mask5]
    # valid_data5 = data_H50[valid_mask5]
    
    # H50_block4 = csr_matrix((valid_data5, (valid_rows5, valid_cols5)), 
                           # shape=(rows4, cols_H36_etc))
    # H51_block4 = H50_block4.copy()
    
    # block4 = hstack([
        # csr_matrix((rows4, cols_T5)),
        # csr_matrix((rows4, cols_H31_etc)),  # H45
        # H46_block4,
        # H47_block4,
        # csr_matrix((rows4, cols_H34_etc)),  # H48
        # csr_matrix((rows4, cols_H35_etc)),  # H49
        # H50_block4,
        # H51_block4
    # ], format='csr')
    
    # # ===== BLOCK 5: eta8 that corresponding to pj_ik (VECTORIZED) =====
    # rows5 = n_d * n_2d * d
    
    # # Create i, l, j combinations
    # i_vals5 = np.repeat(np.arange(n_d), n_2d * d)
    # l_vals5 = np.tile(np.repeat(np.arange(n_2d), d), n_d)
    # j_vals5 = np.tile(np.arange(d), n_d * n_2d)
    # row_indices5 = np.arange(rows5)
    
    # # H55 and H56: if k==l and j==r: w=±1 for q=0,1
    # total_patterns6 = rows5 * n_2d * d * 2
    
    # # Expand l, j
    # l_expanded6 = np.repeat(l_vals5, n_2d * d * 2)
    # j_expanded6 = np.repeat(j_vals5, n_2d * d * 2)
    # rows_expanded6 = np.repeat(row_indices5, n_2d * d * 2)
    
    # # Create k, r, q
    # k_vals6 = np.tile(np.repeat(np.arange(n_2d), d * 2), rows5)
    # r_vals6 = np.tile(np.repeat(np.arange(d), 2), rows5 * n_2d)
    # q_vals6 = np.tile([0, 1], rows5 * n_2d * d)
    
    # # Condition: k==l and j==r
    # condition6 = (k_vals6 == l_expanded6) & (j_expanded6 == r_vals6)
    # valid_mask6 = condition6
    
    # # Column indices and data
    # cols_H55 = (k_vals6 * d + r_vals6) * 2 + q_vals6
    # data_H55 = np.where(q_vals6 == 0, 1.0, -1.0)
    
    # valid_rows6 = rows_expanded6[valid_mask6]
    # valid_cols6 = cols_H55[valid_mask6]
    # valid_data6 = data_H55[valid_mask6]
    
    # H55_block5 = csr_matrix((valid_data6, (valid_rows6, valid_cols6)), 
                           # shape=(rows5, cols_H34_etc))
    # H56_block5 = H55_block5.copy()
    
    # # H57 and H58: if j==k: w=±1 for q=0,1
    # total_patterns7 = rows5 * d * 2
    
    # j_expanded7 = np.repeat(j_vals5, d * 2)
    # rows_expanded7 = np.repeat(row_indices5, d * 2)
    # k_vals7 = np.tile(np.repeat(np.arange(d), 2), rows5)
    # q_vals7 = np.tile([0, 1], rows5 * d)
    
    # condition7 = (j_expanded7 == k_vals7)
    # valid_mask7 = condition7
    
    # cols_H57 = k_vals7 * 2 + q_vals7
    # data_H57 = np.where(q_vals7 == 0, 1.0, -1.0)
    
    # valid_rows7 = rows_expanded7[valid_mask7]
    # valid_cols7 = cols_H57[valid_mask7]
    # valid_data7 = data_H57[valid_mask7]
    
    # H57_block5 = csr_matrix((valid_data7, (valid_rows7, valid_cols7)), 
                           # shape=(rows5, cols_H36_etc))
    # H58_block5 = H57_block5.copy()
    
    # block5 = hstack([
        # csr_matrix((rows5, cols_T5)),
        # csr_matrix((rows5, cols_H31_etc)),  # H52
        # csr_matrix((rows5, cols_H32_etc)),  # H53
        # csr_matrix((rows5, cols_H33_etc)),  # H54
        # H55_block5,
        # H56_block5,
        # H57_block5,
        # H58_block5
    # ], format='csr')
    
    # # ===== ASSEMBLE =====
    # W33 = vstack([block1, block2, block3, block4, block5], format='csr')
    
    # return W33





from scipy.sparse import csr_matrix
import numpy as np

def build_B33(n, d, C):
    """Build B33 as a sparse column vector with minimal memory and no loops."""
    
    n_2d = n + 2*d

    # Section sizes (same logic as original)
    size_u2   = n_2d
    size_tau2 = n_2d * n_2d
    size_eta6 = n_2d * n_2d * d
    size_eta7 = n_2d * d
    size_eta8 = (n + d) * n_2d * d

    total_size = size_u2 + size_tau2 + size_eta6 + size_eta7 + size_eta8
    
    # Number of nonzero entries
    non_zero_count = size_eta6 + size_eta7 + size_eta8

    # All non-zero values are -3 (your original code did NOT use C)
    data = np.full(non_zero_count, -3.0, dtype=np.float64)

    # Compute row indices without loops
    rows = np.arange(size_u2 + size_tau2, total_size, dtype=np.int32)

    # Final sparse column vector
    return csr_matrix((data, (rows, np.zeros(non_zero_count, dtype=np.int32))),
                      shape=(total_size, 1))
