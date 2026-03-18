# -*- coding: utf-8 -*-
"""
Weights and bias of layer 37

@author: Ghafoor
"""
import numpy as np
from scipy.sparse import coo_matrix
import sys
from scipy.sparse import csr_matrix, hstack, vstack


import numpy as np
from scipy.sparse import csr_matrix, hstack, vstack

def build_W37(n, d, eps):
    """
    Vectorized version of build_W37.
    eps must be nonzero (used as divisor). Returns scipy.sparse.csr_matrix.
    """
    if eps == 0:
        raise ValueError("eps must be nonzero")

    n_2d = n + 2 * d
    n_d = n + d
    d2 = d + 1

    W37_blocks = []

    # ---------------- BLOCK 1 ----------------
    # rows1 = n_2d
    rows1 = n_2d
    # T9: identity (rows1 x n_2d)
    T9 = csr_matrix((np.ones(rows1, dtype=np.float64), (np.arange(rows1), np.arange(rows1))),
                    shape=(rows1, n_2d))

    # J1, J2 zeros shapes:
    J1 = csr_matrix((rows1, n_2d * n_2d))
    J2 = csr_matrix((rows1, n_2d * n_2d))
    J3 = csr_matrix((rows1, n_2d))
    J4 = csr_matrix((rows1, n_d * n_2d))

    block1 = hstack([T9, J1, J2, J3, J4], format='csr')
    W37_blocks.append(block1)

    # ---------------- BLOCK 2 ----------------
    # Interpreting tau2 as mapping rows indexed by (i,l) -> single column index (i,l)
    # rows2 should be n_2d * n_2d (one row per (i,l))
    rows2 = n_2d * n_2d
    # Row indices for (i,l)
    i_idx = np.repeat(np.arange(n_2d, dtype=np.int64), n_2d)  # length rows2
    l_idx = np.tile(np.arange(n_2d, dtype=np.int64), n_2d)

    # J1 places a 1 at column col = i * n_2d + l  (zero-based)
    J1_rows = np.arange(rows2, dtype=np.int64)
    J1_cols = (i_idx * n_2d) + l_idx
    J1_data = np.ones(rows2, dtype=np.float64)
    J1 = csr_matrix((J1_data, (J1_rows, J1_cols)), shape=(rows2, n_2d * n_2d))

    T9_b2 = csr_matrix((rows2, n_2d))
    J2 = csr_matrix((rows2, n_2d * n_2d))
    J3 = csr_matrix((rows2, n_2d))
    J4 = csr_matrix((rows2, n_d * n_2d))

    block2 = hstack([T9_b2, J1, J2, J3, J4], format='csr')
    W37_blocks.append(block2)

    # ---------------- BLOCK 3 ----------------
    # rows3 = n_d * n_2d * d2 * 2 with row ordering (i, k, j, q)
    rows3 = n_d * n_2d * d2 * 2
    # produce index arrays matching loop order in original: for i in 1..n_d, k in 1..n_2d, j in 1..d2, q in 0..1
    i3 = np.repeat(np.arange(n_d, dtype=np.int64), n_2d * d2 * 2)
    k3 = np.tile(np.repeat(np.arange(n_2d, dtype=np.int64), d2 * 2), n_d)
    j3 = np.tile(np.repeat(np.arange(d2, dtype=np.int64), 2), n_d * n_2d)
    q3 = np.tile(np.arange(2, dtype=np.int64), n_d * n_2d * d2)

    # Condition in original: if i + j - 1 == r and p == k  -> column index (r-1)*n_2d + (p-1)
    # That means r = i + j - 1  (1-based). Zero-based r0 = i3 + j3
    r0 = i3 + j3
    # valid r0 in [0, n_2d-1]
    mask_valid = (r0 >= 0) & (r0 < n_2d)
    # For each valid row, column = r0 * n_2d + p0 where p0 = k3
    if mask_valid.any():
        Gcol_base = (r0[mask_valid] * n_2d) + k3[mask_valid]
        Grow = np.nonzero(mask_valid)[0].astype(np.int64)
        Gdata = np.full(Gcol_base.size, 1.0 / eps, dtype=np.float64)
    else:
        Grow = np.array([], dtype=np.int64)
        Gcol_base = np.array([], dtype=np.int64)
        Gdata = np.array([], dtype=np.float64)

    J6 = csr_matrix((Gdata, (Grow, Gcol_base)), shape=(rows3, n_2d * n_2d))

    T9_b3 = csr_matrix((rows3, n_2d))
    J5 = csr_matrix((rows3, n_2d * n_2d))
    J7 = csr_matrix((rows3, n_2d))
    J8 = csr_matrix((rows3, n_d * n_2d))

    block3 = hstack([T9_b3, J5, J6, J7, J8], format='csr')
    W37_blocks.append(block3)

    # ---------------- BLOCK 4 ----------------
    # rows4 = n_d * n_2d * d2 * 2, condition: if i+j-1 == k and p == l => column (k-1)*n_2d + (p-1)
    rows4 = n_d * n_2d * d2 * 2
    # row ordering (i, l, j, q) in original; we'll follow that
    i4 = np.repeat(np.arange(n_d, dtype=np.int64), n_2d * d2 * 2)
    l4 = np.tile(np.repeat(np.arange(n_2d, dtype=np.int64), d2 * 2), n_d)
    j4 = np.tile(np.repeat(np.arange(d2, dtype=np.int64), 2), n_d * n_2d)
    q4 = np.tile(np.arange(2, dtype=np.int64), n_d * n_2d * d2)

    k0 = i4 + j4                # zero-based k
    mask_valid4 = (k0 >= 0) & (k0 < n_2d)
    if mask_valid4.any():
        # col = k0 * n_2d + p where p = l4
        col4 = (k0[mask_valid4] * n_2d) + l4[mask_valid4]
        row4 = np.nonzero(mask_valid4)[0].astype(np.int64)
        data4 = np.full(col4.size, -1.0 / eps, dtype=np.float64)
    else:
        row4 = np.array([], dtype=np.int64)
        col4 = np.array([], dtype=np.int64)
        data4 = np.array([], dtype=np.float64)

    J10 = csr_matrix((data4, (row4, col4)), shape=(rows4, n_2d * n_2d))
    T9_b4 = csr_matrix((rows4, n_2d))
    J9 = csr_matrix((rows4, n_2d * n_2d))
    J11 = csr_matrix((rows4, n_2d))
    J12 = csr_matrix((rows4, n_d * n_2d))

    block4 = hstack([T9_b4, J9, J10, J11, J12], format='csr')
    W37_blocks.append(block4)

    # ---------------- BLOCK 5 ----------------
    # rows5 = n_d * d2 * 2, condition: if i + j - 1 == l -> column l-1
    rows5 = n_d * d2 * 2
    i5 = np.repeat(np.arange(n_d, dtype=np.int64), d2 * 2)
    j5 = np.tile(np.repeat(np.arange(d2, dtype=np.int64), 2), n_d)
    q5 = np.tile(np.arange(2, dtype=np.int64), n_d * d2)

    l0 = i5 + j5
    mask5 = (l0 >= 0) & (l0 < n_2d)
    if mask5.any():
        rows_H15 = np.nonzero(mask5)[0].astype(np.int64)
        cols_H15 = l0[mask5].astype(np.int64)
        data_H15 = np.full(cols_H15.size, 1.0 / eps, dtype=np.float64)
    else:
        rows_H15 = np.array([], dtype=np.int64)
        cols_H15 = np.array([], dtype=np.int64)
        data_H15 = np.array([], dtype=np.float64)

    J15 = csr_matrix((data_H15, (rows_H15, cols_H15)), shape=(rows5, n_2d))
    T9_b5 = csr_matrix((rows5, n_2d))
    J13 = csr_matrix((rows5, n_2d * n_2d))
    J14 = csr_matrix((rows5, n_2d * n_2d))
    J16 = csr_matrix((rows5, n_d * n_2d))

    block5 = hstack([T9_b5, J13, J14, J15, J16], format='csr')
    W37_blocks.append(block5)

    # ---------------- BLOCK 6 ----------------
    # rows6 = n_d * d2 * 2, negative counterpart of block5
    rows6 = n_d * d2 * 2
    i6 = np.repeat(np.arange(n_d, dtype=np.int64), d2 * 2)
    j6 = np.tile(np.repeat(np.arange(d2, dtype=np.int64), 2), n_d)
    l0_6 = i6 + j6
    mask6 = (l0_6 >= 0) & (l0_6 < n_2d)
    if mask6.any():
        rows_H19 = np.nonzero(mask6)[0].astype(np.int64)
        cols_H19 = l0_6[mask6].astype(np.int64)
        data_H19 = np.full(cols_H19.size, -1.0 / eps, dtype=np.float64)
    else:
        rows_H19 = np.array([], dtype=np.int64)
        cols_H19 = np.array([], dtype=np.int64)
        data_H19 = np.array([], dtype=np.float64)

    J19 = csr_matrix((data_H19, (rows_H19, cols_H19)), shape=(rows6, n_2d))
    T9_b6 = csr_matrix((rows6, n_2d))
    J17 = csr_matrix((rows6, n_2d * n_2d))
    J18 = csr_matrix((rows6, n_2d * n_2d))
    J20 = csr_matrix((rows6, n_d * n_2d))

    block6 = hstack([T9_b6, J17, J18, J19, J20], format='csr')
    W37_blocks.append(block6)

    # ---------------- BLOCK 7 ----------------
    # rows7 = n_d * n_d * d2 * 2; condition l + j - 1 == p and i == k -> col = (k-1)*n_2d + (p-1)
    rows7 = n_d * n_d * d2 * 2
    i7 = np.repeat(np.arange(n_d, dtype=np.int64), n_d * d2 * 2)
    l7 = np.tile(np.repeat(np.arange(n_d, dtype=np.int64), d2 * 2), n_d)
    j7 = np.tile(np.repeat(np.arange(d2, dtype=np.int64), 2), n_d * n_d)
    q7 = np.tile(np.arange(2, dtype=np.int64), n_d * n_d * d2)

    p0 = l7 + j7
    mask7 = (p0 >= 0) & (p0 < n_2d)
    if mask7.any():
        rows_J24 = np.nonzero(mask7)[0].astype(np.int64)
        cols_J24 = (i7[mask7] * n_2d) + p0[mask7]   # k = i7, p = p0
        data_J24 = np.full(cols_J24.size, 1.0 / eps, dtype=np.float64)
    else:
        rows_J24 = np.array([], dtype=np.int64)
        cols_J24 = np.array([], dtype=np.int64)
        data_J24 = np.array([], dtype=np.float64)

    J24 = csr_matrix((data_J24, (rows_J24, cols_J24)), shape=(rows7, n_d * n_2d))
    T9_b7 = csr_matrix((rows7, n_2d))
    J21 = csr_matrix((rows7, n_2d * n_2d))
    J22 = csr_matrix((rows7, n_2d * n_2d))
    J23 = csr_matrix((rows7, n_2d))

    block7 = hstack([T9_b7, J21, J22, J23, J24], format='csr')
    W37_blocks.append(block7)

    # ---------------- BLOCK 8 ----------------
    # rows8 is same as rows7, negative counterpart
    rows8 = n_d * n_d * d2 * 2
    i8 = np.repeat(np.arange(n_d, dtype=np.int64), n_d * d2 * 2)
    l8 = np.tile(np.repeat(np.arange(n_d, dtype=np.int64), d2 * 2), n_d)
    j8 = np.tile(np.repeat(np.arange(d2, dtype=np.int64), 2), n_d * n_d)

    p0_8 = l8 + j8
    mask8 = (p0_8 >= 0) & (p0_8 < n_2d)
    if mask8.any():
        rows_J28 = np.nonzero(mask8)[0].astype(np.int64)
        cols_J28 = (i8[mask8] * n_2d) + p0_8[mask8]   # k = i8, p = p0_8
        data_J28 = np.full(cols_J28.size, -1.0 / eps, dtype=np.float64)
    else:
        rows_J28 = np.array([], dtype=np.int64)
        cols_J28 = np.array([], dtype=np.int64)
        data_J28 = np.array([], dtype=np.float64)

    J28 = csr_matrix((data_J28, (rows_J28, cols_J28)), shape=(rows8, n_d * n_2d))
    T9_b8 = csr_matrix((rows8, n_2d))
    J25 = csr_matrix((rows8, n_2d * n_2d))
    J26 = csr_matrix((rows8, n_2d * n_2d))
    J27 = csr_matrix((rows8, n_2d))

    block8 = hstack([T9_b8, J25, J26, J27, J28], format='csr')
    W37_blocks.append(block8)

    # final assembly
    W37 = vstack(W37_blocks, format='csr')
    return W37


def build_B37(n, d, eps, B):
    
    n_2d = n + 2*d
    n_d = n + d
    d2 = d + 1  # CORRECTED: d+1 not d+2
    
    # Calculate sizes
    rows_U2 = n_2d
    rows_tau2 = n_2d * n_2d
    rows_psi1_psi2 = n_d * n_2d * d2 * 2
    rows_rho1_rho2 = n_d * n_2d * d2 * 2
    rows_psi3_psi4 = n_d * d2 * 2
    rows_rho3_rho4 = n_d * d2 * 2
    rows_psi5_psi6 = n_d * n_d * d2 * 2
    rows_rho5_rho6 = n_d * n_d * d2 * 2

    total_rows = (rows_U2 + rows_tau2 + rows_psi1_psi2 + rows_rho1_rho2 +
                  rows_psi3_psi4 + rows_rho3_rho4 + rows_psi5_psi6 + rows_rho5_rho6)

    #print(f"B37 size breakdown:")
    #print(f"  U2: {rows_U2}, tau2: {rows_tau2}")
    #print(f"  psi1_psi2: {rows_psi1_psi2}, rho1_rho2: {rows_rho1_rho2}")
    #print(f"  psi3_psi4: {rows_psi3_psi4}, rho3_rho4: {rows_rho3_rho4}")
    #print(f"  psi5_psi6: {rows_psi5_psi6}, rho5_rho6: {rows_rho5_rho6}")
    #print(f"  TOTAL: {total_rows}")

    B37 = np.zeros(total_rows)
    idx = 0


    # Section 1: U2 nodes (all zeros)
    B37[idx:idx + rows_U2] = 0.0
    idx += rows_U2

    # Section 2: tau2 nodes (all zeros)
    B37[idx:idx + rows_tau2] = 0.0
    idx += rows_tau2

    # Section 3: psi1 psi2 nodes
    for i in range(1, n_d + 1):
        for l in range(1, n_2d + 1):
            for j in range(1, d2 + 1):
                B37[idx] = -i*B/eps + 1  # q=0
                idx += 1
                B37[idx] = -i*B/eps      # q=1
                idx += 1

    # Section 4: rho1 rho2 nodes
    for i in range(1, n_d + 1):
        for l in range(1, n_2d + 1):
            for j in range(1, d2 + 1):
                B37[idx] = (i*B + 1)/eps + 1  # q=0
                idx += 1
                B37[idx] = (i*B + 1)/eps      # q=1
                idx += 1

    # Section 5: psi3 psi4 nodes
    for i in range(1, n_d + 1):
        for j in range(1, d2 + 1):
            B37[idx] = -i*B/eps + 1  # q=0
            idx += 1
            B37[idx] = -i*B/eps      # q=1
            idx += 1

    # Section 6: rho3 rho4 nodes
    for i in range(1, n_d + 1):
        for j in range(1, d2 + 1):
            B37[idx] = (i*B + 1)/eps + 1  # q=0
            idx += 1
            B37[idx] = (i*B + 1)/eps      # q=1
            idx += 1

    # Section 7: psi5 psi6 nodes
    for i in range(1, n_d + 1):
        for l in range(1, n_d + 1):
            for j in range(1, d2 + 1):
                B37[idx] = -l*B/eps + 1  # q=0
                idx += 1
                B37[idx] = -l*B/eps      # q=1
                idx += 1

    # Section 8: rho5 rho6 nodes
    for i in range(1, n_d + 1):
        for l in range(1, n_d + 1):
            for j in range(1, d2 + 1):
                B37[idx] = (l*B + 1)/eps + 1  # q=0
                idx += 1
                B37[idx] = (l*B + 1)/eps      # q=1
                idx += 1

    return B37