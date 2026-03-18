# -*- coding: utf-8 -*-
"""
Weights and bias of layer 4

@author: Ghafoor
"""


import numpy as np
from scipy.sparse import csr_matrix

def build_W4(d, eps):
    rows = []
    cols = []
    data = []
    
    current_row = 0
    W_cols = 7 * d   # total columns

    # ----- helper for adding a single non-zero -----
    def add(r, c, val):
        rows.append(r)
        cols.append(c)
        data.append(val)

    # ======================================================
    # 1. SUBSTITUTION
    # ======================================================

    # ψ 11,12 nodes
    for k in range(1, d + 1):
        for l in range(1, d + 1):
            for q in range(2):
                if k != l:
                    add(current_row, l - 1, 1/eps)
                    add(current_row, k - 1, -1/eps)
                current_row += 1

    # ρ 11,12 nodes
    for k in range(1, d + 1):
        for l in range(1, d + 1):
            for q in range(2):
                if k != l:
                    add(current_row, l - 1, -1/eps)
                    add(current_row, k - 1,  1/eps)
                current_row += 1

    # identity map x_j
    for k in range(1, 2*d + 1):
        add(current_row, k - 1, 1)
        current_row += 1

    # ======================================================
    # 2. INSERTION
    # ======================================================

    # η1, η2
    for j in range(1, d + 1):
        for q in range(2):
            add(current_row, j + 2*d - 1,  1/eps)
            add(current_row, j + 3*d - 1, -1/eps)
            current_row += 1

    # η3, η4
    for j in range(1, d + 1):
        for q in range(2):
            add(current_row, j + 2*d - 1, -1/eps)
            add(current_row, j + 3*d - 1,  1/eps)
            current_row += 1

    # α1, α2 : delta(x_j, x_k)
    for j in range(1, d + 1):
        for k in range(1, d + 1):
            for q in range(2):
                if j != k:
                    add(current_row, j + 2*d - 1,  1/eps)
                    add(current_row, k + 2*d - 1, -1/eps)
                current_row += 1

    # β1, β2
    for j in range(1, d + 1):
        for k in range(1, d + 1):
            for q in range(2):
                if j != k:
                    add(current_row, j + 2*d - 1, -1/eps)
                    add(current_row, k + 2*d - 1,  1/eps)
                current_row += 1

    # α'1, α'2 : x_j+d
    for j in range(d+1, 2*d+1):
        for k in range(d+1, 2*d+1):
            for q in range(2):
                if j != k:
                    add(current_row, j + 2*d - 1,  1/eps)
                    add(current_row, k + 2*d - 1, -1/eps)
                current_row += 1

    # β'1, β'2
    for j in range(d+1, 2*d+1):
        for k in range(d+1, 2*d+1):
            for q in range(2):
                if j != k:
                    add(current_row, j + 2*d - 1, -1/eps)
                    add(current_row, k + 2*d - 1,  1/eps)
                current_row += 1

    # identity (insertion)
    for j in range(1, 3*d + 1):
        add(current_row, j + 2*d - 1, 1)
        current_row += 1

    # ======================================================
    # 3. DELETION
    # ======================================================

    # α'1 α'2 : x'_j
    for j in range(1, d + 1):
        for k in range(1, d + 1):
            for q in range(2):
                if j != k:
                    add(current_row, j + 5*d - 1,  1/eps)
                    add(current_row, k + 5*d - 1, -1/eps)
                current_row += 1

    # β'1 β'2
    for j in range(1, d + 1):
        for k in range(1, d + 1):
            for q in range(2):
                if j != k:
                    add(current_row, j + 5*d - 1, -1/eps)
                    add(current_row, k + 5*d - 1,  1/eps)
                current_row += 1

    # α''1 α''2
    for j in range(d+1, 2*d+1):
        for k in range(d+1, 2*d+1):
            for q in range(2):
                if j != k:
                    add(current_row, j + 5*d - 1,  1/eps)
                    add(current_row, k + 5*d - 1, -1/eps)
                current_row += 1

    # β''1 β''2
    for j in range(d+1, 2*d+1):
        for k in range(d+1, 2*d+1):
            for q in range(2):
                if j != k:
                    add(current_row, j + 5*d - 1, -1/eps)
                    add(current_row, k + 5*d - 1,  1/eps)
                current_row += 1

    # identity (deletion)
    for j in range(1, 2*d + 1):
        add(current_row, j + 5*d - 1, 1)
        current_row += 1

    # --- build sparse matrix ---
    W4_sparse = csr_matrix((data, (rows, cols)), shape=(current_row, W_cols))
    return W4_sparse




def build_B4(d):
    total_length = (d * d * 2) + (d * d * 2) + (2 * d) + (d * 2) + (d * 2) + (d * d * 2) + (d * d * 2) + (d * d * 2) + (d * d * 2) + (3 * d) + (d * d * 2) + (d * d * 2) + (d * d * 2) + (d * d * 2) + (2 * d)
    
    B4 = np.zeros(total_length)
    idx = 0
    
    # psi nodes
    for k in range(1, d+1):
        for l in range(1, d+1):
            for q in range(2):
                if q == 0:
                    B4[idx] = 1
                idx += 1
    
    # rho nodes
    for k in range(1, d+1):
        for l in range(1, d+1):
            for q in range(2):
                if q == 0:
                    B4[idx] = 1
                idx += 1
    
    # identity map x_j (zeros)
    idx += 2 * d
    
    # eta1, eta2
    for j in range(1, d+1):
        for q in range(2):
            if q == 0:
                B4[idx] = 1
            idx += 1
    
    # eta3, eta4
    for j in range(1, d+1):
        for q in range(2):
            if q == 0:
                B4[idx] = 1
            idx += 1
    
    # alpha1, alpha2
    for j in range(1, d+1):
        for k in range(1, d+1):
            for q in range(2):
                if q == 0 and j != k:
                    B4[idx] = 1
                idx += 1
    
    # beta1, beta2
    for j in range(1, d+1):
        for k in range(1, d+1):
            for q in range(2):
                if q == 0 and j != k:
                    B4[idx] = 1
                idx += 1
    
    # alpha'1, alpha'2
    for j in range(d+1, 2*d+1):
        for k in range(d+1, 2*d+1):
            for q in range(2):
                if q == 0 and j != k:
                    B4[idx] = 1
                idx += 1
    
    # beta'1, beta'2
    for j in range(d+1, 2*d+1):
        for k in range(d+1, 2*d+1):
            for q in range(2):
                if q == 0 and j != k:
                    B4[idx] = 1
                idx += 1
    
    # identity map x_j (zeros)
    idx += 3 * d
    
    # alpha'1, alpha'2 (deletion)
    for j in range(1, d+1):
        for k in range(1, d+1):
            for q in range(2):
                if q == 0 and j != k:
                    B4[idx] = 1
                idx += 1
    
    # beta'1, beta'2 (deletion)
    for j in range(1, d+1):
        for k in range(1, d+1):
            for q in range(2):
                if q == 0 and j != k:
                    B4[idx] = 1
                idx += 1
    
    # alpha''1, alpha''2
    for j in range(d+1, 2*d+1):
        for k in range(d+1, 2*d+1):
            for q in range(2):
                if q == 0 and j != k:
                    B4[idx] = 1
                idx += 1
    
    # beta''1, beta''2
    for j in range(d+1, 2*d+1):
        for k in range(d+1, 2*d+1):
            for q in range(2):
                if q == 0 and j != k:
                    B4[idx] = 1
                idx += 1
    
    # identity map x'_j (zeros)
    idx += 2 * d
    
    return B4