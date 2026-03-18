# -*- coding: utf-8 -*-
"""
Weights and bias of layer 35_c

@author: Ghafoor
"""
import numpy as np
from scipy.sparse import coo_matrix
import sys

from scipy.sparse import eye, csr_matrix, hstack, vstack, lil_matrix
import numpy as np

import numpy as np
from scipy.sparse import csr_matrix, hstack, vstack

import numpy as np


from scipy.sparse import csr_matrix, hstack, vstack

from scipy.sparse import csr_matrix, hstack, vstack

import numpy as np
from scipy.sparse import csr_matrix, hstack
import numpy as np
from scipy.sparse import csr_matrix, hstack, vstack, lil_matrix


import numpy as np
from scipy.sparse import csr_matrix, hstack, vstack

import numpy as np
from scipy.sparse import csr_matrix, hstack, vstack


import numpy as np
from scipy.sparse import csr_matrix, coo_matrix, hstack, vstack


def build_W35(n, d, eps):
    n_2d = n + 2 * d
    n_d = n + d

    blocks = []

    # ---------- BLOCK 1 ----------
    rows = n_2d
    T7 = csr_matrix((np.ones(rows), (np.arange(rows), np.arange(rows))),
                    shape=(rows, n_2d))
    blocks.append(hstack([
        T7,
        csr_matrix((rows, n_2d * n_2d)),
        csr_matrix((rows, n_2d * n_2d)),
        csr_matrix((rows, n_2d)),
        csr_matrix((rows, n_d * n_2d))
    ], format="csr"))

    # ---------- BLOCK 2 ----------
    rows = n_2d * n_2d
    i = np.repeat(np.arange(n_2d), n_2d)
    l = np.tile(np.arange(n_2d), n_2d)
    I17 = csr_matrix((np.ones(rows), (np.arange(rows), i * n_2d + l)),
                     shape=(rows, n_2d * n_2d))

    blocks.append(hstack([
        csr_matrix((rows, n_2d)),
        I17,
        csr_matrix((rows, n_2d * n_2d)),
        csr_matrix((rows, n_2d)),
        csr_matrix((rows, n_d * n_2d))
    ], format="csr"))

    # ---------- BLOCK 3 (triangular k ≤ i) ----------
    rows = n_2d * n_2d
    row_idx = []
    col_idx = []

    for i in range(n_2d):
        base_rows = i * n_2d + np.arange(n_2d)
        for k in range(i + 1):
            row_idx.append(base_rows)
            col_idx.append(k * n_2d + np.arange(n_2d))

    row_idx = np.concatenate(row_idx)
    col_idx = np.concatenate(col_idx)

    I22 = coo_matrix(
        (np.ones(len(row_idx)), (row_idx, col_idx)),
        shape=(rows, n_2d * n_2d)
    ).tocsr()

    blocks.append(hstack([
        csr_matrix((rows, n_2d)),
        csr_matrix((rows, n_2d * n_2d)),
        I22,
        csr_matrix((rows, n_2d)),
        csr_matrix((rows, n_d * n_2d))
    ], format="csr"))

    # ---------- BLOCK 4 ----------
    rows = n_2d
    I27 = csr_matrix(
        (np.ones(n_2d * (n_2d + 1) // 2),
         (np.repeat(np.arange(n_2d), np.arange(1, n_2d + 1)),
          np.concatenate([np.arange(i + 1) for i in range(n_2d)]))),
        shape=(rows, n_2d)
    )

    blocks.append(hstack([
        csr_matrix((rows, n_2d)),
        csr_matrix((rows, n_2d * n_2d)),
        csr_matrix((rows, n_2d * n_2d)),
        I27,
        csr_matrix((rows, n_d * n_2d))
    ], format="csr"))

    # ---------- BLOCKS 5–6 (±1/eps diagonal replication) ----------
    for sign in (+1, -1):
        rows = n_2d * n_2d * 2
        col = np.repeat(np.arange(n_2d * n_2d), 2)
        D = csr_matrix(
            (np.full(rows, sign / eps),
             (np.arange(rows), col)),
            shape=(rows, n_2d * n_2d)
        )
        blocks.append(hstack([
            csr_matrix((rows, n_2d)),
            csr_matrix((rows, n_2d * n_2d)),
            D,
            csr_matrix((rows, n_2d)),
            csr_matrix((rows, n_d * n_2d))
        ], format="csr"))

    # ---------- BLOCKS 7–8 (±1/eps on n_2d) ----------
    for sign in (+1, -1):
        rows = n_2d * 2
        col = np.repeat(np.arange(n_2d), 2)
        D = csr_matrix(
            (np.full(rows, sign / eps),
             (np.arange(rows), col)),
            shape=(rows, n_2d)
        )
        blocks.append(hstack([
            csr_matrix((rows, n_2d)),
            csr_matrix((rows, n_2d * n_2d)),
            csr_matrix((rows, n_2d * n_2d)),
            D,
            csr_matrix((rows, n_d * n_2d))
        ], format="csr"))

    # ---------- BLOCK 9 (triangular k ≤ l) ----------
    rows = n_d * n_2d
    row_idx = []
    col_idx = []

    for l in range(n_2d):
        for k in range(l + 1):
            row_idx.append(np.arange(n_d) * n_2d + l)
            col_idx.append(np.arange(n_d) * n_2d + k)

    row_idx = np.concatenate(row_idx)
    col_idx = np.concatenate(col_idx)

    D39 = coo_matrix(
        (np.ones(len(row_idx)), (row_idx, col_idx)),
        shape=(rows, n_d * n_2d)
    ).tocsr()

    blocks.append(hstack([
        csr_matrix((rows, n_2d)),
        csr_matrix((rows, n_2d * n_2d)),
        csr_matrix((rows, n_2d * n_2d)),
        csr_matrix((rows, n_2d)),
        D39
    ], format="csr"))

    # ---------- BLOCKS 10–11 (±1/eps) ----------
    for sign in (+1, -1):
        rows = n_d * n_2d * 2
        col = np.repeat(np.arange(n_d * n_2d), 2)
        D = csr_matrix(
            (np.full(rows, sign / eps),
             (np.arange(rows), col)),
            shape=(rows, n_d * n_2d)
        )
        blocks.append(hstack([
            csr_matrix((rows, n_2d)),
            csr_matrix((rows, n_2d * n_2d)),
            csr_matrix((rows, n_2d * n_2d)),
            csr_matrix((rows, n_2d)),
            D
        ], format="csr"))

    return vstack(blocks, format="csr")


def build_B35(n, d):
    """Build B35 as sparse vector."""
    from scipy.sparse import csr_matrix
    import numpy as np
    
    n_2d = n + 2*d
    
    # Calculate section sizes
    # 1. U2 nodes: n_2d entries (all zeros)
    size_u2 = n_2d
    
    # 2. tau2 nodes: n_2d * n_2d entries (all zeros)
    size_tau2 = n_2d * n_2d
    
    # 3. gamma5 nodes: n_2d * n_2d entries (all zeros)
    size_gamma5 = n_2d * n_2d
    
    # 4. gamma6 nodes: n_2d entries (all zeros)
    size_gamma6 = n_2d
    
    # 5. lambda1, lambda2 nodes (delta(e_il,0)): n_2d * n_2d * 2 entries
    #    Pattern: when q == 0 -> 1, when q == 1 -> 0
    size_lambda12 = n_2d * n_2d * 2
    
    # 6. mu1, mu2 nodes (delta(e_il,0)): n_2d * n_2d * 2 entries
    #    Pattern: when q == 0 -> 1, when q == 1 -> 0
    size_mu12 = n_2d * n_2d * 2
    
    # 7. lambda3, lambda4 nodes (delta(e'_i,0)): n_2d * 2 entries
    #    Pattern: when q == 0 -> 1, when q == 1 -> 0
    size_lambda34 = n_2d * 2
    
    # 8. mu3, mu4 nodes (delta(e'_i,0)): n_2d * 2 entries
    #    Pattern: when q == 0 -> 1, when q == 1 -> 0
    size_mu34 = n_2d * 2
    
    # 9. gamma7 nodes: (n+d) * n_2d entries (all zeros)
    size_gamma7 = (n + d) * n_2d
    
    # 10. lambda5, lambda6 nodes (delta(p_il,0)): (n+d) * n_2d * 2 entries
    #     Pattern: when q == 0 -> 1, when q == 1 -> 0
    size_lambda56 = (n + d) * n_2d * 2
    
    # 11. mu5, mu6 nodes (delta(p_il,0)): (n+d) * n_2d * 2 entries
    #     Pattern: when q == 0 -> 1, when q == 1 -> 0
    size_mu56 = (n + d) * n_2d * 2
    
    total_size = (size_u2 + size_tau2 + size_gamma5 + size_gamma6 + 
                  size_lambda12 + size_mu12 + size_lambda34 + size_mu34 + 
                  size_gamma7 + size_lambda56 + size_mu56)
    
    #print(f"B35 total size calculated: {total_size}")
    #print(f"Sections: u2={size_u2}, tau2={size_tau2}, gamma5={size_gamma5}, gamma6={size_gamma6}")
    #print(f"lambda12={size_lambda12}, mu12={size_mu12}, lambda34={size_lambda34}, mu34={size_mu34}")
    #print(f"gamma7={size_gamma7}, lambda56={size_lambda56}, mu56={size_mu56}")
    
    # Non-zero sections: lambda12, mu12, lambda34, mu34, lambda56, mu56
    # For each pair section (q in range(2)): half are 1s (q==0), half are 0s (q==1)
    
    # Calculate non-zero count: for each pair section, half are 1s
    non_zero_count = (size_lambda12 // 2 + size_mu12 // 2 + 
                      size_lambda34 // 2 + size_mu34 // 2 + 
                      size_lambda56 // 2 + size_mu56 // 2)
    
    data = np.zeros(non_zero_count, dtype=np.float64)
    rows = np.zeros(non_zero_count, dtype=np.int32)
    
    idx = 0  # Current position in the full vector
    data_idx = 0  # Current position in the data/rows arrays
    
    # Skip zeros sections: u2, tau2, gamma5, gamma6
    idx += size_u2 + size_tau2 + size_gamma5 + size_gamma6
    
    # lambda12 section (pattern: q==0 -> 1, q==1 -> 0)
    for i in range(size_lambda12):
        if i % 2 == 0:  # q == 0
            data[data_idx] = 1.0
            rows[data_idx] = idx
            data_idx += 1
        idx += 1
    
    # mu12 section (pattern: q==0 -> 1, q==1 -> 0)
    for i in range(size_mu12):
        if i % 2 == 0:  # q == 0
            data[data_idx] = 1.0
            rows[data_idx] = idx
            data_idx += 1
        idx += 1
    
    # lambda34 section (pattern: q==0 -> 1, q==1 -> 0)
    for i in range(size_lambda34):
        if i % 2 == 0:  # q == 0
            data[data_idx] = 1.0
            rows[data_idx] = idx
            data_idx += 1
        idx += 1
    
    # mu34 section (pattern: q==0 -> 1, q==1 -> 0)
    for i in range(size_mu34):
        if i % 2 == 0:  # q == 0
            data[data_idx] = 1.0
            rows[data_idx] = idx
            data_idx += 1
        idx += 1
    
    # Skip gamma7 zeros section
    idx += size_gamma7
    
    # lambda56 section (pattern: q==0 -> 1, q==1 -> 0)
    for i in range(size_lambda56):
        if i % 2 == 0:  # q == 0
            data[data_idx] = 1.0
            rows[data_idx] = idx
            data_idx += 1
        idx += 1
    
    # mu56 section (pattern: q==0 -> 1, q==1 -> 0)
    for i in range(size_mu56):
        if i % 2 == 0:  # q == 0
            data[data_idx] = 1.0
            rows[data_idx] = idx
            data_idx += 1
        idx += 1
    
    # Create sparse column vector
    B35 = csr_matrix((data, (rows, np.zeros(len(data), dtype=np.int32))), 
                     shape=(total_size, 1))
    
    #print(f"B35 shape: {B35.shape}, Non-zeros: {B35.nnz}")
    return B35