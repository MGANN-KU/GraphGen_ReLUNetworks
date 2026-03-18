# -*- coding: utf-8 -*-
"""
Weights and bias of layer 25

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

def build_W25_block1_sparse(n, d):
    rows = n + 2*d
    n_2d = rows
    
    # N1: Identity matrix
    N1_data = np.ones(rows, dtype=np.float64)
    N1_row = np.arange(rows)
    N1_col = np.arange(rows)
    N1 = csr_matrix((N1_data, (N1_row, N1_col)), shape=(rows, n_2d))
    
    # All others are zero matrices
    N2 = csr_matrix((rows, n_2d * d))
    N3 = csr_matrix((rows, n_2d * n_2d))
    N4 = csr_matrix((rows, n_2d * n_2d))
    N5 = csr_matrix((rows, n_2d))
    N6 = csr_matrix((rows, n_2d))
    N7 = csr_matrix((rows, d * 2))
    N8 = csr_matrix((rows, d * 2))
    N9 = csr_matrix((rows, n_2d * d * 2))
    N10 = csr_matrix((rows, n_2d * d * 2))
    N11 = csr_matrix((rows, n_2d * d * 2))
    N12 = csr_matrix((rows, n_2d * d * 2))
    N13 = csr_matrix((rows, n_2d * d * 2))
    N14 = csr_matrix((rows, n_2d * d * 2))
    N15 = csr_matrix((rows, n_2d * d * 2))
    N16 = csr_matrix((rows, n_2d * d * 2))
    N17 = csr_matrix((rows, 2*d))
    
    block1 = hstack([N1, N2, N3, N4, N5, N6, N7, N8, N9, N10, N11, N12, N13, N14, N15, N16, N17], format='csr')
    return block1

def build_W25_block2_sparse(n, d):
    rows = n + 2*d
    n_2d = rows
    
    # N1: All zeros
    N1 = csr_matrix((rows, n_2d))
    
    # N2: Identity pattern for each i,k with j loop
    N2_nnz = rows * d
    N2_data = np.ones(N2_nnz, dtype=np.float64)
    N2_row = np.repeat(np.arange(rows), d)
    N2_col = np.arange(N2_nnz)
    N2 = csr_matrix((N2_data, (N2_row, N2_col)), shape=(rows, n_2d * d))
    
    # All others are zero matrices
    N3 = csr_matrix((rows, n_2d * n_2d))
    N4 = csr_matrix((rows, n_2d * n_2d))
    N5 = csr_matrix((rows, n_2d))
    N6 = csr_matrix((rows, n_2d))
    N7 = csr_matrix((rows, d * 2))
    N8 = csr_matrix((rows, d * 2))
    N9 = csr_matrix((rows, n_2d * d * 2))
    N10 = csr_matrix((rows, n_2d * d * 2))
    N11 = csr_matrix((rows, n_2d * d * 2))
    N12 = csr_matrix((rows, n_2d * d * 2))
    N13 = csr_matrix((rows, n_2d * d * 2))
    N14 = csr_matrix((rows, n_2d * d * 2))
    N15 = csr_matrix((rows, n_2d * d * 2))
    N16 = csr_matrix((rows, n_2d * d * 2))
    N17 = csr_matrix((rows, 2*d))
    
    block2 = hstack([N1, N2, N3, N4, N5, N6, N7, N8, N9, N10, N11, N12, N13, N14, N15, N16, N17], format='csr')
    return block2

def build_W25_block3_sparse(n, d):
    rows = (n + 2*d) * (n + 2*d)
    n_2d = n + 2*d
    
    # N1, N2: All zeros
    N1 = csr_matrix((rows, n_2d))
    N2 = csr_matrix((rows, n_2d * d))
    
    # N3: Identity at (i==k and p==j) positions
    N3_nnz = rows
    N3_data = np.ones(N3_nnz, dtype=np.float64)
    N3_row = np.arange(rows)
    N3_col = np.arange(rows)
    N3 = csr_matrix((N3_data, (N3_row, N3_col)), shape=(rows, n_2d * n_2d))
    
    # N4: Same as N3
    N4 = N3.copy()
    
    # All others are zero matrices
    N5 = csr_matrix((rows, n_2d))
    N6 = csr_matrix((rows, n_2d))
    N7 = csr_matrix((rows, d * 2))
    N8 = csr_matrix((rows, d * 2))
    N9 = csr_matrix((rows, n_2d * d * 2))
    N10 = csr_matrix((rows, n_2d * d * 2))
    N11 = csr_matrix((rows, n_2d * d * 2))
    N12 = csr_matrix((rows, n_2d * d * 2))
    N13 = csr_matrix((rows, n_2d * d * 2))
    N14 = csr_matrix((rows, n_2d * d * 2))
    N15 = csr_matrix((rows, n_2d * d * 2))
    N16 = csr_matrix((rows, n_2d * d * 2))
    N17 = csr_matrix((rows, 2*d))
    
    block3 = hstack([N1, N2, N3, N4, N5, N6, N7, N8, N9, N10, N11, N12, N13, N14, N15, N16, N17], format='csr')
    return block3

def build_W25_block4_sparse(n, d):
    rows = n + 2*d
    n_2d = rows
    
    # N1-N4: All zeros
    N1 = csr_matrix((rows, n_2d))
    N2 = csr_matrix((rows, n_2d * d))
    N3 = csr_matrix((rows, n_2d * n_2d))
    N4 = csr_matrix((rows, n_2d * n_2d))
    
    # N5: Identity matrix
    N5_data = np.ones(rows, dtype=np.float64)
    N5_row = np.arange(rows)
    N5_col = np.arange(rows)
    N5 = csr_matrix((N5_data, (N5_row, N5_col)), shape=(rows, n_2d))
    
    # All others are zero matrices
    N6 = csr_matrix((rows, n_2d))
    N7 = csr_matrix((rows, d * 2))
    N8 = csr_matrix((rows, d * 2))
    N9 = csr_matrix((rows, n_2d * d * 2))
    N10 = csr_matrix((rows, n_2d * d * 2))
    N11 = csr_matrix((rows, n_2d * d * 2))
    N12 = csr_matrix((rows, n_2d * d * 2))
    N13 = csr_matrix((rows, n_2d * d * 2))
    N14 = csr_matrix((rows, n_2d * d * 2))
    N15 = csr_matrix((rows, n_2d * d * 2))
    N16 = csr_matrix((rows, n_2d * d * 2))
    N17 = csr_matrix((rows, 2*d))
    
    block4 = hstack([N1, N2, N3, N4, N5, N6, N7, N8, N9, N10, N11, N12, N13, N14, N15, N16, N17], format='csr')
    return block4

def build_W25_block5_sparse(n, d):
    rows = n + 2*d
    n_2d = rows
    
    # N1-N5: All zeros
    N1 = csr_matrix((rows, n_2d))
    N2 = csr_matrix((rows, n_2d * d))
    N3 = csr_matrix((rows, n_2d * n_2d))
    N4 = csr_matrix((rows, n_2d * n_2d))
    N5 = csr_matrix((rows, n_2d))
    
    # N6: Identity matrix
    N6_data = np.ones(rows, dtype=np.float64)
    N6_row = np.arange(rows)
    N6_col = np.arange(rows)
    N6 = csr_matrix((N6_data, (N6_row, N6_col)), shape=(rows, n_2d))
    
    # All others are zero matrices
    N7 = csr_matrix((rows, d * 2))
    N8 = csr_matrix((rows, d * 2))
    N9 = csr_matrix((rows, n_2d * d * 2))
    N10 = csr_matrix((rows, n_2d * d * 2))
    N11 = csr_matrix((rows, n_2d * d * 2))
    N12 = csr_matrix((rows, n_2d * d * 2))
    N13 = csr_matrix((rows, n_2d * d * 2))
    N14 = csr_matrix((rows, n_2d * d * 2))
    N15 = csr_matrix((rows, n_2d * d * 2))
    N16 = csr_matrix((rows, n_2d * d * 2))
    N17 = csr_matrix((rows, 2*d))
    
    block5 = hstack([N1, N2, N3, N4, N5, N6, N7, N8, N9, N10, N11, N12, N13, N14, N15, N16, N17], format='csr')
    return block5

def build_W25_block6_sparse(n, d):
    # Total rows from original loops:
    # for i in range(n+2d) → (n+2d)
    # for l in range(n+2d) → (n+2d)
    # for j in range(d)    → d
    rows = (n + 2*d) * (n + 2*d) * d
    n_2d = n + 2*d

    # -----------------------------
    # N1–N6 are all zeros in dense version
    # -----------------------------
    N1 = csr_matrix((rows, n_2d))
    N2 = csr_matrix((rows, n_2d * d))
    N3 = csr_matrix((rows, n_2d * n_2d))
    N4 = csr_matrix((rows, n_2d * n_2d))
    N5 = csr_matrix((rows, n_2d))
    N6 = csr_matrix((rows, n_2d))

    # -----------------------------
    # N7: Depends ONLY on j (NOT on i or l)
    #
    # Original nested loops:
    # for i:
    #   for l:
    #     for j:
    #
    # That means j increases EVERY row → j = row % d
    # -----------------------------
    N7_data = []
    N7_row = []
    N7_col = []

    for row in range(rows):
        j = row % d  # FIXED: correct mapping for N7

        # Only when k == j, so we only place 2 entries at positions (2*j, 2*j+1)
        for q in range(2):
            N7_row.append(row)
            N7_col.append(j * 2 + q)
            if q == 0:
                N7_data.append(-1.0)
            else:
                N7_data.append(1.0)

    N7 = csr_matrix((N7_data, (N7_row, N7_col)), shape=(rows, d * 2))

    # -----------------------------
    # N8 is identical to N7 in dense code
    # -----------------------------
    N8 = N7.copy()

    # -----------------------------
    # N9: Values only when i == p and j == k
    # Must map (i,l,j) from row index:
    #
    # i changes slowest → step d*(n+2d)
    # l changes next    → step d
    # j changes fastest → step 1
    # -----------------------------
    N9_data = []
    N9_row = []
    N9_col = []

    for row in range(rows):
        i = row // (n_2d * d)
        l = (row // d) % n_2d   # computed for consistency (not used here)
        j = row % d

        # Only p=i and k=j → block position:
        base_col = i * (d * 2) + j * 2

        for q in range(2):
            N9_row.append(row)
            N9_col.append(base_col + q)
            if q == 0:
                N9_data.append(1.0)
            else:
                N9_data.append(-1.0)

    N9 = csr_matrix((N9_data, (N9_row, N9_col)), shape=(rows, n_2d * d * 2))

    # -----------------------------
    # N10 identical to N9
    # -----------------------------
    N10 = N9.copy()

    # -----------------------------
    # N11: Values only when l == p and j == k
    # Here p corresponds to l index
    # -----------------------------
    N11_data = []
    N11_row = []
    N11_col = []

    for row in range(rows):
        i = row // (n_2d * d)
        l = (row // d) % n_2d     # <- USED here
        j = row % d

        base_col = l * (d * 2) + j * 2

        for q in range(2):
            N11_row.append(row)
            N11_col.append(base_col + q)
            if q == 0:
                N11_data.append(1.0)
            else:
                N11_data.append(-1.0)

    N11 = csr_matrix((N11_data, (N11_row, N11_col)), shape=(rows, n_2d * d * 2))

    # -----------------------------
    # N12 identical to N11
    # -----------------------------
    N12 = N11.copy()

    # -----------------------------
    # N13–N16: Always zeros in dense
    # -----------------------------
    N13 = csr_matrix((rows, n_2d * d * 2))
    N14 = csr_matrix((rows, n_2d * d * 2))
    N15 = csr_matrix((rows, n_2d * d * 2))
    N16 = csr_matrix((rows, n_2d * d * 2))

    # -----------------------------
    # N17: zeros (same size as dense)
    # -----------------------------
    N17 = csr_matrix((rows, 2*d))

    # -----------------------------
    # Concatenate horizontally (same order as dense)
    # -----------------------------
    block6 = hstack([
        N1, N2, N3, N4, N5, N6,
        N7, N8, N9, N10, N11, N12,
        N13, N14, N15, N16, N17
    ], format='csr')

    return block6


def build_W25_block7_sparse(n, d):
    rows = (n + 2*d) * (n + 2*d) * d
    n_2d = n + 2*d
    
    # N1-N6: All zeros
    N1 = csr_matrix((rows, n_2d))
    N2 = csr_matrix((rows, n_2d * d))
    N3 = csr_matrix((rows, n_2d * n_2d))
    N4 = csr_matrix((rows, n_2d * n_2d))
    N5 = csr_matrix((rows, n_2d))
    N6 = csr_matrix((rows, n_2d))
    
    # N7: Values when j == k - FIXED: j = row % d
    N7_data = []
    N7_row = []
    N7_col = []
    
    for row in range(rows):
        j = row % d  # FIXED!
        # Only for k = j
        for q in range(2):
            N7_row.append(row)
            N7_col.append(j * 2 + q)
            if q == 0:
                N7_data.append(-1.0)
            else:
                N7_data.append(1.0)
    
    N7 = csr_matrix((N7_data, (N7_row, N7_col)), shape=(rows, d * 2))
    
    # N8: Same as N7
    N8 = N7.copy()
    
    # N9-N12: All zeros
    N9 = csr_matrix((rows, n_2d * d * 2))
    N10 = csr_matrix((rows, n_2d * d * 2))
    N11 = csr_matrix((rows, n_2d * d * 2))
    N12 = csr_matrix((rows, n_2d * d * 2))
    
    # N13: Values when l == p and j == k - Need to fix j calculation too
    N13_data = []
    N13_row = []
    N13_col = []
    
    for row in range(rows):
        l = (row // d) % n_2d  # FIXED: consistent with Block 6
        j = row % d  # FIXED!
        # Only when p = l and k = j
        base_col = l * (d * 2) + j * 2
        for q in range(2):
            N13_row.append(row)
            N13_col.append(base_col + q)
            if q == 0:
                N13_data.append(1.0)
            else:
                N13_data.append(-1.0)
    
    N13 = csr_matrix((N13_data, (N13_row, N13_col)), shape=(rows, n_2d * d * 2))
    
    # N14: Same as N13
    N14 = N13.copy()
    
    # N15: Values when i == p and j == k - Need to fix j calculation too
    N15_data = []
    N15_row = []
    N15_col = []
    
    for row in range(rows):
        i = row // (n_2d * d)
        j = row % d  # FIXED!
        # Only when p = i and k = j
        base_col = i * (d * 2) + j * 2
        for q in range(2):
            N15_row.append(row)
            N15_col.append(base_col + q)
            if q == 0:
                N15_data.append(1.0)
            else:
                N15_data.append(-1.0)
    
    N15 = csr_matrix((N15_data, (N15_row, N15_col)), shape=(rows, n_2d * d * 2))
    
    # N16: Same as N15
    N16 = N15.copy()
    
    # N17: All zeros
    N17 = csr_matrix((rows, 2*d))
    
    block7 = hstack([N1, N2, N3, N4, N5, N6, N7, N8, N9, N10, N11, N12, N13, N14, N15, N16, N17], format='csr')
    return block7
    
def build_W25_block8_sparse(n, d):
    rows = n * d
    n_2d = n + 2*d
    
    # N1-N8: All zeros
    N1 = csr_matrix((rows, n_2d))
    N2 = csr_matrix((rows, n_2d * d))
    N3 = csr_matrix((rows, n_2d * n_2d))
    N4 = csr_matrix((rows, n_2d * n_2d))
    N5 = csr_matrix((rows, n_2d))
    N6 = csr_matrix((rows, n_2d))
    N7 = csr_matrix((rows, d * 2))
    N8 = csr_matrix((rows, d * 2))
    
    # N9: Values when i == p and j == k
    N9_data = []
    N9_row = []
    N9_col = []
    
    for row in range(rows):
        i = row // d
        j = row % d
        # Only when p = i and k = j
        base_col = i * (d * 2) + j * 2
        for q in range(2):
            N9_row.append(row)
            N9_col.append(base_col + q)
            if q == 0:
                N9_data.append(1.0)
            else:
                N9_data.append(-1.0)
    
    N9 = csr_matrix((N9_data, (N9_row, N9_col)), shape=(rows, n_2d * d * 2))
    
    # N10: Same as N9
    N10 = N9.copy()
    
    # N11-N16: All zeros
    N11 = csr_matrix((rows, n_2d * d * 2))
    N12 = csr_matrix((rows, n_2d * d * 2))
    N13 = csr_matrix((rows, n_2d * d * 2))
    N14 = csr_matrix((rows, n_2d * d * 2))
    N15 = csr_matrix((rows, n_2d * d * 2))
    N16 = csr_matrix((rows, n_2d * d * 2))
    
    # N17: All zeros
    N17 = csr_matrix((rows, 2*d))
    
    block8 = hstack([N1, N2, N3, N4, N5, N6, N7, N8, N9, N10, N11, N12, N13, N14, N15, N16, N17], format='csr')
    return block8

def build_W25_block9_sparse(n, d):
    rows = 2*d
    n_2d = n + 2*d
    
    # N1-N16: All zeros
    N1 = csr_matrix((rows, n_2d))
    N2 = csr_matrix((rows, n_2d * d))
    N3 = csr_matrix((rows, n_2d * n_2d))
    N4 = csr_matrix((rows, n_2d * n_2d))
    N5 = csr_matrix((rows, n_2d))
    N6 = csr_matrix((rows, n_2d))
    N7 = csr_matrix((rows, d * 2))
    N8 = csr_matrix((rows, d * 2))
    N9 = csr_matrix((rows, n_2d * d * 2))
    N10 = csr_matrix((rows, n_2d * d * 2))
    N11 = csr_matrix((rows, n_2d * d * 2))
    N12 = csr_matrix((rows, n_2d * d * 2))
    N13 = csr_matrix((rows, n_2d * d * 2))
    N14 = csr_matrix((rows, n_2d * d * 2))
    N15 = csr_matrix((rows, n_2d * d * 2))
    N16 = csr_matrix((rows, n_2d * d * 2))
    
    # N17: Identity matrix
    N17_data = np.ones(rows, dtype=np.float64)
    N17_row = np.arange(rows)
    N17_col = np.arange(rows)
    N17 = csr_matrix((N17_data, (N17_row, N17_col)), shape=(rows, 2*d))
    
    block9 = hstack([N1, N2, N3, N4, N5, N6, N7, N8, N9, N10, N11, N12, N13, N14, N15, N16, N17], format='csr')
    return block9

def build_W25(n, d):
    """
    Build complete W25 sparse matrix by combining all blocks.
    Prints dimensions after each block.
    """
    W25_blocks = []
    
    # Build all blocks
    block1 = build_W25_block1_sparse(n, d)
    W25_blocks.append(block1)
    #print(f"Block 1: {block1.shape}")
    
    block2 = build_W25_block2_sparse(n, d)
    W25_blocks.append(block2)
    #print(f"Block 2: {block2.shape}")
    
    block3 = build_W25_block3_sparse(n, d)
    W25_blocks.append(block3)
    #print(f"Block 3: {block3.shape}")
    
    block4 = build_W25_block4_sparse(n, d)
    W25_blocks.append(block4)
    #print(f"Block 4: {block4.shape}")
    
    block5 = build_W25_block5_sparse(n, d)
    W25_blocks.append(block5)
    #print(f"Block 5: {block5.shape}")
    
    block6 = build_W25_block6_sparse(n, d)
    W25_blocks.append(block6)
    #print(f"Block 6: {block6.shape}")
    
    block7 = build_W25_block7_sparse(n, d)
    W25_blocks.append(block7)
    #print(f"Block 7: {block7.shape}")
    
    block8 = build_W25_block8_sparse(n, d)
    W25_blocks.append(block8)
    #print(f"Block 8: {block8.shape}")
    
    block9 = build_W25_block9_sparse(n, d)
    W25_blocks.append(block9)
    #print(f"Block 9: {block9.shape}")
    
    # Vertically stack all blocks
    W25 = vstack(W25_blocks, format='csr')
    
    # #print summary
    #print(f"\nTotal W25 shape: {W25.shape}")
    #print(f"Number of blocks: {len(W25_blocks)}")
    #print(f"Total non-zero entries: {W25.nnz}")
    
    return W25


def build_B25(n, d, C):
    """Build B25 as sparse vector."""
    from scipy.sparse import csr_matrix
    import numpy as np
    
    n_2d = n + 2*d
    
    # Calculate section sizes
    # 1. First section (subs?): n_2d entries
    size_first = n_2d
    
    # 2. Tau9 section: n_2d entries
    size_tau9 = n_2d
    
    # 3. nu1 section: n_2d * n_2d entries
    size_nu1 = n_2d * n_2d
    
    # 4. eta22 section: n_2d entries
    size_eta22 = n_2d
    
    # 5. eta23 section: n_2d entries
    size_eta23 = n_2d
    
    # 6. eta5 section (b^1_il): n_2d * n_2d * d entries
    size_eta5 = n_2d * n_2d * d
    
    # 7. eta_prime5 section (b^2_il): n_2d * n_2d * d entries
    size_eta_prime5 = n_2d * n_2d * d
    
    # 8. delta(x_j, i) section: n * d entries
    size_delta = n * d
    
    # 9. x_j section: 2*d entries
    size_xj = 2*d
    
    total_size = (size_first + size_tau9 + size_nu1 + size_eta22 + size_eta23 + 
                  size_eta5 + size_eta_prime5 + size_delta + size_xj)
    
    # #print(f"B25 total size calculated: {total_size}")
    # #print(f"Sections: first={size_first}, tau9={size_tau9}, nu1={size_nu1}, "
          # f"eta22={size_eta22}, eta23={size_eta23}, eta5={size_eta5}, "
          # f"eta_prime5={size_eta_prime5}, delta={size_delta}, xj={size_xj}")
    
    # Non-zero sections: eta5, eta_prime5, delta
    # eta5: -2 for all entries
    # eta_prime5: -2 for all entries  
    # delta: -1 for all entries
    non_zero_count = size_eta5 + size_eta_prime5 + size_delta
    
    data = np.zeros(non_zero_count, dtype=np.float64)
    rows = np.zeros(non_zero_count, dtype=np.int32)
    
    idx = 0  # Current position in the full vector
    data_idx = 0  # Current position in the data/rows arrays
    
    # Skip zeros sections: first, tau9, nu1, eta22, eta23
    idx += size_first + size_tau9 + size_nu1 + size_eta22 + size_eta23
    
    # eta5 section (-2)
    for i in range(size_eta5):
        data[data_idx] = -2.0
        rows[data_idx] = idx
        data_idx += 1
        idx += 1
    
    # eta_prime5 section (-2)
    for i in range(size_eta_prime5):
        data[data_idx] = -2.0
        rows[data_idx] = idx
        data_idx += 1
        idx += 1
    
    # delta section (-1)
    for i in range(size_delta):
        data[data_idx] = -1.0
        rows[data_idx] = idx
        data_idx += 1
        idx += 1
    
    # xj section is zeros, skip (already accounted for by idx increment)
    
    # Create sparse column vector
    B25 = csr_matrix((data, (rows, np.zeros(len(data), dtype=np.int32))), 
                     shape=(total_size, 1))
    
    #print(f"B25 shape: {B25.shape}, Non-zeros: {B25.nnz}")
    return B25