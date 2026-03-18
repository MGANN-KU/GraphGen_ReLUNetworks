# -*- coding: utf-8 -*-
"""
Weights and bias of layer 22_b-->22_c

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
from scipy.sparse import csr_matrix, hstack, vstack
import numpy as np
from scipy.sparse import csr_matrix, hstack, vstack

import numpy as np
from scipy.sparse import csr_matrix, hstack, vstack, lil_matrix

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix

import numpy as np
from scipy.sparse import coo_matrix

import numpy as np
from scipy.sparse import coo_matrix


import numpy as np
from scipy.sparse import coo_matrix


def build_W22(n, d, C):
    n_2d = n + 2*d

    # -------------------------------
    # Column sizes (exactly as dense)
    # -------------------------------
    cols_M11 = n_2d
    cols_M12 = n_2d * n_2d
    cols_M13 = d
    cols_M14 = d * d * 2
    cols_M15 = d * d * 2
    cols_M16 = n_2d
    cols_M17 = n_2d
    cols_M18 = d * 2
    cols_M19 = d * 2

    cols_M20 = n_2d * d * 2
    cols_M21 = n_2d * d * 2
    cols_M22 = n_2d * d * 2
    cols_M23 = n_2d * d * 2
    cols_M24 = n_2d * d * 2
    cols_M25 = n_2d * d * 2
    cols_M26 = n_2d * d * 2
    cols_M27 = n_2d * d * 2
    cols_M28 = n_2d * d * 2
    cols_M29 = n_2d * d * 2

    cols_M30 = 2 * d

    # -------------------------------
    # Column offsets (SEQUENTIAL!)
    # -------------------------------
    offset = 0
    off_M11 = offset; offset += cols_M11
    off_M12 = offset; offset += cols_M12
    off_M13 = offset; offset += cols_M13
    off_M14 = offset; offset += cols_M14
    off_M15 = offset; offset += cols_M15
    off_M16 = offset; offset += cols_M16
    off_M17 = offset; offset += cols_M17
    off_M18 = offset; offset += cols_M18
    off_M19 = offset; offset += cols_M19
    off_M20 = offset; offset += cols_M20
    off_M21 = offset; offset += cols_M21
    off_M22 = offset; offset += cols_M22
    off_M23 = offset; offset += cols_M23
    off_M24 = offset; offset += cols_M24
    off_M25 = offset; offset += cols_M25
    off_M26 = offset; offset += cols_M26
    off_M27 = offset; offset += cols_M27
    off_M28 = offset; offset += cols_M28
    off_M29 = offset; offset += cols_M29
    off_M30 = offset; offset += cols_M30

    total_cols = offset

    # -------------------------------
    # Row counts
    # -------------------------------
    rows1 = n_2d
    rows2 = n_2d * n_2d
    rows3 = d * d
    rows4 = n_2d
    rows5 = n_2d
    rows6 = n_2d * n_2d * d
    rows7 = n_2d * n_2d * d
    rows8 = n_2d * d
    rows9 = 2 * d

    total_rows = rows1 + rows2 + rows3 + rows4 + rows5 + rows6 + rows7 + rows8 + rows9

    rows = []
    cols = []
    data = []

    r = 0

    # -------------------------------
    # Block 1: M11
    # -------------------------------
    for i in range(n_2d):
        rows.append(r+i)
        cols.append(off_M11 + i)
        data.append(1.0)
    r += rows1

    # -------------------------------
    # Block 2: M12
    # -------------------------------
    for idx in range(rows2):
        rows.append(r+idx)
        cols.append(off_M12 + idx)
        data.append(1.0)
    r += rows2

    # -------------------------------
    # Block 3: M13, M14, M15
    # -------------------------------
    for j in range(d):
        for i in range(d):
            rr = r + j*d + i

            rows.append(rr)
            cols.append(off_M13 + i)
            data.append(1.0)

            base14 = off_M14 + (j*d + i) * 2
            base15 = off_M15 + (j*d + i) * 2

            rows += [rr, rr]
            cols += [base14, base14+1]
            data += [C, -C]

            rows += [rr, rr]
            cols += [base15, base15+1]
            data += [C, -C]

    r += rows3

    # -------------------------------
    # Block 4: M16
    # -------------------------------
    for i in range(n_2d):
        rows.append(r+i)
        cols.append(off_M16 + i)
        data.append(1.0)
    r += rows4

    # -------------------------------
    # Block 5: M17
    # -------------------------------
    for i in range(n_2d):
        rows.append(r+i)
        cols.append(off_M17 + i)
        data.append(1.0)
    r += rows5

    # -------------------------------
    # Block 6: zeta
    # -------------------------------
    for i in range(n_2d):
        for k in range(n_2d):
            for j in range(d):
                rr = r + (i*n_2d + k)*d + j

                rows += [rr, rr]
                cols += [off_M18 + 2*j, off_M18 + 2*j + 1]
                data += [-1, 1]

                rows += [rr, rr]
                cols += [off_M19 + 2*j, off_M19 + 2*j + 1]
                data += [-1, 1]

                base = (i*d + j) * 2
                rows += [rr, rr]
                cols += [off_M20 + base, off_M20 + base + 1]
                data += [1, -1]

                rows += [rr, rr]
                cols += [off_M21 + base, off_M21 + base + 1]
                data += [1, -1]

                base = (k*d + j) * 2
                rows += [rr, rr]
                cols += [off_M24 + base, off_M24 + base + 1]
                data += [1, -1]

                rows += [rr, rr]
                cols += [off_M25 + base, off_M25 + base + 1]
                data += [1, -1]

    r += rows6

    # -------------------------------
    # Block 7: zeta'
    # -------------------------------
    for i in range(n_2d):
        for k in range(n_2d):
            for j in range(d):
                rr = r + (i*n_2d + k)*d + j

                rows += [rr, rr]
                cols += [off_M18 + 2*j, off_M18 + 2*j + 1]
                data += [-1, 1]

                rows += [rr, rr]
                cols += [off_M19 + 2*j, off_M19 + 2*j + 1]
                data += [-1, 1]

                base = (k*d + j) * 2
                rows += [rr, rr]
                cols += [off_M22 + base, off_M22 + base + 1]
                data += [1, -1]

                rows += [rr, rr]
                cols += [off_M23 + base, off_M23 + base + 1]
                data += [1, -1]

                base = (i*d + j) * 2
                rows += [rr, rr]
                cols += [off_M26 + base, off_M26 + base + 1]
                data += [1, -1]

                rows += [rr, rr]
                cols += [off_M27 + base, off_M27 + base + 1]
                data += [1, -1]

    r += rows7

    # -------------------------------
    # Block 8: M28, M29
    # -------------------------------
    for i in range(n_2d):
        for j in range(d):
            rr = r + i*d + j
            base = (i*d + j) * 2

            rows += [rr, rr]
            cols += [off_M28 + base, off_M28 + base + 1]
            data += [1, -1]

            rows += [rr, rr]
            cols += [off_M29 + base, off_M29 + base + 1]
            data += [1, -1]

    r += rows8

    # -------------------------------
    # Block 9: M30
    # -------------------------------
    for i in range(2*d):
        rows.append(r+i)
        cols.append(off_M30 + i)
        data.append(1.0)

    return coo_matrix(
        (data, (rows, cols)),
        shape=(total_rows, total_cols),
        dtype=np.float64
    ).tocsr()


def build_B22(n, d, C):

    n_2d = n + 2*d
    d1 = d   # because loops run from 1..d

    # --- Section sizes ---
    size_subs        = n_2d
    size_mu1         = n_2d * n_2d
    size_tau6        = d * d
    size_eta22       = n_2d
    size_eta23       = n_2d
    size_zeta1       = n_2d * n_2d * d1
    size_zeta_prime  = n_2d * n_2d * d1
    size_tau7        = n_2d * d1
    size_deletion    = 2 * d

    total_rows = (size_subs + size_mu1 + size_tau6 + size_eta22 + size_eta23 +
                  size_zeta1 + size_zeta_prime + size_tau7 + size_deletion)

    # --- Non-zero regions only ---
    nonzero_counts = size_tau6 + size_zeta1 + size_zeta_prime + size_tau7

    # Pre-allocate arrays
    data = np.empty(nonzero_counts, dtype=np.float32)
    rows = np.empty(nonzero_counts, dtype=np.int32)

    write_pos = 0  # pointer for writing into sparse structure
    cursor = 0     # current row index in the full vector


    # -----------------------------
    # 1) Skip: subs + mu1 (all zero)
    # -----------------------------
    cursor += size_subs + size_mu1


    # -----------------------------
    # 2) tau6 block  (value = -2C, length = d*d)
    # -----------------------------
    end = write_pos + size_tau6
    data[write_pos:end] = -2.0 * C
    rows[write_pos:end] = np.arange(cursor, cursor + size_tau6, dtype=np.int32)
    write_pos = end
    cursor += size_tau6


    # -----------------------------
    # 3) Skip eta22 + eta23 (zeros)
    # -----------------------------
    cursor += size_eta22 + size_eta23


    # -----------------------------
    # 4) zeta1 block (value = -2)
    # -----------------------------
    end = write_pos + size_zeta1
    data[write_pos:end] = -2.0
    rows[write_pos:end] = np.arange(cursor, cursor + size_zeta1, dtype=np.int32)
    write_pos = end
    cursor += size_zeta1


    # -----------------------------
    # 5) zeta_prime block (value = -2)
    # -----------------------------
    end = write_pos + size_zeta_prime
    data[write_pos:end] = -2.0
    rows[write_pos:end] = np.arange(cursor, cursor + size_zeta_prime, dtype=np.int32)
    write_pos = end
    cursor += size_zeta_prime


    # -----------------------------
    # 6) tau7 block (value = -1)
    # -----------------------------
    end = write_pos + size_tau7
    data[write_pos:end] = -1.0
    rows[write_pos:end] = np.arange(cursor, cursor + size_tau7, dtype=np.int32)
    write_pos = end
    cursor += size_tau7


    # -----------------------------
    # 7) Delete block (zeros → skipped)
    # -----------------------------
    # cursor += size_deletion   # not needed because we only care about total shape


    # Build sparse CSR column vector
    col_idx = np.zeros(nonzero_counts, dtype=np.int32)
    return csr_matrix((data, (rows, col_idx)), shape=(total_rows, 1))
