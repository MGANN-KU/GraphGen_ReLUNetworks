# -*- coding: utf-8 -*-
"""
Weights and bias of layer 39_c

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
from scipy.sparse import csr_matrix, hstack, vstack

import numpy as np
from scipy.sparse import csr_matrix, hstack, vstack


def build_W39(n, d, eps):
    n_d  = n + d
    n_2d = n + 2 * d
    d2   = d + 1

    blocks = []

    # ============================================================
    # BLOCK 1
    # rows = n_d * n_2d
    # maps (i,l) → all j at fixed (p=i, r=l)
    # ============================================================

    rows1 = n_d * n_2d
    cols1 = n_d * n_2d * d2

    rows = []
    cols = []
    data = []

    row = 0
    for i in range(n_d):
        for l in range(n_2d):
            base = (i * n_2d + l) * d2
            for j in range(d2):
                rows.append(row)
                cols.append(base + j)
                data.append(1.0)
            row += 1

    H1 = csr_matrix((data, (rows, cols)), shape=(rows1, cols1))
    H2 = csr_matrix((rows1, n_d * d2))
    H3 = csr_matrix((rows1, n_d * n_d * d2))

    block1 = hstack([H1, H2, H3], format="csr")
    blocks.append(block1)

    # ============================================================
    # BLOCK 2
    # rows = n_d
    # maps i → (p=i, all j)
    # ============================================================

    rows2 = n_d
    cols2 = n_d * d2

    rows = []
    cols = []
    data = []

    for i in range(n_d):
        base = i * d2
        for j in range(d2):
            rows.append(i)
            cols.append(base + j)
            data.append(1.0)

    H4 = csr_matrix((rows2, cols1))
    H5 = csr_matrix((data, (rows, cols)), shape=(rows2, cols2))
    H6 = csr_matrix((rows2, n_d * n_d * d2))

    block2 = hstack([H4, H5, H6], format="csr")
    blocks.append(block2)

    # ============================================================
    # BLOCK 3
    # rows = n_d * n_d * d2
    # exact identity mapping
    # ============================================================

    rows3 = n_d * n_d * d2
    cols3 = n_d * n_d * d2

    idx = np.arange(rows3)

    H7 = csr_matrix((rows3, cols1))
    H8 = csr_matrix((rows3, cols2))
    H9 = csr_matrix(
        (np.ones(rows3), (idx, idx)),
        shape=(rows3, cols3)
    )

    block3 = hstack([H7, H8, H9], format="csr")
    blocks.append(block3)

    # ============================================================
    # FINAL ASSEMBLY
    # ============================================================

    W39 = vstack(blocks, format="csr")
    return W39
