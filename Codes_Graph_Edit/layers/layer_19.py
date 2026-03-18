# -*- coding: utf-8 -*-
"""
Weights and bias of layer 19

@author: Ghafoor
"""
import numpy as np
from scipy.sparse import coo_matrix

from scipy.sparse import eye, csr_matrix, hstack, vstack, lil_matrix
import numpy as np

def build_W19(n, d, eps):
    m = n + 2*d
    W19_blocks = []
    
    # Block 1: subs nodes (m rows)
    I6 = eye(m, format='csr')
    I16 = csr_matrix((m, d))
    I7 = csr_matrix((m, d))
    I8 = csr_matrix((m, 2*d))
    I9 = csr_matrix((m, 2*d))
    I10 = csr_matrix((m, 2*m))
    I11 = csr_matrix((m, 2*m))
    I12 = csr_matrix((m, 2*m))
    I13 = csr_matrix((m, 2*m))
    I14 = csr_matrix((m, 2*m))
    I15 = csr_matrix((m, 2*d))
    I17 = csr_matrix((m, 2*d))
    # block = hstack([I6, I16, I7, I8, I9, I10, I11, I12, I13, I14, I15, I17], format='csr')
    # #print(f"Block {len(W19_blocks)+1}: {block.shape}")
    # W19_blocks.append(block)
    W19_blocks.append(hstack([I6, I16, I7, I8, I9, I10, I11, I12, I13, I14, I15, I17], format='csr'))
    
    # Block 2: tau4(gj) nodes (d rows)  
    I6 = csr_matrix((d, m))
    I16 = eye(d, format='csr')
    I7 = csr_matrix((d, d))
    I8 = csr_matrix((d, 2*d))
    I9 = csr_matrix((d, 2*d))
    I10 = csr_matrix((d, 2*m))
    I11 = csr_matrix((d, 2*m))
    I12 = csr_matrix((d, 2*m))
    I13 = csr_matrix((d, 2*m))
    I14 = csr_matrix((d, 2*m))
    I15 = csr_matrix((d, 2*d))
    I17 = csr_matrix((d, 2*d))
    # block = hstack([I6, I16, I7, I8, I9, I10, I11, I12, I13, I14, I15, I17], format='csr')
    # #print(f"Block {len(W19_blocks)+1}: {block.shape}")
    # W19_blocks.append(block)
    W19_blocks.append(hstack([I6, I16, I7, I8, I9, I10, I11, I12, I13, I14, I15, I17], format='csr'))
    
    # Block 3 & 4: alpha5, alpha6 and beta5, beta6 for delta(gj,gk) (4*d*d rows total)
    # This is the complex block with ±1/eps values
    num_delta_rows = 2 * d * d  # for alpha5, alpha6
    I6 = csr_matrix((num_delta_rows, m))
    
    # Build I16 with ±1/eps values
    I16_data = []
    I16_row = []
    I16_col = []
    
    row_idx = 0
    for j in range(d):
        for k in range(d):
            for q in range(2):  # alpha5, alpha6
                if j != k:
                    # For alpha: j=i -> 1/eps, k=i -> -1/eps
                    I16_data.append(1/eps)
                    I16_row.append(row_idx)
                    I16_col.append(j)
                    
                    I16_data.append(-1/eps)
                    I16_row.append(row_idx)
                    I16_col.append(k)
                row_idx += 1
    
    I16 = csr_matrix((I16_data, (I16_row, I16_col)), shape=(num_delta_rows, d))
    I7 = csr_matrix((num_delta_rows, d))
    I8 = csr_matrix((num_delta_rows, 2*d))
    I9 = csr_matrix((num_delta_rows, 2*d))
    I10 = csr_matrix((num_delta_rows, 2*m))
    I11 = csr_matrix((num_delta_rows, 2*m))
    I12 = csr_matrix((num_delta_rows, 2*m))
    I13 = csr_matrix((num_delta_rows, 2*m))
    I14 = csr_matrix((num_delta_rows, 2*m))
    I15 = csr_matrix((num_delta_rows, 2*d))
    I17 = csr_matrix((num_delta_rows, 2*d))
        # block = hstack([I6, I16, I7, I8, I9, I10, I11, I12, I13, I14, I15, I17], format='csr')
    # #print(f"Block {len(W19_blocks)+1}: {block.shape}")
    # W19_blocks.append(block)
    W19_blocks.append(hstack([I6, I16, I7, I8, I9, I10, I11, I12, I13, I14, I15, I17], format='csr'))
    
    # Block 4: beta5, beta6 (same structure, signs reversed)
    I6 = csr_matrix((num_delta_rows, m))
    
    I16_data = []
    I16_row = []
    I16_col = []
    
    row_idx = 0
    for j in range(d):
        for k in range(d):
            for q in range(2):  # beta5, beta6
                if j != k:
                    # For beta: j=i -> -1/eps, k=i -> 1/eps
                    I16_data.append(-1/eps)
                    I16_row.append(row_idx)
                    I16_col.append(j)
                    
                    I16_data.append(1/eps)
                    I16_row.append(row_idx)
                    I16_col.append(k)
                row_idx += 1
    
    I16 = csr_matrix((I16_data, (I16_row, I16_col)), shape=(num_delta_rows, d))
    I7 = csr_matrix((num_delta_rows, d))
    I8 = csr_matrix((num_delta_rows, 2*d))
    I9 = csr_matrix((num_delta_rows, 2*d))
    I10 = csr_matrix((num_delta_rows, 2*m))
    I11 = csr_matrix((num_delta_rows, 2*m))
    I12 = csr_matrix((num_delta_rows, 2*m))
    I13 = csr_matrix((num_delta_rows, 2*m))
    I14 = csr_matrix((num_delta_rows, 2*m))
    I15 = csr_matrix((num_delta_rows, 2*d))
    I17 = csr_matrix((num_delta_rows, 2*d))
        # block = hstack([I6, I16, I7, I8, I9, I10, I11, I12, I13, I14, I15, I17], format='csr')
    # #print(f"Block {len(W19_blocks)+1}: {block.shape}")
    # W19_blocks.append(block)
    W19_blocks.append(hstack([I6, I16, I7, I8, I9, I10, I11, I12, I13, I14, I15, I17], format='csr'))
    
    
    # Block 5: tau6 nodes for (1-δ(e'_j, x_{j+d})) ∧ H(n+d'-e'_j) (d rows)
    I6 = csr_matrix((d, m))
    I16 = csr_matrix((d, d))
    
    # I7: -1 on diagonal
    I7_data = -np.ones(d)
    I7_row = np.arange(d)
    I7_col = np.arange(d)
    I7 = csr_matrix((I7_data, (I7_row, I7_col)), shape=(d, d))
    
    # I8: +1 and -1 on diagonal for q=0 and q=1
    I8_data = np.concatenate([np.ones(d), -np.ones(d)])
    I8_row = np.concatenate([np.arange(d), np.arange(d)])
    I8_col = np.concatenate([2*np.arange(d), 2*np.arange(d) + 1])
    I8 = csr_matrix((I8_data, (I8_row, I8_col)), shape=(d, 2*d))
    
    I9 = csr_matrix((d, 2*d))
    I10 = csr_matrix((d, 2*m))
    I11 = csr_matrix((d, 2*m))
    I12 = csr_matrix((d, 2*m))
    I13 = csr_matrix((d, 2*m))
    I14 = csr_matrix((d, 2*m))
    I15 = csr_matrix((d, 2*d))
    I17 = csr_matrix((d, 2*d))
        # block = hstack([I6, I16, I7, I8, I9, I10, I11, I12, I13, I14, I15, I17], format='csr')
    # #print(f"Block {len(W19_blocks)+1}: {block.shape}")
    # W19_blocks.append(block)
    W19_blocks.append(hstack([I6, I16, I7, I8, I9, I10, I11, I12, I13, I14, I15, I17], format='csr'))
    
    
    
    # Block 6: tau'_6 nodes (d rows) - similar to Block 5 but with pattern in I9 instead of I8
    I6 = csr_matrix((d, m))
    I16 = csr_matrix((d, d))
    
    # I7: -1 on diagonal
    I7_data = -np.ones(d)
    I7_row = np.arange(d)
    I7_col = np.arange(d)
    I7 = csr_matrix((I7_data, (I7_row, I7_col)), shape=(d, d))
    
    I8 = csr_matrix((d, 2*d))
    
    # I9: +1 and -1 on diagonal for q=0 and q=1
    I9_data = np.concatenate([np.ones(d), -np.ones(d)])
    I9_row = np.concatenate([np.arange(d), np.arange(d)])
    I9_col = np.concatenate([2*np.arange(d), 2*np.arange(d) + 1])
    I9 = csr_matrix((I9_data, (I9_row, I9_col)), shape=(d, 2*d))
    
    I10 = csr_matrix((d, 2*m))
    I10 = csr_matrix((d, 2*m))
    I11 = csr_matrix((d, 2*m))
    I12 = csr_matrix((d, 2*m))
    I13 = csr_matrix((d, 2*m))
    I14 = csr_matrix((d, 2*m))
    I15 = csr_matrix((d, 2*d))
    I17 = csr_matrix((d, 2*d))
        # block = hstack([I6, I16, I7, I8, I9, I10, I11, I12, I13, I14, I15, I17], format='csr')
    # #print(f"Block {len(W19_blocks)+1}: {block.shape}")
    # W19_blocks.append(block)
    W19_blocks.append(hstack([I6, I16, I7, I8, I9, I10, I11, I12, I13, I14, I15, I17], format='csr'))
    
    
    
    # Block 7: eta22 - H(n-i) formed by eta9,eta10 (m rows)
    I6 = csr_matrix((m, m))
    I16 = csr_matrix((m, d))
    I7 = csr_matrix((m, d))
    I8 = csr_matrix((m, 2*d))
    I9 = csr_matrix((m, 2*d))
    
    # I10: +1 and -1 on diagonal
    I10_data = np.concatenate([np.ones(m), -np.ones(m)])
    I10_row = np.concatenate([np.arange(m), np.arange(m)])
    I10_col = np.concatenate([2*np.arange(m), 2*np.arange(m) + 1])
    I10 = csr_matrix((I10_data, (I10_row, I10_col)), shape=(m, 2*m))
    
    I11 = csr_matrix((m, 2*m))
    I12 = csr_matrix((m, 2*m))
    I13 = csr_matrix((m, 2*m))
    I14 = csr_matrix((m, 2*m))
    I15 = csr_matrix((m, 2*d))
    I17 = csr_matrix((m, 2*d))
       # block = hstack([I6, I16, I7, I8, I9, I10, I11, I12, I13, I14, I15, I17], format='csr')
    # #print(f"Block {len(W19_blocks)+1}: {block.shape}")
    # W19_blocks.append(block)
    W19_blocks.append(hstack([I6, I16, I7, I8, I9, I10, I11, I12, I13, I14, I15, I17], format='csr'))
    
    
    
    # Block 8: eta23 - H(i-n-1) formed by eta11,eta12 (m rows)
    I6 = csr_matrix((m, m))
    I16 = csr_matrix((m, d))
    I7 = csr_matrix((m, d))
    I8 = csr_matrix((m, 2*d))
    I9 = csr_matrix((m, 2*d))
    I10 = csr_matrix((m, 2*m))
    
    # I11: +1 and -1 on diagonal
    I11_data = np.concatenate([np.ones(m), -np.ones(m)])
    I11_row = np.concatenate([np.arange(m), np.arange(m)])
    I11_col = np.concatenate([2*np.arange(m), 2*np.arange(m) + 1])
    I11 = csr_matrix((I11_data, (I11_row, I11_col)), shape=(m, 2*m))
    
    I12 = csr_matrix((m, 2*m))
    I13 = csr_matrix((m, 2*m))
    I14 = csr_matrix((m, 2*m))
    I15 = csr_matrix((m, 2*d))
    I17 = csr_matrix((m, 2*d))
       # block = hstack([I6, I16, I7, I8, I9, I10, I11, I12, I13, I14, I15, I17], format='csr')
    # #print(f"Block {len(W19_blocks)+1}: {block.shape}")
    # W19_blocks.append(block)
    W19_blocks.append(hstack([I6, I16, I7, I8, I9, I10, I11, I12, I13, I14, I15, I17], format='csr'))
    
    
    
    # Block 9: phi1 (p_ik) nodes (m*m rows)
    num_phi_rows = m * m
    I6 = csr_matrix((num_phi_rows, m))
    I16 = csr_matrix((num_phi_rows, d))
    I7 = csr_matrix((num_phi_rows, d))
    I8 = csr_matrix((num_phi_rows, 2*d))
    I9 = csr_matrix((num_phi_rows, 2*d))
    I10 = csr_matrix((num_phi_rows, 2*m))
    
    # I11: +1 and -1 for each i (row-wise pattern)
    I11_data = []
    I11_row = []
    I11_col = []
    
    row_idx = 0
    for i in range(m):
        for k in range(m):
            # For each (i,k) pair: +1 and -1 at position i
            I11_data.extend([1.0, -1.0])
            I11_row.extend([row_idx, row_idx])
            I11_col.extend([2*i, 2*i + 1])
            row_idx += 1
    
    I11 = csr_matrix((I11_data, (I11_row, I11_col)), shape=(num_phi_rows, 2*m))
    
    # I12: similar pattern for i
    I12_data = []
    I12_row = []
    I12_col = []
    
    row_idx = 0
    for i in range(m):
        for k in range(m):
            I12_data.extend([1.0, -1.0])
            I12_row.extend([row_idx, row_idx])
            I12_col.extend([2*i, 2*i + 1])
            row_idx += 1
    
    I12 = csr_matrix((I12_data, (I12_row, I12_col)), shape=(num_phi_rows, 2*m))
    
    # I13: +1 and -1 for each k
    I13_data = []
    I13_row = []
    I13_col = []
    
    row_idx = 0
    for i in range(m):
        for k in range(m):
            I13_data.extend([1.0, -1.0])
            I13_row.extend([row_idx, row_idx])
            I13_col.extend([2*k, 2*k + 1])
            row_idx += 1
    
    I13 = csr_matrix((I13_data, (I13_row, I13_col)), shape=(num_phi_rows, 2*m))
    
    I14 = csr_matrix((num_phi_rows, 2*m))
    I15 = csr_matrix((num_phi_rows, 2*d))
    I17 = csr_matrix((num_phi_rows, 2*d))
    # block = hstack([I6, I16, I7, I8, I9, I10, I11, I12, I13, I14, I15, I17], format='csr')
    # #print(f"Block {len(W19_blocks)+1}: {block.shape}")
    # W19_blocks.append(block)
    W19_blocks.append(hstack([I6, I16, I7, I8, I9, I10, I11, I12, I13, I14, I15, I17], format='csr'))
    
    
    
    # Block 10: phi2 (q_ik) nodes (m*m rows)
    I6 = csr_matrix((num_phi_rows, m))
    I16 = csr_matrix((num_phi_rows, d))
    I7 = csr_matrix((num_phi_rows, d))
    I8 = csr_matrix((num_phi_rows, 2*d))
    I9 = csr_matrix((num_phi_rows, 2*d))
    
    # I10: +1 and -1 for each i
    I10_data = []
    I10_row = []
    I10_col = []
    
    row_idx = 0
    for i in range(m):
        for k in range(m):
            I10_data.extend([1.0, -1.0])
            I10_row.extend([row_idx, row_idx])
            I10_col.extend([2*i, 2*i + 1])
            row_idx += 1
    
    I10 = csr_matrix((I10_data, (I10_row, I10_col)), shape=(num_phi_rows, 2*m))
    
    I11 = csr_matrix((num_phi_rows, 2*m))
    I12 = csr_matrix((num_phi_rows, 2*m))
    
    # I13: +1 and -1 for each k
    I13_data = []
    I13_row = []
    I13_col = []
    
    row_idx = 0
    for i in range(m):
        for k in range(m):
            I13_data.extend([1.0, -1.0])
            I13_row.extend([row_idx, row_idx])
            I13_col.extend([2*k, 2*k + 1])
            row_idx += 1
    
    I13 = csr_matrix((I13_data, (I13_row, I13_col)), shape=(num_phi_rows, 2*m))
    
    # I14: +1 and -1 for each k
    I14_data = []
    I14_row = []
    I14_col = []
    
    row_idx = 0
    for i in range(m):
        for k in range(m):
            I14_data.extend([1.0, -1.0])
            I14_row.extend([row_idx, row_idx])
            I14_col.extend([2*k, 2*k + 1])
            row_idx += 1
    
    I14 = csr_matrix((I14_data, (I14_row, I14_col)), shape=(num_phi_rows, 2*m))
    
    I15 = csr_matrix((num_phi_rows, 2*d))
    I17 = csr_matrix((num_phi_rows, 2*d))
    # block = hstack([I6, I16, I7, I8, I9, I10, I11, I12, I13, I14, I15, I17], format='csr')
    # #print(f"Block {len(W19_blocks)+1}: {block.shape}")
    # W19_blocks.append(block)
    W19_blocks.append(hstack([I6, I16, I7, I8, I9, I10, I11, I12, I13, I14, I15, I17], format='csr'))
    
    
    
    
    # Block 11: xj nodes for insertion (2*d rows)
    num_xj_rows = 2*d
    I6 = csr_matrix((num_xj_rows, m))
    I16 = csr_matrix((num_xj_rows, d))
    I7 = csr_matrix((num_xj_rows, d))
    I8 = csr_matrix((num_xj_rows, 2*d))
    I9 = csr_matrix((num_xj_rows, 2*d))
    I10 = csr_matrix((num_xj_rows, 2*m))
    I11 = csr_matrix((num_xj_rows, 2*m))
    I12 = csr_matrix((num_xj_rows, 2*m))
    I13 = csr_matrix((num_xj_rows, 2*m))
    I14 = csr_matrix((num_xj_rows, 2*m))
    
    # I15: identity on first half
    I15_data = np.ones(2*d)
    I15_row = np.arange(2*d)
    I15_col = np.arange(2*d)
    I15 = csr_matrix((I15_data, (I15_row, I15_col)), shape=(num_xj_rows, 2*d))
    
    I17 = csr_matrix((num_xj_rows, 2*d))
    # block = hstack([I6, I16, I7, I8, I9, I10, I11, I12, I13, I14, I15, I17], format='csr')
    # #print(f"Block {len(W19_blocks)+1}: {block.shape}")
    # W19_blocks.append(block)
    W19_blocks.append(hstack([I6, I16, I7, I8, I9, I10, I11, I12, I13, I14, I15, I17], format='csr'))
    
    
    
    
    # Block 12: xj nodes for deletion (2*d rows)
    I6 = csr_matrix((num_xj_rows, m))
    I16 = csr_matrix((num_xj_rows, d))
    I7 = csr_matrix((num_xj_rows, d))
    I8 = csr_matrix((num_xj_rows, 2*d))
    I9 = csr_matrix((num_xj_rows, 2*d))
    I10 = csr_matrix((num_xj_rows, 2*m))
    I11 = csr_matrix((num_xj_rows, 2*m))
    I12 = csr_matrix((num_xj_rows, 2*m))
    I13 = csr_matrix((num_xj_rows, 2*m))
    I14 = csr_matrix((num_xj_rows, 2*m))
    I15 = csr_matrix((num_xj_rows, 2*d))  # ZEROS
    
    I17_data = np.ones(2*d)  # IDENTITY
    I17_row = np.arange(2*d)
    I17_col = np.arange(2*d)
    I17 = csr_matrix((I17_data, (I17_row, I17_col)), shape=(num_xj_rows, 2*d))##print(f"Block 12 - I17 shape: {I17.shape}, nnz: {I17.nnz}")
    ##print(f"I17 non-zero positions: {list(zip(I17.row, I17.col))}")
    # block = hstack([I6, I16, I7, I8, I9, I10, I11, I12, I13, I14, I15, I17], format='csr')
    # #print(f"Block {len(W19_blocks)+1}: {block.shape}")
    # W19_blocks.append(block)
    W19_blocks.append(hstack([I6, I16, I7, I8, I9, I10, I11, I12, I13, I14, I15, I17], format='csr'))
    
    
    
    # Stack all blocks vertically
    W19 = vstack(W19_blocks, format='csr')
     # Check rows 252-259
# Check rows 252-259
    # for r in range(252, 260):
        # row_nnz = W19[r,:].nonzero()[1]  # Non-zero columns in this row
        # #print(f"Row {r} non-zero cols: {row_nnz.tolist()}")
    return W19


def build_B19(d, n):
    # Calculate total rows
    rows_section1 = n + 2*d  # subs nodes
    rows_section2 = d  # tau4(gj) nodes
    rows_section3 = d * d * 2  # alpha5, alpha6
    rows_section4 = d * d * 2  # beta5, beta6
    rows_section5 = d  # tau6
    rows_section6 = d  # tau'6
    rows_section7 = n + 2*d  # eta22
    rows_section8 = n + 2*d  # eta23
    rows_section9 = (n + 2*d) * (n + 2*d)  # psi1
    rows_section10 = (n + 2*d) * (n + 2*d)  # psi2
    rows_section11 = 2*d  # deletion nodes
    rows_section12 = 2*d  # deletion nodes
    
    rows = (rows_section1 + rows_section2 + rows_section3 + rows_section4 + 
            rows_section5 + rows_section6 + rows_section7 + rows_section8 + 
            rows_section9 + rows_section10 + rows_section11 + rows_section12)
    
    B19 = np.zeros(rows)
    idx = 0
    
    # Sections 1-2, 5-8, 11-12: all zeros (already initialized)
    idx += rows_section1 + rows_section2
    
    # Section 3: alpha5, alpha6 nodes
    for j in range(d):
        for k in range(d):
            for q in range(2):
                if q == 0:
                    B19[idx] = 1
                idx += 1
    
    # Section 4: beta5, beta6 nodes
    for j in range(d):
        for k in range(d):
            for q in range(2):
                if q == 0:
                    B19[idx] = 1
                idx += 1
    
    # Skip sections 5-8 (zeros)
    idx += rows_section5 + rows_section6 + rows_section7 + rows_section8
    
    # Section 9: psi1 nodes
    B19[idx:idx + rows_section9] = -2
    idx += rows_section9
    
    # Section 10: psi2 nodes
    B19[idx:idx + rows_section10] = -2
    idx += rows_section10
    
    # Sections 11-12: zeros (already)
    
    return B19