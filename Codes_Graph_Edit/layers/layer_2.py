# -*- coding: utf-8 -*-
"""
Weights and bias of layer 1

@author: Ghafoor
"""

import numpy as np
from scipy import sparse



import numpy as np
from scipy.sparse import lil_matrix, csr_matrix

def build_W2(n, d, m):
    """
    Efficient construction of W2 matrix using sparse matrices
    
    Parameters:
    n, d, m: network parameters
    
    Returns:
    W2: sparse matrix of appropriate dimensions
    """
    # Dynamically compute dimensions
    rows_total = 21 * d  # 7d rho + 7d rho_prime + 7d psi
    
    # Compute columns dynamically
    cols_gamma1 = 4 * (7 * d) * (n + 1) * 2      # 4 gamma types × 7d × (n+1) × 2
    cols_gamma_prime = 4 * (7 * d) * (n + d) * 2  # 4 gamma prime types × 7d × (n+d) × 2  
    cols_gamma2 = 4 * (7 * d) * m * 2            # 4 gamma types × 7d × m × 2
    cols_total = cols_gamma1 + cols_gamma_prime + cols_gamma2
    
    # print(f"Constructing W2: {rows_total} × {cols_total}")
    
    # Initialize sparse matrix
    W2 = lil_matrix((rows_total, cols_total))
    
    # Helper function to compute column offsets
    def get_col_offset(gamma_type, k_offset):
        """Calculate column offset for specific gamma type and k position"""
        if gamma_type in [0, 1, 2, 3]:  # gamma 11-42 (n+1 range)
            return k_offset * (n+1) * 2 + gamma_type * (7*d) * (n+1) * 2
        elif gamma_type in [4, 5, 6, 7]:  # gamma prime 11-42 (n+d range)
            base = 4 * (7*d) * (n+1) * 2
            gamma_prime_idx = gamma_type - 4
            return base + k_offset * (n+d) * 2 + gamma_prime_idx * (7*d) * (n+d) * 2
        else:  # gamma 51-82 (m range)
            base = 4 * (7*d) * (n+1) * 2 + 4 * (7*d) * (n+d) * 2
            gamma_idx = gamma_type - 8
            return base + k_offset * m * 2 + gamma_idx * (7*d) * m * 2
    
    row_idx = 0
    
    # 1. Rho nodes (1, d) - rows 0 to d-1
    for j in range(1, d+1):
        for k in range(1, 7*d+1):
            if j == k:
                # Gamma 11,12, 21,22, 31,32, 41,42
                for gamma_type in [0, 1, 2, 3]:
                    sign1 = 1 if gamma_type in [0, 1] else -1
                    sign2 = -1 if gamma_type in [0, 1] else 1
                    for i in range(n+1):
                        W2[row_idx, get_col_offset(gamma_type, k-1) + i*2 + 0] = sign1 * i
                        W2[row_idx, get_col_offset(gamma_type, k-1) + i*2 + 1] = sign2 * i
        row_idx += 1
    
    # 2. Rho nodes (d+1, 5d+1) - all zeros - rows d to 5d-1
    row_idx += 4 * d
    
    # 3. Rho nodes (5d+1, 7d+1) - rows 5d to 7d-1
    for j in range(5*d+1, 7*d+1):
        for k in range(1, 7*d+1):
            if j == k:
                for gamma_type in [0, 1, 2, 3]:
                    sign1 = 1 if gamma_type in [0, 1] else -1
                    sign2 = -1 if gamma_type in [0, 1] else 1
                    for i in range(n+1):
                        W2[row_idx, get_col_offset(gamma_type, k-1) + i*2 + 0] = sign1 * i
                        W2[row_idx, get_col_offset(gamma_type, k-1) + i*2 + 1] = sign2 * i
        row_idx += 1
    
    # 4. Rho' nodes (1, 2d+1) - all zeros - rows 7d to 9d-1
    row_idx += 2 * d
    
    # 5. Rho' nodes (2d+1, 4d+1) - rows 9d to 11d-1
    for j in range(2*d+1, 4*d+1):
        for k in range(1, 7*d+1):
            if j == k:
                # Gamma prime nodes
                for gamma_type in [4, 5, 6, 7]:
                    sign1 = 1 if gamma_type in [4, 5] else -1
                    sign2 = -1 if gamma_type in [4, 5] else 1
                    for i in range(n+d):
                        W2[row_idx, get_col_offset(gamma_type, k-1) + i*2 + 0] = sign1 * i
                        W2[row_idx, get_col_offset(gamma_type, k-1) + i*2 + 1] = sign2 * i
        row_idx += 1
    
    # 6. Rho' nodes (4d+1, 7d+1) - rows 11d to 14d-1
    for j in range(4*d+1, 7*d+1):
        for k in range(1, 7*d+1):
            if j == k:
                # Gamma prime nodes
                for gamma_type in [4, 5, 6, 7]:
                    sign1 = 1 if gamma_type in [4, 5] else -1
                    sign2 = -1 if gamma_type in [4, 5] else 1
                    for i in range(n+d):
                        W2[row_idx, get_col_offset(gamma_type, k-1) + i*2 + 0] = sign1 * i
                        W2[row_idx, get_col_offset(gamma_type, k-1) + i*2 + 1] = sign2 * i
        row_idx += 1
    
    # 7. Psi nodes (1, d) - all zeros - rows 14d to 15d-1
    row_idx += d
    
    # 8. Psi nodes (d+1, 2d+1) - rows 15d to 16d-1
    for j in range(d+1, 2*d+1):
        for k in range(1, 7*d+1):
            if j == k:
                # Gamma 51,52, 61,62
                for gamma_type in [8, 9]:
                    for i in range(1, m+1):
                        W2[row_idx, get_col_offset(gamma_type, k-1) + (i-1)*2 + 0] = i
                        W2[row_idx, get_col_offset(gamma_type, k-1) + (i-1)*2 + 1] = -i
                
                # Gamma 71,72, 81,82  
                for gamma_type in [10, 11]:
                    for i in range(1, m+1):
                        if i != 1:  # l != 1 condition
                            W2[row_idx, get_col_offset(gamma_type, k-1) + (i-1)*2 + 0] = -i
                            W2[row_idx, get_col_offset(gamma_type, k-1) + (i-1)*2 + 1] = i
        row_idx += 1
    
    # 9. Psi nodes (2d+1, 4d+1) - all zeros - rows 16d to 18d-1
    row_idx += 2 * d
    
    # 10. Psi nodes (4d+1, 5d+1) - rows 18d to 19d-1
    for j in range(4*d+1, 5*d+1):
        for k in range(1, 7*d+1):
            if j == k:
                # Gamma 51,52, 61,62
                for gamma_type in [8, 9]:
                    for i in range(1, m+1):
                        W2[row_idx, get_col_offset(gamma_type, k-1) + (i-1)*2 + 0] = i
                        W2[row_idx, get_col_offset(gamma_type, k-1) + (i-1)*2 + 1] = -i
                
                # Gamma 71,72, 81,82
                for gamma_type in [10, 11]:
                    for i in range(1, m+1):
                        if i != 1:  # l != 1 condition
                            W2[row_idx, get_col_offset(gamma_type, k-1) + (i-1)*2 + 0] = -i
                            W2[row_idx, get_col_offset(gamma_type, k-1) + (i-1)*2 + 1] = i
        row_idx += 1
    
    # 11. Psi nodes (5d+1, 7d+1) - all zeros - rows 19d to 21d-1
    # No need to increment since we're at the end
    
    # Convert to CSR format for efficient operations
    W2_csr = W2.tocsr()
    
    return W2_csr


def build_B2(d):
    """
    Build B2 bias vector as 1D numpy array (similar to B1 format)
    """
    # Total length = 21*d (7d rho + 7d rho_prime + 7d psi)
    total_length = 21 * d
    
    # Initialize bias vector with zeros
    B2 = np.zeros(total_length)
    
    current_idx = 0
    
    # 1. ρ nodes (1, 7d+1) - all zeros (already initialized)
    current_idx += 7 * d
    
    # 2. ρ' nodes (1, 7d+1) - all zeros (already initialized)  
    current_idx += 7 * d
    
    # 3. ψ nodes 
    # (1, d) - zeros (already initialized)
    current_idx += d
    
    # (d+1, 2d+1) - negative ones
    for j in range(d+1, 2*d+1):
        B2[current_idx] = -1
        current_idx += 1
    
    # (2d+1, 4d+1) - zeros (already initialized)
    current_idx += 2 * d
    
    # (4d+1, 5d+1) - negative ones
    for j in range(4*d+1, 5*d+1):
        B2[current_idx] = -1
        current_idx += 1
    
    # (5d+1, 7d+1) - zeros (already initialized)
    # current_idx += 2*d (not needed since we're at the end)
    
    # print(f"B2 length: {len(B2)}")
    # print(f"B2 non-zero values: {np.sum(B2 != 0)}")
    
    return B2