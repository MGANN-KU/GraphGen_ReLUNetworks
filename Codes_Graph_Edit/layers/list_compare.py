import numpy as np

def compare_array_lists(list1, list2):
    if len(list1) != len(list2):
        print(f" Length mismatch: {len(list1)} vs {len(list2)}")
        return False
    
    differences = []
    for i, (a, b) in enumerate(zip(list1, list2)):
        if not np.array_equal(a, b):
            differences.append((i, a, b))
    
    if differences:
        print(f" Found {len(differences)} differences:")
        for idx, a, b in differences[:5]:  # Show first 5
            print(f"  Index {idx}: {a} != {b}")
        return False
    else:
        print("Lists are equal")
        return True

# Usage
#compare_array_lists(L19_p_list, L19)

import numpy as np

def get_all_mismatches(dense_mat, sparse_mat):
    """
    Find all mismatches between dense (list of lists) and sparse matrices.
    
    Parameters:
        dense_mat: list of lists or numpy array
        sparse_mat: scipy sparse matrix
    
    Returns:
        list of tuples (row, col, dense_value, sparse_value) for all mismatches
    """
    dense_array = np.array(dense_mat, dtype=float)
    sparse_dense = sparse_mat.toarray()
    
    diff_mask = dense_array != sparse_dense
    diff_indices = np.where(diff_mask)
    
    mismatches = []
    for i in range(len(diff_indices[0])):
        r, c = diff_indices[0][i], diff_indices[1][i]
        mismatches.append((int(r), int(c), 
                          float(dense_array[r,c]), 
                          float(sparse_dense[r,c])))
    
    return mismatches