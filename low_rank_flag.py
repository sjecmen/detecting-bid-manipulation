import numpy as np
import scipy.linalg

# Threshold low singular values of S so it has given rank.
def threshold(S, rank):
    U, s, Vh = scipy.linalg.svd(S, full_matrices=False)
    k = s.size
    if k < rank:
        return S
    for i in range(rank, k):
        s[i] = 0
    return U @ np.diag(s) @ Vh
    
# Flag the top n_flag entries in S based on the difference with the 
# thresholded matrix at rank. M is the mask matrix.
def flag(S, M, rank, n_flag):
    L = threshold(S, rank)
    diff = np.abs(L - S)
    flat_idxs = np.argsort(diff, axis=None) # flattened index, increasing
    unraveled = np.unravel_index(flat_idxs, S.shape) # unflattens
    idxs = [(unraveled[0][i], unraveled[1][i]) for i in range(flat_idxs.size)]
    flags = []
    for idx in reversed(idxs): # decreasing order
        if M[idx] == 0 and S[idx] == 1:
            flags.append(idx)
            n_flag -= 1
            if n_flag == 0:
                break
    return flags


