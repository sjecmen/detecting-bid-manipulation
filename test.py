from low_rank_flag import *
import numpy as np
import matplotlib.pyplot as plt
import sys
from scipy.stats import sem
from itertools import product

rng = np.random.default_rng()

'''
Return the probability of detecting the manipulation (averaged over 
all r,p) when flagging nflag bids using a threshold at rank.
    - rank: rank threshold
    - nflag: number of positive bids to flag
    - weak: whether to perform weak or strong manipulation
    - T: number of random (rev, pap) samples to run; 
            if None then average over all pairs
    - mult: if False perform manipulation by setting bids to {0, 1}
            if True perform manipulation by multiplying/dividing
                base similarity by 2
'''
def run(rank, nflag, weak=False, T=200, mult=False):
    correct = 0
    i = 0
    if T==None:
        for r, p in product(range(S.shape[0]), range(S.shape[1])):
            if M[r, p] == 1:
                continue
            
            if weak and S[r, p] == 1:
                continue
    
            Sm = np.copy(S)
            if not weak:
                if mult:
                    tmp = Sm[r, p]
                    Sm[r, :] /= 2
                    Sm[r, p] = tmp
                else:
                    Sm[r, :] = 0
            if mult:
                Sm[r, p] *= 2
            else:
                Sm[r, p] = 1
            flags = flag(Sm, M, rank, nflag)
            for f in flags:
                if f == (r, p):
                    correct += 1
                    break
            i += 1
    else:
       while i < T:
            r = rng.integers(S.shape[0]) 
            p = rng.integers(S.shape[1])
   
            # TODO duplicates above code
            if M[r, p] == 1:
                continue
            
            if weak and S[r, p] == 1:
                continue
   
            Sm = np.copy(S)
            if not weak:
                if mult:
                    tmp = Sm[r, p]
                    Sm[r, :] /= 2
                    Sm[r, p] = tmp
                else:
                    Sm[r, :] = 0
            if mult:
                Sm[r, p] *= 2
            else:
                Sm[r, p] = 1
            flags = flag(Sm, M, rank, nflag)
            for f in flags:
                if f == (r, p):
                    correct += 1
                    break
            i += 1
    se = sem([1]*correct + [0]*(i-correct)) if T != None else 0
    return correct/i, se


fname = 'preflib1' # change for different dataset


# set parameters based on dataset
if fname == 'preflib1' or fname == 'preflib2':
    rank_step = 1
elif fname == 'preflib3':
    rank_step = 5
else:
    rank_step = 10
mult = (fname == 'iclr2018')
T = None if (fname == 'preflib1' or fname == 'preflib2') else 200


pflags = [0.05, 0.01]
weaks = [True, False]

f = np.load("data/" + fname + ".npz")
S, M = f['similarity_matrix'], f['mask_matrix']


print("total rank", np.linalg.matrix_rank(S), 'step', rank_step)
num_ones = np.sum(S == 1)
print('number positive bids', num_ones, 'number entries', S.size, 'mult', mult, 'T', T)

for pflag in pflags:
    for weak in weaks:
        nflag = int(num_ones*pflag)
        print('percent flags', pflag, "number flags", nflag, 'weak', weak)
        
        ranks = np.arange(1, np.linalg.matrix_rank(S), rank_step)
        ps = []
        ses = []
        for rank in ranks:
            p,se = run(rank, nflag, weak, T, mult)
            ps.append(p)
            ses.append(se)
            print(rank, p, se)
        
        np.savez('bidexp2_' +fname + '_weak'+str(weak)+'_p'+str(pflag*100) +'.npz', ranks=ranks, ps=ps, ses=ses, T=T)
