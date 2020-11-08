from low_rank_flag import *
import numpy as np
import matplotlib.pyplot as plt
import sys

# Return the probability of detecting the manipulation (averaged over 
# all r,p) when flagging nflag bids using a threshold at rank.
def run(rank, nflag):
    correct = 0
    i = 0
    for r in range(S.shape[0]):
        for p in range(S.shape[1]):
            if M[r, p] == 1:
                continue
            
            '''
            # uncomment for weaker manipulation case
            if S[r, p] == 1:
                continue
            '''
            

            Sm = np.copy(S)
            Sm[r, :] = 0 # comment for weaker manipulation case
            Sm[r, p] = 1
            flags = flag(Sm, M, rank, nflag)
            for f in flags:
                if f == (r, p):
                    correct += 1
                    break
            i += 1
    return correct/i

f = np.load("data/preflib1.npz")
S, M = f['similarity_matrix'], f['mask_matrix']

print("total rank", np.linalg.matrix_rank(S))

nflag = 3#int(S.size * 0.05)
print("number flags", nflag)

ranks = np.arange(1, np.linalg.matrix_rank(S))
ps = []
for rank in ranks:
    p = run(rank, nflag)
    ps.append(p)
    print(rank, p)

np.savez('bidexp_'+str(nflag) +'.npz', ranks=ranks, ps=ps)

# plotting
#plt.plot(ranks, ps, '-o')
#plt.xlabel('rank threshold')
#plt.ylabel('detection probability')
#plt.ylim([0, 1])
#plt.savefig('bidexp_'+str(nflag) +'.png')
