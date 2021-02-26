from low_rank_flag import *
import numpy as np
import matplotlib.pyplot as plt
import sys

fname = 'preflib1'

d5 = np.load('bidexp2_'+fname+'_weakFalse_p5.0.npz')
d1 = np.load('bidexp2_'+fname+'_weakFalse_p1.0.npz')
d5w = np.load('bidexp2_'+fname+'_weakTrue_p5.0.npz')
d1w = np.load('bidexp2_'+fname+'_weakTrue_p1.0.npz')


plt.rcParams.update({'font.size': 15.5})
plt.errorbar(d5['ranks'], d5['ps'], yerr=d5['ses'], fmt='-o', label="5% positive flags, strong manipulation", ms=7, linewidth=2)
plt.errorbar(d1['ranks'], d1['ps'], yerr=d1['ses'], fmt='-^', label="1% positive flags, strong manipulation", ms=7, linewidth=2)
plt.errorbar(d5w['ranks'], d5w['ps'], yerr=d5w['ses'], fmt='-x', label="5% positive flags, weak manipulation", ms=7, linewidth=2)
plt.errorbar(d1w['ranks'], d1w['ps'], yerr=d1w['ses'], fmt='-*', label="1% positive flags, weak manipulation", ms=7, linewidth=2)
plt.legend(prop={'size':12})
plt.title(fname)
plt.xlabel('Rank Threshold')
plt.ylabel('Detection Probability')
plt.tight_layout()
plt.ylim([-.05, 1.05])
plt.xticks()
plt.savefig('bidexp2_'+fname+'.png')
plt.show()
