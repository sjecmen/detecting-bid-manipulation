from low_rank_flag import *
import numpy as np
import matplotlib.pyplot as plt
import sys

d3 = np.load('bidexp_3.npz')
d83 = np.load('bidexp_83_weak.npz')


plt.rcParams.update({'font.size': 15.5})
plt.plot(d3['ranks'], d3['ps'], '-o', label="3 flags", ms=7, linewidth=2)
plt.plot(d83['ranks'], d83['ps'], '--^', label="83 flags, weak manipulation", ms=7, linewidth=2)
plt.legend(loc='lower left')
plt.xlabel('Rank Threshold')
plt.ylabel('Detection Probability')
plt.tight_layout()
plt.ylim([0, 1])
plt.xticks([1, 6, 11, 16, 21, 26, 31])
plt.savefig('bidexp.png')
plt.show()
