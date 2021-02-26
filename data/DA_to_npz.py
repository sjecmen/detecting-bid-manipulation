import json
import numpy as np

# translate DA datasets to our format
# assign yes (cost in [0, 1]) -> 4
# maybe (cost in [1, 2]) -> 2
# no (cost in [2, 8]) -> 1
# scaled to [0, 1], so / 4

fnames = ['DA1', 'DA2', 'DA3']

for fname in fnames:
    with open(fname + '.json') as f:
        o = json.load(f)
        S = np.asarray(o["cost_matrix"])
        assert S.shape == (o["total_reviewers"], o["total_papers"])
        yes_bids = (S <= 1)
        maybe_bids = np.logical_and(S > 1, S <= 2)
        no_bids = (S > 2)
        S[yes_bids] = 1
        S[maybe_bids] = 1#1 / 2
        S[no_bids] = 0#1 / 4
        assert np.all(S >= 0) and np.all(S <= 1)
        M = np.zeros(S.shape)
        np.savez(fname, similarity_matrix=S, mask_matrix=M)
