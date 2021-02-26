import json
import numpy as np

# translate DA datasets to our format
# assign yes (cost in [0, 1]) -> yes_value
# maybe (cost in [1, 2]) -> maybe_value
# no (cost in [2, 8]) -> no_value

yes_value = 1
maybe_value = 1
no_value = 0

fnames = ['DA1', 'DA2', 'DA3']

for fname in fnames:
    with open(fname + '.json') as f:
        o = json.load(f)
        S = np.asarray(o["cost_matrix"])
        assert S.shape == (o["total_reviewers"], o["total_papers"])
        yes_bids = (S <= 1)
        maybe_bids = np.logical_and(S > 1, S <= 2)
        no_bids = (S > 2)
        S[yes_bids] = yes_value
        S[maybe_bids] = maybe_value
        S[no_bids] = no_value
        assert np.all(S >= 0) and np.all(S <= 1)
        M = np.zeros(S.shape)
        np.savez(fname, similarity_matrix=S, mask_matrix=M)
