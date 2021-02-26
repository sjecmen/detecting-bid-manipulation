test.py can be executed to run an experiment and save data.
plot.py plots the experiment.
low_rank_flag.py contains functions to flag suspicious bids.

data/ contains datasets and code to modify their format.
	- npz files contain similarity/bid matrices under the keyword 'similarity_matrix' and conflict-of-interest matrices under the keyword 'mask_matrix'
	- bid values for DA/preflib datasets can be changed with the files preflib_to_npz.py or DA_to_npz.py
