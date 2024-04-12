'''
Preprocess MCNC dataset to get head-tail node pairs that is within 2-hop.

Since the MCNC dataset has a very large amount of items, we filter out very frequent nodes to limit the number of pairs.

It takes around 12 hours for a AWS c6i.metal instance (with 128 cpus) to process all shortest paths on this data.
'''

import pickle
from retrieval_pipeline import UndirectedShortestPathFinder

from aser_utils import load_aser
aser = load_aser(aser_path='/path/to/data/core_100_normed_nocoocr.pkl', remove_top_degree=True)

# Initialize shortest path finder
shortest_path_finder = UndirectedShortestPathFinder(aser)


with open('/path/to/data/aser_shortest_paths/NEEG_pairs.pkl', 'rb') as f:
    pairs = pickle.load(f)


# save_dir = '/path/to/data/aser_shortest_paths/NEEG_sp_notop.pkl'
save_dir = '/path/to/data/aser_shortest_paths/NEEG_sp_notop_cutoff-3.pkl'

# preprocess pairs set to False, since they have been processed in preprocess_MCNC_pairs.py
# shortest_path_finder.find_shortest_paths_multiproc(pairs, cutoff=2, sub_list_size=8, preprocess_pairs=False)   
shortest_path_finder.find_shortest_paths_multiproc(pairs, cutoff=3, sub_list_size=8, preprocess_pairs=False)  
shortest_path_finder.save(save_dir)