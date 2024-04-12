# # Preprocess the Story Cloze Test data 
# Process the SCT dataset to get event anchors.

import pickle
from aser_utils import load_aser
from retrieval_pipeline import UndirectedShortestPathFinder

pairs_dir = '/path/to/data/aser_shortest_paths/SCT_pairs_cutoff-3.pkl'
save_dir = '/path/to/data/aser_shortest_paths/SCT_sp_notop_cutoff-3.pkl'


# load ASER 
# aser = load_aser(aser_path='/path/to/data/core_100_normed_nocoocr.pkl')
aser = load_aser(aser_path='/path/to/data/core_100_normed_nocoocr.pkl', remove_top_degree=True)

# Initialize shortest path finder
shortest_path_finder = UndirectedShortestPathFinder(aser)

# load paths from saved precomputed path files
with open(pairs_dir, 'rb') as f:
    import pickle
    pairs = pickle.load(f)

shortest_path_finder.find_shortest_paths_multiproc(pairs, cutoff=3, sub_list_size=8, preprocess_pairs=True)
shortest_path_finder.save(save_dir)