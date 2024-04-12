'''
Preprocess MCNC dataset to get head-tail node pairs that is within 2-hop.

Since the MCNC dataset has a very large amount of items, we filter out very frequent nodes to limit the number of pairs.
'''

import pickle
from tqdm import tqdm
from retrieval_pipeline import UndirectedShortestPathFinder

def get_event_pairs(data):
    results = set()
    for item in tqdm(data):
        for cand in item:
            events = [i['anchor_events'][0][0] for i in cand]
            for i in range(len(events)):
                for j in range(i+1, len(events)):
                    results.add((events[i], events[j]))
    return results

def get_all_event_pairs(filename_list):
    pairs = set()

    for filename in filename_list:
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        print(filename, '\n', len(data))
        sub_pairs = get_event_pairs(data)
        pairs = pairs.union(sub_pairs)

    pairs = list(pairs)
    print(f'# pairs: {len(pairs)}')
    return pairs


from aser_utils import load_aser
aser = load_aser(aser_path='/path/to/data/core_100_normed_nocoocr.pkl', remove_top_degree=True)

# Initialize shortest path finder
shortest_path_finder = UndirectedShortestPathFinder(aser)

fn_list = ['/path/to/dataset/NEEG_data/NEEG_events/train.pickle', '/path/to/dataset/NEEG_data/NEEG_events/test.pickle', '/path/to/dataset/NEEG_data/NEEG_events/dev.pickle']
save_dir = '/path/to/data/aser_shortest_paths/NEEG_sp.pkl'
pair_dir = '/path/to/data/aser_shortest_paths/NEEG_pairs.pkl'

pairs = get_all_event_pairs(fn_list)
pairs = shortest_path_finder.preprocess_pairs(pairs, cutoff=2, n_proc=128)

with open(pair_dir, 'wb') as f:
    import pickle
    pickle.dump(pairs, f)