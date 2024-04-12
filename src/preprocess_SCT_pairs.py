# # Preprocess the Story Cloze Test data 
# Process the SCT dataset to get event anchors.

import pickle
import itertools
from tqdm import tqdm
from retrieval_pipeline import UndirectedShortestPathFinder

def get_group_comparison(event_groups):
    all_pairs = set()
    for i in range(len(event_groups)):
        for j in range(i+1, len(event_groups)):
            head = event_groups[i]
            tail = event_groups[j]
            for pair in itertools.product(head, tail):
                all_pairs.add(pair)
    return all_pairs

def get_event_pairs(data):
    results = list()
    for item in tqdm(data):
        events = [list(set([i['anchor_events'][key][0][0] for key in i['anchor_events']])) for i in data[item][0]['events']['events']]
        all_pairs = get_group_comparison(events)

        events = [list(set([i['anchor_events'][key][0][0] for key in i['anchor_events']])) for i in data[item][1]['events']['events']]
        all_pairs = all_pairs.union(get_group_comparison(events))

        results.extend(list(all_pairs))
    results = set(results)
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

if __name__ == '__main__':
    # load ASER 
    from aser_utils import load_aser
    aser = load_aser(aser_path='/path/to/data/core_100_normed_nocoocr.pkl')

    # Initialize shortest path finder
    shortest_path_finder = UndirectedShortestPathFinder(aser)

    # find paths from SCT dataset events
    fn_list = [f'/path/to/dataset/StoryClozeTest/raw/event_{i}{j}.pickle' for i in ['test', 'valid'] for j in [2016, 2018]]
    pairs = get_all_event_pairs(fn_list)
    pairs = shortest_path_finder.preprocess_pairs(pairs, cutoff=3)
    pairs_dir = '/path/to/data/aser_shortest_paths/SCT_pairs_cutoff-3.pkl'
    
    with open(pairs_dir, 'wb') as f:
        import pickle
        pickle.dump(pairs, f)