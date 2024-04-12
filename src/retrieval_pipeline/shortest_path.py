import os
import pickle
import networkx as nx
import multiprocessing as mp
from tqdm import tqdm

class UndirectedShortestPathFinder:
    def __init__(self, graph=None):
        if graph:
            self.graph = graph.to_undirected(as_view=True)
        else:
            self.graph = None

        self.manager = mp.Manager()
        self.shortest_paths = self.manager.dict()
        self.shortest_path_lengths = self.manager.dict()
    
    def get(self, pair):
        pair = self.pair2key(pair)
        return self.shortest_paths.get(pair, self.shortest_path_lengths.get(pair, None))
    
    def find_shortest_paths_multiproc(self, pairs, cutoff=2, n_proc=None, sub_list_size=4, preprocess_pairs=True):
        ''' The main entry of the class. 
        '''

        if n_proc is None:
            n_proc = mp.cpu_count()
        if preprocess_pairs:
            pairs = self.preprocess_pairs(pairs, cutoff, n_proc)

        print('Start finding all shortest paths...')
        batch_size = n_proc * sub_list_size
        for st in tqdm(range(0, len(pairs), batch_size)):
            # start mp
            jobs = []
            for i in range(n_proc):
                if st+i*sub_list_size >= len(pairs): continue
                sub_pairs = pairs[st+i*sub_list_size: st+(i+1)*sub_list_size]
                job = mp.Process(target=self.sub_find_shortest_paths, args=(sub_pairs, ))
                job.start()
                jobs.append(job)
            # sync
            for job in jobs:
                job.join()

    def save(self, fn='undirected_sp.pkl'):
        with open(fn, 'wb') as f:
            pickle.dump({'shortest_paths': dict(self.shortest_paths), 
                        'shortest_path_lengths': dict(self.shortest_path_lengths)}, f)
    
    def load(self, fn='undirected_sp.pkl'):
        with open(fn, 'rb') as f:
            saves = pickle.load(f)
            self.shortest_paths.update(saves['shortest_paths'])
            self.shortest_path_lengths.update(saves['shortest_path_lengths'])
    
    def preprocess_pairs(self, pairs, cutoff=float('inf'), n_proc=None):
        ''' Preprocess the head, tail node pairs by 
            (1) normalizing pairs, 
            (2) finding shortest path lengths and filtering by cutoff-length. 
        '''
        if n_proc is None:
            n_proc = mp.cpu_count()
        print(f'Multiprocessing shortest path finding, \nTotal # pairs={len(pairs)}\n# processes={n_proc}.')

        print('Normalizing source-target pairs (sorting source-target order)...')
        pairs = self.normalize_pairs(pairs)

        # First, filter out bad pairs (shortest path length > cutoff)
        print('Computing shortest path lengths...')
        if len(pairs) < 5000:
            self.find_shortest_path_length(pairs)
        else:
            self.find_shortest_path_length_multiproc(pairs, n_proc=n_proc, sub_list_size=10000)

        print(f'Filtering out too-long path or unreachable pairs...\nBefore: {len(pairs)}')
        pairs = [p for p in pairs if self.shortest_path_lengths[p] <= cutoff]
        print(f'After: {len(pairs)}')
        return pairs

    def pair2key(self, pair):
        source, target = pair
        return (max(source, target), min(source, target))

    def normalize_pairs(self, pairs):
        pairs = list(set([self.pair2key(p) for p in pairs]))
        return pairs
    
    def find_shortest_path_length(self, pairs):
        print('Finding shortest path lengths')
        for p in tqdm(pairs):
            try:
                length = nx.shortest_path_length(self.graph, p[0], p[1])
            except:
                length = float('inf')
            self.shortest_path_lengths[p] = length

    def find_shortest_path_length_multiproc(self, pairs, n_proc=None, sub_list_size=4):
        if n_proc is None:
            n_proc = mp.cpu_count()

        batch_size = n_proc * sub_list_size
        for st in tqdm(range(0, len(pairs), batch_size)):
            # start mp
            jobs = []
            for i in range(n_proc):
                if st+i*sub_list_size >= len(pairs): continue
                sub_pairs = pairs[st+i*sub_list_size: st+(i+1)*sub_list_size]
                job = mp.Process(target=self.sub_find_shortest_path_length, args=(sub_pairs, ))
                job.start()
                jobs.append(job)
            # sync
            for job in jobs:
                job.join()

    def sub_find_shortest_path_length(self, pairs):
        for p in pairs:
            try:
                length = nx.shortest_path_length(self.graph, p[0], p[1])
            except:
                length = float('inf')
            self.shortest_path_lengths[p] = length

    def sub_find_shortest_paths(self, pairs):
        for p in pairs:
            all_shortest_paths = []
            for path in nx.all_shortest_paths(self.graph, p[0], p[1]):
                all_shortest_paths.append(path[1:-1])   # only save the intermediate nodes on paths
            self.shortest_paths[p] = all_shortest_paths