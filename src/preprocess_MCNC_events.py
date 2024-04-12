''' Preprocess MCNC data to get events '''

import os
import pickle
import networkx as nx
from tqdm import tqdm

from retrieval_pipeline import FreeTextProcessor, SemanticRetriever, free_text_pipeline
from retrieval_pipeline.retrieval_utils import MCNC_script_pipeline

aser_path = '/path/to/data/core_100_normed_nocoocr.pkl'
data = {'train': '/path/to/dataset/NEEG_data/clean/train.pickle',
        'test': '/path/to/dataset/NEEG_data/clean/test.pickle',
        'dev': '/path/to/dataset/NEEG_data/clean/dev.pickle'}
output_dir = '/path/to/dataset/NEEG_data/NEEG_events'

aser = nx.read_gpickle(aser_path)
# graph statistics
print('# node', len(aser.nodes))
print('# edge', len(aser.edges))

proc = FreeTextProcessor()
semretriever = SemanticRetriever(aser, ranker_args={'sbert_model':'all-MiniLM-L6-v2', 
                                                'use_gpu':True, 
                                                'nprobe':64, 
                                                'faiss_index_fn': '/path/to/data/aser_retrieval_data/core_100_normed/faiss.index',
                                                'mapping_fn': "/path/to/data/aser_embedding/sbert-all-MiniLM-L6-v2-core100-normed.hdf5"})

def preprocess(dataset):
    def tuple2text(tpl):
        wrds = []
        for idx in [3, 0, 4, 5]:
            if tpl[idx] is not None:
                wrds.append(tpl[idx].replace('+', ' '))
        txt = ' '.join(wrds)
        return txt
    
    def subjobjset(tpl):
        wrds = set()
        for idx in [3, 4, 5]:
            if tpl[idx] is not None:
                wrds.add(tpl[idx])
        return wrds

    processed = []
    all_protag = []

    for item in tqdm(dataset):
        # print(item)
        contexts, candidates, _ = item

        unique = None
        for k in contexts+candidates:
            subjobj = subjobjset(k)
            if unique is None:
                unique = subjobj
            unique = unique.intersection(subjobj)
        if len(unique) == 1:
            pass
        else:
            unique = [None]
        all_protag.append(list(unique)[0])

        contexts = [tuple2text(i) for i in contexts]
        candidates = [tuple2text(i) for i in candidates]

        context = '. '.join(contexts)
        per_candidate_items = [context+'. '+i+'. ' for i in candidates]

        processed.append(per_candidate_items)
    
    return processed, all_protag

def process_MCNC_data(fn, freetextprocessor, retriever):

    import pickle
    with open(fn, 'rb') as f:
        data = pickle.load(f)

    # preprocess
    data, all_protag = preprocess(data)

    returns = []
    for i in tqdm(range(len(data))):
        item, protag = data[i], all_protag[i]
        events = MCNC_script_pipeline(item, protag, freetextprocessor, retriever)
        returns.append(events)
    
    return returns

for lbl in data:
    print(f'Processing the {lbl} set...')
    events = process_MCNC_data(data[lbl], proc, semretriever)
    with open(os.path.join(output_dir, lbl+'.pickle'), 'wb') as f:
        pickle.dump(events, f)