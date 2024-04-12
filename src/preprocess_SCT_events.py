# # Preprocess the Story Cloze Test data 
# Process the SCT dataset to get event anchors.

import networkx as nx
from tqdm import tqdm
import os
os.environ['TOKENIZERS_PARALLELISM'] = 'true'
import pandas as pd
import multiprocessing as mp
from retrieval_pipeline import FreeTextProcessor, SemanticRetriever, free_text_pipeline

aser_path = '/path/to/data/core_100_normed_nocoocr.pkl'
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

# two data versions
data_2016 = {'valid': '/path/to/dataset/StoryClozeTest/raw/val_spring2016.csv',
     'test': '/path/to/dataset/StoryClozeTest/raw/test_spring2016.csv'}
data_2018 = {'valid': '/path/to/dataset/StoryClozeTest/raw/val_winter2018.csv',
     'test': '/path/to/dataset/StoryClozeTest/raw/test_winter2018.csv'}

def process_SCT_data(fn, freetextprocessor, retriever):
     data = pd.read_csv(fn)
     id2text = {}
     for line in tqdm(data.iterrows()):
          item = line[1]
          context = ' '.join([item['InputSentence1'], item['InputSentence2'], item['InputSentence3'], item['InputSentence4']])
          text = [context+' '+item['RandomFifthSentenceQuiz1'], context+' '+item['RandomFifthSentenceQuiz2']]
          id2text[item['InputStoryid']] = text
     
     # processing data
     id2data = {}
     for idx in tqdm(id2text):
          text = id2text[idx]
          id2data[idx] = [free_text_pipeline(text[0], freetextprocessor, retriever, topk=1), free_text_pipeline(text[1], freetextprocessor, retriever, topk=1)]

     return id2data

id2data = process_SCT_data(data_2016['valid'], proc, semretriever)
with open(os.path.join('/path/to/dataset/StoryClozeTest/raw/', 'event_valid2016.pickle'), 'wb') as f:
    import pickle
    pickle.dump(id2data, f)

id2data = process_SCT_data(data_2016['test'], proc, semretriever)
with open(os.path.join('/path/to/dataset/StoryClozeTest/raw/', 'event_test2016.pickle'), 'wb') as f:
    import pickle
    pickle.dump(id2data, f)

id2data = process_SCT_data(data_2018['valid'], proc, semretriever)
with open(os.path.join('/path/to/dataset/StoryClozeTest/raw/', 'event_valid2018.pickle'), 'wb') as f:
    import pickle
    pickle.dump(id2data, f)

id2data = process_SCT_data(data_2018['test'], proc, semretriever)
with open(os.path.join('/path/to/dataset/StoryClozeTest/raw/', 'event_test2018.pickle'), 'wb') as f:
    import pickle
    pickle.dump(id2data, f)