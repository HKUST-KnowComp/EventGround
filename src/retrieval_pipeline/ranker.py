# coding: utf-8

import h5py
import numpy as np
from abc import abstractclassmethod, abstractmethod
from sentence_transformers import SentenceTransformer
from .faiss_utils import FaissWrapper

from collections import defaultdict

class Ranker:
    @abstractclassmethod
    def closest_docs(self, query, k=1):
        pass

    @abstractclassmethod
    def batch_closest_docs(self, queries, k=1):
        pass

    @abstractclassmethod
    def text2vec(self, queries):
        pass



class SemanticDocRanker(Ranker):
    def __init__(self, args={'sbert_model':'all-MiniLM-L6-v2', 
                            'use_gpu':True, 
                            'nprobe':64, 
                            'faiss_index_fn': '/path/to/data/aser_retrieval_data/core_100/faiss.index',
                            'mapping_fn': "/path/to/data/aser_embedding/sbert-all-MiniLM-L6-v2-core100.hdf5"}):
        
        self.args = args
        self.init_faiss()
        self.init_encoder()
        self.init_mapping()

    def init_faiss(self):
        self.nprobe = self.args['nprobe'] if self.args['nprobe'] is not None else 64
        self.faiss_index = FaissWrapper(use_gpu=True)
        self.faiss_index.load(self.args['faiss_index_fn'])
        if self.args['use_gpu'] is True:
            self.faiss_index.to_gpu() 

    def init_encoder(self):
        self.encoder = SentenceTransformer(self.args['sbert_model'])

    def init_mapping(self):
        f = h5py.File(self.args['mapping_fn'], 'r')
        self.words = f['words']
        self.mapping = lambda x: self.words[x].decode()

    def text2vec(self, queries):
        if isinstance(queries, str):
            return self.encoder.encode([queries])
        elif isinstance(queries, list):
            return self.encoder.encode(queries)
        else:
            raise TypeError("Wrong type for the text2vec input")

    def closest_docs(self, query, k=5):
        query = self.text2vec(query)
        # indices = self.faiss_index.search(query, k)[1].tolist()[0]
        scores, indices = self.faiss_index.search(query, k)
        scores = scores[0]
        result = [self.mapping(i) for i in indices[0]]
        return list(zip(result, scores))

    def batch_closest_docs(self, queries, k=5):
        queries = self.text2vec(queries)
        results = []
        for i in range(queries.shape[0]):
            scores, indices = self.faiss_index.search(np.expand_dims(queries[i], 0), k)
            scores = scores[0]
            result = [self.mapping(i) for i in indices[0]]
            # indices = self.faiss_index.search(np.expand_dims(queries[i], 0), k)[1].tolist()[0]
            results.append(list(zip(result, scores)))
        return results

