# coding: utf-8

import networkx as nx
from random import sample
from copy import deepcopy
from drqa.pipeline import TfidfDocRanker
from .ranker import SemanticDocRanker
from abc import abstractclassmethod

class BaseRetriever(object):
    @abstractclassmethod
    def _get_events(self, queries, n_docs=5):
        ''' Get events with retriever. Receives both str (single query) and list of str (batched queries) as input for ``queries``.
        '''
        raise NotImplementedError("_get_events not implemented")
    
    @abstractclassmethod
    def _filter(self, queries, events):
        ''' Filter the retrieved events given queries, according to defined rules.
        '''
        raise NotImplementedError("_filter not implemented")
    
    @abstractclassmethod
    def _init_ranker(self):
        ''' Initialize ranker. This should result in a self.ranker object which have closest_docs and batch_closest_docs apis.
        '''
        raise NotImplementedError("_init_ranker not implemented")

    def __init__(self, aser, ranker_args):
        self.ranker_args = ranker_args
        self._init_ranker()

        if not aser: raise FileNotFoundError('ASER not fed as input')
        else: self.aser = aser

    def __call__(self, input_query, n_docs=5, neighbors_per_event=5, n_return=10, rank_by='random', returns='event'):
        # input a piece of text, return the retrieval results (list)
        events = self._get_events(input_query, n_docs)

        if returns == 'event':
            return events

        # filtering stage
        if returns == 'filter':
            events = self._rule_filter(input_query, events)
            return events

        # get triples from the given events
        triples = []
        for event in events:
            neighbors = self.get_neighbors(event, with_cooccur=False, rank_by=rank_by, top_n=neighbors_per_event)
            if len(neighbors) > 0:
                triples.append(neighbors)

        triples = sum(triples, [])
        if len(triples) > n_return:
            triples = sample(triples, n_return)

        return triples

    def get_neighbors(self, target, with_cooccur=False, rank_by='random', top_n=5):
        succ = {}
        all_triples = []
        for n in self.aser.successors(target):
            succ[n] = deepcopy(list(self.aser.edges[target, n].values())[0])

            if not with_cooccur: 
                succ[n].pop('Co_Occurrence', None)
                if len(succ[n]) == 0:
                    del succ[n]
                    continue
            
            for rel in succ[n]:
                score = succ[n][rel]
                all_triples.append(((target, rel, n), score))

        pred = {}
        for n in self.aser.predecessors(target):
            pred[n] = deepcopy(list(self.aser.edges[n, target].values())[0])

            if not with_cooccur: 
                pred[n].pop('Co_Occurrence', None)
                if len(pred[n]) == 0:
                    del pred[n]
                    continue

            for rel in pred[n]:
                score = pred[n][rel]
                all_triples.append(((n, rel, target), score))
        
        if len(all_triples) > top_n:
            if rank_by=='random':
                all_triples = sample(all_triples, top_n)
            if rank_by == 'score':
                all_triples = list(sorted(all_triples, key=lambda x: x[-1], reverse=True)[:top_n])

        return all_triples

    def get_subgraph(self, ordered_events, cutoff=2, n_shortest_paths=3):
        assert len(ordered_events) > 1

        labels = {e:0 for e in ordered_events} # for distinguishing the original nodes and the later ones
        labels[ordered_events[-1]] = -1
        added_events = []
        
        for i in range(1, len(ordered_events)):
            for j in range(i):
                st, ed = ordered_events[j], ordered_events[i]
                try:
                    len_sp = nx.shortest_path_length(self.aser, st, ed)
                except:
                    len_sp = float('inf')
                
                if len_sp > cutoff or len_sp < 2: # when the shortest path is too long or too short
                    continue
                else:
                    path_ct = 0
                    for path in nx.all_shortest_paths(self.aser, st, ed):
                        # sometimes the paths are too much
                        if path_ct >= n_shortest_paths:
                            print(f'Too much shortest paths (>{path_ct}), skipping the rest...')
                            break

                        for node in path[1:-1]:
                            if node not in ordered_events and node not in added_events:
                                added_events.append(node)
                                labels[node] = len_sp
                        path_ct += 1
        
        all_events = ordered_events + added_events

        subgraph = self.aser.subgraph(all_events)

        assert len(subgraph.nodes) == len(labels)
        return {'subgraph': subgraph, 'labels': labels, 'all_events': all_events, 'input_events': ordered_events, 'intermediate_events': added_events}



class TfidfRetriever(BaseRetriever):
    ''' Tf-idf based document retriever, with DrQA's Tf-idf document ranker as backend. '''
    def __init__(self, aser=None, ranker_args={'tfidf_path': '/path/to/data/aser_retrieval_data/core_100/db/tfidf-tfidf-ngram=3-hash=16777216-tokenizer=simple.npz', 
                                                    'use_rule_filter': False}):
        super(TfidfRetriever, self).__init__(aser, ranker_args)

        self.use_rule_filter = ranker_args['use_rule_filter']

    def _init_ranker(self):
        self.ranker = TfidfDocRanker(tfidf_path=self.ranker_args['tfidf_path'])

    def _get_events(self, queries, n_docs):
        try:
            if isinstance(queries, str):
                queries = [queries]
            all_docids, all_doc_scores = zip(*self.ranker.batch_closest_docs(queries, k=n_docs))
        except Exception as e:
            print(e)
            all_docids, all_doc_scores = zip(*self.ranker.batch_closest_docs(['default' for _ in range(len(queries))], k=n_docs))
            # all_docids = []
            # all_doc_scores = []

        # retrieval_results = all_docids
        formatter = lambda ids, scores: [[(id_[i], score[i]) for i in range(len(id_))] for id_, score in zip(ids, scores)]
        retrieval_results = formatter(all_docids, all_doc_scores)
        return retrieval_results

    def _filter(self, input_query, results):
        output_results = []
        for item in results:
            # verb filter
            if input_query.split(' ')[1] not in item:
                continue
            # length filter
            if len(input_query) >= 2*len(item) or len(input_query) < len(item)/2:
                continue
            output_results.append(item)
        return output_results
        



class SemanticRetriever(BaseRetriever):
    ''' Semantic matching based retriever, with Faiss accelerated L2-distance matching and SBERT encoding '''
    def __init__(self, aser=None, ranker_args={'sbert_model':'all-MiniLM-L6-v2', 
                                                'use_gpu':True, 
                                                'nprobe':64, 
                                                'faiss_index_fn': '/path/to/data/aser_retrieval_data/core_100/faiss.index',
                                                'mapping_fn': "/path/to/data/aser_embedding/sbert-all-MiniLM-L6-v2-core100.hdf5"}):
        super(SemanticRetriever, self).__init__(aser, ranker_args)
    
    def _init_ranker(self):
        self.ranker = SemanticDocRanker(self.ranker_args)

    def _get_events(self, queries, n_docs=5):
        res = self.ranker.batch_closest_docs(queries, k=n_docs)

        return res

    def _filter(self, queries, events):
        return events




class Retriever:
    ''' Old retriever with tfidf. Deprecated. '''
    def __init__(self, tfidf_path='', db_path='', aser=None, use_rule_filter=False):
        print("WARNING: This retriever class is deprecated since it does not follow the inheritance. Use retrieval.TfidfRetreiver instead.")
        self.ranker = TfidfDocRanker(tfidf_path=tfidf_path)
        # self.db = DocDB(db_path=db_path)
        if not aser: raise FileNotFoundError('aser not found')
        else: self.aser = aser

        self.use_rule_filter = use_rule_filter

    def __call__(self, input_query, n_docs=5, neighbors_per_event=5, n_return=10, rank_by='random', returns=None):
        # input a piece of text, return the retrieval results (list)
        events = self._get_events(input_query, n_docs)

        if returns == 'event':
            return events

        # filtering stage
        if self.use_rule_filter:
            events = self._rule_filter(input_query, events)

        if returns == 'filter':
            return events

        # get triples from the given events
        triples = []
        for event in events:
            neighbors = self._get_neighbors(event, with_cooccur=False, rank_by=rank_by, top_n=neighbors_per_event)
            if len(neighbors) > 0:
                triples.append(neighbors)

        triples = sum(triples, [])
        if len(triples) > n_return:
            triples = sample(triples, n_return)

        return triples


    def _get_events(self, input_query, n_docs):
        try:
            all_docids, all_doc_scores = self.ranker.closest_docs(input_query, k=n_docs)
        except:
            all_docids = []

        retrieval_results = all_docids

        return retrieval_results

    def _rule_filter(self, input_query, results):
        output_results = []
        for item in results:
            # verb filter
            if input_query.split(' ')[1] not in item:
                continue
            # length filter
            if len(input_query) >= 2*len(item) or len(input_query) < len(item)/2:
                continue
            output_results.append(item)
        return output_results
        
    def _get_neighbors(self, target, with_cooccur=False, rank_by='random', top_n=5):
        succ = {}
        all_triples = []
        for n in self.aser.successors(target):
            succ[n] = deepcopy(list(self.aser.edges[target, n].values())[0])

            if not with_cooccur: 
                succ[n].pop('Co_Occurrence', None)
                if len(succ[n]) == 0:
                    del succ[n]
                    continue
            
            for rel in succ[n]:
                score = succ[n][rel]
                all_triples.append(((target, rel, n), score))

        pred = {}
        for n in self.aser.predecessors(target):
            pred[n] = deepcopy(list(self.aser.edges[n, target].values())[0])

            if not with_cooccur: 
                pred[n].pop('Co_Occurrence', None)
                if len(pred[n]) == 0:
                    del pred[n]
                    continue

            for rel in pred[n]:
                score = pred[n][rel]
                all_triples.append(((n, rel, target), score))
        
        if len(all_triples) > top_n:
            if rank_by=='random':
                all_triples = sample(all_triples, top_n)
            if rank_by == 'score':
                all_triples = list(sorted(all_triples, key=lambda x: x[-1], reverse=True)[:top_n])

        return all_triples
