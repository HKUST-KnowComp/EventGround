import os
import sys
sys.path.append('..')

import pickle
import dgl
import torch
import datasets
from torch.utils.data.dataset import Dataset
from collections import defaultdict
from datasets import load_dataset
from aser_utils import NTYPE2ID, EDGE2ID, ID2NTYPE_LM_STR

class GraphMCDataset(Dataset):
    def __init__(self, data=None, add_rev_edges=False, preprocess=True):
        if preprocess:
            for item in data:
                self.preprocess_item(item, add_rev_edges)
        self.data = data
    
    def __len__(self):
        return len(self.data)

    def preprocess_item(self, item, add_rev_edges=False):
        item['graph0']['graph'] = self.get_dgl_graph(item['graph0']['graph'], add_rev_edges)
        item['graph1']['graph'] = self.get_dgl_graph(item['graph1']['graph'], add_rev_edges)
        
    def get_dgl_graph(self, data_dict, add_rev_edges=False):
        """ Re-order the input graph data_dict to generate dgl format graphs (with node and edge types)
        """
        new_data_dict = defaultdict(list)
        ntypes = {}
        etypes = defaultdict(list)

        # re-order the data_dict so that node types are merged;
        # then get edge types, node types
        for edge_type in data_dict:
            edges = data_dict[edge_type]
            n_edges = len(edges[0])
            edge_type_str = edge_type[1]

            # get edge types and node types
            etype = EDGE2ID[edge_type_str]
            rev_etype = EDGE2ID.get('rev-'+edge_type_str, None)
            src_node_type, dst_node_type = NTYPE2ID[edge_type[0]], NTYPE2ID[edge_type[2]]

            # reorder data dict
            new_edge_type = ('node', edge_type_str, 'node')
            if add_rev_edges and (edge_type_str not in {'context', 'ground'}):
                # add rev edges to dict
                src, dst = edges
                all_edges = (src+dst, dst+src)
                # collect edge types
                etypes[new_edge_type].extend([etype for _ in range(n_edges)]+[rev_etype for _ in range(n_edges)])
            else:
                all_edges = edges
                # collect edge types
                etypes[new_edge_type].extend([etype for _ in range(n_edges)])
            new_data_dict[new_edge_type].append(all_edges)

            # collect node types
            for src, dst in zip(*edges):
                ntypes[src] = src_node_type
                ntypes[dst] = dst_node_type
        
        # merge edges (since the edge type is merged after ignoring node types)
        for edge_type in new_data_dict:
            edges = new_data_dict[edge_type]
            sts = [item[0] for item in edges]
            eds = [item[1] for item in edges]
            new_data_dict[edge_type] = (sum(sts, []), sum(eds, []))

        g = dgl.heterograph(new_data_dict)
        # print(ntypes, len(ntypes))
        g.ndata['ntypes'] = torch.tensor([ntypes.get(i, NTYPE2ID['KG']) for i in range(g.num_nodes())])
        if len(etypes) > 1:
            g.edata['etypes'] = {key: torch.tensor(value) for key, value in etypes.items()}
        else:
            # when there is only one edge type, HeteroEdgeDataView requires that the tensor be passed directly
            for key in etypes:  # which only has one key
                g.edata['etypes'] = torch.tensor(etypes[key])
        g = dgl.to_homogeneous(g, ndata=['ntypes'], edata=['etypes'])
        return g

    def __getitem__(self, x):
        return self.data[x]

    @classmethod
    def make_datasets(cls, seed=2022, data_version=2016, add_rev_edges=False,
        sct_data_dir = '/path/to/dataset/StoryClozeTest/raw/',
        graph_data_dir = '/path/to/dataset/StoryClozeTest/raw/graphs', 
        graph_name = 'graph_notop_directed_nothresh_core100'):
        """ Preprocess and save dataset to file
        """
        output_dir = os.path.join(sct_data_dir, 'graph_datasets')
        # output dataset file name
        output_dataset_fn = os.path.join(output_dir, graph_name+'{}_{}_{}.pickle'.format(data_version, seed, 'rev' if add_rev_edges else 'no-rev'))
        # output graph file name
        output_graph_fn = os.path.join(output_dir, graph_name+'{}_{}.pickle'.format(data_version, seed))

        # if there already is a dataset file, load it and go ahead
        if os.path.exists(output_dataset_fn):
            with open(output_dataset_fn, 'rb') as f:
                all_data = pickle.load(f)
            return {key: cls(all_data[key], add_rev_edges, preprocess=False) for key in all_data}

        # if there is saved graphs, make datasets from them
        if os.path.exists(output_graph_fn):
            with open(output_graph_fn, 'rb') as f:
                data = pickle.load(f)
            returns = {key: cls(data[key], add_rev_edges) for key in data}
        else:
            os.makedirs(output_dir, exist_ok=True)
            train, valid, test = get_datasets(seed, data_version, sct_data_dir, graph_data_dir, graph_name)
            data = {'train': train, 'valid': valid, 'test': test}
            # cache graphs
            with open(output_graph_fn, 'wb') as f:
                pickle.dump(data, f)

            returns = {key: cls(data[key], add_rev_edges) for key in data}

        # cache datasets
        with open(output_dataset_fn, 'wb') as f:
            pickle.dump({key: returns[key].data for key in returns}, f)

        return returns


class GraphCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        """ Collate a list of graphs, where the nodes have text features.
        """
        all_graphs = []
        all_node_info = []
        all_node2textid = []
        unique_text2id = {}
        all_labels = []

        ' prepare graph info '
        for item in batch:
            all_graphs.extend([item['graph0']['graph'], item['graph1']['graph']])
            all_node_info.extend([item['graph0']['node_info'], item['graph1']['node_info']])
            all_labels.append(item['label'])

        for node_info in all_node_info:
            nodes = list(node_info)
            nodes.sort(key=lambda x: node_info[x])
            # print(nodes, node_info)

            for node in nodes:
                if node not in unique_text2id:
                    unique_text2id[node] = len(unique_text2id)
                
                all_node2textid.append(unique_text2id[node])
        batched_graph = dgl.batch(all_graphs)

        ' node text info '
        all_texts = list(unique_text2id)
        all_texts.sort(key=lambda x:unique_text2id[x])
        tokenized = self.tokenizer(all_texts, truncation=True, padding='longest', return_tensors='pt')
        tokenized = (tokenized['input_ids'], tokenized['attention_mask'])
        
        all_labels = torch.tensor(all_labels)

        return {'graph': batched_graph, 'texts': all_texts, 'tokenized': tokenized, 'node2textid': all_node2textid, 'labels': all_labels}


class TextGraphCollator:
    def __init__(self, tokenizer, specify_node_type=False, max_num_nodes=160):
        # make sure your tokenizer has added the node type tokens
        self.tokenizer = tokenizer
        self.specify_node_type = specify_node_type
        self.max_num_nodes = max_num_nodes

    def __call__(self, batch):
        """ Collate a list of graphs, where the nodes have text features.
        """
        all_graphs = []
        all_node_info = []
        all_node2textid = []
        unique_text2id = {}     # node text -> unique text id
        unique_text2ntype = {}  # node text -> node type id
        all_labels = []

        all_original_texts = []

        ' prepare graph info '
        for item in batch:
            for i in range(2):
                cand_graph = item['graph'+str(i)]['graph']
                cand_node_info = item['graph'+str(i)]['node_info']
                cand_orig_text = item['text'+str(i)]
                if cand_graph.num_nodes() > self.max_num_nodes:
                    cand_graph = cand_graph.subgraph(range(self.max_num_nodes))
                    cand_node_info = {key: cand_node_info[key] for key in cand_node_info if cand_node_info[key] < self.max_num_nodes}
                all_graphs.append(cand_graph)
                all_node_info.append(cand_node_info)
                all_original_texts.append(cand_orig_text)
            # all_graphs.extend([item['graph0']['graph'], item['graph1']['graph']])
            # all_node_info.extend([item['graph0']['node_info'], item['graph1']['node_info']])
            all_labels.append(item['label'])
            # original sct texts
            # all_original_texts.extend([item['text0'], item['text1']])

        for graph, node_info in zip(all_graphs, all_node_info):
            ntypes = graph.ndata['ntypes']  # an index tensor indicating the node types
            nodes = list(node_info)

            nodes.sort(key=lambda x: node_info[x])

            for node, ntype in zip(nodes, ntypes.tolist()):
                if node not in unique_text2id:
                    unique_text2id[node] = len(unique_text2id)
                    unique_text2ntype[node] = ntype 
                
                all_node2textid.append(unique_text2id[node])
        batched_graph = dgl.batch(all_graphs)
        
        'node texts'
        all_texts = list(unique_text2id)
        all_texts.sort(key=lambda x:unique_text2id[x])

        if self.specify_node_type:
            # add special tokens for each node text to distinguish them
            all_texts = [ID2NTYPE_LM_STR[unique_text2ntype[text]]+' '+text for text in all_texts]

        tokenized = self.tokenizer(all_texts, truncation=True, padding='longest', return_tensors='pt', max_length=512)
        tokenized = (tokenized['input_ids'], tokenized['attention_mask'])

        'original instance texts'
        orig_tokenized = self.tokenizer(all_original_texts, truncation=True, padding='longest', return_tensors='pt', max_length=512)
        orig_tokenized = (orig_tokenized['input_ids'].view(len(batch), 2, -1), orig_tokenized['attention_mask'].view(len(batch), 2, -1))

        all_labels = torch.tensor(all_labels)

        return {'graph': batched_graph, 'tokenized': tokenized, 'text_tokenized': orig_tokenized,
                 'node2textid': all_node2textid, 'etypes': batched_graph.edata.get('etypes', None), 'labels': all_labels}


def get_datasets(seed, data_version=2016, 
        sct_data_dir = '/path/to/dataset/StoryClozeTest/raw/',
        graph_data_dir = '/path/to/dataset/StoryClozeTest/raw/graphs', 
        graph_name = 'graph_notop_directed_nothresh_core100'):
    """ Load original SCT datasets and the preprocessed graph data to compose the graph multiple choice dataset.
    """

    sct_datasets = {2016: {'valid': 'val_spring2016.csv',
                            'test': 'test_spring2016.csv'},
                    2018: {'valid': 'val_winter2018.csv',
                            'test': 'test_winter2018.csv'}}

    data_files = {key: os.path.join(sct_data_dir, sct_datasets[data_version][key]) for key in sct_datasets[data_version]}
    print(data_files)

    if data_version == 2016:
        dataset = load_dataset('csv', data_files=data_files)
        # split the original valid set into train/valid sets
        returns = dataset['valid'].train_test_split(test_size=100, train_size=len(dataset['valid'])-100, seed=seed)
        train, valid, test = returns['train'], returns['test'], dataset['test']
    elif data_version == 2018:
        # 2018 debiased version: test set does not have labels
        dataset_valid = load_dataset('csv', data_files={'valid': data_files['valid']})['valid']
        dataset_test = load_dataset('csv', data_files={'test': data_files['test']})['test']

        returns = dataset_valid.train_test_split(test_size=100, train_size=len(dataset_valid)-100, seed=seed)
        # returns['valid'], returns['test'] = returns['test'], dataset_test
        train, valid, test = returns['train'], returns['test'], dataset_test

    # compose the new dataset
    dataset = datasets.dataset_dict.DatasetDict({'train': train, 
                                             'valid': valid, 
                                             'test': test})

    train_set = list(dataset['train'])
    valid_set = list(dataset['valid'])
    test_set = list(dataset['test'])

    # prepare graph data filenames

    filenames = {2016: {'valid': 'graph_valid2016.pickle',
                        'test': 'graph_test2016.pickle'},
                2018: {'valid': 'graph_valid2018.pickle',
                        'test': 'graph_test2018.pickle'}
    }

    for yr in filenames:
        for lbl in filenames[yr]:
            filenames[yr][lbl] = os.path.join(graph_data_dir, graph_name, filenames[yr][lbl])
    
    print(filenames[data_version])

    # load graph data and make SCT datasets
    with open(filenames[data_version]['valid'], 'rb') as f:
        graph_valid = pickle.load(f)
    with open(filenames[data_version]['test'], 'rb') as f:
        graph_test = pickle.load(f)

    def get_candidate_texts(item):
        context = ' '.join([item['InputSentence1'], 
                item['InputSentence2'], 
                item['InputSentence3'], 
                item['InputSentence4']])
        candidates = [context + ' ' + item['RandomFifthSentenceQuiz1'], context + ' ' + item['RandomFifthSentenceQuiz2']]
        return candidates

    train_valid_test = []
    for data in [train_set, valid_set]:
        new_set = []
        for i in range(len(data)):
            id_ = data[i]['InputStoryid']
            candidates = get_candidate_texts(data[i])
            tmp = {'label': data[i]['AnswerRightEnding']-1,
                    'text0': candidates[0],
                    'text1': candidates[1],
                    'id': id_}
            tmp['graph0'] = graph_valid[id_][0]
            tmp['graph1'] = graph_valid[id_][1]
            new_set.append(tmp)
        train_valid_test.append(new_set)

    new_set = []
    for i in range(len(test_set)):
        id_ = test_set[i]['InputStoryid']

        if 'AnswerRightEnding' not in test_set[i]:
            test_set[i]['AnswerRightEnding'] = 1
        
        candidates = get_candidate_texts(test_set[i])
        tmp = {'label': test_set[i]['AnswerRightEnding']-1,
                'text0': candidates[0],
                'text1': candidates[1],
                'id': id_}
        tmp['graph0'] = graph_test[id_][0]
        tmp['graph1'] = graph_test[id_][1]
        new_set.append(tmp)
    train_valid_test.append(new_set)

    train_set, valid_set, test_set = train_valid_test
    return train_set, valid_set, test_set