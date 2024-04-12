import os
import sys
sys.path.append('..')
import pickle
import dgl
import torch
from tqdm import tqdm
from torch.utils.data.dataset import Dataset
from collections import defaultdict
from aser_utils import NTYPE2ID, EDGE2ID, ID2NTYPE_LM_STR
from dataset import load_NEEG_datasets

class GraphMCDataset(Dataset):
    def __init__(self, data=None, add_rev_edges=False, preprocess=True):
        if preprocess:
            for item in tqdm(data):
                self.preprocess_item(item, add_rev_edges)
        self.data = data
    
    def __len__(self):
        return len(self.data)

    def preprocess_item(self, item, add_rev_edges=False):
        for i in range(5):
            item['graph'+str(i)]['graph'] = self.get_dgl_graph(item['graph'+str(i)]['graph'], add_rev_edges)
        
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
    def make_datasets(cls, add_rev_edges=False,
        neeg_data_dir = '/path/to/dataset/NEEG_data/clean',
        graph_data_dir = '/path/to/dataset/NEEG_data/graphs', 
        graph_name = 'graph_notop_directed_nothresh_core100'):
        """ Preprocess and save dataset to file
        """
        output_dir = os.path.join(neeg_data_dir, 'graph_datasets')
        # output dataset file name
        output_dataset_fn = os.path.join(output_dir, graph_name+'_{}.pickle'.format('rev' if add_rev_edges else 'no-rev'))
        # output graph file name
        output_graph_fn = os.path.join(output_dir, graph_name+'.pickle')

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
            train, valid, test = get_datasets(neeg_data_dir, graph_data_dir, graph_name)
            data = {'train': train, 'valid': valid, 'test': test}
            # cache graphs
            with open(output_graph_fn, 'wb') as f:
                pickle.dump(data, f)

            returns = {key: cls(data[key], add_rev_edges) for key in data}

        # cache datasets
        with open(output_dataset_fn, 'wb') as f:
            pickle.dump({key: returns[key].data for key in returns}, f)

        return returns

def get_datasets(neeg_data_dir = '/path/to/dataset/NEEG_data/clean',
        graph_data_dir = '/path/to/dataset/NEEG_data/graphs', 
        graph_name = 'graph_notop_directed_nothresh_core100'):
    """ Load original NEEG datasets and the preprocessed graph data to compose the graph multiple choice dataset.
    """
    datasets = load_NEEG_datasets(neeg_data_dir, use_lemmatized_verb=False)

    # load graph data and make neeg datasets
    filenames = {'train': 'train.pickle', 
                'dev': 'dev.pickle',
                'test': 'test.pickle'}
    for key in filenames:
        filenames[key] = os.path.join(graph_data_dir, graph_name, filenames[key])

    all_datasets = {}
    for key in filenames:
        new_set = []
        data = datasets[key]
        with open(filenames[key], 'rb') as f:
            graph = pickle.load(f)
        for i in range(len(data)):
            id_ = i
            candidates = data[i]['text']
            tmp = {'label': data[i]['label'],
                    'text0': candidates[0],
                    'text1': candidates[1],
                    'text2': candidates[2],
                    'text3': candidates[3],
                    'text4': candidates[4],
                    'graph0': graph[i][0],
                    'graph1': graph[i][1],
                    'graph2': graph[i][2],
                    'graph3': graph[i][3],
                    'graph4': graph[i][4],
                    'id': id_}
            new_set.append(tmp)
        all_datasets[key] = new_set
    return all_datasets['train'], all_datasets['dev'], all_datasets['test']



class TextGraphCollator:
    def __init__(self, tokenizer, specify_node_type=False, max_num_nodes=80):
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
            for i in range(5):
                cand_graph = item['graph'+str(i)]['graph']
                cand_node_info = item['graph'+str(i)]['node_info']
                cand_orig_text = item['text'+str(i)]
                # ###### MODIFIED HERE ######
                # cand_orig_text = '. '.join([txt.capitalize() for txt in cand_orig_text.split(' ## ')])
                # ###########################
                if cand_graph.num_nodes() > self.max_num_nodes:
                    cand_graph = cand_graph.subgraph(range(self.max_num_nodes))
                    cand_node_info = {key: cand_node_info[key] for key in cand_node_info if cand_node_info[key] < self.max_num_nodes}
                all_graphs.append(cand_graph)
                all_node_info.append(cand_node_info)
                all_original_texts.append(cand_orig_text)
                    
            # all_graphs.extend([item['graph'+str(i)]['graph'] for i in range(5)])
            # all_node_info.extend([item['graph'+str(i)]['node_info'] for i in range(5)])
            all_labels.append(item['label'])
            # original sct texts
            # all_original_texts.extend([item['text'+str(i)] for i in range(5)])

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

        tokenized = self.tokenizer(all_texts, truncation=True, padding='longest', return_tensors='pt')
        tokenized = (tokenized['input_ids'], tokenized['attention_mask'])

        'original instance texts'
        orig_tokenized = self.tokenizer(all_original_texts, truncation=True, padding='longest', return_tensors='pt')
        orig_tokenized = (orig_tokenized['input_ids'].view(len(batch), 5, -1), orig_tokenized['attention_mask'].view(len(batch), 5, -1))

        all_labels = torch.tensor(all_labels)

        return {'graph': batched_graph, 'tokenized': tokenized, 'text_tokenized': orig_tokenized,
                 'node2textid': all_node2textid, 'etypes': batched_graph.edata.get('etypes', None), 'labels': all_labels}



if __name__ == '__main__':
    # datasets = GraphMCDataset.make_datasets(add_rev_edges=True, 
    #                 neeg_data_dir = '/path/to/dataset/NEEG_data/clean',
    #                 graph_data_dir = '/path/to/dataset/NEEG_data/graphs', 
    #                 graph_name = 'graph_notop_directed_nothresh_core100')
    
    # datasets = GraphMCDataset.make_datasets(add_rev_edges=False, 
    #                 neeg_data_dir = '/path/to/dataset/NEEG_data/clean',
    #                 graph_data_dir = '/path/to/dataset/NEEG_data/graphs', 
    #                 graph_name = 'graph_notop_directed_nothresh_core100')



    # datasets = GraphMCDataset.make_datasets(add_rev_edges=True, 
    #                 neeg_data_dir = '/path/to/dataset/NEEG_data/clean',
    #                 graph_data_dir = '/path/to/dataset/NEEG_data/graphs', 
    #                 graph_name = 'graph_notop_directed_thresh_core100')
    
    datasets = GraphMCDataset.make_datasets(add_rev_edges=False, 
                    neeg_data_dir = '/path/to/dataset/NEEG_data/clean',
                    graph_data_dir = '/path/to/dataset/NEEG_data/graphs', 
                    graph_name = 'graph_notop_directed_thresh_core100')