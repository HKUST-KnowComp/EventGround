import os
import pickle
import networkx as nx
from tqdm import tqdm
from collections import defaultdict
from retrieval_pipeline import UndirectedShortestPathFinder

def process_file(event_fn, shortest_path_finder, aser, is_directed=False, quality_thresh=1):
    import pickle
    with open(event_fn, 'rb') as f:
        events = pickle.load(f)

    graph_info = process_graph(events, shortest_path_finder, aser, is_directed, quality_thresh)
    return graph_info

def process_graph(events, shortest_path_finder, aser, is_directed=False, quality_thresh=1):
    graph_info = {}
    for item_id in tqdm(range(len(events))):
        info = events[item_id]
        graph_list = []
        for cand_id in range(5):
            item = info[cand_id]

            context_events = [i['event'] for i in item]
            anchor_events = [i['anchor_events'][0] for i in item]

            subgraph, node_info = get_event_graph(context_events, 
                                        anchor_events, 
                                        shortest_path_finder,
                                        aser,
                                        is_directed=is_directed, quality_thresh=quality_thresh,
                                        node_info={})
            
            graph_list.append({'graph': subgraph, 'node_info': node_info})
        graph_info[item_id] = graph_list
        # print(graph_list[0]['graph'])
    return graph_info

def get_event_graph(context_events, anchor_events, 
                   shortest_path_finder, aser, 
                   edges={'context', 'ground', 'KG'},
                   add_context_node=True,
                   is_directed=True,
                   quality_thresh=1,
                   node_info=None):
    '''
    Args:
        context_events, anchor_events: the preprocessed event info
        shortest_path_finder: path finder which has a .get method, used for determining the shortest path between a pair of events
        KG: the event knowledge graph
        edge2id: A mapping from edge names to their indices
        edges: types of edges to include in the graph. 'context': between context events, 'hierarchy': between parent and child, 'KG': edges in the event KG
        is_directed: whether to return a directed graph or not
        quality_thresh: retrieval events' threshold, the events with L2-distance higher than this threshold will be discarded
    
    Outputs:
        graph (nodes-with type, edges-with type)
            node type: {type-0: context events, type-1: grounded events}
            edge type: Mapped to index with aser_utils.EDGE2ID/EDGE2ID_cooccur + special edges (subevent, event_progress, ...)
        node info (texts), with the pronouns [P0], ... recovered
    '''
    # data_dict = {(0, 0, 0): ([0,1,2],[1,2,0])}
    data_dict = defaultdict(list)
    if node_info is None:
        node_info = {}
    def get_node_id(node_info, node):
        if node in node_info:
            return node_info[node]
        else:
            tmp = len(node_info)
            node_info[node] = tmp
            return tmp

    # filter anchor events by quality_thresh
    anchor_events = [event[0] if event[1] < quality_thresh else None for event in anchor_events]

    if add_context_node:
        for i in range(len(context_events)-1):
            # add context edges to graph
            head, tail = context_events[i], context_events[i+1]
            head = get_node_id(node_info, head)
            tail = get_node_id(node_info, tail)

            data_dict[('context', 'context', 'context')].extend([(head, tail), (tail, head)])
    
        for i in range(len(context_events)):
            # add edge between context node and the anchors
            context_event = context_events[i]
            anchor_event = anchor_events[i]

            if anchor_event is None: 
                continue

            context_node_id = get_node_id(node_info, context_event)
            anchor_event_id = get_node_id(node_info, anchor_event)

            data_dict[('context', 'ground', 'KG')].append((context_node_id, anchor_event_id))
            data_dict[('KG', 'ground', 'context')].append((anchor_event_id, context_node_id))

    # find all shortest paths among the events (and determine the direction & edge type & coreference in aser)
    anchor_events = [event for event in anchor_events if event is not None]
    for i in range(len(anchor_events)):
        for j in range(i+1, len(anchor_events)):
            head, tail = anchor_events[i], anchor_events[j]
            shortest_paths = shortest_path_finder.get((head, tail))

            if isinstance(shortest_paths, list):
                # when there is shortest path
                for intermediate_nodes in shortest_paths:
                    nodes = [head]+intermediate_nodes+[tail]
                    # for all the pairs on the shortest path
                    for st in range(len(nodes)-1):
                        start, end = nodes[st], nodes[st+1]

                        # judge the direction
                        if (start, end) in aser.edges:
                            pass
                        elif (end, start) in aser.edges:
                            start, end = end, start
                        else:   # not in aser
                            continue
                        aser_key = (start, end)

                        # recover the personal id based on ASER's coreference and the event coreference
                        coref = eval(list(aser.edges[start, end]['coreference'])[0])
                        if coref['same']:
                            # e.g. head_tail_map = {'0': '1'} means the [P0] in head is the same person as [P1] in tail
                            head_tail_map = {map_h[-1]: map_t[-1] for map_h, map_t in coref['same']}
                            if st == 0:
                                # start is the head, end is an intermediate; can be align with the head
                                # head p == tail q && head p == z => tail q == z
                                for h_pers_id, t_pers_id in head_tail_map.items():
                                    end = end.replace(t_pers_id, h_pers_id)

                            elif st == len(nodes)-2:
                                # can be aligned with the tail
                                tail_head_map = {map_t[-1]: map_h[-1] for map_h, map_t in coref['same']}
                                for t_pers_id, h_pers_id in tail_head_map.items():
                                    start = start.replace(h_pers_id, t_pers_id)

                        # add edges
                        start_id, end_id = get_node_id(node_info, start), get_node_id(node_info, end)
                        for rel in aser.edges[aser_key]['relations']:
                            data_dict[('KG', rel, 'KG')].append((start_id, end_id))
                            if not is_directed:
                                data_dict[('KG', rel, 'KG')].append((end_id, start_id))

    for edge_type in data_dict:
        # eliminate repetition
        edges = list(set(data_dict[edge_type]))
        st, ed = list(zip(*edges))

        data_dict[edge_type] = (list(st), list(ed))

    return data_dict, node_info




import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--graph_output_name", type=str, default='graph_notop_undirected_nothresh_core100')
parser.add_argument("--is_directed", type=eval, default=False)
parser.add_argument("--quality_thresh", type=float, default=1)

args = parser.parse_args()

if __name__ == '__main__':
    upload_dir = '/path/to/dataset/NEEG_data/graphs/'
    # graph_output_name = 'graph_notop_undirected_nothresh_core100'
    graph_output_name = args.graph_output_name

    # hypter-parameters
    sp_fn = '/path/to/data/aser_shortest_paths/NEEG_sp_notop.pkl'
    aser_path = '/path/to/data/core_100_normed_nocoocr.pkl'

    # is_directed = True
    # is_directed = False 
    is_directed = args.is_directed
    # quality_thresh = 0.65
    # quality_thresh = 1 # no threshold
    quality_thresh = args.quality_thresh


    event_fn_list = ['/path/to/dataset/NEEG_data/NEEG_events/{}.pickle'.format(setname) for setname in ['dev', 'test', 'train']]
    graph_output_dir = os.path.join(upload_dir, graph_output_name)
    if not os.path.exists(graph_output_dir): os.mkdir(graph_output_dir)

    # start processing
    print('Loading shortest paths:', sp_fn.split('/')[-1])
    shortest_path_finder = UndirectedShortestPathFinder()
    shortest_path_finder.load(sp_fn)

    print('Loading ASER:', aser_path.split('/')[-1])
    aser = nx.read_gpickle(aser_path)
    # graph statistics
    print('# node', len(aser.nodes))
    print('# edge', len(aser.edges))


    for fn in event_fn_list:
        print('Processing', fn)

        out_fn = os.path.join(graph_output_dir, fn.split('/')[-1].replace('event', 'graph'))
        if not os.path.exists(out_fn):
            graph_info = process_file(fn, shortest_path_finder, aser, is_directed, quality_thresh)
            with open(out_fn, 'wb') as f:
                pickle.dump(graph_info, f)
        else:
            print('File already exists. Skipped...')
