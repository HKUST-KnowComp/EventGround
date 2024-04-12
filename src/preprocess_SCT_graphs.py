# # Preprocess the Story Cloze Test data 
# Process the SCT dataset to get subgraphs.

import os
import pickle
import networkx as nx
from tqdm import tqdm
from collections import defaultdict
from itertools import product
from retrieval_pipeline import UndirectedShortestPathFinder

def process_file(event_fn, shortest_path_finder, aser, is_directed=False, quality_thresh=1, abstract_level_range=(0, 3)):
    import pickle
    with open(event_fn, 'rb') as f:
        events = pickle.load(f)

    graph_info = process_graph(events, shortest_path_finder, aser, is_directed, quality_thresh, abstract_level_range)
    return graph_info

def get_event_graph(hierarchies, anchor_events, anchor_persons, 
                   shortest_path_finder, aser, 
                   edges={'context', 'ground', 'KG'},
                   add_context_node=True,
                   abstract_level_range=(0, 3),
                   is_directed=True,
                   quality_thresh=1,
                   node_info=None):
    '''
    Args:
        hierarchies, anchor_events, anchor_persons: the preprocessed event info
        shortest_path_finder: path finder which has a .get method, used for determining the shortest path between a pair of events
        KG: the event knowledge graph
        edge2id: A mapping from edge names to their indices
        edges: types of edges to include in the graph. 'context': between context events, 'hierarchy': between parent and child, 'KG': edges in the event KG
        abstract_level_range: a range (both inclusive), the events whose hierarchy level within this range will be considered
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

    # select event by the abstract level
    remv_levels = [i for i in range(5) if i < abstract_level_range[0] or i > abstract_level_range[1]]
    for hierarchy, anchor_event in zip(hierarchies, anchor_events):
        for level in remv_levels:
            if level in hierarchy:
                del hierarchy[level]
            if level in anchor_event:
                del anchor_event[level]

    if add_context_node:
        # add context edges to graph
        for i in range(len(hierarchies)-1):
            head, tail = ' '.join(hierarchies[i][0]), ' '.join(hierarchies[i+1][0])
            head = get_node_id(node_info, head)
            tail = get_node_id(node_info, tail)

            data_dict[('context', 'context', 'context')].extend([(head, tail), (tail, head)])

    # process the anchor events and anchored events graph
    anchors_list = []
    persons_list = []
    for hierarchy, anchor_event, anchor_person in zip(hierarchies, anchor_events, anchor_persons):
        context_node = ' '.join(hierarchy[0])
        context_node_id = get_node_id(node_info, context_node)
        # print(context_node, context_node_id)
        anchors = []
        persons = []

        for i in anchor_event:
            person_map = anchor_person[i]

            if anchor_event[i][0][1] < quality_thresh:
                tmp_anchor_event = anchor_event[i][0][0]

                # recover index in anchor events ([P0] -> [Pn])
                for to_, from_ in person_map.items():
                    tmp_anchor_event = tmp_anchor_event.replace(from_, to_)
                
                anchors.append(tmp_anchor_event)
                persons.append(person_map)

                if add_context_node:
                    # add edge between context node and the anchors
                    tmp_anchor_event_id = get_node_id(node_info, tmp_anchor_event)
                    data_dict[('context', 'ground', 'KG')].append((context_node_id, tmp_anchor_event_id))
                    data_dict[('KG', 'ground', 'context')].append((tmp_anchor_event_id, context_node_id))
        
        anchors_list.append(anchors)
        persons_list.append(persons)
    
    # find all shortest paths among the anchor events (and determine the direction & edge type & coreference in aser)
    for i in range(len(anchors_list)):
        for j in range(i+1, len(anchors_list)):
            heads, tails = anchors_list[i], anchors_list[j]

            for head_idx, tail_idx in product(range(len(heads)), range(len(tails))):
                head, tail = heads[head_idx], tails[tail_idx]

                shortest_paths = shortest_path_finder.get((head, tail))
                if isinstance(shortest_paths, list):
                    # when there is shortest path
                    for intermediate_nodes in shortest_paths:
                        nodes = [head]+intermediate_nodes+[tail]
                        # for all the pairs on the shortest path
                        for st in range(len(nodes)-1):
                            start, end = nodes[st], nodes[st+1]

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
                                    person_map = persons_list[i][head_idx]
                                    # head p == tail q && head p == z => tail q == z
                                    for to_, from_ in person_map.items():
                                        if from_ in head_tail_map:
                                            from_ = head_tail_map[from_]
                                            end = end.replace(from_, to_)

                                elif st == len(nodes)-2:
                                    # can be aligned with the tail
                                    tail_head_map = {map_t[-1]: map_h[-1] for map_h, map_t in coref['same']}
                                    person_map = persons_list[j][tail_idx]
                                    for to_, from_ in person_map.items():
                                        if from_ in tail_head_map:
                                            from_ = tail_head_map[from_]
                                            start = start.replace(from_, to_)

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

def process_graph(events, shortest_path_finder, aser, is_directed=False, quality_thresh=1, abstract_level_range=(0, 3)):
    graph_info = {}
    for item_id in tqdm(events):
        info = events[item_id]
        graph_list = []
        for cand_id in range(2):

            hierarchies = []
            anchor_events = []
            anchor_persons = []

            for event_id in range(len(info[cand_id]['events']['events'])):
                event_info = info[cand_id]['events']['events'][event_id]
                hierarchies.append(event_info['hierarchy'])
                anchor_events.append(event_info['anchor_events'])
                anchor_persons.append(event_info['anchor_persons'])

            subgraph, node_info = get_event_graph(hierarchies, 
                                        anchor_events, 
                                        anchor_persons, 
                                        shortest_path_finder,
                                        aser,
                                        is_directed=is_directed, quality_thresh=quality_thresh,
                                        abstract_level_range=abstract_level_range,
                                        node_info={})
            
            graph_list.append({'graph': subgraph, 'node_info': node_info})
        graph_info[item_id] = graph_list
    return graph_info


if __name__ == '__main__':
    # hypter-parameters
    sp_fn = '/path/to/data/aser_shortest_paths/SCT_sp_notop.pkl'
    aser_path = '/path/to/data/core_100_normed_nocoocr.pkl'

    is_directed = True
    # is_directed = False 
    quality_thresh = 0.65
    # quality_thresh = 1 # no threshold

    event_fn_list = ['/path/to/dataset/StoryClozeTest/raw/event_{}{}.pickle'.format(setname, year) for setname in ['valid', 'test'] for year in [2016, 2018]]
    graph_output_dir = '/path/to/dataset/StoryClozeTest/raw/graphs/graph_notop_directed_thresh_core100/'
    if not os.path.exists(graph_output_dir): os.mkdir(graph_output_dir)

    upload_dir = '/path/to/dataset/StoryClozeTest/raw/'

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
        graph_info = process_file(fn, shortest_path_finder, aser, is_directed, quality_thresh)

        out_fn = os.path.join(graph_output_dir, fn.split('/')[-1].replace('event', 'graph'))
        with open(out_fn, 'wb') as f:
            pickle.dump(graph_info, f)