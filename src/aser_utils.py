
import networkx as nx
from heapq import nlargest

NUM_EDGETYPE = {
    'uni': 16,     # w/o reverse edges of KG edges
    'bi': 30,       # add rev-edges to KG related edge
}

EDGE2ID = {
    'Alternative': 0,
    'ChosenAlternative': 1,
    'Concession': 2,
    'Condition': 3,
    'Conjunction': 4,
    'Contrast': 5,
    'Exception': 6,
    'Instantiation': 7,
    'Precedence': 8,
    'Reason': 9,
    'Restatement': 10,
    'Result': 11,
    'Succession': 12,
    'Synchronous': 13,
    'context': 14,
    'ground': 15, 
    'rev-Alternative': 16,
    'rev-ChosenAlternative': 17,
    'rev-Concession': 18,
    'rev-Condition': 19,
    'rev-Conjunction': 20,
    'rev-Contrast': 21,
    'rev-Exception': 22,
    'rev-Instantiation': 23,
    'rev-Precedence': 24,
    'rev-Reason': 25,
    'rev-Restatement': 26,
    'rev-Result': 27,
    'rev-Succession': 28,
    'rev-Synchronous': 29,
}

NTYPE2ID = {
    'context': 0,
    'KG': 1
}

ID2NTYPE_LM_STR = {
    0: '<CTX>',
    1: '<KG>'
}

EDGE2ID_cooccur = {
    'Alternative': 0,
    'ChosenAlternative': 1,
    'Concession': 2,
    'Condition': 3,
    'Conjunction': 4,
    'Contrast': 5,
    'Exception': 6,
    'Instantiation': 7,
    'Precedence': 8,
    'Reason': 9,
    'Restatement': 10,
    'Result': 11,
    'Succession': 12,
    'Synchronous': 13,
    'Co_Occurrence': 14
}

top1000_kept_nodes = ['PersonX be sorry', 'PersonX die', 'PersonY be glad', 'PersonY love PersonX',
                      'PersonX love PersonY', 'PersonX do not care', 'PersonY do not care', 'PersonY be pretty sure',
                      'PersonX be pretty sure', 'PersonX wake up', 'PersonY wake up', 'PersonX smile', 'PersonY smile',
                      'PersonX be interested', 'PersonY be interested', 'PersonY be so glad', 'PersonX be so glad',
                      'PersonX sleep', 'PersonY sleep', 'PersonY like PersonX', 'PersonX like PersonY',
                      'PersonX go out',
                      'PersonY go out', 'PersonY like PersonX', 'PersonX like PersonY', 'PersonX go back',
                      'PersonY go back',
                      'PersonX be hungry', 'PersonY be hungry', 'PersonY kill PersonX', 'PersonX kill PersonY',
                      'PersonY meet PersonX', 'PersonX meet PersonY', 'PersonX be ready', 'PersonY be ready',
                      'PersonY apologize', 'PersonX apologize',
                      'PersonX feel better', 'PersonY feel better', 'the food be delicious', 'PersonY be pregnant',
                      'PersonX be pregnant',
                      'PersonY sit down', 'PersonX sit down', 'PersonX need to know', 'PersonY need to know',
                      'PersonY leave PersonX', 'PersonX leave PersonY',
                      'PersonX lie', 'PersonY lie', 'PersonY can understand', 'PersonX can understand',
                      'PersonX be sick',
                      'PersonY be sick',
                      'PersonX have time', 'PersonY have time', 'PersonX be curious', 'PersonY be curious',
                      'PersonY go away', 'PersonX go away', 'PersonX get up', 'PersonY get up',
                      'PersonX come home', 'PersonY come home', 'PersonY call PersonX', 'PersonX call PersonY',
                      'PersonX be a child', 'PersonY be a child', 'PersonY feel bad', 'PersonX feel bad',
                      'PersonY be crazy', 'PersonX be crazy', 'PersonX come out', 'PersonY come out',
                      'PersonY be worry', 'PersonX be worry', 'PersonX be marry', 'PersonY be marry',
                      'PersonX need PersonY', 'PersonY need PersonX', 'PersonX be drunk', 'PersonY be drunk',
                      'PersonX be okay', 'PersonY be okay', 'PersonX get out', 'PersonY get out',
                      'PersonX can not see', 'PersonY can not see', 'PersonX go home', 'PersonY go home',
                      'PersonY be surprise', 'PersonX be surprise', 'PersonX agree with PersonY',
                      'PersonY agree with PersonX']


def load_aser(aser_path='/path/to/data/core_100_normed_nocoocr.pkl', remove_top_degree=False):
    aser = nx.read_gpickle(aser_path)

    if remove_top_degree:
        kept_parts = set()
        for i in top1000_kept_nodes:
            i = i.split()
            if 'PersonX' in i: i.remove('PersonX')
            if 'PersonY' in i: i.remove('PersonY')
            i = ' '.join(i)
            kept_parts.add(i)
        nl = nlargest(1000, aser.degree, key=lambda x: x[1])
        remove_nodes = []
        for i, _ in nl:
            for part in kept_parts:
                if part in i:
                    break
            else:
                remove_nodes.append(i)

        aser.remove_nodes_from(remove_nodes)

    # graph statistics
    print('# node', len(aser.nodes))
    print('# edge', len(aser.edges))
    return aser






