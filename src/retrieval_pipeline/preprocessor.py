import nltk

from collections import defaultdict
from .nlp_utils import AllenNLPWrapper, CoreNLPWrapper
from .normalization import ParsingBasedNormalizer

class FreeTextProcessor:
    def __init__(self, args={'allennlp_args':{'coref_model': "https://storage.googleapis.com/allennlp-public-models/coref-spanbert-large-2021.03.10.tar.gz",
                                                'srl_model': "https://storage.googleapis.com/allennlp-public-models/structured-prediction-srl-bert.2020.12.15.tar.gz"}, 
                            'corenlp_args':{'path': "/home/ubuntu/stanfordcorenlp/stanford-corenlp-4.4.0",
                                            'port': 10086,
                                            'annotators': ["tokenize", "ssplit", "pos", "lemma", "ner", "parse", "coref"]}}):
        self.args = args
        self.init_models()

    def init_models(self):
        self.allennlp = AllenNLPWrapper(self.args['allennlp_args'])
        self.corenlp = CoreNLPWrapper(self.args['corenlp_args'])
        self.nmlzer = ParsingBasedNormalizer()

    def __call__(self, inputs):
        return self.predict_and_normalize_and_generalize(inputs)

    def predict(self, inputs):
        if isinstance(inputs, str):
            srl_results = self.allennlp.predict_srl(nltk.sent_tokenize(inputs), True)
            allen_coref_results = self.allennlp.predict_coref(inputs, True)
            core_coref_results = self.corenlp.predict_coref(inputs, True)
        elif isinstance(inputs, list):
            srl_results = self.allennlp.predict_srl(inputs, True)
            inputs = ' '.join(inputs)
            allen_coref_results = self.allennlp.predict_coref(inputs, True)
            core_coref_results = self.corenlp.predict_coref(inputs, True)
        # print(srl_results)

        # get person mentions and their spans
        persons = self.find_persons(allen_coref_results, core_coref_results)
        
        # get events
        events = self.find_events(srl_results)

        return persons, events, allen_coref_results, core_coref_results

    def predict_and_normalize(self, inputs):
        persons, events, _, core_coref_results = self.predict(inputs)
        document = sum([item['words'] for item in core_coref_results['parsed_info']], [])
        find_subevent_relations(events)
        normalize_events(events, persons, document=document, save_original_event=True)
        return {'persons': persons, 'events': events}

    def predict_and_generalize(self, inputs):
        results = self.predict_and_normalize(inputs)
        generalize_events(results['events'], from_='original')
        return results

    def predict_and_normalize_and_generalize(self, inputs):
        results = self.predict_and_normalize(inputs)
        generalize_events(results['events'], from_='normalized')
        return results
    
    def find_events(self, srl_results):
        document = []
        event_list = []
        sent_len = 0
        for item in srl_results:
            document.extend(item['words'])
            for event in item['verbs']:
                if len(event['span']) <= 1 or \
                    'V' not in event['span'] or \
                    ('ARG0' not in event['span'] and 'ARG1' not in event['span']): 
                    # no-verb or only-verb events or verb & modifiers events
                    continue
                event_info = {}
                event_info['verb'] = event['verb']
                spans = event['span']
                for arg in spans:
                    st, ed = spans[arg]
                    spans[arg] = (st+sent_len, ed+sent_len)
                event_info['span'] = spans
                event_list.append(event_info)
            
            sent_len += len(item['words'])
        return {'events': event_list, 'document': document}


    def find_persons(self, allen_coref_results, core_coref_results):
        # find personal word spans + classify (per sentence), re-calculate indices
        info_list = core_coref_results['parsed_info']
        person_list = []
        bias = 0
        
        for info in info_list:
            person_spans = self.nmlzer.get_personal_words(info)
            coref = self.nmlzer.node_person_coref(person_spans, info)

            # recalculate indices
            new_coref = {}
            new_coref['subset'] = coref['subset']
            persons = {}
            for p in coref['persons']:
                # (((0, 1), {'target': [0], 'target_word': ['I']}, ['I']), 'default')
                mention_list = []
                for mention in coref['persons'][p]:
                    # print(mention)
                    span = (mention[0][0][0]+bias, mention[0][0][1]+bias)
                    target = [i+bias for i in mention[0][1]['target']]
                    label = mention[1]

                    mention_list.append({'span': span, 'target': target, 'label': label})
                persons[p] = mention_list
            new_coref['persons'] = persons
            person_list.append(new_coref)
            
            bias += len(info['words'])
        
        # cross sentence coref
        document, clusters = allen_coref_results[0]['document'], allen_coref_results[0]['clusters']

        # gather local and cross-sentence coref to {person -> mention_list}
        persons = defaultdict(dict)
        clusters = [set([tuple(i) for i in cluster]) for cluster in clusters]
        clusters_is_person = [False for _ in range(len(clusters))]

        for sent_info in person_list:
            for p in sent_info['persons']:
                mention_list = sent_info['persons'][p]
                # map mention span to cluster
                target_pid = None
                for mention in mention_list:
                    for pid, cluster in enumerate(clusters):
                        if mention['span'] in cluster:
                            # success: this person in cluster i
                            target_pid = pid
                            break
                    else:
                        # this mention not in any cluster
                        assert target_pid is None
                        continue
                    
                    if target_pid is not None:
                        break

                if target_pid is None:   # p not in cluster, create one
                    new_cluster = set([mention['span'] for mention in mention_list])
                    clusters.append(new_cluster)
                    clusters_is_person.append(False)
                    target_pid = len(clusters) - 1

                # p in cluster
                # update cluster spans in ``persons`` if necessary
                if clusters_is_person[target_pid] is False:
                    clusters_is_person[target_pid] = True
                    for span in clusters[target_pid]:
                        persons[target_pid][span] = 'default'
                # update corenlp spans in ``persons``
                for mention in mention_list:
                    persons[target_pid][mention['span']] = mention['label']

        # identify subset by spans
        is_subspan = lambda span1, span2: span1[0] >= span2[0] and span1[1] <= span2[1]
        def _find_subset_and_record(i, j, is_subset):
            for span_i in persons[i]:
                if persons[i][span_i] == 'possessive':
                    continue
                for span_j in persons[j]:
                    if is_subspan(span_i, span_j):
                        is_subset[i] = j
                        break
                else:
                    continue
                break

        is_subset = {}
        for i in persons:
            for j in persons:
                if i >= j: continue
                _find_subset_and_record(i, j, is_subset)
                _find_subset_and_record(j, i, is_subset)

        return {'persons': persons, 'is_subset': is_subset, 'document': document}



def is_sub_span(span_a, span_b):
    # check whether span-a is contained in span-b, return True if so
    return span_a[0]>=span_b[0] and span_a[1]<=span_b[1]

def is_overlapped(span_a, span_b):
    # check whether span-a is overlapped on span-b
    return not(span_a[0]>=span_b[1] or span_a[1]<=span_b[0])
    
def span_to_word(words, span, normalize=False, normalize_info={'span2person': None, 'person2freq': None}):
    ''' Given basic SRL events info, retrieve the words and (if normalize is set to True) normalize the event.
    '''
    res = {}

    span2person = normalize_info['span2person']
    person2freq = normalize_info['person2freq']
    
    for key in span:
        st, ed = span[key]
        wrds = words[st:ed]

        if normalize:
            # normalize the event spans with personal corefs
            hits = []
            if key != 'V': 
                for sp in span2person:
                    if is_sub_span(sp, (st,ed)):
                        hits.append((sp, span2person[sp]))
            
            if hits:
                hits.sort(key=lambda x:x[0][0])

                if len(hits) > 1:
                    # resolve conflicts: sometimes there are overlapped person mentions
                    distinct_hits = [hits[0]]

                    for mention in hits[1:]:
                        target = distinct_hits[-1]
                        if is_overlapped(target[0], mention[0]):
                            # resolve by global frequency
                            if person2freq[target[1][0]] < person2freq[mention[1][0]]:
                                distinct_hits.pop()
                                distinct_hits.append(mention)
                        else:
                            distinct_hits.append(mention)
                    hits = distinct_hits
                
                # get normalized word list for each argument
                pre_st = 0
                normed_wrds = []
                for mention in hits:
                    m_st, m_ed = mention[0]
                    m_st -= st
                    m_ed -= st
                    normed_wrds.extend(wrds[pre_st: m_st])
                    
                    pid, tag = mention[1]
                    postfix = "'s" if tag == 'possessive' else ""
                    normed_wrds.append(f"[P{pid}{postfix}]")

                    pre_st = m_ed
                normed_wrds.extend(wrds[pre_st:])
                
                res[key] = normed_wrds

            else:
                res[key] = wrds
            
        else:
            res[key] = wrds
    return res

def find_subevent_relations(events):
    ''' Find event-subevent relations based on verb-argument overlap. 

    Arg:
        events: the output from FreeTextProcessor.predict
    '''
    all_events = events['events']

    for i, event in enumerate(all_events):
        verb_span, arg_spans = None, []
        spans = event['span']
        for name in spans:
            if name == 'V':
                verb_span = spans[name]
            else:
                arg_spans.append(spans[name])
        if verb_span is None:
            # when there is no verb
            continue
        
        # compare to all following events
        for j in range(i+1, len(all_events)):
            target_spans = all_events[j]['span']

            # check whether event i is subevent of j
            for _, target_span in target_spans.items():
                if is_sub_span(verb_span, target_span):
                    # event i is subevent of event j
                    if 'subevent_of' not in all_events[i]:
                        all_events[i]['subevent_of'] = [j]
                    else:
                        all_events[i]['subevent_of'].append(j)
                    break
            
            # check the other way around
            for arg_span in arg_spans:
                if 'V' in target_spans and is_sub_span(target_spans['V'], arg_span):
                    # event j is subevent of event i
                    if 'subevent_of' not in all_events[j]:
                        all_events[j]['subevent_of'] = [i]
                    else:
                        all_events[j]['subevent_of'].append(i)
                    break

def normalize_events(events, persons, document=None, save_original_event=False):
    ''' Normalize events using personal coreference info. 
    
    Args:
        events & persons: the output from FreeTextProcessor.predict
        document <list of str>: if specified, use the words in document to process events.
        save_original_event <bool>: whether to store the original events or not.
    '''
    if not document:
        document = persons['document']

    span2person = {}
    for p in persons['persons']:
        for span in persons['persons'][p]:
            span2person[span] = p, persons['persons'][p][span]

    person2freq = {p: len(persons['persons'][p]) for p in persons['persons']}

    for event in events['events']:
        span = event['span']
        
        if save_original_event:
            orig_event = span_to_word(document, span)
            event['original'] = orig_event

        normed_event = span_to_word(document, span, True, {'span2person': span2person, 'person2freq': person2freq})
        event['normalized'] = normed_event


def get_verb_args_list(event):
    ''' get ordered verb/args labels from event dicts '''
    verb_args = list(event['span'].items())
    verb_args.sort(key=lambda x: x[1][0])
    verb_args = [va[0] for va in verb_args]
    return verb_args

def get_event_wrds(event, verb_args, from_='normalized'):
    ''' get event words list according to its verb-arg ordered list '''
    wrds = []
    for key in verb_args:
        wrds.extend(event[from_][key])
    return wrds

def generalize_events(events, from_='normalized'):
    ''' Generalize events by dropping some arguments. This method should be called AFTER normalize_events.

    Three levels:
    1. Drop modifier-arguments (ARGM-XXX) except for ARGM-NEG and ARGM-MOD
    2. Drop ARG2, ARG3, ARG4
    3. Drop ARG1 if there is ARG0
    4. Only verb is preserved
    '''

    for event in events['events']:
        verb_args = get_verb_args_list(event)
        hierarchy = {0: get_event_wrds(event, verb_args, from_)}

        # 1. drop modifiers except NEG and MOD
        i = 0
        changed = False
        while i < len(verb_args):
            if verb_args[i].startswith('ARGM') and verb_args[i] not in {'ARGM-NEG', 'ARGM-MOD'}:
                verb_args.pop(i)
                changed = True
            else:
                i += 1
        if changed:
            hierarchy[1] = get_event_wrds(event, verb_args, from_)

        # 2. drop arg2, arg3, arg4
        i = 0
        changed = False
        while i < len(verb_args):
            if verb_args[i] in {'ARG2', 'ARG3', 'ARG4'}:
                verb_args.pop(i)
                changed = True
            else:
                i += 1
        if changed:
            hierarchy[2] = get_event_wrds(event, verb_args, from_)

        # 3. drop ARG1 if there is ARG0
        if 'ARG1' in verb_args and 'ARG0' in verb_args:
            verb_args.remove('ARG1')
            hierarchy[3] = get_event_wrds(event, verb_args, from_)

        if 'V' in verb_args:
            hierarchy[4] = get_event_wrds(event, ['V'], from_)
        else:
            hierarchy[4] = ['']

        event['hierarchy'] = hierarchy
        

