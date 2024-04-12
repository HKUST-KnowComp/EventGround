import re

def reorder_persons(event_wrds, person_pattern=r'\[P([0123456789]+?)[\']?s?\]'):
    ''' Reorder the person ids in input event locally.
    E.g. for ['[P3]', 'tell', '[P2]', 'something'], this function will return ['[P0]', 'tell', '[P1]', 'something'] and a mapping {'3': '0', '2': '1'}
    '''
    persons = {}
    max_person = 0
    new_wrds = []
    for wrd in event_wrds:
        pid = re.findall(person_pattern, wrd)
        if len(pid) > 0:
            pid = pid[0]
            if pid not in persons:
                persons[pid] = str(max_person)
                max_person += 1
            
            new_wrds.append(wrd.replace(pid, persons[pid]))
        else:
            new_wrds.append(wrd)
    return new_wrds, persons

def map_events_from_KG(events, retriever, topk=1, hier_level_thresh=4):
    ''' Given event results from FreeTextProcessor, retrieve events for the hierarchy of events from KG, this will add a new key-value pair ("anchor_events") 
    Currently, only retrieve events with hierarchy level < 4 (events that have at least one argument)
    
    This function will first replace the [Pn] markers with a sequence of [P0], [P1], ... as in ASER-normed, then start retrieval.
    '''

    for event in events['events']:
        anchor_events = {}
        person_maps = {}
        for hier_level in event['hierarchy']:
            if hier_level >= hier_level_thresh: continue

            event_wrds = event['hierarchy'][hier_level]

            # re-order persons in event wrds for the convenience of retrieval
            reordered_wrds, persons = reorder_persons(event_wrds)
            
            # retrieve
            query = ' '.join(reordered_wrds)
            retrieved_events = retriever(query, topk)
            anchor_events[hier_level] = retrieved_events[0]
            person_maps[hier_level] = persons
        event['anchor_events'] = anchor_events
        event['anchor_persons'] = person_maps



def free_text_pipeline(text, free_text_processor, retriever, topk=1):
    ''' The pipeline for processing free text to get events.
    '''
    # parse free-text to get SRL-basd events
    result = free_text_processor(text)

    # map the (hierarchical) events to external KG
    map_events_from_KG(result['events'], retriever, topk)

    return result



""" pipeline for MCNC-like script """
def MCNC_script_pipeline(script_list, protag, free_text_processor, retriever, topk=1):
    ''' The pipeline for processing MCNC script (s-v-o-o) like data to get events.
    '''
    try:
        is_human = protag_is_human(script_list[0], protag, free_text_processor)
    except Exception as e:
        print(e)
        is_human = True
    return [sub_MCNC_script_pipeline(text, protag, is_human, retriever) for text in script_list]

def sub_MCNC_script_pipeline(script, protag, is_human, retriever, topk=1):
    if protag is not None:
        # normalize (replace with [P0] if the protagonist is a person)
        if is_human:
            script = script.replace(protag, '[P0]')
        
    # retrieve 
    events = [{'event': event, 'protag': protag} for event in script.split('. ')]
    map_events_from_KG_MCNC(events, retriever, topk)
    return events

def protag_is_human(script, protag, free_text_processor):
    results = free_text_processor.corenlp.predict_coref(script)['corefs']
    is_human = False

    for k in results:
        mentions = results[k]
        for mention in mentions:
            if mention['text'].lower() == protag:
                if mention['gender'] not in {'UNKNOWN', 'NEUTRAL'} or mention['animacy'] != 'INANIMATE':
                    is_human = True
                    break
        if is_human is True:
            break
    return is_human

def map_events_from_KG_MCNC(events, retriever, topk=1):
    ''' Map events from the external KG for s-v-o-o like scripts (no hierarchy due to the simple structure).
    '''

    for event in events:
        query = event['event']
        retrieved_events = retriever(query, topk)
        anchor_events = retrieved_events[0]
        
        event['anchor_events'] = anchor_events