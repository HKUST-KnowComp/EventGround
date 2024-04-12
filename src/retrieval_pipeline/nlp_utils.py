# coding: utf-8

import nltk
import allennlp
from allennlp.predictors.predictor import Predictor

class AllenNLPWrapper:
    def __init__(self, args={'coref_model': "https://storage.googleapis.com/allennlp-public-models/coref-spanbert-large-2021.03.10.tar.gz",
                            'srl_model': "https://storage.googleapis.com/allennlp-public-models/structured-prediction-srl-bert.2020.12.15.tar.gz"}):
        self.args = args
        self.init_models()

    def init_models(self):
        if 'coref_model' in self.args:
            self.coref_parser = Predictor.from_path(self.args['coref_model'])
        if 'srl_model' in self.args:
            self.srl_parser = Predictor.from_path(self.args['srl_model'])

    def predict(self, inputs):
        raise NotImplementedError('predict: not implemented yet')

    def predict_srl(self, inputs, post_process=False):
        ''' Parse the input sentence (str) or batch of sentences (list of str).

        post_process: 
            The original SRL pipeline does not return the argment / verb spans. Add the spans to each returning dict.

        Important keys of the returning items:
            verbs
                description
                tags
                span
            words
        '''
        if isinstance(inputs, str):
            batch = [{'sentence': inputs}]
        elif isinstance(inputs, list):
            batch = [{'sentence': i} for i in inputs]

        results = self.srl_parser.predict_batch_json(batch)
        if post_process:
            # Add 'span' (start and end indexes, python slice style)
            for item in results:
                for verb_info in item['verbs']:
                    verb_info['span'] = find_tag_span(verb_info['tags'])

        return results

    def predict_coref(self, inputs, post_process=False):
        ''' Parse the input sentence (str) or batch of sentences (list of str).

        post_process:
            Modify the returning cluster indexes into python slicing style

        Important keys of the returning items:
            document -> list of words for each input
            clusters -> all identified spans, grouped by coreference
        '''
        if isinstance(inputs, str):
            batch = [{'document': inputs}]
        elif isinstance(inputs, list):
            batch = [{'document': i} for i in inputs]

        results = self.coref_parser.predict_batch_json(batch)
        if post_process:
            for item in results:
                for spans in item['clusters']:
                    for span in spans:
                        span[1] += 1

        return results


class CoreNLPWrapper:
    def __init__(self, args={'path': "/home/ubuntu/stanfordcorenlp/stanford-corenlp-4.4.0",
                            'port': 10086,
                            'annotators': ["tokenize", "ssplit", "pos", "lemma", "ner", "parse", "coref"]}):
        self.args = args    
        self.init_models()

    def init_models(self):
        self.corenlp, _ = get_corenlp_client(
                        corenlp_path=self.args['path'], 
                        corenlp_port=self.args['port'], 
                        annotators=self.args['annotators']
                    )
        self.annotators = self.args['annotators']
        
    def predict(self, inputs):
        raise NotImplementedError('predict: not implemented yet')
    
    def predict_coref(self, inputs, post_process=False):
        results = parse_sentence(inputs, self.corenlp, self.annotators)
        return results


def find_tag_span(tags):
    ''' Find argument/verb spans from the given SRL tags
    '''
    result = {}
    st = None
    current = None
    for i, wrd in enumerate(tags):
        if wrd.startswith('B'):
            if current is None:
                current = wrd[2:]
                st = i
            else:
                # end the previous span detection
                result[current] = (st, i)
                # start the new one
                current = wrd[2:]
                st = i
        elif wrd.startswith('I'):
            continue
        elif wrd.startswith('O'):
            if current is None:
                continue
            else:
                # end the previous span detection
                result[current] = (st, i)
                current = None
    if current is not None:
        result[current] = (st, len(tags))
    
    return result

def get_words_from_span(document, spans, ed_plus=0):
    ''' get words (list of str) from document (str) by spans (list of tuples)
    ed_plus <int>: The end index style. The span will be retrieved by document[spans[i][0]: spans[i][1]], so ed_plus=0 means the end index is normal python slicing.
    '''
    return [document[i[0]: i[1]+ed_plus] for i in spans]





##################
# CoreNLP utils  #
##################
import re
import os
import socket
from stanza.server import CoreNLPClient, TimeoutException

ANNOTATORS = ["tokenize", "ssplit", "pos", "lemma", "ner", "coref"]

TYPE_SET = frozenset(["CITY", "ORGANIZATION", "COUNTRY", "STATE_OR_PROVINCE", "LOCATION", "NATIONALITY", "PERSON"])

PRONOUN_SET = frozenset(
    [
        "i", "I", "me", "my", "mine", "myself",
        "we", "us", "our", "ours", "ourselves",
        "you", "your", "yours",
        "yourself", "yourselves",
        "he", "him", "his", "himself",
        "she", "her", "hers", "herself",
        "it", "its", "itself",
        "they", "them", "their", "theirs", "themself", "themselves"
    ]
)

def is_port_occupied(ip="127.0.0.1", port=80):
    """ Check whether the ip:port is occupied
    :param ip: the ip address
    :type ip: str (default = "127.0.0.1")
    :param port: the port
    :type port: int (default = 80)
    :return: whether is occupied
    :rtype: bool
    """

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.connect((ip, int(port)))
        s.shutdown(2)
        return True
    except:
        return False

def get_corenlp_client(corenlp_path="", corenlp_port=0, annotators=None, memory='4G'):
    """
    :param corenlp_path: corenlp path, e.g., /home/xliucr/stanford-corenlp-3.9.2
    :type corenlp_path: str (default = "")
    :param corenlp_port: corenlp port, e.g., 9000
    :type corenlp_port: int (default = 0)
    :param annotators: annotators for corenlp, please refer to https://stanfordnlp.github.io/CoreNLP/annotators.html
    :type annotators: Union[List, None] (default = None)
    :return: the corenlp client and whether the client is external
    :rtype: Tuple[stanfordnlp.server.CoreNLPClient, bool]
    """

    if corenlp_port == 0:
        return None, True

    if not annotators:
        annotators = list(ANNOTATORS)

    if is_port_occupied(port=corenlp_port):
        try:
            os.environ["CORENLP_HOME"] = corenlp_path
            corenlp_client = CoreNLPClient(
                annotators=annotators,
                timeout=99999,
                memory=memory,
                endpoint="http://localhost:%d" % corenlp_port,
                start_server=False,
                be_quiet=False
            )
            # corenlp_client.annotate("hello world", annotators=list(annotators), output_format="json")
            return corenlp_client, True
        except BaseException as err:
            raise err
    elif corenlp_path != "":
        print("Starting corenlp client at port {}".format(corenlp_port))
        corenlp_client = CoreNLPClient(
            annotators=annotators,
            timeout=99999,
            memory=memory,
            endpoint="http://localhost:%d" % corenlp_port,
            start_server=True,
            be_quiet=False
        )
        corenlp_client.annotate("hello world", annotators=list(annotators), output_format="json")
        return corenlp_client, False
    else:
        return None, True

def parse_sentence(sentence, corenlp_client, annotators=None):
    """
    :param input_sentence: a raw sentence
    :type input_sentence: str
    :param corenlp_client: the given corenlp client
    :type corenlp_client: stanfordnlp.server.CoreNLPClient
    :param annotators: annotators for corenlp, please refer to https://stanfordnlp.github.io/CoreNLP/annotators.html
    :type annotators: Union[List, None] (default = None)
    :param max_len: the max length of a paragraph (constituency parsing cannot handle super-long sentences)
    :type max_len: int (default = 1024)
    :return: the parsed result
    :rtype: List[Dict[str, object]]
    """

    if not annotators:
        annotators = list(ANNOTATORS)

    parsed_sentences = list()
    raw_texts = list()

    try:
        returns = corenlp_client.annotate(sentence, annotators=annotators,
                                                    output_format="json")

        parsed_sentence = returns["sentences"]
                            
    except TimeoutException as e:
        print(e)
        exit()

    for sent in parsed_sentence:
        if sent["tokens"]:
            char_st = sent["tokens"][0]["characterOffsetBegin"]
            char_end = sent["tokens"][-1]["characterOffsetEnd"]
        else:
            char_st, char_end = 0, 0
        raw_text = sentence[char_st:char_end]
        raw_texts.append(raw_text)
    parsed_sentences.extend(parsed_sentence)


    parsed_rst_list = list()
    for sent, text in zip(parsed_sentences, raw_texts):
        # words
        words = [t["word"] for t in sent["tokens"]]
        x = {
            "text": text,
            # "dependencies": dependencies,    
            "words": words,
        }

        # dependencies
        enhanced_dependency_list = sent["enhancedPlusPlusDependencies"]
        dependencies = set()
        for relation in enhanced_dependency_list:
            if relation["dep"] == "ROOT":
                continue
            governor_pos = relation["governor"]
            dependent_pos = relation["dependent"]
            dependencies.add((governor_pos - 1, relation["dep"], dependent_pos - 1))
        dependencies = list(dependencies)
        dependencies.sort(key=lambda x: (x[0], x[2]))

        if "pos" in annotators:
            pos_tags = [t["pos"] for t in sent["tokens"]]
            x["pos_tags"] = pos_tags
            dependencies = [((i, words[i], pos_tags[i]), rel, (j, words[j], pos_tags[j])) for i, rel, j in dependencies]

        x["dependencies"] = dependencies

        if "lemma" in annotators:
            x["lemmas"] = [t["lemma"] for t in sent["tokens"]]
            x["words"] = x["lemmas"]
        if "ner" in annotators:
            mentions = []
            for m in sent["entitymentions"]:
                if m["ner"] in TYPE_SET and m["text"].lower().strip() not in PRONOUN_SET:
                    mentions.append(
                        {
                            "start": m["tokenBegin"],
                            "end": m["tokenEnd"],
                            "text": m["text"],
                            "ner": m["ner"],
                            "link": None,
                            "entity": None
                        }
                    )

            x["ners"] = [t["ner"] for t in sent["tokens"]]

            # reorganize mentions to dict
            tmp_mentions = {}
            for mention in mentions:
                st, ed = mention['start'], mention['end']
                tmp_mentions[(st, ed)] = mention
            x["mentions"] = tmp_mentions
        if "parse" in annotators:
            x["parse"] = re.sub(r"\s+", " ", sent["parse"])

        parsed_rst_list.append(x)
    res = {'parsed_info': parsed_rst_list}
    if 'coref' in annotators:
        res['corefs'] = returns['corefs']
    return res
