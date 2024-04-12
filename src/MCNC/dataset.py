# coding: utf-8

import os
import pickle
from tqdm import tqdm
from torch.utils.data import Dataset

class NEEGDataset(Dataset):
    ''' NEEG Dataset object
    '''
    def __init__(self, data_dir, which='dev.pickle', use_lemmatized_verb=False, event_separator=' ## '):
        ''''''
        super(NEEGDataset, self).__init__()
        self.path = os.path.join(data_dir, which)
        self.use_lemmatized_verb = use_lemmatized_verb

        with open(self.path, 'rb') as f:
            self.data = pickle.load(f)
        print(f'Dataset loaded: \n{self.path}')

        self.data = self.__process_data(self.data, separator=event_separator)

    def __len__(self):
        ''' Total length of dataset '''
        return len(self.data)

    def __getitem__(self, index):
        ''' Returns the corresponding items '''
        
        return self.data[index]

    def event2text(self, event):
        ''' Concat the event tuple to text 
        If self.use_lemmatized_verb is True, use the lemmatized version
        '''
        verb, lem_verb, deprel, subj, obj, p_obj = event
        if self.use_lemmatized_verb: verb = lem_verb
        verb = verb.replace('+', ' ')

        return ' '.join([i for i in (subj, verb, obj, p_obj) if i])

    def __process_data(self, data, separator=' ## '):
        ''' Process the data into natural language form '''
        new_data = []
        print('Processing data into natural language texts...')
        for item in tqdm(data):
            context_events, candidate_events, label = item
            
            context = separator.join([self.event2text(event) for event in context_events])

            tmp_list = []
            for event in candidate_events:
                context_cand = context + separator + self.event2text(event)
                tmp_list.append(context_cand)
            
            new_data.append({'text': tmp_list,
                            'label': label})

        return new_data


def load_NEEG_datasets(data_dir, use_lemmatized_verb=False):
    ''' Load the train, valid and test dataset from disk '''
    train, valid, test = [NEEGDataset(data_dir, i, use_lemmatized_verb) for i in ['train.pickle', 'dev.pickle', 'test.pickle']]
    return {'train': train, 'dev': valid, 'test': test}


def resave_to_csv(dataset, output_dir, output_name):
    with open(os.path.join(output_dir, output_name), 'w') as f:
        for item in tqdm(dataset):
            f.write(','.join([i.replace(',', ' ') for i in item['text']]) + ',' + str(item['label']) + '\n')
        


if __name__ == '__main__':
    # path_to_dir = '/path/to/dataset/NEEG_data/clean'
    # # devset = NEEGDataset(path_to_dir, 'dev.pickle')

    # # print(devset[100])
    # datasets = load_NEEG_datasets(path_to_dir)
    # [resave_to_csv(datasets[key], '/path/to/dataset/NEEG_data/NEEG_HF', key+'.csv') for key in datasets]


    path_to_dir = '/path/to/dataset/NEEG_data/clean'
    # devset = NEEGDataset(path_to_dir, 'dev.pickle')

    # print(devset[100])
    datasets = load_NEEG_datasets(path_to_dir, use_lemmatized_verb=True)
    [resave_to_csv(datasets[key], '/path/to/dataset/NEEG_data/NEEG_LEMMATIZED_HF', key+'.csv') for key in datasets]

