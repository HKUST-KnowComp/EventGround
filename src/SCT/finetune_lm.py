import torch
import random
import argparse
import numpy as np
import transformers
import datasets
from datasets import load_dataset
from transformers import AutoTokenizer
print(transformers.__version__)

parser = argparse.ArgumentParser()

parser.add_argument("--model_checkpoint", type=str, default="roberta-base")
parser.add_argument("--dataset", type=int, default=2016)
parser.add_argument("--random_seed", type=int, default=2022)
parser.add_argument("--learning_rate", type=float, default=1e-5)
parser.add_argument("--train_batch_size", type=int, default=32)
parser.add_argument("--eval_batch_size", type=int, default=32)
parser.add_argument("--eval_steps", type=int, default=100)
parser.add_argument("--save_steps", type=int, default=100)
parser.add_argument("--max_steps", type=int, default=500)
parser.add_argument("--save_total_limit", type=int, default=1)
parser.add_argument("--data_2016_valid", type=str, default='/path/to/dataset/StoryClozeTest/raw/val_spring2016.csv')
parser.add_argument("--data_2016_test", type=str, default='/path/to/dataset/StoryClozeTest/raw/test_spring2016.csv')
parser.add_argument("--data_2018_valid", type=str, default='/path/to/dataset/StoryClozeTest/raw/val_winter2018.csv')
parser.add_argument("--data_2018_test", type=str, default='/path/to/dataset/StoryClozeTest/raw/test_winter2018.csv')

args = parser.parse_args()

configs = vars(args)
configs['name'] = configs['model_checkpoint']+str(configs['dataset'])+str(configs['random_seed'])

print(configs)

data_2016 = {'valid': configs['data_2016_valid'],
            'test': configs['data_2016_test']}
data_2018 = {'valid': configs['data_2018_valid'],
            'test': configs['data_2018_test']}

random_seed = configs['random_seed']

np.random.seed(random_seed)
random.seed(random_seed)
torch.manual_seed(random_seed)

if configs['dataset'] == 2016:
     dataset = load_dataset('csv', data_files=data_2016)
     # split the original valid set into train/valid sets
     returns = dataset['valid'].train_test_split(test_size=100, train_size=len(dataset['valid'])-100, seed=random_seed)
     # compose the new dataset
     dataset = datasets.dataset_dict.DatasetDict({'train': returns['train'], 
                                             'valid': returns['test'], 
                                             'test': dataset['test']})
elif configs['dataset'] == 2018:
     # 2018 debiased version: test set does not have labels
     dataset_valid = load_dataset('csv', data_files={'valid': data_2018['valid']})['valid']
     dataset_test = load_dataset('csv', data_files={'test': data_2018['test']})['test']

     returns = dataset_valid.train_test_split(test_size=100, train_size=len(dataset_valid)-100, seed=random_seed)

     # compose the new dataset
     dataset = datasets.dataset_dict.DatasetDict({'train': returns['train'], 
                                             'valid': returns['test'], 
                                             'test': dataset_test})

def preprocess_item(item):
    texts = []
    
    for sents in zip(item['InputSentence1'], 
                item['InputSentence2'], 
                item['InputSentence3'], 
                item['InputSentence4'], 
                item['RandomFifthSentenceQuiz1'], 
                item['RandomFifthSentenceQuiz2']):
        sents = list(sents)
        contexts = ' '.join(sents[:4])
        candidates = [contexts + ' ' + sents[4], contexts + ' ' + sents[5]]

        texts.append(candidates)

    flatten_sents = sum(texts, [])
    tokenized = tokenizer(flatten_sents, truncation=True)
    processed_info = {k: [v[i:i+2] for i in range(0, len(v), 2)] for k, v in tokenized.items()}

    # label: 1 -> 0, 2 -> 1
    if 'AnswerRightEnding' in item:
        processed_info['label'] = [i-1 for i in item['AnswerRightEnding']]

    return processed_info

tokenizer = AutoTokenizer.from_pretrained(configs['model_checkpoint'])

encoded_dataset = dataset.map(preprocess_item, batched=True)

from transformers import AutoModelForMultipleChoice, TrainingArguments, Trainer
import sys
sys.path.append('..')
from models import TextMCModel_SCT
# model = AutoModelForMultipleChoice.from_pretrained(configs['model_checkpoint'])
model = TextMCModel_SCT(configs, num_choices=2)

# https://huggingface.co/docs/transformers/v4.17.0/en/main_classes/trainer#transformers.TrainingArguments
args = TrainingArguments(
    configs['name'],
    evaluation_strategy = 'steps',
    eval_steps = configs['eval_steps'],
    learning_rate = configs['learning_rate'],
    per_device_train_batch_size = configs['train_batch_size'],
    per_device_eval_batch_size = configs['eval_batch_size'],
    max_steps = configs['max_steps'],
    save_steps = configs['save_steps'],
    load_best_model_at_end=True,
    seed=configs['random_seed'],
    save_total_limit=configs['save_total_limit']
)

from dataclasses import dataclass
from transformers.tokenization_utils_base import PreTrainedTokenizerBase, PaddingStrategy
from typing import Optional, Union
import torch

@dataclass
class DataCollatorForMultipleChoice:
    """
    Data collator that will dynamically pad the inputs for multiple choice received.
    """

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = 128 # larger than the maximum length of training set
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features):
        label_name = "label" if "label" in features[0].keys() else "labels"
        labels = [feature.pop(label_name) if label_name in feature else 0 for feature in features]
        batch_size = len(features)
        num_choices = len(features[0]["input_ids"])

        flattened_features = [[{k: v[i] for k, v in feature.items()} for i in range(num_choices)] for feature in features]
        flattened_features = sum(flattened_features, [])
        
        batch = self.tokenizer.pad(
            flattened_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        
        # Un-flatten
        batch = {k: v.view(batch_size, num_choices, -1) for k, v in batch.items()}
        # Add back labels
        batch["labels"] = torch.tensor(labels, dtype=torch.int64)
        return batch

import numpy as np

def compute_metrics(eval_predictions):
    predictions, label_ids = eval_predictions
    preds = np.argmax(predictions, axis=1)
    return {"accuracy": (preds == label_ids).astype(np.float32).mean().item()}

trainer = Trainer(
    model,
    args,
    train_dataset=encoded_dataset["train"],
    eval_dataset=encoded_dataset["valid"],
    tokenizer=tokenizer,
    data_collator=DataCollatorForMultipleChoice(tokenizer),
    compute_metrics=compute_metrics,
)

import wandb
wandb.login()
wandb.init(project='story-cloze-test', config=configs, name=configs['name'], group='rerun-baseline')

trainer.train()

if configs['dataset'] == 2016:
    test_acc = trainer.evaluate(encoded_dataset['test'])
    wandb.run.summary['test_accuracy'] = test_acc['eval_accuracy']
    print(test_acc)

preds = trainer.predict(encoded_dataset['test'])
predictions = preds.predictions.argmax(-1) + 1
prediction_ids = encoded_dataset['test']['InputStoryid']

import os
import pandas as pd
df = pd.DataFrame({'InputStoryid': prediction_ids, 'AnswerRightEnding': predictions})

df_path = os.path.join(configs['name'],'test_preds.txt')
df.to_csv(df_path, index=False)

artifact = wandb.Artifact('artifact'+'-'+'test_preds', type='result')
artifact.add_file(df_path)
wandb.run.log_artifact(artifact)
wandb.finish()


