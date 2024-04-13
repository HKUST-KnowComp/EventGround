import os
import sys
sys.path.append('..')
import warnings
warnings.filterwarnings('ignore')
import torch
import numpy as np

from transformers import AutoTokenizer
from utils import Trainer, Arguments
from aser_utils import ID2NTYPE_LM_STR
from datasets import load_dataset

import argparse
parser = argparse.ArgumentParser()

parser.add_argument("--learning_rate", type=float, default=1e-5)
parser.add_argument("--per_device_train_batch_size", type=int, default=32)
parser.add_argument("--per_device_eval_batch_size", type=int, default=64)
parser.add_argument("--max_steps", type=int, default=80000)
parser.add_argument("--eval_steps", type=int, default=100)
parser.add_argument("--early_stopping_steps", type=float, default=500)
parser.add_argument("--conv_hidden", type=int, default=128)
parser.add_argument("--conv_layers", type=int, default=2)
parser.add_argument("--fusion_type", type=str, default='add') # add, concat
parser.add_argument("--group", type=str, default='baseline')
parser.add_argument("--name", type=str, default='mcnc')
parser.add_argument("--model_name", type=str, default='text-graph')
parser.add_argument("--encoder_name", type=str, default='roberta-base')
parser.add_argument("--seed", type=int, default=2022)
parser.add_argument("--warmup_steps", type=int, default=100)
parser.add_argument("--lr_scheduler_type", type=str, default='cosine')

configs = parser.parse_args()
configs = vars(configs)

def get_tokenizer(args):
    tokenizer = AutoTokenizer.from_pretrained(args.encoder_name)
    special_tokens = [f'[P{i}]' for i in range(10)] + [f"[P{i}'s]" for i in range(10)] + list(ID2NTYPE_LM_STR.values())
    special_tokens_dict = {'additional_special_tokens': special_tokens}
    # print(special_tokens)
    tokenizer.add_special_tokens(special_tokens_dict)
    return tokenizer
    
def compute_metrics(eval_predictions):
    predictions, label_ids = eval_predictions
    preds = np.argmax(predictions, axis=1)
    return {"accuracy": (preds == label_ids).astype(np.float32).mean().item()}

class TextCollator:
    def __init__(self, tokenizer):
        # make sure your tokenizer has added the node type tokens
        self.tokenizer = tokenizer

    def __call__(self, batch):
        """ Collate a list of graphs, where the nodes have text features.
        """
        all_labels = []

        all_original_texts = []

        ' prepare graph info '
        for item in batch:
            for i in range(5):
                cand_orig_text = item['text'+str(i)]
                ###### MODIFIED HERE ######
                cand_orig_text = '. '.join([txt.capitalize() for txt in cand_orig_text.split(' ## ')])
                ###########################

                all_original_texts.append(cand_orig_text)
                    
            all_labels.append(item['label'])

        'original instance texts'
        orig_tokenized = self.tokenizer(all_original_texts, truncation=True, padding='longest', return_tensors='pt')
        orig_tokenized = (orig_tokenized['input_ids'].view(len(batch), 5, -1), orig_tokenized['attention_mask'].view(len(batch), 5, -1))

        all_labels = torch.tensor(all_labels)

        return {'text_tokenized': orig_tokenized,'labels': all_labels}

args = Arguments(
        seed=configs['seed'],
        # the following are basic params
        name=configs['name'],
        run_name=None,
        group=configs['group'],
        model_name=configs['model_name'],
        encoder_name=configs['encoder_name'],
        conv_hidden=configs['conv_hidden'],
        evaluation_strategy = 'steps',
        save_strategy = 'steps',
        eval_steps = configs['eval_steps'],
        save_steps = None,
        early_stopping_steps = configs['early_stopping_steps'],
        lr_scheduler_type = configs['lr_scheduler_type'],
        warmup_steps = configs['warmup_steps'],
        learning_rate = configs['learning_rate'],
        max_steps = configs['max_steps'],
        per_device_train_batch_size = configs['per_device_train_batch_size'],
        per_device_eval_batch_size = configs['per_device_eval_batch_size'],
        load_best_model_at_end=True,
        # load_best_model_at_end=False,
        metric_for_best_model='accuracy',
        greater_is_better=True,
        # metric_for_best_model='loss',
        # greater_is_better=False,
        save_total_limit=1,
    )
for key in configs:
    setattr(args, key, configs[key])

datasets = load_dataset("/path/to/dataset/NEEG_data/NEEG_HF/")
train_set, valid_set, test_set = datasets['train'], datasets['validation'], datasets['test']
tokenizer = get_tokenizer(args)
args.tokenizer_len = len(tokenizer)

from models import TextMCModel
model = TextMCModel(args, num_choices=5)
collator = TextCollator(tokenizer)

trainer = Trainer(model, args, train_set, valid_set, collator, compute_metrics)
trainer.init_wandb()
trainer.train()
result = trainer.evaluate(test_set, get_prediction=True)
if trainer.is_master:
    summary = result['metrics']
    summary['best_metric'] = trainer.state.best_metric
    trainer.log(summary=summary)
trainer.finish_wandb()
import wandb
wandb.finish()