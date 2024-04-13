import os
import sys
sys.path.append('..')
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import torch.distributed as dist
import torch.multiprocessing as mp

from transformers import AutoTokenizer
from models import TextGraphMCModel_NEEG as TextGraphMCModel
from neeg_dataset import GraphMCDataset, TextGraphCollator
from utils import Trainer, Arguments, init_process_group
from aser_utils import ID2NTYPE_LM_STR

model_collator_map = {
    'text-graph': (TextGraphMCModel, TextGraphCollator)
}


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

def save_preds(logits, dataset, directory=''):
    print('hi')
    print(logits, dataset, directory)
    if directory:
        out_zip_name = os.path.join(directory, 'answer.zip')
        out_txt_name = os.path.join(directory, 'answer.txt')
    else:
        out_zip_name = 'answer.zip'
        out_txt_name = 'answer.txt'
    preds = logits.argmax(1)+1
    ids = [i['id'] for i in dataset]
    prediction = pd.DataFrame({'InputStoryid':ids, 'AnswerRightEnding':preds})
    prediction.to_csv(out_txt_name, index=False)
    os.system('zip {} {}'.format(out_zip_name, out_txt_name))
    return out_zip_name

def main(rank, world_size, configs):
    init_process_group(rank, world_size, timeout=1400)
    # print("WARNING: You are using Distributed DP, where error msgs are prone to be hidden inside processes (the processes hang up forever sometimes). Try banning wandb or logging stdout for each process!")
    args = Arguments(
        use_ddp=True,
        rank=rank,
        world_size=world_size,
        seed=configs['seed'],
        graph_name=configs['graph_name'],
        # the following are basic params
        name=configs['name'],
        run_name=None,
        group=configs['group'],
        model_name=configs['model_name'],
        encoder_name=configs['encoder_name'],
        conv_hidden=configs['conv_hidden'],
        conv_layers=configs['conv_layers'],
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
        num_bases=configs['num_bases'],
        conv_type=configs['conv_type'],
        add_rev_edges=configs['add_rev_edges'],
        homogenous=configs['homogeneous'],
        max_num_nodes=configs['max_num_nodes']
    )
    for key in configs:
        setattr(args, key, configs[key])

    datasets = GraphMCDataset.make_datasets(graph_name=args.graph_name, add_rev_edges=args.add_rev_edges)
    train_set, valid_set, test_set = datasets['train'], datasets['valid'], datasets['test']

    tokenizer = get_tokenizer(args)
    args.tokenizer_len = len(tokenizer)

    model, collator = model_collator_map[args.model_name]
    model = model(args, num_choices=5)

    specify_node_type = not args.homogeneous
    collator = collator(tokenizer, specify_node_type=specify_node_type, max_num_nodes=args.max_num_nodes)
    trainer = Trainer(model, args, train_set, valid_set, collator, compute_metrics)
    
    try:
        trainer.init_wandb()
        trainer.train()
        result = trainer.evaluate(test_set, get_prediction=True)
        # result = trainer.evaluate(valid_set, get_prediction=True)

        if trainer.is_master:
            summary = result['metrics']
            summary['best_metric'] = trainer.state.best_metric
            trainer.log(summary=summary)
            # filepath = save_preds(result['logits'], test_set, directory=trainer.args.wandb_folder)
            # trainer.log(artifacts={'filepaths': filepath, 'type': 'result'})

        trainer.finish_wandb()
        import wandb
        wandb.finish()
    except Exception as e:
        print(f'Rank {rank} ERROR MESSAGE:', e)

    dist.destroy_process_group()




import argparse
parser = argparse.ArgumentParser()

parser.add_argument("--learning_rate", type=float, default=1e-5)
parser.add_argument("--per_device_train_batch_size", type=int, default=4)
parser.add_argument("--per_device_eval_batch_size", type=int, default=8)
parser.add_argument("--max_steps", type=int, default=80000)
parser.add_argument("--eval_steps", type=int, default=100)
parser.add_argument("--early_stopping_steps", type=float, default=float('inf'))
parser.add_argument("--conv_hidden", type=int, default=128)
parser.add_argument("--conv_layers", type=int, default=2)
parser.add_argument("--fusion_type", type=str, default='add') # add, concat
parser.add_argument("--group", type=str, default='prelim')
parser.add_argument("--name", type=str, default='mcnc')
parser.add_argument("--graph_name", type=str, default='graph_notop_directed_thresh_core100')
parser.add_argument("--model_name", type=str, default='text-graph')
parser.add_argument("--encoder_name", type=str, default='roberta-base')
parser.add_argument("--seed", type=int, default=2022)
parser.add_argument("--warmup_steps", type=int, default=100)
parser.add_argument("--lr_scheduler_type", type=str, default='cosine')
parser.add_argument("--conv_type", type=str, default='GIN')
parser.add_argument("--num_bases", type=int, default=30, help='number of bases parameter for RGCN')
parser.add_argument("--add_rev_edges", type=eval, default=False)
parser.add_argument("--homogeneous", type=eval, default=True, help='whether to distinguish node types in LM or not.')
parser.add_argument("--max_num_nodes", type=int, default=9999999, help='max number of nodes in a graph')

configs = parser.parse_args()
configs = vars(configs)

if __name__ == '__main__':
    import os
    os.environ['TOKENIZERS_PARALLELISM']='true'
    import torch.multiprocessing as mp

    import torch
    num_gpus = torch.cuda.device_count()
    # num_gpus = 1

    mp.spawn(main, args=(num_gpus, configs), nprocs=num_gpus, join=True)







    


