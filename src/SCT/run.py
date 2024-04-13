import os
import sys
sys.path.append('..')
from utils import get_runstr_combinations

if __name__ == '__main__':
    header =  'python3 train.py'
    # fixed_configs = {
    #     'name': 'story-cloze-test',
    #     'group': 'deberta-v3-large+RGCN',
    #     'model_name': 'text-graph',
    #     'encoder_name': 'microsoft/deberta-v3-large',
    #       'conv_type': 'RGCN',
    #     'fusion_type': 'add',
    #     'learning_rate': 5e-6,
    #     'warmup_steps': 80,
    #     'eval_steps': 80,
    #     'max_steps': 1600,
    #     'lr_scheduler_type': 'cosine',
    #     'graph_name': 'graph_notop_directed_thresh_core100',
    #     "per_device_train_batch_size": 2,
    #     "per_device_eval_batch_size": 8,
    #     "max_num_nodes": 40,
    #     'add_rev_edges': True,
    #     'homogeneous': True,
    # }
    # 8 v100 * 32GB
    fixed_configs = {
        'name': 'story-cloze-test',
        'group': 'deberta-v3-large+RGCN',
        'model_name': 'text-graph',
        'encoder_name': 'microsoft/deberta-v3-large',
        'conv_type': 'RGCN',
        'fusion_type': 'add',
        'learning_rate': 5e-6,
        'warmup_steps': 40,
        'eval_steps': 40,
        'max_steps': 800,
        'lr_scheduler_type': 'cosine',
        'graph_name': 'graph_notop_directed_thresh_core100',
        "per_device_train_batch_size": 2,
        "per_device_eval_batch_size": 8,
        "max_num_nodes": 40,
        'add_rev_edges': True,
        'homogeneous': True,
    }
    comb_configs = {
        # 'learning_rate': [1e-4, 1e-5],
        # 'conv_hidden': [64, 128, 256, 512],
        'dataset': [2016, 2018],
        'conv_hidden': [128, 256, 512],
        'conv_layers': [2],
        'num_bases': [-1, 10],
        # "fusion_type": ['add', 'concat'],
        'seed': [2022],
        # 'graph_name': ['graph_notop_directed_nothresh_core100','graph_notop_directed_thresh_core100',
            # 'graph_notop_undirected_nothresh_core100', 'graph_notop_undirected_thresh_core100'],
    }
    all_runstrs = get_runstr_combinations(header, comb_configs=comb_configs, fixed_configs=fixed_configs)

    for i, string in enumerate(all_runstrs):
        print(f'[{i+1}/{len(all_runstrs)}]', string)
        os.system(string)

    import wandb
    wandb.init(project='test', name='alert')
    wandb.alert(
        title='run finished',
        text='[deberta-v3-large SCT]The assigned runs finished.'
    )