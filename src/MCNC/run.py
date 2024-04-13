import os
import sys
sys.path.append('..')
from utils import get_runstr_combinations

if __name__ == '__main__':
    header =  'python3 train.py'
    # fixed_configs = {
    #     # 'name': 'tmp',
    #     'name': 'mcnc',
    #     # 'group': 'roberta-large+RGCN-final-opt',
    #     'group': 'deberta-v3-large-RGCN',
    #     'model_name': 'text-graph',
    #     # 'encoder_name': 'roberta-large',
    #     'encoder_name': 'microsoft/deberta-v3-large',
    #     'conv_type': 'RGCN',
    #     'fusion_type': 'add',
    #     'learning_rate': 5e-6,
    #     'warmup_steps': 3200,
    #     'eval_steps': 3200,
    #     'lr_scheduler_type': 'cosine',
    #     'max_steps': 160000,
    #     'early_stopping_steps': 32000,
    #     # 'conv_hidden': 128,
    #     # 'conv_layers': 2,
    #     # 'seed': 2022,
    #     "per_device_train_batch_size": 2,
    #     "per_device_eval_batch_size": 8,
    #     "max_num_nodes": 40,
    # }

    # 8 32GB gpus
    fixed_configs = {
        # 'name': 'tmp',
        'name': 'mcnc',
        # 'group': 'roberta-large+RGCN-final-opt',
        'group': 'deberta-v3-large-RGCN',
        'model_name': 'text-graph',
        # 'encoder_name': 'roberta-large',
        'encoder_name': 'microsoft/deberta-v3-large',
        'conv_type': 'RGCN',
        'fusion_type': 'add',
        'learning_rate': 5e-6,
        'warmup_steps': 1600,
        'eval_steps': 1600,
        'lr_scheduler_type': 'cosine',
        'max_steps': 80000,
        'early_stopping_steps': 8000,
        # 'conv_hidden': 128,
        # 'conv_layers': 2,
        # 'seed': 2022,
        "per_device_train_batch_size": 2,
        "per_device_eval_batch_size": 16,
        "max_num_nodes": 30,
    }
    comb_configs = {
        # 'learning_rate': [1e-4, 1e-5],
        # 'conv_hidden': [256],
        'conv_hidden': [256, 512],
        'conv_layers': [2],
        'seed': [2022],
        'graph_name': ['graph_notop_directed_thresh_core100'],
        'num_bases': [10], 
        'add_rev_edges': [True],
        'homogeneous': [True]
    }
    all_runstrs = get_runstr_combinations(header, comb_configs=comb_configs, fixed_configs=fixed_configs)

    comb_configs = {
        # 'learning_rate': [1e-4, 1e-5],
        # 'conv_hidden': [256],
        'conv_hidden': [128, 256, 512],
        'conv_layers': [3],
        'seed': [2022],
        'graph_name': ['graph_notop_directed_thresh_core100'],
        'num_bases': [10], 
        'add_rev_edges': [True],
        'homogeneous': [True]
    }
    all_runstrs.extend(get_runstr_combinations(header, comb_configs=comb_configs, fixed_configs=fixed_configs))

    for i, string in enumerate(all_runstrs):
        print('[{}/{}]'.format(i+1, len(all_runstrs)), string)
        os.system(string)

    import wandb
    wandb.init(project='test', name='alert')
    wandb.alert(
        title='run finished',
        text='[deberta-v3-large mcnc] The assigned runs finished.'
    )