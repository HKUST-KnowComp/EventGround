import os

def get_runstr_combinations(header='python3 finetune_lm.py', fixed_configs={}, comb_configs={}):
    """ Get all the combinations in `comb_configs`. 
    This is useful when trying to conduct multiple experiments with different params.

    Args:
        header <str>: The starting part of the command, e.g., python3 run.py 
        fixed_configs <dict: str->any>: The fixed part of parameters, you may also write these params in the header manually.
        comb_configs <dict: str-> list of any>: The combinitorial part of parameters. All the combinations within this dictionary will be considered.

    Returns:
        A list of all the combinations of runstrings.
    """
    from itertools import product

    all_runstrs = []

    # add fixed arguments
    string = header
    for key in fixed_configs:
        string += ' --'+key+' '+str(fixed_configs[key])

    # add all combinations
    key_list = list(comb_configs.keys())
    value_list = [comb_configs[key] for key in key_list]
    for vals in list(product(*value_list)):
        tmp_str = string
        for key, val in zip(key_list, vals):
            tmp_str += ' --'+key+' '+str(val)
        all_runstrs.append(tmp_str)
    
    return all_runstrs



header = "python3 finetune_lm.py "

base_fixed_configs = {
    'train_batch_size': 32,
    'eval_batch_size': 32,
    'eval_steps': 100,
    'save_steps': 100,
    'max_steps': 500,
    'learning_rate': 1e-5
}

# base_comb_configs = {
#     'model_checkpoint': ['roberta-base', 'bert-base-cased'],
#     'dataset': [2016, 2018],
#     'random_seed': [2022, 4044, 8088],
# }
base_comb_configs = {
    'model_checkpoint': ['roberta-base', 'bert-base-cased'],
    'dataset': [2016],
    'random_seed': [2022, 4044, 8088],
}



all_runstrs = get_runstr_combinations(header, base_fixed_configs, base_comb_configs)



large_fixed_configs = {
    'train_batch_size': 4,
    'eval_batch_size': 8,
    'eval_steps': 800,
    'save_steps': 800,
    'max_steps': 4000,
    'learning_rate': 1e-5
}

large_comb_configs = {
    'model_checkpoint': ['roberta-large', 'bert-large-cased'],
    'dataset': [2016, 2018],
    'random_seed': [2022, 4044, 8088],
}

all_runstrs += get_runstr_combinations(header, large_fixed_configs, large_comb_configs)


for string in all_runstrs:
    # print(string)
    os.system(string)

os.system('sudo shutdown -h now')