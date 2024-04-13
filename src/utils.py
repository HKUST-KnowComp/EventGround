# coding: utf-8
""" RUN string combinations """
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


""" Dynamic table """
try:
    from prettytable import PrettyTable
except Exception as e:
    print(e, '\nTrying to install the required packages...')
    import os
    os.system('pip3 install prettytable')
    from prettytable import PrettyTable
from tqdm import tqdm

class DynamicTable:
    """ A pretty table that can dynamically append row values and print to stdout.

    Example usage:
        columns = ['Name', 'Age']
        widths = [5, 10]
        alignments = ['l', 'l']

        table = DynamicTable(columns, widths=widths, alignments=alignments)
        table.add_row(['I am your father', 19.123124124])
        table.add_row(['He is good', 'why'])
        table.add_row(['Now it is all clear!', '1203321421412541251241254213412387129874921739812379'])
    """
    def __init__(self, columns, widths=10, alignments='l', print_func='tqdm'):
        widths_map = {}
        if isinstance(widths, int):
            for col in columns:
                widths_map[col] = widths
        elif isinstance(widths, list):
            assert len(columns) == len(widths)
            for col, width in zip(columns, widths):
                widths_map[col] = width
        else:
            raise TypeError('wrong type for input "widths"')

        if isinstance(alignments, str):
            alignments = [alignments for _ in range(len(columns))]
        elif isinstance(alignments, list):
            assert len(alignments) == len(columns)
        else:
            raise TypeError('wrong type for input "alignments"')
        self.columns = columns
        self.table = PrettyTable(columns)
        # set width
        self.table._max_width = widths_map
        self.table._min_width = widths_map
        # set alignment
        for col, align in zip(columns, alignments):
            self.table.align[col] = align
        # get initial width
        self.last_row_index = 0
        self.shown = False

        if print_func == 'tqdm':
            self.print = tqdm.write
        else:
            self.print = print
    
    def show(self, return_string=False):
        if self.shown is False:
            self.shown = True
            string = self.table.get_string()
            self.last_row_index = self.get_current_row_index()
            if return_string:
                return string
            else:
                self.print(string)
    
    def get_current_row_index(self):
        return self.table.get_string().count('\n')
    
    def normalize(self, d):
        if isinstance(d, float):
            return f"{d:.4f}"
        return d

    def add_row(self, row_data, return_string=False):
        """ row_data is a list (which must be of the same length as the columns) 
        or a dict (containing the values)
        """
        assert (isinstance(row_data, list) and len(row_data)==len(self.columns)) or \
            isinstance(row_data, dict)
        if isinstance(row_data, dict):
            row_data = [row_data.get(key, 'No data') for key in self.columns]
        row_data = [self.normalize(d) for d in row_data]

        self.table.add_row(row_data)
        if not self.shown:
            string = self.show(return_string)

        else:
            cur_index = self.get_current_row_index()
            row_height = cur_index - self.last_row_index + 1
            string = "\n".join(self.table.get_string().splitlines()[-1*row_height:])
            self.last_row_index = cur_index
        if return_string:
            return(string)
        else:
            self.print(string)

    def add_rows(self, row_data_list, return_string=False):
        full_string = ''
        for row_data in row_data_list:
            string = self.add_row(row_data, return_string)
            if isinstance(string, str):
                full_string += '\n'+string
        return full_string

""" Wandb wrapper """
import time
import wandb

class wandbWrapper:
    """ An easy to use wandb wrapper.

    Pass all the arguments in terms of keyword args to initialize this wrapper.
    A required key is: name, which defines the project name.
    You may also pass 'run_name' to specify the run name under the project.

    self.log(log_dict={}, prefix='train_')  >>> call this function to log the necessary info during training/evaluation

    self.add_summary(**{'test_accuracy': 0.91})  >>> add the additional summary info to this run 
                                                    (you may call this after the training is finished)
    
    self.add_file('/home/ubuntu/my.txt', )
    
    """
    def __init__(self, **kwargs):
        self.args = kwargs
        self.init_wandb()

    def init_wandb(self):
        """ Initialize the wandb project to enable cloud recording of performance.
        """
        project_name = self.args['name']

        self.run = wandb.init(project=project_name, 
                    config=self.args, 
                    name=self.args.get('run_name', None), 
                    group=self.args.get('group', None),
                    tags=self.args.get('tags', None),
                    notes=self.args.get('notes', None))

    def log(self, log_dict=None, step=None, commit=True, prefix=''):
        """ Add a log to wandb. This should be called when a metric datapoint is to be uploaded to cloud.
        Setting argument ``prefix`` will add an extra prefix to all the key names in ``log_dict``.

        This is useful when you would like to classify the metrics by folders. 
        E.g., setting prefix='train/' or 'test/' will put all the corresponding metrics inside a 'train' or 'test' folder.
        """
        if isinstance(log_dict, dict) and len(log_dict):
            prefix_dict = {prefix+key: log_dict[key] for key in log_dict}
            wandb.log(prefix_dict, step=step, commit=commit)

    def add_summary(self, **kwargs):
        """ Add the summary info to the current run. 
        For example, you may call this function to save your final test accuracy in the run.
        """
        for key in kwargs:
            wandb.run.summary[key] = kwargs[key]
    
    def add_file(self, filepaths, filename=None, type='result'):
        """ Upload the specified file to cloud. (along with the current run)
        filepaths: the target file's local path (str or list of str)
        filename: the name you would like to show in wandb (must be unique across all runs in the project)
        type: type of this file
        """
        if filename is None:
            filename = wandb.run.id + time.strftime("-ymdhms-%Y-%m-%d-%H-%M-%S")
        artifact = wandb.Artifact(filename, type=type)
        if isinstance(filepaths, str):
            filepaths = [filepaths]
        for filepath in filepaths:
            artifact.add_file(filepath)
        wandb.run.log_artifact(artifact)

    def finish(self):
        """ Finish logging the current run.
        """
        wandb.finish()

""" Trainer & Arguments """
import re
import os
import json
import math
import torch
import random
import shutil
import warnings
import datetime
import numpy as np

from tqdm import tqdm
from pathlib import Path

from dataclasses import asdict, dataclass
from abc import abstractmethod
from typing import Optional

from torch.optim.lr_scheduler import LambdaLR

""" Default names for checkpointing files.
"""

DEFAULT_CKPT_NAME = "checkpoint"
TRAINER_ARGS_NAME = "trainer_args.json"
TRAINER_STATE_NAME = "trainer_state.json"
WEIGHTS_NAME = "pytorch_model.bin"
OPTIMIZER_NAME = "optimizer.pt"
SCHEDULER_NAME = "scheduler.pt"

def set_random_seed(random_seed):
    np.random.seed(random_seed)
    random.seed(random_seed)
    torch.manual_seed(random_seed)

class AbstractTrainer:
    """ Abstract trainer class. 
    """
    @abstractmethod
    def __init__(self):
        """ Initialize the trainer with arguments, model and datasets.
        """
        pass

    @abstractmethod
    def train(self):
        """ Define the training process.
        """
        pass

    @abstractmethod
    def evaluate(self):
        """ Define the evaluation process.
        """
        pass

@dataclass
class AbstractArguments:
    """ Abstract arguments class.
    """
    @abstractmethod
    def __init__(self):
        """ Initialize the arguments class by providing necessary arguments explicitly.
        """
        pass

    def update(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def to_dict(self):
        """ Convert self arguments to dict type.
        """
        d = asdict(self)
        d.update(vars(self))
        return d

    def to_json_string(self):
        """ Serializes this instance to a JSON string.
        """
        return json.dumps(self.to_dict(), indent=4, sort_keys=True, ensure_ascii=False)

    def save_json(self, out_fn):
        """ Save self to a json file.
        """
        with open(out_fn, 'w') as f:
            f.write(self.to_json_string())
    
    def load_json(self, out_fn):
        """ Load self params from a json file.
        """
        with open(out_fn, 'r') as f:
            args = json.loads(f.read())
            self.update(**args)

class Trainer(AbstractTrainer):
    """ The basic trainer class.

    All the fixed parameters are updated during init and stored in self.args
    All the state parameters are updated dynamically and stored in self.state

    Use self.train() / self.evaluate() / self.evaluate(get_prediction=True) to conduct train / evaluate / predict operations.
    """
    def __init__(self, model, args,
            train_dataset=None, eval_dataset=None,
            data_collator=None,
            compute_metrics=None):
        # The two argument classes: args for static params; state for dynamic params.
        self.args = args
        # init model state params: global step == 1
        self.init_state()

        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.collator = data_collator
        self.compute_metrics = compute_metrics

        # Initialize training parameters
        # 0. Folder & Device & Wrap model by parallel wrapper
        if not os.path.exists(self.args.output_dir): os.mkdir(self.args.output_dir)
        if self.args.n_gpus == 0:
            self.device = 'cpu'
        elif self.args.use_ddp is False:
            self.device = f'cuda:{self.args.devices[0]}'
        elif self.args.use_ddp is True:
            self.device = f'cuda:{self.args.devices[self.args.rank]}'
        self.device = torch.device(self.device)

        self.model = self.init_model_parallel(model)

        # 1. get total #training steps and #eval steps
        self.init_steps(train_dataset, eval_dataset)

        # 2. build dataloader from dataset, collator and batchsize
        if train_dataset:
            self.train_loader = self.get_dataloader(train_dataset, data_collator, is_train=True)
        if eval_dataset:
            self.eval_loader = self.get_dataloader(eval_dataset, data_collator, is_train=False)

        # 3. init optimizer & scheduler
        self.init_optimizer()
        self.init_scheduler()

        # 4. setup random seed
        set_random_seed(self.args.seed)

    @property
    def is_master(self):
        """ Indicate whether the current process is the master.
        The master is responsible for outputing messages and checkpointing.

        This is useful in DistributedDataParallel.
        """
        if self.args.use_ddp:
            if self.args.rank == 0:
                # the rank-0 process is master
                return True
            else:
                return False
        else:
            return True

    def train_step(self, batch):
        """ Overwrite this to customize the training process. 
        """
        # IMPORTANT: global training step += 1, will lead to infinite loop if unspecified
        self.state.global_step += 1

        self.model.train()
        self.optimizer.zero_grad()

        # forward
        outputs = self.model(**batch)

        # back prop
        if hasattr(outputs, 'loss'):
            loss = outputs.loss.mean()
        else:
            loss = outputs['loss'].mean()
        loss.backward()
        loss = loss.detach().item()
        
        # gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
        # update params
        self.optimizer.step()
        self.lr_scheduler.step()

        return loss

    def eval_step(self, batch):
        """ Overwrite this to customize the evaluating process. 
        """
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(**batch)

            if hasattr(outputs, 'loss'):
                losses = outputs.loss.detach().cpu()
            else:
                losses = outputs['loss'].detach().cpu()

            if hasattr(outputs, 'logits'):
                logits = outputs.logits.detach().cpu()
            else:
                logits = outputs['logits'].detach().cpu()

        labels = batch['labels'].detach().cpu()
        return logits, labels, losses
    
    def trigger_perf_check(self, metrics: Optional[dict]=None):
        """ Check whether the input metrics is the best. And log corresponding params if necessary.
        Return:
            is_best <bool>: whether the current model is the best
        """
        is_best = False
        if metrics is not None and self.args.metric_for_best_model is not None:
            metric_value = metrics[self.args.metric_for_best_model]
            # We use "greater equal" and "less equal" here because we may call this method multiple times
            # and metric that has the same value with best metric is also the best
            operator = np.greater_equal if self.args.greater_is_better else np.less_equal

            if (self.state.best_metric is None
                or self.state.best_model_checkpoint is None
                or operator(metric_value, self.state.best_metric)):
                # new best metrics
                is_best = True
                # log best model info
                self.state.best_metric = float(metric_value)
                self.state.best_model_step = self.state.global_step
        return is_best

    def trigger_eval_or_save(self, eval_dataset=None, end_of_epoch=False):
        """ Define the evaluation and checkpointing behaviors here.
        """
        # evaluation strategy
        if self.state.global_step != self.state.last_eval_step and \
        (    
            (end_of_epoch and self.args.evaluation_strategy == 'epoch')
                    or 
                (self.args.evaluation_strategy == 'steps' and self.state.global_step % self.args.eval_steps == 0)
        ):
            metrics = self.evaluate(eval_dataset)
            self.state.last_eval_step = self.state.global_step
            # check whether eval performance is the best, and log the best step & metric value
            self.trigger_perf_check(metrics)

        else:
            metrics = None
        
        # save strategy
        if self.is_master and \
            (    (end_of_epoch and self.args.save_strategy == 'epoch')
                    or
                (self.args.save_strategy == 'steps' and self.state.global_step % self.args.save_steps == 0)
        ):
            self.save_checkpoint(self.model, metrics)

        return metrics
    
    def trigger_leave(self):
        """ Check whether it is time to end the training process.
        Case:
            1. if self.state.global_step > self.args.num_training_steps: True
            2. (early stopping) if 
                (1) self.args.early_stopping_steps is not None
                (2) self.state.global_step - self.state.best_model_step >= self.args.early_stopping_steps
        """
        if self.state.global_step > self.args.num_training_steps:
            return True
        if self.args.early_stopping_steps is not None:
            if (
                self.state.global_step - self.state.best_model_step >= self.args.early_stopping_steps
                ):
                return True

        return False   

    def log(self, metrics: Optional[dict]=None, 
                summary: Optional[dict]=None, 
                artifacts: Optional[dict]=None, quiet=True):
        """ Log metrics/ summary/ artifacts.

        artifacts: {'filepaths': str/list of str,  'filename': None/str, 'type': None/str}

        Example usage:
        >>> trainer.log(metrics={'test/accuracy': 0.9788})
        >>> trainer.log(summary={'test/accuracy': 0.9788, 'running time': 32})
        >>> trainer.log(artifacts={'filepaths': 'save.pth', 'type': 'result'})
        """
        # only log on the master process
        if not self.is_master:
            return 

        if metrics is not None:
            if hasattr(self, 'wandb'):
                self.wandb.log(metrics, step=self.state.global_step)

            if not quiet:
                    # print by table
                print_columns = ['global_step']
                print_dict = {'global_step': self.state.global_step}
                for key in metrics:
                    for metric_name in metrics[key]:
                        new_key = key+'-'+metric_name
                        print_columns.append(new_key)
                        print_dict[new_key] = metrics[key][metric_name]

                if not hasattr(self, 'print_table'):
                    self.print_table = DynamicTable(print_columns, print_func='tqdm')
                table_str = self.print_table.add_row(print_dict, return_string=True)
                print('\r', end='')
                print(table_str)
                
        if summary is not None:
            if hasattr(self, 'wandb'):
                self.wandb.add_summary(**summary)

        if artifacts is not None:
            if hasattr(self, 'wandb'):
                if 'filepaths' in artifacts:
                    self.wandb.add_file(**artifacts)

    def train(self, train_dataset=None, eval_dataset=None):
        """ Main entry for training the model using the given train loader.
        If train dataloader not specified, will use the train dataset in initialization.
        """
        if train_dataset is None:
            train_loader = self.train_loader
        else:
            train_loader = self.get_dataloader(train_dataset, self.collator, is_train=True)
            self.init_steps(train_dataset=train_dataset)
        
        total_loss = 0
        with tqdm(total=self.args.num_training_steps, initial=self.state.global_step, 
                    leave=True, disable=not self.is_master) as pbar:
            while True:
                total_loss = 0
                for i, batch in enumerate(train_loader):
                    loss = self.train_step(batch)
                    total_loss += loss * self.args.train_batch_size

                    pbar.update(1)
                    self.state.train_loss = total_loss/((i+1)*self.args.train_batch_size)
                    pbar.set_description(f'Train avg. loss: {self.state.train_loss:.4f}')

                    metrics = self.trigger_eval_or_save(eval_dataset, end_of_epoch=False)
                    if metrics:
                        metrics = {'train': {'loss': self.state.train_loss}, 
                                    'eval': metrics}
                        self.log(metrics=metrics, quiet=False)
                    if self.trigger_leave(): break

                metrics = self.trigger_eval_or_save(eval_dataset, end_of_epoch=True)
                if metrics:
                    metrics = {'train': {'loss': self.state.train_loss}, 
                                'eval': metrics}
                    self.log(metrics=metrics, quiet=False)
                if self.trigger_leave(): break
                
        if self.args.load_best_model_at_end and self.is_master:
            self.model = self.load_model(self.model, self.state.best_model_checkpoint)

    def evaluate(self, eval_dataset=None, get_prediction=False):
        """ Main entry for evaluating the model using the given train loader.
        If train dataloader not specified, will use the train dataset in initialization.

        get_prediction: bool, if set to True, will return a dict object containing 
                    the predicted logits, labels, losses and metrics
        """
        if eval_dataset is None:
            eval_loader = self.eval_loader
        else:
            eval_loader = self.get_dataloader(eval_dataset, self.collator, is_train=False)
            self.init_steps(eval_dataset=eval_dataset)

        with tqdm(total=self.args.num_eval_steps, leave=False, disable=not self.is_master) as pbar:
            total_loss = 0
            all_logits, all_labels, all_losses = [], [], []
            for i, batch in enumerate(eval_loader):
                logits, labels, losses = self.eval_step(batch)
                total_loss += losses.sum()

                pbar.update(1)
                pbar.set_description(f'Eval avg. loss: {total_loss/((i+1)*self.args.eval_batch_size):.4f}')

                all_logits.append(logits)
                all_labels.append(labels)
                all_losses.append(losses.reshape(-1, 1))
            all_logits = torch.cat(all_logits).numpy()
            all_labels = torch.cat(all_labels).numpy()
            all_losses = torch.cat(all_losses).numpy()
        
        metrics = self.compute_metrics((all_logits, all_labels))
        metrics['loss'] = all_losses.mean()

        if get_prediction:
            return {'logits': all_logits, 'labels': all_labels, 'losses': all_losses, 'metrics': metrics}
        else:
            return metrics
    
    def init_state(self):
        """ Override this to customize model parameters initialization.
        """
        self.state = ModelState(
            global_step = 0,
            best_metric = None,
            best_model_checkpoint = None,
            best_model_step = 0,
            last_eval_step = -1,
        )       
    
    def init_steps(self, train_dataset=None, eval_dataset=None):
        """ Initialize self.args.num_training_steps and self.args.num_eval_steps.

        self.args.num_training_steps are fixed once initialized, while self.args.num_eval_steps changes 
        according to the input eval_dataset (e.g., adapt to valid/test set respectively)
        """
        if train_dataset is not None and not hasattr(self.args, 'num_training_steps'):
            assert hasattr(self.args, 'num_train_epochs') and self.args.num_train_epochs > 0
            num_training_steps = math.ceil(self.args.num_train_epochs * len(train_dataset) / self.args.train_batch_size)
            setattr(self.args, 'num_training_steps', num_training_steps)

        if eval_dataset is not None:
            num_eval_steps = math.ceil(len(eval_dataset) / self.args.eval_batch_size)
            setattr(self.args, 'num_eval_steps', num_eval_steps)

    def init_wandb(self, project_name: Optional[str]=None, 
                        run_name: Optional[str]=None, group: Optional[str]=None,
                        tags: Optional[list]=None, notes: Optional[str]=None):
        """ Initialize the wandb project to enable cloud recording of performance.

        Example usage:
        >>> trainer.init_wandb(project_name='test', run_name='test_2', group='preliminary', tags=['test', 'fast'], notes='A really testy test')

        Remember to manually close wandb runs after all train/test finished by
        >>> trainer.finish_wandb()
        """
        # no need to init wandb if self is not in master
        if not self.is_master:
            return

        kwargs = self.args.to_dict()

        if project_name is not None:
            kwargs['name'] = project_name
        if run_name is not None:
            kwargs['run_name'] = run_name
        if group is not None:
            kwargs['group'] = group
        if tags is not None:
            kwargs['tags'] = tags
        if notes is not None:
            kwargs['notes'] = notes

        self.wandb = wandbWrapper(**kwargs)

        if run_name is None and kwargs['run_name'] is None:
            self.args.run_name = self.wandb.run.name
    
    def finish_wandb(self):
        """ Finish the wandb logging。
        """
        # no need to init wandb if self is not in master
        if not self.is_master:
            return
        
        self.wandb.finish()

    def init_model_parallel(self, model):
        """ Override this function to customize model parallel, e.g. use torch_geometric/dgl wrapper.s
        """
        if self.args.use_ddp:
            if self.args.n_gpus > 0:
                model.to(self.device)
                model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[self.device], output_device=self.device)
            else:
                model = torch.nn.parallel.DistributedDataParallel(model)
                
        else:
            if self.args.n_gpus >= 1:
                # do not repeatedly wrap the model
                if not isinstance(model, torch.nn.DataParallel):
                    model = torch.nn.DataParallel(model, device_ids=self.args.devices)
                model.to(self.device)

        return model

    def get_dataloader(self, dataset, collator, is_train=False):
        if is_train:
            if self.args.use_ddp:
                # the batch_size per process equals per_device batch_size (in ddp)
                batch_size = self.args.per_device_train_batch_size
                sampler = torch.utils.data.DistributedSampler(dataset, rank=self.args.rank, 
                                        num_replicas=self.args.world_size, shuffle=True)
            else:
                # the batch_size equals total batch_size (in dataparallel or vanilla)
                batch_size = self.args.train_batch_size
                sampler = torch.utils.data.RandomSampler(dataset)
        else:
            batch_size = self.args.eval_batch_size
            sampler = torch.utils.data.SequentialSampler(dataset)
        return torch.utils.data.DataLoader(dataset, batch_size=batch_size, 
                                        collate_fn=collator, sampler=sampler, pin_memory=self.args.pin_memory)

    def init_optimizer(self):
        if not hasattr(self, 'optimizer'):
            optim = {'adamw': torch.optim.AdamW, 
                    'adam': torch.optim.Adam, 
                    'sgd': torch.optim.SGD}
            optim = optim[self.args.optim]
            self.optimizer = optim(self.model.parameters(), lr=self.args.learning_rate, weight_decay=self.args.weight_decay)
        return self.optimizer
        
    def init_scheduler(self):
        if not hasattr(self, 'lr_scheduler'):
            self.lr_scheduler = get_lr_scheduler(
                self.args.lr_scheduler_type,
                optimizer=self.optimizer,
                num_warmup_steps=self.args.get_warmup_steps(self.args.num_training_steps),
                num_training_steps=self.args.num_training_steps,
            )
        return self.lr_scheduler

    def save_checkpoint(self, model=None, metrics: Optional[dict]=None):
        output_dir = self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        checkpoint_folder = os.path.join(output_dir, f"{DEFAULT_CKPT_NAME}-{self.state.global_step}")
        os.makedirs(checkpoint_folder, exist_ok=True)
        if hasattr(self, 'wandb'):
            wandb_folder = os.path.join(output_dir, self.args.run_name)
            os.makedirs(wandb_folder, exist_ok=True)
            setattr(self.args, 'wandb_folder', wandb_folder)
        
        # save model checkpoint
        self.save_model(model, checkpoint_folder)

        # save arguments and states
        self.state.save_json(os.path.join(checkpoint_folder, TRAINER_STATE_NAME))
        self.args.save_json(os.path.join(checkpoint_folder, TRAINER_ARGS_NAME))

        # save optimizer & scheduler
        torch.save(self.optimizer.state_dict(), os.path.join(checkpoint_folder, OPTIMIZER_NAME))
        with warnings.catch_warnings(record=True) as caught_warnings:
            torch.save(self.lr_scheduler.state_dict(), os.path.join(checkpoint_folder, SCHEDULER_NAME))

        # determine whether it is the best model
        if self.trigger_perf_check(metrics):
            # if it is the best
            self.state.best_model_checkpoint = checkpoint_folder
        
        # delete old checkpoints if reaches maximum # checkpoints
        self.keep_checkpoints(output_dir)

    def load_checkpoint(self, folder: Optional[str]=None):
        if folder is None:
            folder = f"{self.args.name}-{self.state.global_step}"

    def save_model(self, model=None, checkpoint_folder: Optional[str]=None):
        if model is None:
            model = self.model
        if checkpoint_folder is None:
            output_dir = self.args.output_dir
            checkpoint_folder = os.path.join(output_dir, f"{DEFAULT_CKPT_NAME}-{self.state.global_step}")
            os.makedirs(checkpoint_folder, exist_ok=True)
        
        # different behavior on DataParallel wrapper and torch modules.
        if isinstance(model, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel)):
            state_dict = model.module.state_dict()
        else:
            state_dict = model.state_dict()

        torch.save(state_dict, os.path.join(checkpoint_folder, WEIGHTS_NAME))
    
    def load_model(self, model=None, checkpoint_folder: Optional[str]=None):
        if model is None:
            model = self.model
        if checkpoint_folder is not None and os.path.exists(checkpoint_folder):
            # only load the model checkpoint when the checkpoint_folder param is a valid directory
            state_dict = torch.load(os.path.join(checkpoint_folder, WEIGHTS_NAME))
            
            # different behavior on DataParallel wrapper and torch modules
            if isinstance(model, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel)):
                model.module.load_state_dict(state_dict)
            else:
                model.load_state_dict(state_dict)
            
        return model

    def get_sorted_checkpoints(self, output_dir: Optional[str]=None):
        ordering_and_checkpoint_path = []

        glob_checkpoints = [str(x) for x in Path(output_dir).glob(f"{DEFAULT_CKPT_NAME}-*") if os.path.isdir(x)]

        for path in glob_checkpoints:
            regex_match = re.match(f".*{DEFAULT_CKPT_NAME}-([0-9]+)", path)
            if regex_match is not None and regex_match.groups() is not None:
                ordering_and_checkpoint_path.append((int(regex_match.groups()[0]), path))

        checkpoints_sorted = sorted(ordering_and_checkpoint_path)
        checkpoints_sorted = [checkpoint[1] for checkpoint in checkpoints_sorted]
        
        # do not delete the best model
        if self.state.best_model_checkpoint is not None:
            best_model_index = checkpoints_sorted.index(str(Path(self.state.best_model_checkpoint)))
            # replace the best to second in list
            for i in range(best_model_index, len(checkpoints_sorted) - 2):
                checkpoints_sorted[i], checkpoints_sorted[i + 1] = checkpoints_sorted[i + 1], checkpoints_sorted[i]
        return checkpoints_sorted

    def keep_checkpoints(self, output_dir: Optional[str]=None):
        """ Keep the number of checkpoints under self.args.save_total_limit.
        Special case: when both self.args.save_total_limit == 1 and self.args.load_best_model_at_end == True, 
                    will reset self.args.save_total_limit = 2 so that the current checkpoint will not be deleted.
        """
        if output_dir is None:
            output_dir = self.args.output_dir

        if self.args.save_total_limit is None or self.args.save_total_limit <= 0:
            return
        else:
            sorted_checkpoints = self.get_sorted_checkpoints(output_dir)
            if len(sorted_checkpoints) <= self.args.save_total_limit:
                return
            else:
                # If save_total_limit=1 with load_best_model_at_end=True, we could end up deleting the last checkpoint, which
                # we don't do to allow resuming.
                save_total_limit = self.args.save_total_limit
                if (
                    self.state.best_model_checkpoint is not None
                    and self.args.save_total_limit == 1
                    and sorted_checkpoints[-1] != self.state.best_model_checkpoint
                ):
                    save_total_limit = 2
                number_to_delete = max(0, len(sorted_checkpoints) - save_total_limit)
                checkpoints_to_be_deleted = sorted_checkpoints[:number_to_delete]
                for checkpoint in checkpoints_to_be_deleted:
                    shutil.rmtree(checkpoint)


@dataclass
class ModelState(AbstractArguments):
    """ The basic model state class, used for storing model state info in trainers.
    """
    global_step: int    = 0

    best_metric: Optional[float] = None
    best_model_checkpoint: Optional[str] = None

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


@dataclass
class Arguments(AbstractArguments):
    """ The basic arguments class.
    All the specified arguments should be generic (task-agnostic).


    >>> basics

    name = 'checkpoint'                 # the project name
    run_name = None                     # the run-name used for wandb logging, if left None, will be randomly assigned by wandb
    group = None                        # the group name of this run (e.g., 'preliminary-exp'), also used for wandb logging
    output_dir = None                   # the output directory where all checkpoints will be saved to, if left None, will be set to ``name``
    devices: Optional[list] = None      # -None: automatically choose all gpus/ cpu;  
                                          -'cpu': use cpu only;  
                                          -[0,1,2,3], use all specified gpus, the first device-id in the list is supposed to be the hub
    
    >>> distributed training 

    use_ddp: bool = False               # whether to use DistributedDataParallel or not
    rank: int     = 0                   # if use_ddp is True, this is the rank of process where this trainer is in
    world_size: int = len(devices)      # if use_ddp is True, this specifies the number of processes

    >>> training strategy

    per_device_train_batch_size = 8     # train batch size = this * len(devices)
    per_device_eval_batch_size = 8      # eval batch size = this * len(devices)

    learning_rate = 1e-5                # learning rate
    optim = 'adamw'                     # optimizer name that will be passed to Trainer.init_optimizer()
    lr_scheduler_type = 'linear'        # types of learning rate scheduler that will be passed to Trainer.init_scheduler()
    warmup_steps = 0                    # warmup steps param for many schedulers
    warmup_ratio = 0.0                  # [override] if positive, will override the warmup_steps=training_steps*warmup_ratio
    weight_decay = 0.0                  # L2-regularization 
    max_grad_norm = 1.0                 # gradient clipping norm

    num_train_epochs =  3.0             # total number of training epochs
    max_steps =         -1              # [override] num_train_epochs if specified > 0
    early_stopping_steps = None         # the early stopping steps; default: None (not using early stopping)

    >>> evaluation strategy

    evaluation_strategy =   'steps'     # 'no' -> no eval | 'steps' -> eval_steps | 'epoch' -> at the end of each epoch
    eval_steps =            500         # only effective when evaluation_strategy == 'steps'


    >>> saving strategy

    save_strategy =         'steps'     # same as above
    save_steps =            500         # only effective when save_strategy == 'steps'
    save_total_limit =      None        # None / int, maximum number of checkpoints to save (will save the best one and the latest one)    

    load_best_model_at_end= True        # whether or not to load the best model checkpoint at the end of training
    metric_for_best_model = 'loss'      # the metrics by which the ``best`` model is determined
    greater_is_better = False           # for ``metric_for_best_model``, whether greater is better


    >>> reproducibility

    seed = 1999                         # random seed
    data_seed = None                    # random seed for data. If left None, will be set to ``seed``

    >>> other params

    pin_memory: bool            = True

    # ### other parameters (not used by the Trainer class, maybe useful in command line)  
    # resume_from_checkpoint = ''  
    # do_train = False  
    # do_eval = False  
    # do_predict = False
    """
    
    ''' basics '''
    name: str                   = 'checkpoint'
    run_name: Optional[str]     = None
    group: Optional[str]        = None
    output_dir: Optional[str]   = None
    devices: Optional[list]     = None

    ''' distributed training '''
    use_ddp: bool               = False
    rank: Optional[int]         = None
    world_size: Optional[int]   = None

    ''' training strategy '''
    per_device_train_batch_size: int    = 8
    per_device_eval_batch_size: int     = 8

    learning_rate: float = 1e-5
    optim: str           = 'adamw'
    lr_scheduler_type: str  = 'linear'
    warmup_steps: int       = 0  
    warmup_ratio: float     = 0.0 
    weight_decay: float  = 0.0
    max_grad_norm: float = 1.0

    num_train_epochs: float   = 3.0
    max_steps: int            = -1
    early_stopping_steps: Optional[int] = None
    
    ''' evaluation strategy '''
    evaluation_strategy: str            = 'steps'   
    eval_steps: Optional[int]           = None

    ''' saving strategy '''
    save_strategy: str                  = 'steps'
    save_steps: Optional[int]           = None
    save_total_limit: Optional[int]     = None

    load_best_model_at_end: bool        = True
    metric_for_best_model: str          = 'loss'
    greater_is_better: bool             = False

    ''' reproducibility '''
    seed: int                   = 1999
    data_seed: Optional[int]    = None

    ''' other params '''
    pin_memory: bool            = True

    # # other parameters (not used by the Trainer class, maybe useful in command line)
    # resume_from_checkpoint: Optional[str] = None
    # do_train: bool   = False
    # do_eval: bool    = False
    # do_predict: bool = False

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

        # TODO: check params validity

        # init training params (devices, warmup-steps)
        self.init_devices()

        # get real batch_size & ddp world size
        if isinstance(self.devices, str):
            n_device = 1
        else:
            n_device = len(self.devices)
        if self.use_ddp:
            if self.world_size is None:
                self.world_size = n_device
            assert self.rank is not None and self.rank in range(self.world_size)
        # but make sure to pass the "per_device_train_batch_size" to train data sampler instead of "train_batch_size"
        self.train_batch_size = self.per_device_train_batch_size * n_device
        if self.use_ddp:
            self.eval_batch_size = self.per_device_eval_batch_size
        else:
            self.eval_batch_size = self.per_device_eval_batch_size * n_device

        # evaluation and saving steps
        if self.evaluation_strategy == 'steps':
            assert self.eval_steps is not None and self.eval_steps > 0
            if self.save_steps is None and self.save_strategy == 'steps':
                # use the same strategy as eval_steps
                self.save_steps = self.eval_steps
        elif self.evaluation_strategy == 'epoch':
            self.eval_steps = None

        if self.save_strategy == 'steps':
            assert self.save_steps is not None and self.save_steps > 0
        elif self.save_strategy == 'epoch':
            self.save_steps = None

        # get warmup steps if possible
        if self.max_steps > 0:
            # override num_train_epochs
            self.num_training_steps = self.max_steps
            self.warmup_steps = self.get_warmup_steps(self.num_training_steps)
        
        # get data random seed
        if self.data_seed is None:
            self.data_seed = self.seed

        # get output_dir & names
        if self.output_dir is None:
            self.output_dir = self.name

    def init_devices(self):
        if self.devices is None:
            if torch.cuda.is_available():
                self.devices = list(range(torch.cuda.device_count()))
            else:
                self.devices = 'cpu'

        if self.devices == 'cpu':
            self.n_gpus = 0
        else:
            self.n_gpus = len(self.devices)

    def get_warmup_steps(self, num_training_steps: int):
        """
        Get number of steps used for a linear warmup.
        """
        warmup_steps = (
            self.warmup_steps if self.warmup_steps > 0 else math.ceil(num_training_steps * self.warmup_ratio)
        )
        return warmup_steps


# Schedulers: copied from huggingface
def get_lr_scheduler(lr_scheduler_type, optimizer, num_warmup_steps: Optional[int]=None, num_training_steps: Optional[int]=None, last_epoch: int=-1):
    if lr_scheduler_type == 'constant':
        lr_lambda = lambda _: 1

    elif lr_scheduler_type == 'constant_with_warmup':
        assert isinstance(num_warmup_steps, int) and num_warmup_steps > 0
        def lr_lambda(current_step: int):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1.0, num_warmup_steps))
            return 1.0

    elif lr_scheduler_type == 'linear':
        def lr_lambda(current_step: int):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            return max(
                0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
            )
    
    elif lr_scheduler_type == 'cosine':
        num_cycles = 0.5
        def lr_lambda(current_step):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    elif lr_scheduler_type == 'polynomial':
        lr_end=1e-7
        power=1.0
        lr_init = optimizer.defaults["lr"]
        if not (lr_init > lr_end):
            raise ValueError(f"lr_end ({lr_end}) must be be smaller than initial lr ({lr_init})")

        def lr_lambda(current_step: int):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            elif current_step > num_training_steps:
                return lr_end / lr_init  # as LambdaLR multiplies by lr_init
            else:
                lr_range = lr_init - lr_end
                decay_steps = num_training_steps - num_warmup_steps
                pct_remaining = 1 - (current_step - num_warmup_steps) / decay_steps
                decay = lr_range * pct_remaining**power + lr_end
                return decay / lr_init  # as LambdaLR multiplies by lr_init
    
    else:
        raise ValueError('Wrong lr-scheduler type. Valid values include: constant, constant_with_warmup, linear, cosine, polynomial')

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def init_process_group(rank, size, master_addr='localhost', master_port='29500', backend='nccl', timeout=600):
    """ Initialize the distributed environment. 

    It is extremely important to set the ``timeout'' , since the DDP will hang training when one of the process faces an error.
    To solve this, we need to:
    (1) Set the environment variable NCCL_ASYNC_ERROR_HANDLING=1 during training.
    (2) Set the timeout param upon dist.init_process_group (in secs).
    (3) If possible, catch the error message of training process and log it. This is because the faulty process does not print the message.

    Reference:
    https://github.com/pytorch/pytorch/issues/50820#issuecomment-763973361
    https://pytorch.org/docs/stable/distributed.html?highlight=init_process_group#torch.distributed.init_process_group
    """
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = master_port
    os.environ['NCCL_ASYNC_ERROR_HANDLING'] = '1'
    torch.distributed.init_process_group(backend, rank=rank, world_size=size, timeout=datetime.timedelta(seconds=timeout))



if __name__ == '__main__':
    ''' Below is an example. You may run this file to see the outputs. '''

    header = "python3 finetune_lm.py "

    fixed_configs = {
        'train_batch_size': 32,
        'eval_batch_size': 32,
        'eval_steps': 100,
        'save_steps': 100,
        'max_steps': 500
    }

    comb_configs = {
        'model_checkpoint': ['roberta-base', 'bert-base-cased'],
        'dataset': [2016, 2018],
        'random_seed': [2022, 4044, 8088],
    }

    all_runstrs = get_runstr_combinations(header, fixed_configs, comb_configs)

    for string in all_runstrs:
        print(string)



    args = Arguments()
    print(args.__dict__, args.do_predict, vars(args))
    # print(args.to_dict())
    print(args.to_json_string())