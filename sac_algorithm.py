from collections import OrderedDict, namedtuple
from typing import Tuple

import numpy as np
import torch
import torch.optim as optim
from rlkit.core.loss import LossFunction, LossStatistics
from torch import nn as nn

import rlkit.torch.pytorch_util as ptu
from rlkit.core.eval_util import create_stats_ordered_dict
from rlkit.torch.torch_rl_algorithm import TorchTrainer
from rlkit.core.logging import add_prefix
import gtimer as gt
from rlkit.core.rl_algorithm import BaseRLAlgorithm
from rlkit.data_management.replay_buffer import ReplayBuffer
from rlkit.samplers.data_collector import PathCollector
import abc
import tqdm


class BatchRLAlgorithm(BaseRLAlgorithm, metaclass=abc.ABCMeta):
    def __init__(
            self,
            trainer,
            exploration_env,
            evaluation_env,
            exploration_data_collector: PathCollector,
            evaluation_data_collector: PathCollector,
            replay_buffer: ReplayBuffer,
            batch_size,
            max_path_length,
            num_epochs,
            num_eval_steps_per_epoch,
            num_expl_steps_per_train_loop,
            num_trains_per_train_loop,
            num_train_loops_per_epoch=1,
            min_num_steps_before_training=0,
            start_epoch=0,  # negative epochs are offline, positive epochs are online
            name='default'
    ):
        super().__init__(
            trainer,
            exploration_env,
            evaluation_env,
            exploration_data_collector,
            evaluation_data_collector,
            replay_buffer,
            name
        )
        self.batch_size = batch_size
        self.max_path_length = max_path_length
        self.num_epochs = num_epochs
        self.num_eval_steps_per_epoch = num_eval_steps_per_epoch
        self.num_trains_per_train_loop = num_trains_per_train_loop
        self.num_train_loops_per_epoch = num_train_loops_per_epoch
        self.num_expl_steps_per_train_loop = num_expl_steps_per_train_loop
        self.min_num_steps_before_training = min_num_steps_before_training
        self._start_epoch = start_epoch
        self.name = name
    
    def fuse(self, other):
        self.trainer.fuse(other.trainer)
    
    def get_networks(self):
        return self.trainer.networks[1:]
    
    def get_stats(self):
        return self.trainer.get_stats()
    
    def set_networks(self, networks):
        self.trainer.set_networks(networks)
       
    def step(self, epoch):
        """Negative epochs are offline, positive epochs are online"""
        # for self.epoch in gt.timed_for(
        #         range(self._start_epoch, self.num_epochs),
        #         save_itrs=True,
        # ):

        offline_rl = epoch < 0
        #self.trainer.to('cuda:0')
        self._begin_epoch(epoch)
        self._step(epoch, offline_rl)
        self._end_epoch(epoch)
        #self.trainer.to('cpu')

    def _step(self, epoch, offline_rl):
        if epoch == 0 and self.min_num_steps_before_training > 0:
            init_expl_paths = self.expl_data_collector.collect_new_paths(
                self.max_path_length,
                self.min_num_steps_before_training,
                discard_incomplete_paths=False,
            )
            if not offline_rl:
                self.replay_buffer.add_paths(init_expl_paths)
            self.expl_data_collector.end_epoch(-1)

        self.eval_data_collector.collect_new_paths(
            self.max_path_length,
            self.num_eval_steps_per_epoch,
            discard_incomplete_paths=True,
        )
        gt.stamp(f'{self.name} - evaluation sampling')

        for _ in tqdm.tqdm(range(self.num_train_loops_per_epoch), desc='num_train_loops'):
            new_expl_paths = self.expl_data_collector.collect_new_paths(
                self.max_path_length,
                self.num_expl_steps_per_train_loop,
                discard_incomplete_paths=False,
            )
            gt.stamp(f'{self.name} - exploration sampling', unique=False)

            if not offline_rl:
                self.replay_buffer.add_paths(new_expl_paths)
            gt.stamp(f'{self.name} - data storing', unique=False)

            self.training_mode(True)
            for itr in tqdm.tqdm(range(self.num_trains_per_train_loop), desc='trains per train loop'):
                train_data = self.replay_buffer.random_batch(self.batch_size)
                #self.logger.log_batch(itr, train_data, self.trainer.discount)
                self.trainer.train(train_data)

            gt.stamp(f'{self.name} - training', unique=False)
            self.training_mode(False)


class TorchBatchRLAlgorithm(BatchRLAlgorithm):
    def to(self, device):
        for net in self.trainer.networks:
            net.to(device)

    def training_mode(self, mode):
        for net in self.trainer.networks:
            net.train(mode)
