from typing import Tuple, Iterable, OrderedDict
import gtimer as gt
import tqdm
from sac_algorithm import TorchBatchRLAlgorithm
from joblib import Parallel, delayed
import torch 
import numpy as np
from rlkit.core import Logger 

class FedAlgorithm:
    def __init__(self,
                 algorithms: Iterable[TorchBatchRLAlgorithm],
                 num_epochs,
                 fedFormer=False,
                 max_jobs_per_gpu=1):
        self.num_epochs = num_epochs
        self.algorithms = algorithms
        self.fedFormer = fedFormer
        self.logger = Logger(name='default')
        self.max_jobs_per_gpu=max_jobs_per_gpu

    def train(self, start_epoch=0):
     
        for epoch in gt.timed_for(range(start_epoch, self.num_epochs), save_itrs=True):
            i = 0
            num_gpus = torch.cuda.device_count()

            with Parallel(n_jobs=num_gpus * self.max_jobs_per_gpu, prefer="threads") as parallel:
                parallel(delayed(x.step)(gpu, epoch) for gpu, x in zip(list(range(num_gpus)) * len(self.algorithms), self.algorithms))

            self.logger.log("Epoch {} finished".format(epoch), with_timestamp=True)
            if self.fedFormer: 
                for k in tqdm.tqdm(range(len(self.algorithms)), desc='fusing'):
                    curr = self.algorithms[k]
                    for j in range(k + 1, len(self.algorithms)):
                        other = self.algorithms[j]
                        curr.fuse(other)
            
            else:
                qf1 = []
                qf2 = []
                target_qf1 = []
                target_qf2 = []
                stats = []
                
                for k in tqdm.tqdm(range(len(self.algorithms)), desc='gathering'):
                    self.algorithm.to('cuda:0')
                    networks = self.algorithms[k].get_networks()
                    qf1 += [networks[0]]
                    qf2 += [networks[1]]
                    target_qf1 += [networks[2]]
                    target_qf2 += [networks[3]]
                    stats += [self.algorithms[k].get_stats()]

                # qf1_loss = []
                # qf2_loss = []
                # policy_loss = []
                rewards = []
                keys = []
                for stat in stats:
                    # qf1_loss += [stat['QF1 Loss']]
                    # qf2_loss += [stat['QF2 Loss']]
                    # policy_loss += [stat['Policy Loss']]
                    rewards += [stat['Reward']]
                    keys += [[stat['QF1 Loss'], stat['QF2 Loss'], stat['Policy Loss']]]
                
                keys = torch.Tensor(keys)
                rewards = torch.Tensor(rewards)
                
                query = torch.concat((keys.min(axis=0).values, rewards.max().unsqueeze(0)))
                keys = torch.concat((keys, rewards.unsqueeze(1)), axis=1)

                weights = torch.nn.functional.softmax((keys @ query) / 2, dim=0)

                new_qf1 = self.merge_networks(qf1, weights)
                new_qf2 = self.merge_networks(qf2, weights)
                new_target_qf1 = self.merge_networks(target_qf1, weights)
                new_target_qf2 = self.merge_networks(target_qf2, weights)

                for k in tqdm.tqdm(range(len(self.algorithms)), desc='sending'): 
                    curr = self.algorithms[k]
                    curr.set_networks([new_qf1, new_qf2, new_target_qf1, new_target_qf2])

        if self.fedFormer:
            networks = self.algorithms[0].get_networks()
            qf1 = networks[0].get_encoders()
            qf2 = networks[1].get_encoders()
            target_qf1 = networks[2].get_encoders()
            target_qf2 = networks[3].get_encoders()

            for i in range(len(qf1)):
                torch.save(qf1[i], f'./networks/qf1/encoder-{i}.pt')
                torch.save(qf2[i], f'./networks/qf2/encoder-{i}.pt')
                torch.save(target_qf1[i], f'./networks/target_qf1/encoder-{i}.pt')
                torch.save(target_qf2[i], f'./networks/target_qf2/encoder-{i}.pt')


    def merge_networks(self, networks, weights):
        # Uncomment this for unweighted FedAvg
        # network_params = [net.parameters() for net in networks]
        # ratio = 1 / len(networks)
        # for params in enumerate(zip(*network_params)):
        #     target = params[0] 
        #     target.data.copy_(
        #         sum([param.data * ratio for param in params])
        #     )
        
        averaged_weights = OrderedDict()
        keys = networks[0].state_dict().keys()
        for it, net in enumerate(networks):
            local_weights = net.state_dict()
            for key in keys:
                if it == 0:
                    averaged_weights[key] = weights[it] * local_weights[key]
                else:
                    averaged_weights[key] += weights[it] * local_weights[key]
        networks[0].load_state_dict(averaged_weights)

        return networks[0] # new network
