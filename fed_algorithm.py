from typing import Tuple, Iterable
import gtimer as gt
import tqdm
from sac_algorithm import TorchBatchRLAlgorithm
from joblib import Parallel, delayed
import torch 

class FedAlgorithm:
    def __init__(self,
                 algorithms: Iterable[TorchBatchRLAlgorithm],
                 num_epochs,
                 fedFormer=False):
        self.num_epochs = num_epochs
        self.algorithms = algorithms
        self.fedFormer = fedFormer

    def train(self, start_epoch=0):
     
        for epoch in gt.timed_for(range(start_epoch, self.num_epochs), save_itrs=True):
            i = 0
            
            for algorithm in tqdm.tqdm(self.algorithms, desc='algorithms'):
                algorithm.step(epoch)
                gt.stamp(f'algorithm-{i}')
                i += 1

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
                for k in tqdm.tqdm(range(len(self.algorithms)), desc='gathering'):
                    networks = self.algorithms[k].get_networks()
                    qf1 += [networks[0]]
                    qf2 += [networks[1]]
                    target_qf1 += [networks[2]]
                    target_qf2 += [networks[3]]
                
                new_qf1 = self.merge_networks(qf1)
                new_qf2 = self.merge_networks(qf2)
                new_target_qf1 = self.merge_networks(target_qf1)
                new_target_qf2 = self.merge_networks(target_qf2)

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
                torch.save(qf1[i], f'networks/qf1/encoder-{i}.pt')
                torch.save(qf2[i], f'networks/qf2/encoder-{i}.pt')
                torch.save(target_qf1[i], f'networks/target_qf1/encoder-{i}.pt')
                torch.save(target_qf2[i], f'networks/target_qf2/encoder-{i}.pt')


    def merge_networks(self, networks):
        #print(networks)
        network_params = [net.parameters() for net in networks]
        ratio = 1 / len(networks)
        for params in zip(*network_params):
            target = params[0] 
            target.data.copy_(
                sum([param.data * ratio for param in params])
            )
        return networks[0] # new network
