from rlkit.samplers.data_collector import MdpPathCollector
from rlkit.samplers.rollout_functions import rollout
import numpy as np
from rlkit.envs.wrappers import NormalizedBoxEnv
import random 

MW_TASKS_PER_ENV = 50

class MetaPathCollector(MdpPathCollector):
    def __init__(
        self,
        benchmark,
        policy,
        max_num_epoch_paths_saved=None,
        render=False,
        render_kwargs=None,
        rollout_fn=rollout,
        save_env_in_snapshot=True,
        task_name='reach-v2',
        kind='train',
        k=1):
        
        super().__init__(None, 
            policy, 
            max_num_epoch_paths_saved, 
            render, render_kwargs,
            rollout_fn,
            save_env_in_snapshot)
        
        

        if kind == 'train':
            self.env_cls = benchmark.train_classes[task_name]()
            # self._classes = benchmark.train_classes
            self.task_list = random.choices([task for task in benchmark.train_tasks if task.env_name == task_name], k=k)
           
        elif kind == 'test':
            self.env_cls = benchmark.test_classes[task_name]()
            # self._classes = benchmark.test_classes
            self.task_list = random.choices([task for task in benchmark.train_tasks if task.env_name == task_name], k=k)
        else:
            raise ValueError('kind must be either "train" or "test", '
                             f'not {kind!r}')

        self.task_order = np.arange(len(self.task_list))
        
        self._next_order_index = 0
        self._shuffle_tasks()

    def _shuffle_tasks(self):
        """Reshuffles the task orders."""
        np.random.shuffle(self.task_order)
        
    
    def collect_new_paths(self, max_path_length, num_steps, discard_incomplete_paths, with_replacement=False):
        paths = []
        num_steps_collected = 0
        order_index = self._next_order_index
        while num_steps_collected < num_steps:
            curr_task = self.task_list[self.task_order[order_index]]
            self.env_cls.set_task(curr_task)

            env = NormalizedBoxEnv(self.env_cls) # TODO - Maybe do this at the start?

            max_path_length_this_loop = min(  # Do not go over num_steps
                max_path_length,
                num_steps - num_steps_collected,
            )
            path = self._rollout_fn(
                env,
                self._policy,
                max_path_length=max_path_length_this_loop,
                render=self._render,
                render_kwargs=self._render_kwargs,
            )
            path_len = len(path['actions'])
            if (
                    path_len != max_path_length
                    and not path['dones'][-1]
                    and discard_incomplete_paths
            ):
                break
            num_steps_collected += path_len
            paths.append(path)

            if with_replacement:
                order_index = np.random.randint(0, len(self.task_list))
            else:
                order_index += 1 
                if order_index >= len(self.task_list):
                    order_index = 0
                    self._shuffle_tasks()

        self._next_order_index = order_index
        
        self._num_paths_total += len(paths)
        self._num_steps_total += num_steps_collected
        self._epoch_paths.extend(paths)
        return paths

    
    def get_snapshot(self):
        snapshot_dict = dict(
            policy=self._policy,
        )
        if self._save_env_in_snapshot:
            snapshot_dict['env'] = self.env_cls
        return snapshot_dict

