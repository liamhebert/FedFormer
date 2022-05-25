from gym.envs.mujoco import HalfCheetahEnv
import metaworld
import rlkit.torch.pytorch_util as ptu
from rlkit.data_management.env_replay_buffer import EnvReplayBuffer
from rlkit.envs.wrappers import NormalizedBoxEnv
from rlkit.launchers.launcher_util import setup_logger
from rlkit.samplers.data_collector import MdpPathCollector
from rlkit.torch.sac.policies import MakeDeterministic
from rlkit.torch.networks import ConcatMlp

import numpy as np 
import torch.nn as nn
from sac_algorithm import TorchBatchRLAlgorithm
from fed_trainer import FedTrainer
from fed_algorithm import FedAlgorithm
from fedformer import FedFormer
from fed_path_collector import FedPathCollector
from policy import TanhGaussianPolicy
import random
import click

def experiment(variant):
    mt10 = metaworld.MT10()
    training_envs = []
    algorithms = []
    envs = [task for task in mt10.train_tasks if task.env_name == variant['task']]
    for env_cls in [mt10.train_classes[variant['task']] for _ in range(variant['num_agents'])]:
        env = env_cls()
        tasks_train = random.sample(envs, k=5)
        if not variant['overlap']:
            envs = [x for x in envs if x not in tasks_train]
        tasks_test = random.sample(envs, k=5)
        if not variant['overlap']:
            envs = [x for x in envs if x not in tasks_test]
        env.set_task(tasks_train[0]) # only used for env information
        training_envs.append((env, tasks_train, tasks_test, variant['task']))

    for i, (env, tasks_train, tasks_test, name) in enumerate(training_envs):
        # Select task
        # Note: These envs are not actually used, but rather just for info
        expl_env = NormalizedBoxEnv(env)
        eval_env = NormalizedBoxEnv(env)
        obs_dim = expl_env.observation_space.low.size
        action_dim = eval_env.action_space.low.size

        # build networks
        M = variant['layer_size']
        if variant['fedFormer']:
            qf1 = FedFormer(
                input_size=obs_dim + action_dim,
                from_saved=variant['from_saved'],
                saved_id='qf1',
                output_size=1,
                hidden_sizes=[M, M, M],
                num_layers=variant['transformer_num_layers'],
                transformer_layer_config=variant['transformer_layer_kwargs'],
                agent_index=i,
                num_agents=variant['num_agents']
            )
            qf2 = FedFormer(
                input_size=obs_dim + action_dim,
                from_saved=variant['from_saved'],
                saved_id='qf2',
                output_size=1,
                hidden_sizes=[M, M, M],
                num_layers=variant['transformer_num_layers'],
                transformer_layer_config=variant['transformer_layer_kwargs'],
                agent_index=i,
                num_agents=variant['num_agents']
            )
            target_qf1 = FedFormer(
                input_size=obs_dim + action_dim,
                from_saved=variant['from_saved'],
                saved_id='target_qf1',
                output_size=1,
                hidden_sizes=[M, M, M],
                num_layers=variant['transformer_num_layers'],
                transformer_layer_config=variant['transformer_layer_kwargs'],
                agent_index=i,
                num_agents=variant['num_agents']
            )
            target_qf2 = FedFormer(
                input_size=obs_dim + action_dim,
                from_saved=variant['from_saved'],
                saved_id='target_qf2',
                output_size=1,
                hidden_sizes=[M, M, M],
                num_layers=variant['transformer_num_layers'],
                transformer_layer_config=variant['transformer_layer_kwargs'],
                agent_index=i,
                num_agents=variant['num_agents']
            )
        else:
            qf1 = ConcatMlp(
                input_size=obs_dim + action_dim,
                output_size=1,
                hidden_sizes=[M, M, M],
            )
            qf2 = ConcatMlp(
                input_size=obs_dim + action_dim,
                output_size=1,
                hidden_sizes=[M, M, M],
            )
            target_qf1 = ConcatMlp(
                input_size=obs_dim + action_dim,
                output_size=1,
                hidden_sizes=[M, M, M],
            )
            target_qf2 = ConcatMlp(
                input_size=obs_dim + action_dim,
                output_size=1,
                hidden_sizes=[M, M, M],
            )
        policy = TanhGaussianPolicy(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_sizes=[M, M, M],
            hidden_nonlinearity=nn.ReLU,
            output_nonlinearity=None,
            min_std=np.exp(-20.),
            max_std=np.exp(2.)
        )

        eval_path_collector = FedPathCollector(
            policy=policy,
            task_list=tasks_test
        )
        expl_path_collector = FedPathCollector(
            policy=policy,
            task_list=tasks_train
        )
        replay_buffer = EnvReplayBuffer(
            variant['replay_buffer_size'],
            expl_env,
        )
        trainer = FedTrainer(
            env=eval_env,
            policy=policy,
            qf1=qf1,
            qf2=qf2,
            target_qf1=target_qf1,
            target_qf2=target_qf2,
            **variant['trainer_kwargs']
        )

        algorithm_instance = TorchBatchRLAlgorithm(
            trainer=trainer,
            exploration_env=expl_env,
            evaluation_env=eval_env,
            exploration_data_collector=expl_path_collector,
            evaluation_data_collector=eval_path_collector,
            replay_buffer=replay_buffer,
            name=f'{variant['run_name']} - agent {name} ({i})',
            **variant['algorithm_kwargs']
        )
        algorithm_instance.to(ptu.device)
        algorithms += [algorithm_instance]
  

    algorithm = FedAlgorithm(algorithms, variant['algorithm_kwargs']['num_epochs'])
    algorithm.train()

@click.command()
@click.option("--task", default="window-close-v2", help="MT10 Task Name")
@click.option("--seed", default=1, help="Random Seed")
def main(task, seed):
    random.seed(seed)
    variant = dict(
        algorithm="FedFormer",
        task=task,
        overlap=False, # whether enviroments should overlap
        fedFormer=True, # Whether to use FedFormer Q-Functions or not
        run_name="FedFormer", # For logging purposes
        version="normal",
        from_saved=5, # How many encoder networks to save 
        layer_size=400, # Hidden layer size
        replay_buffer_size=int(1E6), 
        transformer_num_layers=2, # number of transformer encoder layers to use
        num_agents=5, # number of federation agents to initialize
        transformer_layer_kwargs=dict(
            d_model=400, # hidden size for each transformer layer
            nhead=4 # number of attention heads to initialize
        ),
        algorithm_kwargs=dict(
            num_epochs=250,
            num_eval_steps_per_epoch=5000,
            num_trains_per_train_loop=600,
            num_expl_steps_per_train_loop=400,
            min_num_steps_before_training=400,
            max_path_length=500,
            batch_size=1200,
        ),
        trainer_kwargs=dict(
            discount=0.99,
            soft_target_tau=5e-3,
            target_update_period=1,
            policy_lr=3E-4,
            qf_lr=3E-4,
            reward_scale=1,
            use_automatic_entropy_tuning=True,
        ),
    )
    setup_logger('', variant=variant)
    ptu.set_gpu_mode(True)  # optionally set the GPU (default=False)
    print('DEVICE', ptu.device)
    experiment(variant)
    
if __name__ == "__main__":
    main()
    
