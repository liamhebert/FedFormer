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
from meta_sac_trainer import SACTrainer
from meta_sac_algorithm import MetaFedSACAlgorithm
from meta_sac_networks import FederatedTransformer
from meta_path_collector import MetaPathCollector
from meta_sac_policy import TanhGaussianPolicy
import random


def experiment(variant):
    mt10 = metaworld.MT10()
    training_envs = []
    algorithms = []
    for env_cls in [mt10.train_classes['window-close-v2'] for _ in range(variant['num_agents'])]:
        env = env_cls()
        task = random.choice([task for task in mt10.train_tasks
                                if task.env_name == 'window-close-v2'])
        env.set_task(task)
        training_envs.append((env, 'window-close-v2'))

    for i, (env, name) in enumerate(training_envs):
        # Select task
        # Note: These envs are not actually used, but rather just for info
        expl_env = NormalizedBoxEnv(env)
        eval_env = NormalizedBoxEnv(env)
        obs_dim = expl_env.observation_space.low.size
        action_dim = eval_env.action_space.low.size

        # build networks
        M = variant['layer_size']
        qf1 = FederatedTransformer(
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
        qf2 = FederatedTransformer(
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
        target_qf1 = FederatedTransformer(
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
        target_qf2 = FederatedTransformer(
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
        # qf1 = ConcatMlp(
        #     input_size=obs_dim + action_dim,
        #     output_size=1,
        #     hidden_sizes=[M, M, M],
        #     # num_layers=variant['transformer_num_layers'],
        #     # transformer_layer_config=variant['transformer_layer_kwargs'],
        #     # agent_index=i,
        #     # num_agents=variant['num_agents']
        # )
        # qf2 = ConcatMlp(
        #     input_size=obs_dim + action_dim,
        #     output_size=1,
        #     hidden_sizes=[M, M, M],
        #     # num_layers=variant['transformer_num_layers'],
        #     # transformer_layer_config=variant['transformer_layer_kwargs'],
        #     # agent_index=i,
        #     # num_agents=variant['num_agents']
        # )
        # target_qf1 = ConcatMlp(
        #     input_size=obs_dim + action_dim,
        #     output_size=1,
        #     hidden_sizes=[M, M, M],
        #     # num_layers=variant['transformer_num_layers'],
        #     # transformer_layer_config=variant['transformer_layer_kwargs'],
        #     # agent_index=i,
        #     # num_agents=variant['num_agents']
        # )
        # target_qf2 = ConcatMlp(
        #     input_size=obs_dim + action_dim,
        #     output_size=1,
        #     hidden_sizes=[M, M, M],
        #     # num_layers=variant['transformer_num_layers'],
        #     # transformer_layer_config=variant['transformer_layer_kwargs'],
        #     # agent_index=i,
        #     # num_agents=variant['num_agents']
        # )
        policy = TanhGaussianPolicy(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_sizes=[M, M, M],
            hidden_nonlinearity=nn.ReLU,
            output_nonlinearity=None,
            min_std=np.exp(-20.),
            max_std=np.exp(2.)
        )

        eval_path_collector = MetaPathCollector(
            benchmark=mt10,
            policy=policy,
            kind='train',
            task_name=name,
            k=5
        )
        expl_path_collector = MetaPathCollector(
            benchmark=mt10,
            policy=policy,
            kind='train',
            task_name=name,
            k=5
        )
        replay_buffer = EnvReplayBuffer(
            variant['replay_buffer_size'],
            expl_env,
        )
        trainer = SACTrainer(
            env=eval_env,
            policy=policy,
            qf1=qf1,
            qf2=qf2,
            target_qf1=target_qf1,
            target_qf2=target_qf2,
            **variant['trainer_kwargs']
        )

        # TODO - Implement Meta Algorithm
        # Basically, we want one of these for each env, and we want to
        # be able to manually control each iteration, rather then a free flowing

        algorithm_instance = TorchBatchRLAlgorithm(
            trainer=trainer,
            exploration_env=expl_env,
            evaluation_env=eval_env,
            exploration_data_collector=expl_path_collector,
            evaluation_data_collector=eval_path_collector,
            replay_buffer=replay_buffer,
            name=f'fedAttn 1 agents onboarded - agent {name} ({i})',
            **variant['algorithm_kwargs']
        )
        algorithm_instance.to(ptu.device)
        algorithms += [algorithm_instance]
  

    algorithm = MetaFedSACAlgorithm(algorithms, variant['algorithm_kwargs']['num_epochs'])
    algorithm.train()


if __name__ == "__main__":
    # noinspection PyTypeChecker
    variant = dict(
        algorithm="FedSAC",
        version="normal",
        from_saved=5,
        layer_size=400,
        replay_buffer_size=int(1E6),
        transformer_num_layers=2,
        num_agents=1,
        transformer_layer_kwargs=dict(
            d_model=400,
            nhead=4
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
