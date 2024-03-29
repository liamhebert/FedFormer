# FedFormer: Contextual Federation with Attention in Reinforcement Learning
![Architecture](https://github.com/liamhebert/FedFormer/raw/main/FedFormer%20Architecture.png)
This repository contains for the code for the AAMAS 23' paper FedFormer: Contextual Federation with Attention in Reinforcement Learning. [Paper](https://arxiv.org/abs/2205.13697)

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/liamhebert/FedFormer/blob/main/LICENSE)

**Abstract:**
A core issue in federated reinforcement learning is defining how to aggregate insights from multiple agents into one. This is commonly done by taking the average of each participating agent's model weights into one common model (FedAvg). We instead propose FedFormer, a novel federation strategy that utilizes Transformer Attention to contextually aggregate embeddings from models originating from different learner agents. In so doing, we attentively weigh contributions of other agents with respect to the current agent's environment and learned relationships, thus providing more effective and efficient federation. We evaluate our methods on the Meta-World environment and find that our approach yields significant improvements over FedAvg and non-federated Soft Actor Critique single agent methods. Our results compared to Soft Actor Critique show that FedFormer performs better while still abiding by the privacy constraints of federated learning. In addition, we demonstrate nearly linear improvements in effectiveness with increased agent pools in certain tasks. This is contrasted by FedAvg, which fails to make noticeable improvements when scaled. 

![Results on MT10](https://github.com/liamhebert/FedFormer/raw/main/figure2-all-tasks-results.png)

In these graphs, Soft Actor Critique (SAC) results are trained on all environments (representative of not preserving privacy) wheras FedFormer and FedAvg are trained using a set of 5 agents each with distinct subsets of environments (representative of multiple agents preserving privacy). Our results show that we match or exceed the performance of SAC and vastly outperform other federated methods such as FedAvg. 

![Results of FedFormer Scaled](https://github.com/liamhebert/FedFormer/raw/main/figure3-scaling-fedavg-results.png)

When scaling beyond 5 agents, we find that FedFormer is much more rebust then FedAvg and FedFormer, where performance is consistently higher then FedAvg despite increased agent counts. This demonstrates that are method is able to scale well to large populations of agents and, in some-cases, performs stronger with more agents. 

## To install:
We provide conda env files at 'environment.yml' which contains all of our python dependencies. You can create the environment as 
```shell
conda env create --prefix <env-location> -f environment.yml
```
in addition to [MuJoCo 2.10 via mujoco-py](https://github.com/openai/mujoco-py). As part of our source code, we borrow modules from [RLKit](https://github.com/rail-berkeley/rlkit) and [Garage](https://github.com/rlworkgroup/garage). 

## To run: 
The main entry point to run our code is 'main.py'. Inside that file, you can find a dict containing tunable hyperparameters such as: 
```python
variant = dict(
        algorithm="FedFormer",
        task=task,
        overlap=False, # whether enviroments should overlap
        fedFormer=True, # Whether to use FedFormer Q-Functions or not
        run_name="FedFormer - " + str(agents) + " - " + str(seed), # For logging purposes
        from_saved=0, # How many encoder networks to save 
        layer_size=400, # Hidden layer size
        replay_buffer_size=int(1E6), 
        num_jobs_per_gpu=2, # number of agents to train in parallel per gpu
        transformer_num_layers=2, # number of transformer encoder layers to use
        num_agents=agents, # number of federation agents to initialize
        transformer_layer_kwargs=dict(
            d_model=400, # hidden size for each transformer layer
            nhead=4 # number of attention heads to initialize
        ),
        algorithm_kwargs=dict(
            num_epochs=500, # 250
            num_eval_steps_per_epoch=500, # 5000
            num_trains_per_train_loop=500, # 600
            num_expl_steps_per_train_loop=400, # 400
            min_num_steps_before_training=400, # 400
            max_path_length=500, # 500
            batch_size=500, # 1200
            
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
```
Once modified, run as 
```
python main.py --task=<task> --seed=<seed> --agents=<num_agents> 
```
where "task" is once of the [10 MT10 MetaWorld Tasks](https://meta-world.github.io/figures/ml10.gif), such as `reach-v2` and `window-close-v2`, "seed" is a numerical random seed to set environment distribution, and "agents" is the number of agents to federate
    
Once the experiment has finished running, all results can be seen by running 
```
tensorboard --logdir=runs
```
It is important to note that the results reported in the paper are the average performance of all agents, whereas the tensorboard results will report the performance of each individual agent. That is, for Federated Methods with 5 agents, each run will generate 5 reward curves corresponding to each method. 

## Citation
```
@inproceedings{hebert2023fedformer,
  title={FedFormer: Contextual Federation with Attention in Reinforcement Learning},
  author={Hebert, Liam and Golab, Lukasz and Poupart, Pascal and Cohen, Robin},
  booktitle={Proceedings of the 2023 International Conference on Autonomous Agents and Multiagent Systems},
  pages={810--818},
  year={2023}
}
```
