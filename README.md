# FedFormer: Contextual Federation with Attention in Reinforcement Learning
This repository contains for the code for the NeurIPS 22' pre-print FedFormer: Contextual Federation with Attention in Reinforcement Learning. 

**Abstract:**
A core issue in federated reinforcement learning is defining how to aggregate insights from multiple agents into one. This is commonly done by taking the average of each participating agent's model weights into one common model (FedAvg). We instead propose FedFormer, a novel federation strategy that utilizes Transformer Attention to contextually aggregate embeddings from models originating from different learner agents. In so doing, we attentively weigh contributions of other agents with respect to the current agent's environment and learned relationships, thus providing more effective and efficient federation. We evaluate our methods on the Meta-World environment and find that our approach yields significant improvements over FedAvg and non-federated Soft Actor Critique single agent methods. Our results compared to Soft Actor Critique show that FedFormer performs better while still abiding by the privacy constraints of federated learning. In addition, we demonstrate nearly linear improvements in effectiveness with increased agent pools in certain tasks. This is contrasted by FedAvg, which fails to make noticeable improvements when scaled. 

## To install:
We provide conda env files at 'environment.yml' which contains all of our python dependencies. You can create the environment as 
```shell
conda env create --prefix <env-location> -f environment.yml
```

## To run: 
The main entry point to run our code is 'main.py'. Inside that file, you can find a dict containing tunable hyperparameters such as: 
```python
    variant = dict(
        algorithm="FedFormer",
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
```
Once modified, run as 
```
python main.py --task=<task>
```
where <task> is once of the [10 MT10 MetaWorld Tasks](https://meta-world.github.io/figures/ml10.gif), such as `reach-v2` and `window-close-v2`. 
