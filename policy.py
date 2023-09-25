from rlkit.torch.networks import Mlp
from rlkit.torch.sac.policies.base import TorchStochasticPolicy
import torch.nn as nn
import numpy as np
import rlkit.torch.pytorch_util as ptu
from rlkit.torch.distributions import TanhNormal as RLTanhNormal
from garage.torch.distributions import TanhNormal 
from garage.torch.modules import GaussianMLPTwoHeadedModule

class TanhGaussianPolicy(Mlp, TorchStochasticPolicy):
    """
    Usage:

    ```
    policy = TanhGaussianPolicy(...)
    """

    def __init__(
            self,
            hidden_sizes,
            obs_dim,
            action_dim,
            init_w=1e-3,
            hidden_nonlinearity=nn.ReLU,
            hidden_w_init=nn.init.xavier_uniform_,
            hidden_b_init=nn.init.zeros_,
            output_nonlinearity=None,
            output_w_init=nn.init.xavier_uniform_,
            output_b_init=nn.init.zeros_,
            init_std=1.0,
            min_std=np.exp(-20.),
            max_std=np.exp(2.),
            std_parameterization='exp',
            layer_normalization=False,
            **kwargs
    ):
        super().__init__(
            hidden_sizes,
            input_size=obs_dim,
            output_size=action_dim,
            init_w=init_w,
            **kwargs
        )

        self._module = GaussianMLPTwoHeadedModule(
            input_dim=obs_dim,
            output_dim=action_dim,
            hidden_sizes=hidden_sizes,
            hidden_nonlinearity=hidden_nonlinearity,
            hidden_w_init=hidden_w_init,
            hidden_b_init=hidden_b_init,
            output_nonlinearity=output_nonlinearity,
            output_w_init=output_w_init,
            output_b_init=output_b_init,
            init_std=init_std,
            min_std=min_std,
            max_std=max_std,
            std_parameterization=std_parameterization,
            layer_normalization=layer_normalization,
            normal_distribution_cls=TanhNormal)


    def forward(self, obs):
        
        dist = self._module(obs)

        return dist

    def logprob(self, action, mean, std):
        tanh_normal = TanhNormal(mean, std)
        log_prob = tanh_normal.log_prob(
            action,
        )
        log_prob = log_prob.sum(dim=1, keepdim=True)
        return log_prob
