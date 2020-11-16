from tianshou.policy import SACPolicy

import torch
import numpy as np
from copy import deepcopy
from torch.distributions import Independent, Normal
from typing import Any, Dict, Tuple, Union, Optional, Mapping, List

from tianshou.policy import SACPolicy
from tianshou.exploration import BaseNoise
from tianshou.data import Batch, ReplayBuffer, to_torch_as


class MOPOPolicy(SACPolicy):
    """Implementation of MOPO: Model-based Offline Policy Optimization. arXiv:2005.13239.
    :param torch.nn.Module actor: the actor network following the rules in
        :class:`~tianshou.policy.BasePolicy`. (s -> logits)
    :param ReplayBuffer offline_buffer: Buffer for offline dataset
    :para float offline_ratio: fraction of training data from offline dataset
    :param torch.optim.Optimizer actor_optim: the optimizer for actor network.
    :param torch.nn.Module critic1: the first critic network. (s, a -> Q(s,
        a))
    :param torch.optim.Optimizer critic1_optim: the optimizer for the first
        critic network.
    :param torch.nn.Module critic2: the second critic network. (s, a -> Q(s,
        a))
    :param torch.optim.Optimizer critic2_optim: the optimizer for the second
        critic network.
    :param action_range: the action range (minimum, maximum).
    :type action_range: Tuple[float, float]
    :param float tau: param for soft update of the target network, defaults to
        0.005.
    :param float gamma: discount factor, in [0, 1], defaults to 0.99.
    :param (float, torch.Tensor, torch.optim.Optimizer) or float alpha: entropy
        regularization coefficient, default to 0.2.
        If a tuple (target_entropy, log_alpha, alpha_optim) is provided, then
        alpha is automatatically tuned.
    :param bool reward_normalization: normalize the reward to Normal(0, 1),
        defaults to False.
    :param bool ignore_done: ignore the done flag while training the policy,
        defaults to False.
    :param BaseNoise exploration_noise: add a noise to action for exploration,
        defaults to None. This is useful when solving hard-exploration problem.
    :param bool deterministic_eval: whether to use deterministic action (mean
        of Gaussian policy) instead of stochastic action sampled by the policy,
        defaults to True.
    .. seealso::
        Please refer to :class:`~tianshou.policy.BasePolicy` for more detailed
        explanation.
    """

    def __init__(
        self,
        offline_buffer: ReplayBuffer,
        actor: torch.nn.Module,
        actor_optim: torch.optim.Optimizer,
        critic1: torch.nn.Module,
        critic1_optim: torch.optim.Optimizer,
        critic2: torch.nn.Module,
        critic2_optim: torch.optim.Optimizer,
        action_range: Tuple[float, float],
        offline_ratio: float=0.1,
        tau: float = 0.005,
        gamma: float = 0.99,
        alpha: Union[
            float, Tuple[float, torch.Tensor, torch.optim.Optimizer]
        ] = 0.2,
        reward_normalization: bool = False,
        ignore_done: bool = False,
        estimation_step: int = 1,
        exploration_noise: Optional[BaseNoise] = None,
        deterministic_eval: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(actor=actor, actor_optim=actor_optim, critic1=critic1, critic1_optim=critic1_optim, critic2=critic2, critic2_optim=critic2_optim, action_range=action_range, tau=tau, gamma=gamma,
                         alpha=alpha, reward_normalization=reward_normalization, ignore_done=ignore_done, estimation_step=estimation_step, exploration_noise=exploration_noise,  **kwargs)
        
        self.offline_buffer=offline_buffer
        self.offline_ratio=offline_ratio

    def update(self, sample_size: int, buffer: Optional[ReplayBuffer], **kwargs: Any) -> Mapping[str, Union[float, List[float]]]:
        """Update the  policy network and replay buffer.
        It includes 3 function steps: process_fn, learn, and post_process_fn.
        In addition, this function will change the value of ``self.updating``:
        it will be False before this function and will be True when executing
        :meth:`update`. Please refer to :ref:`policy_state` for more detailed
        explanation.
        :param int sample_size: 0 means it will extract all the data from the
            buffer, otherwise it will sample a batch with given sample_size.
        :param ReplayBuffer buffer: the corresponding replay buffer.
        """
        if buffer is None:
            return {}

        # Aviod sample size=0 which means to sample all the data in buffer
        if self.offline_ratio>0:
            # Process_fn: Calculate n-step return and weight for prior-buffers
            offline_sample_size=int(np.max([1,np.floor(sample_size*self.offline_ratio)]))
            env_sample_size=int(np.max([1,sample_size-offline_sample_size]))

            batch, indice = buffer.sample(env_sample_size)
            batch = self.process_fn(batch, buffer, indice)

            offline_batch, offline_indice=self.offline_buffer.sample(offline_sample_size)
            offline_batch = self.process_fn(offline_batch, self.offline_buffer, offline_indice)

            train_batch=Batch.cat([batch, offline_batch])

            self.updating = True
            result = self.learn(train_batch, **kwargs)

            # Post_process_fn: Update weight in the buffers
            self.post_process_fn(batch, buffer, indice) 
            self.post_process_fn(offline_batch, self.offline_buffer, offline_indice) 
            self.updating = False
        else:
            batch, indice = buffer.sample(sample_size)
            batch = self.process_fn(batch, buffer, indice)

            train_batch=batch

            self.updating = True
            result = self.learn(train_batch, **kwargs)

            self.post_process_fn(batch, buffer, indice) 
            self.updating = False
        return result
