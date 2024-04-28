# Copyright 2023 OmniSafe Team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Model Predictive Control Planner of Cross-Entropy Method optimization algorithm."""


from __future__ import annotations

from typing import Any

import numpy as np

import torch

from omnisafe.algorithms.model_based.base.ensemble import EnsembleDynamicsModel
from omnisafe.utils.config import Config


class CEMPlanner:  # pylint: disable=too-many-instance-attributes
    """The planner of  Cross-Entropy Method optimization (CEM) algorithm.

    References:
        - Title: Sample-efficient Cross-Entropy Method for Real-time Planning
        - Authors: Cristina Pinneri, Shambhuraj Sawant, Sebastian Blaes, Jan Achterhold,
            Joerg Stueckler, Michal Rolinek, Georg Martius
        - URL: `CEM <https://arxiv.org/pdf/2008.06389.pdf>`_
    """

    def __init__(  # pylint: disable=too-many-locals, too-many-arguments
        self,
        dynamics: EnsembleDynamicsModel,
        planner_cfgs: Config,
        gamma: float,
        dynamics_state_shape: tuple[int, ...],
        action_shape: tuple[int, ...],
        action_max: float,
        action_min: float,
        device: torch.device,
        time_limit: int = 1000,
        **kwargs: Any,
    ) -> None:
        """Initializes the planner of Cross-Entropy Method optimization (CEM) algorithm."""
        assert (
            planner_cfgs.num_samples * planner_cfgs.num_particles
        ) % dynamics.num_models == 0, 'num_samples * num_elites should be divisible by num_models'
        assert (
            planner_cfgs.num_samples > planner_cfgs.num_elites
        ), 'num_samples should be greater than num_elites'
        self._dynamics = dynamics
        self._num_models = dynamics.num_models
        self._horizon = planner_cfgs.plan_horizon
        self._num_iterations = planner_cfgs.num_iterations
        self._num_particles = planner_cfgs.num_particles
        self._num_samples = planner_cfgs.num_samples
        self._num_elites = planner_cfgs.num_elites
        self._momentum = planner_cfgs.momentum
        self._epsilon = planner_cfgs.epsilon
        self._dynamics_state_shape = dynamics_state_shape
        self.group_size = self._dynamics._group_size
        self._action_shape = action_shape
        self._action_max = action_max
        self._action_min = action_min
        self._gamma = gamma
        self._device = device
        self._time_limit = time_limit
        self._action_sequence_mean = torch.zeros(
            self._horizon,
            *self._action_shape,
            device=self._device,
        )
        self._init_var = planner_cfgs.init_var
        self._action_sequence_var = self._init_var * torch.ones(
            self._horizon,
            *self._action_shape,
            device=self._device,
        )
        self.kwargs = kwargs
        self.mode = planner_cfgs.mode

    @torch.no_grad()
    def _act_from_last_gaus(self, last_mean: torch.Tensor, last_var: torch.Tensor) -> torch.Tensor:
        """Sample actions from the last gaussian distribution.

        Args:
            last_mean (torch.Tensor): Last mean of the gaussian distribution.
            last_var (torch.Tensor): Last variance of the gaussian distribution.

        Returns:
            sampled actions: Sampled actions from the last gaussian distribution.
        """
        constrained_std = torch.sqrt(last_var)
        actions = torch.clamp(
            last_mean.unsqueeze(1)
            + constrained_std.unsqueeze(1)
            * torch.randn(
                self._horizon,
                self._num_samples,
                # *self._action_shape,
                * last_mean.shape[1:],
                device=self._device,
            ),
            self._action_min,
            self._action_max,
        )
        actions.clamp_(min=self._action_min, max=self._action_max)  # clip action range
        return actions

    @torch.no_grad()
    def _state_action_repeat(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Repeat the state for num_repeat * action.shape[0] times and action for num_repeat times.

        Args:
            state (torch.Tensor): The current state.
            action (torch.Tensor): The sampled actions.

        Returns:
            states: The repeated states.
            actions: The repeated actions.
        """
        assert (
            self._num_particles % self._num_models == 0
        ), 'num_particles should be divisible by num_models'
        assert action.shape == torch.Size(
            [self._horizon, self._num_samples, *self._action_shape],
        ), 'Input action dimension should be equal to (self._num_samples, self._action_shape)'
        assert state.shape == torch.Size(
            [1, *self._dynamics_state_shape],
        ), 'state dimension one should be 1'
        states = state.repeat(int(self._num_particles / self._num_models * self._num_samples), 1)
        actions = action.unsqueeze(1).repeat(1, int(self._num_particles / self._num_models), 1, 1)
        actions = actions.reshape(
            self._horizon,
            int(self._num_particles / self._num_models * self._num_samples),
            *self._action_shape,
        )
        return states, actions

    @torch.no_grad()
    def _select_elites(
        self,
        actions: torch.Tensor,
        traj: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
        """Select elites from the sampled actions.

        Args:
            actions (torch.Tensor): Sampled actions.
            traj (dict[str, torch.Tensor]): Trajectory dictionary.

        Returns:
            elites_value: The value of the elites.
            elites_action: The action of the elites.
            info: The dictionary containing the information of elites value and action.
        """
        rewards = traj['rewards']
        group_size = self.group_size
        assert actions.shape == torch.Size(
            [self._horizon, self._num_samples, *self._action_shape],
        ), 'Input action dimension should be equal to (self._horizon, self._num_samples, self._action_shape)'
        assert rewards.shape == torch.Size(
            [
                self._horizon,
                self._num_models,
                int(self._num_particles / self._num_models * self._num_samples * group_size),
                1,
            ],
            # pylint: disable-next=line-too-long
        ), 'Input rewards dimension should be equal to (self._horizon, self._num_models, self._num_particles / self._num_models * self._num_samples * group_size, 1)'

        rewards = rewards.reshape(self._horizon, self._num_models, group_size, int(self._num_particles / self._num_models * self._num_samples), 1)
        rewards = torch.transpose(rewards, 2, 3)
        # [horizon, num_particles, num_samples, group_size, 1]
        returns = rewards.reshape(self._horizon, -1, self._num_samples, group_size, 1)
        # [horizon, num_particles, group_size, num_samples, 1]
        returns = torch.transpose(returns, 2, 3)
        
        sum_horizon_returns = torch.sum(returns, dim=0)
        mean_particles_returns = sum_horizon_returns.mean(dim=0)
        sum_group_returns = torch.sum(mean_particles_returns, dim=0)
        diff_group_returns = torch.abs(mean_particles_returns[0]-mean_particles_returns[1])
        if self.mode == 1: # InsightFair
            weight = 1.0
            mean_episode_returns = ((sum_group_returns - weight * diff_group_returns) * (self._time_limit / self._horizon))
        elif self.mode == 3: # Fair-S
            weight = 20.0
            obs = traj['states']
            obs = obs.reshape(self._horizon, self._num_models, group_size, int(self._num_particles / self._num_models * self._num_samples), self._dynamics_state_shape[0]//group_size)
            obs = torch.transpose(obs, 2, 3)
            obs = obs.reshape(self._horizon, -1, self._num_samples, group_size, self._dynamics_state_shape[0]//group_size)
            obs = torch.transpose(obs, 2, 3)
            mean_particles_obs = obs[-1].mean(dim=0)
            diff_group_obs = mean_particles_obs.mean(-1, keepdims=True).max(0)[0] - mean_particles_obs.mean(-1, keepdims=True).min(0)[0]
            mean_episode_returns = ((sum_group_returns - weight * diff_group_obs) * (self._time_limit / self._horizon))
        else:
            mean_episode_returns = (sum_group_returns * (self._time_limit / self._horizon))
        assert mean_episode_returns.shape == torch.Size(
            [self._num_samples, 1],
        ), 'Input returns dimension should be equal to (self._num_samples, 1)'
        elite_idxs = torch.topk(mean_episode_returns.squeeze(1), self._num_elites, dim=0).indices
        elite_values, elite_actions = mean_episode_returns[elite_idxs], actions[:, elite_idxs]

        info = {
            'Plan/episode_returns_max': mean_episode_returns.max().item(),
            'Plan/episode_returns_mean': mean_episode_returns.mean().item(),
            'Plan/episode_returns_min': mean_episode_returns.min().item(),
            'Plan/episode_actions_max': elite_actions.max().item(),
            'Plan/episode_actions_mean': elite_actions.mean().item(),
        }

        return elite_values, elite_actions, info

    @torch.no_grad()
    def _update_mean_var(  # pylint: disable=unused-argument
        self,
        elite_actions: torch.Tensor,
        elite_values: torch.Tensor,
        info: dict[str, float],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Update the mean and variance of the elite actions.

        Args:
            elite_actions (torch.Tensor): The elite actions.
            elite_values (torch.Tensor): The elite values.
            info (dict[str, float]): The dictionary containing the information of the elite values and actions.

        Returns:
            new_mean: The new mean of the elite actions.
            new_var: The new variance of the elite actions.
        """
        assert elite_actions.shape == torch.Size(
            [self._horizon, self._num_elites, *self._action_shape],
        ), 'Input elite_actions dimension should be equal to (self._horizon, self._num_elites, self._action_shape)'
        assert elite_values.shape == torch.Size(
            [self._num_elites, 1],
        ), 'Input elite_values dimension should be equal to (self._num_elites, 1)'

        new_mean = elite_actions.mean(dim=1)
        new_var = elite_actions.var(dim=1)

        return new_mean, new_var

    @torch.no_grad()
    def output_action(
        self, 
        state: torch.Tensor,
        NDE: float,
    ) -> tuple[torch.Tensor, dict]:
        """Output the action given the state.

        Args:
            state (torch.Tensor): State of the environment.

        Returns:
            action: The action of the agent.
            info: The dictionary containing the information of the action.
        """
        assert state.shape == torch.Size(
            [1, *self._dynamics_state_shape],
        ), 'Input state dimension should be equal to (1, self._dynamics_state_shape)'
        group_size = self.group_size
        group_state_shape = self._dynamics_state_shape[0]//group_size
        state_disparity = torch.abs((state[0][:group_state_shape]-state[0][group_state_shape:])).sum()
        use_same_act_seq = np.isclose(NDE, 0, atol=1e-1) and state_disparity <= 1e-1

        # [horizon, action_shape]
        last_mean = torch.zeros_like(self._action_sequence_mean)
        last_var = self._action_sequence_var.clone()
        last_mean[:-1] = self._action_sequence_mean[1:].clone()
        last_mean[-1] = self._action_sequence_mean[-1].clone()

        current_iter = 0
        info: dict[str, float | int] = {}
        while current_iter < self._num_iterations:
            # [horizon, num_sample, action_shape]
            if (self.mode == 1 and use_same_act_seq) or self. mode == 2: # use the same action seqs
                individual_action_shape = self._action_shape[0]//group_size
                actions = self._act_from_last_gaus(last_mean=last_mean[:,:individual_action_shape], last_var=last_var[:,:individual_action_shape])
                actions = torch.cat([actions] * group_size, dim=-1)
            else:    # use different action seqs
                actions = self._act_from_last_gaus(last_mean=last_mean, last_var=last_var)
            # pylint: disable-next=line-too-long
            # [num_particles * num_samples / num_ensemble, state_shape], [horizon, num_particles * num_samples / num_ensemble, action_shape]
            states_repeat, actions_repeat = self._state_action_repeat(state, actions)
            # pylint: disable-next=line-too-long
            traj = self._dynamics.imagine(states_repeat, self._horizon, actions_repeat)

            elite_values, elite_actions, info = self._select_elites(actions, traj)
            # [num_sample, 1]
            new_mean, new_var = self._update_mean_var(elite_actions, elite_values, info)
            last_mean = self._momentum * last_mean + (1 - self._momentum) * new_mean
            last_var = self._momentum * last_var + (1 - self._momentum) * new_var
            if (self.mode == 1 and use_same_act_seq) or self. mode == 2:
                last_mean = torch.cat([last_mean[:,:individual_action_shape]] * group_size, dim=-1)
                last_var = torch.cat([last_var[:,:individual_action_shape]] * group_size, dim=-1)
            current_iter += 1
        logger_info = {
            'Plan/iter': current_iter,
            'Plan/last_var_mean': last_var.mean().item(),
            'Plan/last_var_max': last_var.max().item(),
            'Plan/last_var_min': last_var.min().item(),
        }
        logger_info.update(info)
        self._action_sequence_mean = last_mean.clone()
        return last_mean[0].clone().unsqueeze(0), logger_info
