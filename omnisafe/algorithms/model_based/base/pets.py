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
"""Implementation of the Probabilistic Ensembles with Trajectory Sampling algorithm."""


from __future__ import annotations

import os
import time
from typing import Any, Callable

import numpy as np
import torch
from gymnasium.spaces import Box
from gymnasium.utils.save_video import save_video
from matplotlib import pylab

from omnisafe.adapter import ModelBasedAdapter
from omnisafe.algorithms import registry
from omnisafe.algorithms.base_algo import BaseAlgo
from omnisafe.algorithms.model_based.base.ensemble import EnsembleDynamicsModel
from omnisafe.algorithms.model_based.planner.cem import CEMPlanner
from omnisafe.common.buffer import OffPolicyBuffer
from omnisafe.common.logger import Logger
from omnisafe.typing import OmnisafeSpace


@registry.register
# pylint: disable-next=too-many-instance-attributes, too-few-public-methods
class PETS(BaseAlgo):
    """The Probabilistic Ensembles with Trajectory Sampling (PETS) algorithm.

    References:
        - Title: Deep Reinforcement Learning in a Handful of Trials using Probabilistic Dynamics Models
        - Authors: Kurtland Chua, Roberto Calandra, Rowan McAllister, Sergey Levine.
        - URL: `PETS <https://arxiv.org/abs/1805.12114>`_
    """

    def _init_env(self) -> None:
        self._env: ModelBasedAdapter = ModelBasedAdapter(
            self._env_id,
            1,
            self._seed,
            self._cfgs,
            env_cfgs = self._cfgs.env_cfgs,
        )
        self._total_steps: int = int(self._cfgs.train_cfgs.total_steps)
        self._steps_per_epoch: int = int(self._cfgs.algo_cfgs.steps_per_epoch)
        self._epochs: int = self._total_steps // self._cfgs.algo_cfgs.steps_per_epoch
        self._n_groups: int = int(self._env.reward_space.shape[0])
        self.NDE = 1.0

    def _init_model(self) -> None:
        """Initialize dynamics model and planner."""
        self._dynamics_state_space: OmnisafeSpace = self._env.observation_space

        assert self._dynamics_state_space is not None and isinstance(
            self._dynamics_state_space.shape,
            tuple,
        )
        assert self._env.action_space is not None and isinstance(
            self._env.action_space.shape,
            tuple,
        )
        if isinstance(self._env.action_space, Box):
            self._action_space = self._env.action_space
        else:
            raise NotImplementedError
        self._dynamics: EnsembleDynamicsModel = EnsembleDynamicsModel(
            model_cfgs=self._cfgs.dynamics_cfgs,
            device=self._device,
            state_shape=self._dynamics_state_space.shape,
            action_shape=self._action_space.shape,
            group_size = self._n_groups,
            actor_critic=None,
            rew_func=None,
            terminal_func=None,
        )

        self._planner: CEMPlanner = CEMPlanner(
            dynamics=self._dynamics,
            planner_cfgs=self._cfgs.planner_cfgs,
            gamma=float(self._cfgs.algo_cfgs.gamma),
            dynamics_state_shape=self._dynamics_state_space.shape,
            action_shape=self._action_space.shape,
            action_max=1.0,
            action_min=-1.0,
            device=self._device,
            time_limit=self._cfgs.algo_cfgs.time_limit,
        )
        self._use_actor_critic: bool = False
        self._update_dynamics_cycle: int = int(self._cfgs.algo_cfgs.update_dynamics_cycle)

    def _init(self) -> None:
        """Initialize the algorithm."""
        self._dynamics_buf: OffPolicyBuffer = OffPolicyBuffer(
            obs_space=self._dynamics_state_space,
            act_space=self._env.action_space,
            rew_space=self._env.reward_space,
            size=self._cfgs.train_cfgs.total_steps,
            batch_size=self._cfgs.dynamics_cfgs.batch_size,
            device=self._device,
        )
        env_kwargs: dict[str, Any] = {}
        self._eval_env: ModelBasedAdapter = ModelBasedAdapter(
            self._env_id,
            1,
            self._seed,
            self._cfgs,
            env_cfgs = self._cfgs['env_cfgs'],
            **env_kwargs,
        )
        self._eval_fn: Callable[[int, bool], None] = self._evaluation_single_step

    def _init_log(self) -> None:
        """Initialize logger.

        +---------------------------+-------------------------------------------------+
        | Things to log             | Description                                     |
        +===========================+=================================================+
        | Train/Epoch               | Current epoch.                                  |
        +---------------------------+-------------------------------------------------+
        | TotalEnvSteps             | Total steps of the experiment.                  |
        +---------------------------+-------------------------------------------------+
        | Metrics/EpRet             | Average return of the epoch.                    |
        +---------------------------+-------------------------------------------------+
        | Metrics/EpLen             | Average length of the epoch.                    |
        +---------------------------+-------------------------------------------------+
        | EvalMetrics/EpRet         | Average episode return in evaluation.           |
        +---------------------------+-------------------------------------------------+
        | EvalMetrics/EpLen         | Average episode length in evaluation.           |
        +---------------------------+-------------------------------------------------+
        | Loss/DynamicsTrainMseLoss | The training loss of dynamics model.            |
        +---------------------------+-------------------------------------------------+
        | Loss/DynamicsValMseLoss   | The validation loss of dynamics model.          |
        +---------------------------+-------------------------------------------------+
        | Plan/iter                 | The number of iterations in the planner.        |
        +---------------------------+-------------------------------------------------+
        | Plan/last_var_mean        | The mean of the last variance in the planner.   |
        +---------------------------+-------------------------------------------------+
        | Plan/last_var_max         | The max of the last variance in the planner.    |
        +---------------------------+-------------------------------------------------+
        | Plan/last_var_min         | The min of the last variance in the planner.    |
        +---------------------------+-------------------------------------------------+
        | Plan/episode_returns_max  | The max of the episode returns in the planner.  |
        +---------------------------+-------------------------------------------------+
        | Plan/episode_returns_mean | The mean of the episode returns in the planner. |
        +---------------------------+-------------------------------------------------+
        | Plan/episode_returns_min  | The min of the episode returns in the planner.  |
        +---------------------------+-------------------------------------------------+
        | Time/Total                | The total time of the algorithm.                |
        +---------------------------+-------------------------------------------------+
        | Time/Rollout              | The time of the rollout.                        |
        +---------------------------+-------------------------------------------------+
        | Time/UpdateActorCritic    | The time of the actor-critic update.            |
        +---------------------------+-------------------------------------------------+
        | Time/Eval                 | The time of the evaluation.                     |
        +---------------------------+-------------------------------------------------+
        | Time/Epoch                | The time of the epoch.                          |
        +---------------------------+-------------------------------------------------+
        | Time/FPS                  | The FPS of the algorithm.                       |
        +---------------------------+-------------------------------------------------+
        | Time/UpdateDynamics       | The time of the dynamics update.                |
        +---------------------------+-------------------------------------------------+

        """
        self._logger = Logger(
            output_dir=self._cfgs.logger_cfgs.log_dir,
            exp_name=self._cfgs.exp_name,
            seed=self._cfgs.seed,
            use_tensorboard=self._cfgs.logger_cfgs.use_tensorboard,
            use_wandb=self._cfgs.logger_cfgs.use_wandb,
            config=self._cfgs,
        )

        self._logger.register_key('Train/Epoch')
        self._logger.register_key('TotalEnvSteps')
        self._logger.register_key('Metrics/EpRet', window_length=50)
        self._logger.register_key('Metrics/EpGap', window_length=50)
        self._logger.register_key('Metrics/EpLen', window_length=50)
        self._logger.register_key('Metrics/EpOccurr', window_length=50)
        self._logger.register_key('Metrics/EpMiss', window_length=50)
        self._logger.register_key('Metrics/EpAlloc', window_length=50)
        self._logger.register_key('Metrics/LastRateMax', window_length=50)
        self._logger.register_key('Metrics/LastRateMin', window_length=50)
        if self._cfgs.evaluation_cfgs.use_eval:
            for i in range(self._n_groups):
                self._logger.register_key(f'EvalMetrics/EpRet-Group-{i}', window_length=5)
            self._logger.register_key('EvalMetrics/EpLen', window_length=5)
        self._logger.register_key('Loss/DynamicsTrainMseLoss')
        self._logger.register_key('Loss/DynamicsValMseLoss')

        self._logger.register_key('Plan/iter')
        self._logger.register_key('Plan/last_var_mean')
        self._logger.register_key('Plan/last_var_max')
        self._logger.register_key('Plan/last_var_min')
        self._logger.register_key('Plan/episode_returns_max')
        self._logger.register_key('Plan/episode_returns_mean')
        self._logger.register_key('Plan/episode_returns_min')
        self._logger.register_key('Plan/episode_actions_max')
        self._logger.register_key('Plan/episode_actions_mean')

        self._logger.register_key('Time/Total')
        self._logger.register_key('Time/Rollout')
        self._logger.register_key('Time/UpdateDynamics')
        if self._use_actor_critic:
            self._logger.register_key('Time/UpdateActorCritic')
        if self._cfgs.evaluation_cfgs.use_eval:
            self._logger.register_key('Time/Eval')
        self._logger.register_key('Time/Epoch')
        self._logger.register_key('Time/FPS')
        self._save_model()

    def _save_model(self) -> None:
        """Save the model."""
        # set up model saving
        what_to_save: dict[str, Any] = {
            'dynamics': self._dynamics.ensemble_model,
        }
        if self._cfgs.algo_cfgs.obs_normalize:
            obs_normalizer = self._env.save()['obs_normalizer']
            what_to_save['obs_normalizer'] = obs_normalizer
        self._logger.setup_torch_saver(what_to_save)
        self._logger.torch_save()

    def learn(self) -> tuple[float, float, float]:
        """This is main function for algorithm update.

        It is divided into the following steps:

        - :meth:`rollout`: collect interactive data from environment.
        - :meth:`update`: perform actor/critic updates.
        - :meth:`log`: epoch/update information for visualization and terminal log print.

        Returns:
            ep_ret: Average episode return in final epoch.
            ep_len: Average episode length in final epoch.
        """
        self._logger.log('INFO: Start training')
        start_time = time.time()
        current_step = 0
        for epoch in range(self._epochs):
            current_step = self._env.rollout(
                current_step=current_step,
                rollout_step=self._steps_per_epoch,
                use_actor_critic=self._use_actor_critic,
                act_func=self._select_action,
                store_data_func=self._store_real_data,
                update_dynamics_func=self._update_dynamics_model,
                use_eval=self._cfgs.evaluation_cfgs.use_eval,
                eval_func=self._eval_fn,
                logger=self._logger,
                algo_reset_func=self._algo_reset,
                update_actor_func=self._update_policy,
            )
            if current_step > self._cfgs.algo_cfgs.start_learning_steps:
                self._update_epoch()
            # evaluate episode
            self._logger.store(
                {
                    'Train/Epoch': epoch,
                    'TotalEnvSteps': current_step,
                    'Time/Total': time.time() - start_time,
                },
            )
            self._logger.dump_tabular()
            # save model to disk
            if (epoch + 1) % self._cfgs.logger_cfgs.save_model_freq == 0:
                self._logger.torch_save()
            # evaluate dynamics fairness
            self.NDE = self._evluate_dynamics_fairnees()

        ep_ret = self._logger.get_stats('Metrics/EpRet')[0]
        ep_len = self._logger.get_stats('Metrics/EpLen')[0]
        self._logger.close()

        return ep_ret, ep_len, self.NDE

    def _algo_reset(
        self,
    ) -> None:
        """Reset the algorithm."""

    def _update_policy(
        self,
        current_step: int,  # pylint: disable=unused-argument
    ) -> None:
        """Update policy."""

    def _evluate_dynamics_fairnees(
            self,
    ) -> float:
        """Evaluate the fairness of dynamics model."""
        state = self._dynamics_buf.data['obs'][: self._dynamics_buf.size, :]
        action = self._dynamics_buf.data['act'][: self._dynamics_buf.size, :]
        reward = self._dynamics_buf.data['reward'][: self._dynamics_buf.size, :]
        next_state = self._dynamics_buf.data['next_obs'][: self._dynamics_buf.size, :self._dynamics.state_size] # Just take the first group

        # Split into groups
        buf_size = self._dynamics_buf.size
        reward = torch.cat(torch.split(reward, 1, dim=1), dim=0)        # [2*buf_size, 1]
        
        traj = self._dynamics.imagine_for_fairness_eval(
            states = state,
            horizon = 1,
            actions = action,
        )
        dis_reward = reward[:buf_size] # [buf_size, 1]
        adv_reward = traj['rewards'][0].mean(0) # [buf_size, 1], only use the first step (index 0) and mean over ensemble models
        print("dis_reward", dis_reward.mean())
        print("adv_reward", adv_reward.mean())
        # dis_state = state[:buf_size, :group_state_shape].cpu().detach().numpy() # only use the first group
        dis_state = next_state.cpu().detach().numpy()
        adv_state = traj['states'][0].mean(0).cpu().detach().numpy()
        if self._env_id == 'Lending-v0':
            weight = np.arange(self._dynamics.state_size)
            dis_state = (dis_state * weight).sum(1).mean()
            adv_state = (adv_state * weight).sum(1).mean()
        else:
            dis_state = dis_state.mean()
            adv_state = adv_state.mean()
        print("dis_state", dis_state)
        print("adv_state", adv_state)
        NDE_R = (adv_reward.mean() - dis_reward.mean()).cpu().detach().numpy().item()
        NDE_S = (adv_state - dis_state).item()
        if self._cfgs.algo_cfgs.use_nde_r and self._cfgs.algo_cfgs.use_nde_s:
            return max(NDE_R, NDE_S)
        elif self._cfgs.algo_cfgs.use_nde_r:
            return NDE_R
        elif self._cfgs.algo_cfgs.use_nde_s:
            return NDE_S
        else:
            return 1.0

    def _update_dynamics_model(
        self,
    ) -> None:
        """Update dynamics model."""
        state = self._dynamics_buf.data['obs'][: self._dynamics_buf.size, :]
        action = self._dynamics_buf.data['act'][: self._dynamics_buf.size, :]
        reward = self._dynamics_buf.data['reward'][: self._dynamics_buf.size, :]
        next_state = self._dynamics_buf.data['next_obs'][: self._dynamics_buf.size, :]
        delta_state = next_state - state
        assert isinstance(delta_state, torch.Tensor), 'delta_state should be torch.Tensor'

        # Split into groups
        obs_size = self._env.observation_space.shape[0]//self._n_groups
        act_size = self._env.action_space.shape[0]//self._n_groups
        buf_size = self._dynamics_buf.size
        state = torch.cat(torch.split(state, obs_size, dim=1), dim=0)
        action = torch.cat(torch.split(action, act_size, dim=1), dim=0)
        reward = torch.cat(torch.split(reward, 1, dim=1), dim=0)
        delta_state = torch.cat(torch.split(delta_state, obs_size, dim=1), dim=0)

        inputs = torch.cat((state, action), -1)
        inputs = torch.reshape(inputs, (inputs.shape[0], -1))

        labels = torch.reshape(delta_state, (delta_state.shape[0], -1))
        if self._cfgs.dynamics_cfgs.predict_reward:
            labels = torch.cat(((torch.reshape(reward, (reward.shape[0], -1))), labels), -1)

        inputs = inputs.cpu().detach().numpy()
        labels = labels.cpu().detach().numpy()
        train_mse_losses, val_mse_losses = self._dynamics.train(
            inputs,
            labels,
            holdout_ratio=0.2,
        )
        self._logger.store(
            **{
                'Loss/DynamicsTrainMseLoss': train_mse_losses.item(),
                'Loss/DynamicsValMseLoss': val_mse_losses.item(),
            },
        )

    def _update_epoch(self) -> None:
        ...

    def _select_action(  # pylint: disable=unused-argument
        self,
        current_step: int,
        state: torch.Tensor,
    ) -> torch.Tensor:
        """Select action.

        Args:
            current_step (int): The current step.
            state (torch.Tensor): The current state.

        Returns:
            The selected action.
        """
        assert state.shape[0] == 1, 'state shape should be [1, state_dim]'
        if current_step < self._cfgs.algo_cfgs.start_learning_steps:
            action = torch.tensor(self._env.action_space.sample()).to(self._device).unsqueeze(0)
        else:
            action, info = self._planner.output_action(state, self.NDE)
            self._logger.store(**info)
        assert action.shape == torch.Size(
            [1, *self._action_space.shape],
        ), 'action shape should be [batch_size, action_dim]'
        return action

    def _store_real_data(  # pylint: disable=too-many-arguments,unused-argument
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        terminated: torch.Tensor,
        truncated: torch.Tensor,
        next_state: torch.Tensor,
        info: dict[str, Any],
    ) -> None:  # pylint: disable=too-many-arguments
        """Store real data in buffer.

        Args:
            state (torch.Tensor): The state from the environment.
            action (torch.Tensor): The action from the agent.
            reward (torch.Tensor): The reward signal from the environment.
            terminated (torch.Tensor): The terminated signal from the environment.
            truncated (torch.Tensor): The truncated signal from the environment.
            next_state (torch.Tensor): The next state from the environment.
            info (dict[str, Any]): The information from the environment.
        """
        done = terminated or truncated
        goal_met = False if 'goal_met' not in info else info['goal_met']
        if not terminated and not truncated and not goal_met:
           self._dynamics_buf.store(
                obs=state,
                act=action,
                reward=reward,
                next_obs=next_state,
                done=done,
            )

    def _evaluation_single_step(  # pylint: disable=too-many-locals
        self,
        current_step: int,
        use_real_input: bool = True,
    ) -> None:
        """Evaluation dynamics model single step.

        Args:
            current_step (int): The current step.
            use_real_input (bool): Whether to use real input or not.
        """
        obs, _ = self._eval_env.reset()
        obs_dynamics = obs
        ep_len, ep_ret = 0, [0]*self._n_groups
        terminated, truncated = torch.tensor([False]), torch.tensor([False])
        obs_pred: list[float] = []
        obs_true: list[float] = []
        reward_pred: list[float] = []
        reward_true: list[float] = []
        num_episode = 0
        while True:
            if terminated or truncated:
                for i in range(self._n_groups):
                    self._logger.store({'EvalMetrics/EpRet-Group-'+str(i): ep_ret[i]})
                self._logger.store({'EvalMetrics/EpLen': ep_len})
                
                obs_pred, obs_true = [], []
                reward_pred, reward_true = [], []
                ep_len, ep_ret = 0, [0]*self._n_groups
                obs, _ = self._eval_env.reset()
                num_episode += 1
                if num_episode == self._cfgs.evaluation_cfgs.num_episode:
                    break

            action = self._select_action(current_step, obs)

            idx = np.random.choice(self._dynamics.elite_model_idxes, size=1)[0]
            traj = self._dynamics.imagine(
                states=obs_dynamics,
                horizon=1,
                idx=idx,
                actions=action.unsqueeze(0),
            )

            s = traj['states'][0][0].flatten(0, 1)
            r = traj['rewards'][0][0].flatten(0, 1)
            pred_next_obs_mean = s.mean()
            pred_reward = r.sum()

            obs, reward, terminated, truncated, info = self._eval_env.step(action)

            obs_dynamics = obs if use_real_input else s

            true_next_obs_mean = obs.mean()

            obs_pred.append(pred_next_obs_mean.item())
            obs_true.append(true_next_obs_mean.item())

            reward_pred.append(pred_reward.item())
            reward_true.append(reward.sum().item())

            for i in range(self._n_groups):
                ep_ret[i] += reward[0][i].cpu().item()
            ep_len += info['num_step']

    def draw_picture(
        self,
        timestep: int,
        num_episode: int,
        pred_state: list[float],
        true_state: list[float],
        save_replay_path: str = './',
        name: str = 'reward',
    ) -> None:
        """Draw a curve of the predicted value and the ground true value.

        Args:
            timestep (int): The current step.
            num_episode (int): The number of episodes.
            pred_state (list[float]): The predicted state.
            true_state (list[float]): The true state.
            save_replay_path (str): The path for saving replay.
            name (str): The name of the curve.
        """
        target1 = list(pred_state)
        target2 = list(true_state)
        input1 = np.arange(0, np.array(pred_state).shape[0], 1)
        input2 = np.arange(0, np.array(pred_state).shape[0], 1)

        pylab.plot(input1, target1, 'r-', label='pred')
        pylab.plot(input2, target2, 'b-', label='true')
        pylab.xlabel('Step')
        pylab.ylabel(name)
        pylab.xticks(np.arange(0, np.array(pred_state).shape[0], 50))  # set the axis numbers
        if name == 'reward':
            pylab.yticks(np.arange(0, 3, 0.2))
        else:
            pylab.yticks(np.arange(0, 1, 0.2))
        pylab.legend(
            loc=3,
            borderaxespad=2.0,
            bbox_to_anchor=(0.7, 0.7),
        )  # set the position of that box for what each line is
        pylab.grid()  # draw grid
        pylab.savefig(
            os.path.join(
                save_replay_path,
                str(name) + str(timestep) + '_' + str(num_episode) + '.png',
            ),
            dpi=200,
        )  # save as picture
        pylab.close()
