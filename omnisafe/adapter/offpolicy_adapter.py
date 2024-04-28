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
"""OffPolicy Adapter for OmniSafe."""

from __future__ import annotations

from typing import Any

import torch
from gymnasium.spaces import Box

from omnisafe.adapter.online_adapter import OnlineAdapter
from omnisafe.common.buffer import OffPolicyBuffer
from omnisafe.common.logger import Logger
from omnisafe.models.actor_critic.constraint_actor_q_critic import ConstraintActorQCritic
from omnisafe.utils.config import Config


class OffPolicyAdapter(OnlineAdapter):
    """OffPolicy Adapter for OmniSafe.

    :class:`OffPolicyAdapter` is used to adapt the environment to the off-policy training.

    .. note::
        Off-policy training need to update the policy before finish the episode,
        so the :class:`OffPolicyAdapter` will store the current observation in ``_current_obs``.
        After update the policy, the agent will *remember* the current observation and
        use it to interact with the environment.

    Args:
        env_id (str): The environment id.
        num_envs (int): The number of environments.
        seed (int): The random seed.
        cfgs (Config): The configuration.
    """

    reward_space: Box | None
    _current_obs: torch.Tensor
    _ep_ret: torch.Tensor
    _ep_len: torch.Tensor

    def __init__(  # pylint: disable=too-many-arguments
        self,
        env_id: str,
        num_envs: int,
        seed: int,
        cfgs: Config,
    ) -> None:
        """Initialize a instance of :class:`OffPolicyAdapter`."""
        super().__init__(env_id, num_envs, seed, cfgs)
        self._current_obs, _ = self.reset()
        self._max_ep_len: int = 20
        self._reset_log()
        self.reward_space = self._env.reward_space

    def eval_policy(  # pylint: disable=too-many-locals
        self,
        episode: int,
        agent: ConstraintActorQCritic,
        logger: Logger,
    ) -> None:
        """Rollout the environment with deterministic agent action.

        Args:
            episode (int): Number of episodes.
            agent (ConstraintActorCritic): Agent.
            logger (Logger): Logger, to log ``EpRet``, ``EpLen``.
        """
        for _ in range(episode):
            ep_ret, ep_gap, ep_len = 0.0, 0.0, 0
            obs, _ = self._eval_env.reset()
            obs = obs.to(self._device)

            done = False
            while not done:
                act = agent.step(obs, deterministic=True)
                obs, reward, terminated, truncated, info = self._eval_env.step(act)
                obs, reward, terminated, truncated = (
                    torch.as_tensor(x, dtype=torch.float32, device=self._device)
                    for x in (obs, reward, terminated, truncated)
                )
                ep_ret += info.get('original_reward', torch.sum(reward)).cpu()
                ep_gap += (torch.max(reward, -1)[0]-torch.min(reward, -1)[0]).cpu()
                ep_len += 1
                done = bool(terminated[0].item()) or bool(truncated[0].item())

            logger.store(
                {
                    'Metrics/TestEpRet': ep_ret,
                    'Metrics/TestEpLen': ep_len,
                    'Metrics/TestEpGap': ep_gap,
                },
            )

    def rollout(  # pylint: disable=too-many-locals
        self,
        rollout_step: int,
        agent: ConstraintActorQCritic,
        buffer: OffPolicyBuffer,
        logger: Logger,
        use_rand_action: bool,
    ) -> None:
        """Rollout the environment and store the data in the buffer.

        .. warning::
            As OmniSafe uses :class:`AutoReset` wrapper, the environment will be reset automatically,
            so the final observation will be stored in ``info['final_observation']``.

        Args:
            rollout_step (int): Number of rollout steps.
            agent (ConstraintActorCritic): Constraint actor-critic, including actor, reward critic.
            buffer (VectorOnPolicyBuffer): Vector on-policy buffer.
            logger (Logger): Logger, to log ``EpRet``, ``EpLen``.
            use_rand_action (bool): Whether to use random action.
        """
        for _ in range(rollout_step):
            if use_rand_action:
                act = (torch.rand(self.action_space.shape) * 2 - 1).unsqueeze(0).to(self._device)  # type: ignore
            else:
                act = agent.step(self._current_obs, deterministic=False)
            next_obs, reward, terminated, truncated, info = self.step(act)

            self._log_value(reward=reward, info=info)
            real_next_obs = next_obs.clone()
            done = terminated or truncated
            if done:
                if 'final_observation' in info:
                    real_next_obs = info['final_observation']
                self._log_metrics(logger)
                self._reset_log()

            buffer.store(
                obs=self._current_obs,
                act=act,
                reward=reward,
                next_obs=real_next_obs,
                done=done,
            )

            self._current_obs = next_obs

    def _log_value(
        self,
        reward: torch.Tensor,
        info: dict[str, Any],
    ) -> None:
        """Log value.

        .. note::
            OmniSafe uses :class:`RewardNormalizer` wrapper, so the original reward will
            be stored in ``info['original_reward']``.

        Args:
            reward (torch.Tensor): The immediate step reward.
            info (dict[str, Any]): Some information logged by the environment.
        """
        self._ep_ret += info.get('original_reward', torch.sum(reward)).cpu()
        self._ep_gap += (torch.max(reward, -1)[0]-torch.min(reward, -1)[0]).cpu()
        self._ep_len += 1

    def _log_metrics(self, logger: Logger) -> None:
        """Log metrics, including ``EpRet``, ``EpLen``.

        Args:
            logger (Logger): Logger, to log ``EpRet``, ``EpLen``.
            idx (int): The index of the environment.
        """
        if hasattr(self._env, 'spec_log'):
            self._env.spec_log(logger)
        logger.store(
            {
                'Metrics/EpRet': self._ep_ret,
                'Metrics/EpGap': self._ep_gap,
                'Metrics/EpLen': self._ep_len,
            },
        )

    def _reset_log(self, idx: int | None = None) -> None:
        """Reset the episode return and episode length.

        Args:
            idx (int or None, optional): The index of the environment. Defaults to None
                (single environment).
        """
        self._ep_ret = torch.zeros(1)
        self._ep_gap = torch.zeros(1)
        self._ep_len = torch.zeros(1)
