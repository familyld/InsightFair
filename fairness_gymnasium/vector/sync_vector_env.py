# Copyright 2022-2023 OmniSafe Team. All Rights Reserved.
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
"""The sync vectorized environment."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Callable, Iterator

import numpy as np
from numpy.typing import NDArray

import gymnasium
from gymnasium import Env
from gymnasium.spaces import Space
from gymnasium.core import ActType, ObsType
from gymnasium.vector.sync_vector_env import SyncVectorEnv
from gymnasium.vector.utils import batch_space, concatenate, create_empty_array, iterate


__all__ = ['FairnessSyncVectorEnv']


class FairnessSyncVectorEnv(SyncVectorEnv):
    """Vectored fair environment that serially runs multiple fair environments."""

    def __init__(
        self,
        env_fns: Iterator[Callable[[], Env]],
        observation_space: Space | None = None,
        action_space: Space | None = None,
        copy: bool = True,
    ) -> None:
        """Initializes the vectorized fair environment."""
        super().__init__(env_fns, observation_space, action_space, copy)
        # Initialises the single spaces from the sub-environments
        self.single_reward_space = self.envs[0].reward_space
        # Initialise the obs and action space based on the single versions and num of sub-environments
        self.reward_space = batch_space(
            self.single_reward_space, self.num_envs
        )
        # Initialise attributes used in `step` and `reset`
        self._rewards = create_empty_array(
            self.single_reward_space, n=self.num_envs, fn=np.zeros
        )

    @property
    def single_reward_space(self) -> gymnasium.Space:
        """Gets the single reward space of the vector environment."""
        if self._single_reward_space is None:
            return self.env.single_reward_space 
        return self._single_reward_space
    
    @single_reward_space.setter
    def single_reward_space(self, space: gymnasium.Space) -> None:
        """Sets the single reward space of the vector environment."""
        self._single_reward_space = space

    def render(self) -> np.ndarray:
        """Render the environment."""
        raise NotImplementedError

    def step(
        self, actions: ActType
    ) -> tuple[ObsType, NDArray[Any], NDArray[Any], NDArray[Any], dict[str, Any]]:
        """Steps through each of the environments returning the batched results.

        Returns:
            The batched environment step results
        """
        actions = iterate(self.action_space, actions)

        observations, rewards, infos = [], [], {}
        for i, action in enumerate(actions):
            if self._autoreset_envs[i]:
                env_obs, env_info = self.envs[i].reset()

                self._rewards[i] = create_empty_array(
                    self.single_reward_space, n=1, fn=np.zeros
                ).squeeze()
                self._terminations[i] = False
                self._truncations[i] = False
            else:
                (
                    env_obs,
                    self._rewards[i],
                    self._terminations[i],
                    self._truncations[i],
                    env_info,
                ) = self.envs[i].step(action)

            observations.append(env_obs)
            rewards.append(self._rewards[i])
            infos = self._add_info(infos, env_info, i)

        # Concatenate the observations
        self._observations = concatenate(
            self.single_observation_space, observations, self._observations
        )
        self._rewards = concatenate(
            self.single_reward_space, rewards, self._rewards
        )
        self._autoreset_envs = np.logical_or(self._terminations, self._truncations)

        return (
            deepcopy(self._observations) if self.copy else self._observations,
            deepcopy(self._rewards) if self.copy else self._rewards,
            np.copy(self._terminations),
            np.copy(self._truncations),
            infos,
        )

    def step_wait(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[dict]]:
        """Steps through each of the environments returning the batched results."""
        observations, rewards, infos = [], [], {}
        for i, (env, action) in enumerate(zip(self.envs, self._actions)):
            (
                observation,
                rewards,
                self._terminateds[i],
                self._truncateds[i],
                info,
            ) = env.step(action)

            if self._terminateds[i] or self._truncateds[i]:
                old_observation, old_info = observation, info
                observation, info = env.reset()
                info['final_observation'] = old_observation
                info['final_info'] = old_info
            observations.append(observation)
            infos = self._add_info(infos, info, i)
        self.observations = concatenate(
            self.single_observation_space,
            observations,
            self.observations,
        )
        self.rewards = concatenate(
            self.single_reward_space,
            rewards,
            self.rewards,
        )

        return (
            deepcopy(self.observations) if self.copy else self.observations,
            deepcopy(self.rewards) if self.copy else self.rewards,
            np.copy(self._terminateds),
            np.copy(self._truncateds),
            infos,
        )
