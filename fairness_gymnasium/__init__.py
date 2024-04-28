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
"""Fairness-Gymnasium Environments."""

import copy

from gymnasium import make as gymnasium_make
from gymnasium import register as gymnasium_register

from fairness_gymnasium import vector, wrappers
from fairness_gymnasium.utils.registration import make, register


__all__ = [
    'register',
    'make',
    'gymnasium_make',
    'gymnasium_register',
]

# ========================================
# Helper Methods for Easy Registration
# ========================================


def __register_helper(env_id, entry_point, spec_kwargs=None, **kwargs):
    """Register a environment to both Fairness-Gymnasium and Gymnasium registry."""
    env_name, dash, version = env_id.partition('-')
    if spec_kwargs is None:
        spec_kwargs = {}

    register(
        id=env_id,
        entry_point=entry_point,
        kwargs=spec_kwargs,
        **kwargs,
    )

__register_helper(
    env_id='Allocation-v0',
    entry_point='fairness_gymnasium.envs.allocation:AllocationEnv',
    # max_episode_steps=1000,
    # reward_threshold=4800.0,
)

__register_helper(
    env_id='Lending-v0',
    entry_point='fairness_gymnasium.envs.lending:LendingEnv',
    # max_episode_steps=1000,
    # reward_threshold=4800.0,
)
