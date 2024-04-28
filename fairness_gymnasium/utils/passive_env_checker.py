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
"""A set of functions for passively checking environment implementations."""

import numpy as np
from gymnasium import error, logger
from gymnasium.utils.passive_env_checker import check_obs


def env_step_passive_checker(env, action):
    """A passive check for the environment step, investigating the returning data then returning the data unchanged."""
    # We don't check the action as for some environments then out-of-bounds values can be given
    result = env.step(action)
    assert isinstance(
        result,
        tuple,
    ), f'Expects step result to be a tuple, actual type: {type(result)}'
    if len(result) == 4:
        logger.deprecation(
            (
                'Core environment is written in old step API which returns one bool instead of two.'
                ' It is recommended to rewrite the environment with new step API.'
            ),
        )
        obs, reward, done, info = result

        if not isinstance(done, (bool, np.bool_)):
            logger.warn(f'Expects `done` signal to be a boolean, actual type: {type(done)}')
    elif len(result) == 5:
        obs, reward, terminated, truncated, info = result

        # np.bool is actual python bool not np boolean type, therefore bool_ or bool8
        if not isinstance(terminated, (bool, np.bool_)):
            logger.warn(
                f'Expects `terminated` signal to be a boolean, actual type: {type(terminated)}',
            )
        if not isinstance(truncated, (bool, np.bool_)):
            logger.warn(
                f'Expects `truncated` signal to be a boolean, actual type: {type(truncated)}',
            )
    else:
        raise error.Error(
            (
                'Expected `Env.step` to return a four or five element tuple, '
                f'actual number of elements returned: {len(result)}.'
            ),
        )

    check_obs(obs, env.observation_space, 'step')
    check_obs(reward, env.reward_space, 'step')   # Borrow the check_obs function to check the reward 

    assert isinstance(
        info,
        dict,
    ), f'The `info` returned by `step()` must be a python dictionary, actual type: {type(info)}'

    return result
