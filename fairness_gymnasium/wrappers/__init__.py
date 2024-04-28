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
"""Env wrappers."""

from typing import Callable

import gymnasium

from fairness_gymnasium.wrappers.autoreset import FairAutoResetWrapper
from fairness_gymnasium.wrappers.env_checker import FairPassiveEnvChecker
from fairness_gymnasium.wrappers.time_limit import FairTimeLimit


__all__ = [
    'FairAutoResetWrapper',
    'FairPassiveEnvChecker',
    'FairTimeLimit',
]
