# coding=utf-8
# Copyright 2022 The ML Fairness Gym Authors.
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

# Lint as: python2, python3
"""ML Fairness gym location-based attention allocation environment.

This environment is meant to be a general but simple location-based
attention allocation environment.

Situations that could be modeled by this environment are pest-control, or
allocation of social interventions like mobile STD testing clinics.
This is  not a good environment to model predictive policing in part due to the
concerns raised in "Dirty Data, Bad Predictions: How Civil Rights Violations
Impact Police Data, Predictive Policing Systems, and Justice", Richardson et al.
(https://www.nyulawreview.org/wp-content/uploads/2019/04/NYULawReview-94-Richardson-Schultz-Crawford.pdf)

The environment contains k locations. At each timestep, some number of incidents
occur (such as rat infestations) in each location, and the agent has to allocate
some of its attention to each location. Each attention unit may then "see" or
discover some number of the incidents.

Incidents are drawn from a Poisson centered on an underlying incident rate.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import time
from typing import List, Optional, Tuple
import attr
import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import seeding
import numpy as np

import math
import numba
from numba import njit

from fairness_gymnasium.envs import core

@njit
def numba_seed(seed):
    np.random.seed(seed)


@njit
def numba_softmax(x):
    x_max = np.max(x)
    return np.exp(x - x_max) / np.sum(np.exp(x-x_max))

@njit
def numba_sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

@njit
def calculate_total_variation_distance(lambda1, lambda2):
    def poisson_pmf(lam, k):
        return np.exp(-lam) * np.power(lam, k) / math.gamma(k+1)

    tv_distance = 0.0
    upper = max(lambda1, lambda2) * 2
    for k in range(upper):
        tv_distance += np.abs(poisson_pmf(lambda1, k) - poisson_pmf(lambda2, k))
    
    return 0.5 * tv_distance

@attr.s
class Params(core.Params):
    """Attention Allocation parameters."""

    # Number of locations represented in the environment.
    n_locations = attr.ib(default=2)  # type: int
    # Incident rates for each location.
    incident_rates = attr.ib(factory=lambda: [6., 6.])  # type: List[float]
    # Number of attention units that are available to be allocated.
    n_attention_units = attr.ib(default=5)  # type: int
    # Probability of an incident is discovered (observable) for each location
    discover_incident_prob = attr.ib(default=(1.0, 1.0))  # type: Tuple[float, ...]
    # Probability of an incident is missed by an attention unit for each location.'
    fail_incident_prob = attr.ib(default=(0.05, 0.05))  # type: Tuple[float, ...]
    # The rate at which the incident_rates change in response to allocation
    # of attention units.
    dynamic_rate_up   = attr.ib(default=0.2)  # type: float
    dynamic_rate_down = attr.ib(default=0.1)  # type: float
    # The max number of steps in an episode.
    max_timesteps = attr.ib(default=50) # type: int
    burn_in = attr.ib(default=5) # type: int
    render = attr.ib(default=False) # type: bool
    exp_name = attr.ib(default="default") # type: str


@attr.s(cmp=False)
class State(core.State):
    """Attention Allocation state."""

    # Parameters.
    params = attr.ib()  # type: Params

    # A ndarray of integers representing the incidents occurred for each location.
    incidents_occurred = attr.ib()  # type: np.ndarray

    # A ndarray of integers representing the incidents discovered at each location
    incidents_discovered = attr.ib()  # type: np.ndarray

    # A ndarray of integers representing the attention units allocated at each location
    attention_allocated = attr.ib()  # type: np.ndarray
    
    # A ndarray of integers representing the incidents solved at each location
    incidents_solved = attr.ib()  # type: np.ndarray

    # An float number representing the maximum number of Total Variance
    inequality = attr.ib()  # type: float


@njit
def _sample_incidents(incident_rates):
    """Generates new crimeincident occurrences across locations.

    Args:
      incident_rates: a list of floats of incident rates for each location.

    Returns:
      incidents_occurred: a list of integers of number of incidents for each
      location.
    """
    n_locations = len(incident_rates)
    incidents_occurred = [
        np.maximum(0, np.round(np.random.normal(incident_rates[i], 1.0))) for i in range(n_locations)
    ]
    return np.array(incidents_occurred, dtype='int32')


@njit
def _update_state(
    discover_incident_prob, 
    fail_incident_prob,
    incident_rates,
    dynamic_rate_up,
    dynamic_rate_down,
    alphas,
    incidents_occurred, 
    action
):
    """Updates the state given the human_designed_policies' action.

    This function simulates attention discovering incidents in order to determine
    and populate the number of solved incidents in the state.

    Args:
      state: a 'State' object with the state to be updated.
      incidents_occurred: a vector of length equal to n_locations in state.param
        that contains integer counts of incidents that occurred for each location.
      action: an action in the action space of AllocationEnv that is a
        vector of integer counts of attention allocated to each location.
    """
    n_locations = len(discover_incident_prob)
    incidents_discovered = incidents_occurred

    incidents_solved = [0] * n_locations
    for location_ind in range(n_locations):
        unused_attention = action[location_ind]

        # Iterate over crime incidents and determine if each one is "caught".
        for _ in range(incidents_occurred[location_ind]):
            is_incident_solved = np.random.binomial(
                1, 1 - (np.power(fail_incident_prob[location_ind],
                                    unused_attention)))
            unused_attention -= is_incident_solved
            incidents_solved[location_ind] += is_incident_solved
            if unused_attention <= 0:
                break

        # Handle dynamics.
        attention = action[location_ind]
        if incidents_discovered[location_ind] >= 2*incidents_solved[location_ind]: # miss >= 50%
            incident_rates[location_ind] += np.maximum(0.0, np.random.normal(dynamic_rate_up, 0.05))
        else: # miss < 50%
            incident_rates[location_ind] = max(
                0.0, incident_rates[location_ind] - np.maximum(0.0, np.random.normal(dynamic_rate_down * attention, 0.05))
            )

    incident_rates[1] = incident_rates[1] - (alphas[1]-alphas[0]) * 1.0
    return np.asarray(incidents_occurred), np.asarray(incidents_discovered, dtype=np.int32),\
          np.asarray(action, dtype=np.int32), np.asarray(incidents_solved, dtype=np.int32)


class AllocationEnv(gym.Env):
    """Location based allocation environment.

    In each step, agent allocates attention across locations. Environment then
    simulates solved incidents based on incidents that occurred and attention
    distribution.
    Incidents are generated from a poisson distribution of underlying incidents
    rates for each location.
    """

    def __init__(self, params = None, env_cfgs = None):
        if params is None:
            params = Params()
        assert (params.n_locations == len(params.incident_rates))

        if env_cfgs is not None:
            self.alphas = (env_cfgs['alpha1'], env_cfgs['alpha2'])
            self.betas = (env_cfgs['beta1'], env_cfgs['beta2'])
            params.incident_rates = [env_cfgs['rate1'], env_cfgs['rate2']]
        else:
            self.alphas = (0.0, 0.0)
            self.betas = (0.0, 0.0)

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(params.n_locations,),
            dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(params.n_locations*2,),
            dtype=np.float32
        )
        self.reward_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(params.n_locations,),
            dtype=np.float32
        )
        self.initial_params = copy.deepcopy(params)
        self.max_timesteps = params.max_timesteps
        self.max_incident_rate = max(params.incident_rates) + params.dynamic_rate_up*params.max_timesteps
        self.timestep = 0
        self._state_init()
        # Observation history of incidents discovered per site
        self.burn_in = params.burn_in
        
        self.render_env = params.render
        if self.render_env:
            self.state_history = []
            self.lines = {"A":[], "B":[], "C":[]}

    def _state_init(self):
        n_locations = self.initial_params.n_locations
        params = copy.deepcopy(self.initial_params)
        params.incident_rates = np.array(params.incident_rates).tolist()
        params.discover_incident_prob = np.array(params.discover_incident_prob).tolist()
        params.fail_incident_prob = np.array(params.fail_incident_prob).tolist()
        self.state = State(
            params=params,
            incidents_occurred=np.zeros(n_locations, dtype='int32'),
            incidents_discovered=np.zeros(n_locations, dtype='int32'),
            attention_allocated=np.zeros(n_locations, dtype='int32'),
            incidents_solved=np.zeros(n_locations, dtype='int32'),
            inequality=0.0)

    def _is_truncated(self):
        """Never returns true because there is no end case to this environment."""
        return self.timestep >= self.max_timesteps

    def _step_impl(self, state, action):
        """Run one timestep of the environment's dynamics.

        In a step, the agent allocates attention across disctricts. The environement
        then returns incidents solved as an observation based off the actual hidden
        incident occurrences and attention allocation.

        Args:
          state: A 'State' object containing the current state.
          action: An action in 'action space'.

        Returns:
          A 'State' object containing the updated state.
        """
        incidents_occurred = _sample_incidents(np.array(state.params.incident_rates))
        params = state.params
        state.incidents_occurred, state.incidents_discovered,\
        state.attention_allocated, state.incidents_solved \
            =  _update_state(
                np.array(params.discover_incident_prob),
                np.array(params.fail_incident_prob),
                params.incident_rates,
                params.dynamic_rate_up,
                params.dynamic_rate_down,
                self.alphas,
                incidents_occurred, 
                action
            )
        return state
    
    def _get_observable_state(self):
        return np.array(self.state.params.incident_rates, dtype=np.float32)
    
    # @staticmethod
    # @njit
    def _reward_fn(self):
        state = self.state
        # Compute incidents missed
        incidents_missed = state.incidents_occurred - state.incidents_solved

        incidents_solved_term = state.incidents_solved
        incidents_missed_term = -1.0 * incidents_missed
        attention_alloc_term  = -1.0 * state.attention_allocated
        
        lambda1, lambda2 = max(state.params.incident_rates), min(state.params.incident_rates)
        inequality_term = 1.0 * calculate_total_variation_distance(lambda1, lambda2) 

        tot_rew = 0.1 * (incidents_missed_term + 0.2/(incidents_missed+1.0) * attention_alloc_term)
        tot_rew[1] = tot_rew[1] + (self.betas[1]-self.betas[0]) * 1.0  # bonus or penalty for the second group

        self.rew_info = {
            'max_rate': np.max(state.params.incident_rates),
            'min_rate': np.min(state.params.incident_rates),
            'occurr': np.sum(state.incidents_occurred),
            'miss': np.sum(incidents_missed),
            'alloc': np.sum(state.attention_allocated),
            'inequality_term': inequality_term,
            'tot_rew': np.sum(tot_rew),
        }

        return tot_rew.astype(np.float32)
    
    @staticmethod
    @njit
    def _process_action(
        raw_action, 
        n_locations, 
        n_attention_units,
    ):
        """
        Convert actions from logits to attention allocation over sites through the following steps:
        1. Convert logits vector into a probability distribution using softmax
        2. Convert probability distribution into allocation distribution with multinomial distribution

        Args:
        action: n_locations vector of logits

        Returns: n_locations vector of allocations
        """
        action = raw_action.reshape(n_locations, 2).sum(1) / 4.0 # make it between -0.5 to 0.5

        action = 0.5 * (1+np.sin(np.pi * action)) # make it between 0 to 1
        action = np.round(np.multiply(action, n_attention_units))
        return action

    def seed(self, seed = None):
        """Sets the seed for this env's random number generator."""
        rng, seed = seeding.np_random(seed)
        numba_seed(seed)
        return [seed]

    def reset(
            self,
            *,
            seed = None,
            options = None,
    ):
        """Resets the environment."""
        if seed is not None:
            self.seed(seed)

        self.timestep = 0
        self._state_init()
        n_locations = self.initial_params.n_locations
        n_attention_units = self.initial_params.n_attention_units
        # burn in a few steps for generating the initial state
        for _ in range(self.burn_in):
            incidents_occurred = _sample_incidents(np.array(self.state.params.incident_rates))
            action = self.action_space.sample()
            action = self._process_action(action, n_locations, n_attention_units)

            params = self.state.params
            self.state.incidents_occurred, self.state.incidents_discovered,\
            self.state.attention_allocated, self.state.incidents_solved \
                =  _update_state(
                    np.array(params.discover_incident_prob),
                    np.array(params.fail_incident_prob),
                    params.incident_rates,
                    params.dynamic_rate_up,
                    params.dynamic_rate_down,
                    self.alphas,
                    incidents_occurred, 
                    action
                )

            obs = self._get_observable_state()
        
        if self.render_env:
            _ = self._reward_fn()
            self.state.inequality = self.rew_info['inequality_term']
            self.state_history = []
            self.state_history.append((copy.deepcopy(self.state), self.timestep))
            self.lines = {"A":[], "B":[], "C":[]}
        return obs, {}

    def step(self, action):
        """Run one timestep of the environment's dynamics.
        This is part of the openAI gym interface and should not be overridden.
        When writing a new ML fairness gym environment, users should override the
        `_step_impl` method.
        Args:
            action: An action provided by the agent. A member of `action_space`.
        Returns:
            observation: Agent's observation of the current environment. A member
            of `observation_space`.
            reward: Scalar reward returned after previous action. This should be the
            output of a `RewardFn` provided by the agent.
            done: Whether the episode has ended, in which case further step() calls
            will return undefined results.
            info: A dictionary with auxiliary diagnostic information.
        Raises:
        NotInitializedError: If called before first reset().
        gym.error.InvalidAction: If `action` is not in `self.action_space`.
        """
        if not self.action_space.contains(action):
            raise gym.error.InvalidAction('Invalid action: %s' % action)
        n_locations = self.initial_params.n_locations
        n_attention_units = self.initial_params.n_attention_units
        action = self._process_action(action, n_locations, n_attention_units)

        real_state = self.state.params.incident_rates # log before update
        self.state = self._step_impl(self.state, action)
        obs = self._get_observable_state()

        self.timestep += 1
        reward = self._reward_fn()
        self.state.inequality = self.rew_info['inequality_term']
        if self.render_env:
            self.state_history.append((copy.deepcopy(self.state), self.timestep))

        return obs, reward, False, self._is_truncated(), \
            {"cost": self.rew_info['inequality_term'], "real_state": real_state, \
             "max_rate": self.rew_info['max_rate'], "min_rate": self.rew_info['min_rate'], \
             "occurr": self.rew_info['occurr'], \
             "miss": self.rew_info['miss'], "alloc": self.rew_info['alloc']}
        
    def render(self):
        raise NotImplementedError

