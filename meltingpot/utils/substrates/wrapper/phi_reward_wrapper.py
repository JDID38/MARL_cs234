# Copyright 2020 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Wrapper that transforms each agent's reward: u_i(t) = cos(phi_i)*r_i(t) + sin(phi_i)*r_bar_{-i}(t)."""

from typing import Sequence, Union

import dm_env
import numpy as np
from meltingpot.utils.substrates.wrappers import base


class PhiRewardWrapper(base.Lab2dWrapper):
  """Transforms rewards: u_i(t) = cos(phi_i) * r_i(t) + sin(phi_i) * mean(rewards of others)."""

  def __init__(self, env, phis: Sequence[Union[float, int]]):
    """Initializes the wrapper.

    Args:
      env: Environment to wrap.
      phis: One angle (radians) per agent. len(phis) must match num_players.
    """
    super().__init__(env)
    self._phis = np.asarray(phis, dtype=np.float64)
    self._n = len(self._phis)
    self._cos_phi = np.cos(self._phis)
    self._sin_phi = np.sin(self._phis)

  def _transform_rewards(self, reward: Sequence[float]) -> tuple:
    """Compute u_i = cos(phi_i)*r_i + sin(phi_i)*r_bar_{-i} for each agent."""
    r = np.asarray(reward, dtype=np.float64)
    out = np.empty(self._n, dtype=np.float64)
    for i in range(self._n):
      others = np.delete(r, i)
      r_bar_others = np.mean(others) if len(others) > 0 else 0.0
      out[i] = self._cos_phi[i] * r[i] + self._sin_phi[i] * r_bar_others
    return tuple(out)

  def _get_timestep(self, timestep: dm_env.TimeStep) -> dm_env.TimeStep:
    if timestep.reward is None:
      return timestep
    new_reward = self._transform_rewards(timestep.reward)
    return dm_env.TimeStep(
        step_type=timestep.step_type,
        reward=new_reward,
        discount=timestep.discount,
        observation=timestep.observation,
    )

  def reset(self, *args, **kwargs) -> dm_env.TimeStep:
    timestep = self._env.reset(*args, **kwargs)
    return self._get_timestep(timestep)

  def step(self, *args, **kwargs) -> dm_env.TimeStep:
    timestep = self._env.step(*args, **kwargs)
    return self._get_timestep(timestep)
