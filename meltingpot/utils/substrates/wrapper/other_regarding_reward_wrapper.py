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
"""Wrapper that transforms each agent's reward with other-regarding preferences.

Transformed reward:
  u_i = r_i - (alpha_i / (N-1)) * sum_{j != i} max(r_j - r_i, 0)
           - (beta_i / (N-1)) * sum_{j != i} max(r_i - r_j, 0).

- The second term penalizes agent i when others get higher rewards (envy/inequality aversion).
- The third term penalizes agent i when it gets higher rewards than others (guilt/advantage aversion).
"""

from typing import Sequence, Union

import dm_env
import numpy as np
from meltingpot.utils.substrates.wrappers import base


class OtherRegardingRewardWrapper(base.Lab2dWrapper):
  """Transforms rewards with alpha (envy) and beta (guilt) other-regarding terms.

  u_i = r_i - (alpha_i/(N-1)) * sum_{j!=i} max(r_j - r_i, 0)
           - (beta_i/(N-1)) * sum_{j!=i} max(r_i - r_j, 0).
  """

  def __init__(
      self,
      env,
      alphas: Sequence[Union[float, int]],
      betas: Sequence[Union[float, int]],
  ):
    """Initializes the wrapper.

    Args:
      env: Environment to wrap.
      alphas: One coefficient per agent for the envy term (penalty when others
        do better). len(alphas) must match num_players.
      betas: One coefficient per agent for the guilt term (penalty when self
        does better than others). len(betas) must match num_players.
    """
    super().__init__(env)
    self._alphas = np.asarray(alphas, dtype=np.float64)
    self._betas = np.asarray(betas, dtype=np.float64)
    self._n = len(self._alphas)
    if len(self._betas) != self._n:
      raise ValueError(
          f'alphas and betas must have the same length; got {self._n} and {len(self._betas)}.'
      )

  def _transform_rewards(self, reward: Sequence[float]) -> tuple:
    """Compute u_i = r_i - alpha_term - beta_term for each agent."""
    r = np.asarray(reward, dtype=np.float64)
    if len(r) != self._n:
      raise ValueError(
          f'Reward length {len(r)} does not match number of agents {self._n}.'
      )
    out = np.empty(self._n, dtype=np.float64)
    denom = self._n - 1
    if denom <= 0:
      # Single agent: no other-regarding terms
      return tuple(r)
    for i in range(self._n):
      r_i = r[i]
      others = np.delete(r, i)
      # Envy term: sum over j != i of max(r_j - r_i, 0)
      envy = np.sum(np.maximum(others - r_i, 0.0))
      # Guilt term: sum over j != i of max(r_i - r_j, 0)
      guilt = np.sum(np.maximum(r_i - others, 0.0))
      u_i = (
          r_i
          - (self._alphas[i] / denom) * envy
          - (self._betas[i] / denom) * guilt
      )
      out[i] = u_i
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
