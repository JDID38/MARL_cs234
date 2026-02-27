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
"""Runs an example of a self-play training experiment."""

import argparse
import os
from datetime import datetime

from meltingpot import substrate
import ray
from ray import air
from ray import tune
from ray.rllib.algorithms import ppo
from ray.rllib.models import ModelCatalog
from ray.rllib.policy import policy

from . import utils
from .custom_model import MeltingPotModel


def get_config(
    substrate_name: str = "bach_or_stravinsky_in_the_matrix__repeated",
    num_rollout_workers: int = 12,
    rollout_fragment_length: int = 100,
    train_batch_size: int = 12800,
    fcnet_hiddens=(64, 64),
    post_fcnet_hiddens=(256,),
    lstm_cell_size: int = 256,
    sgd_minibatch_size: int = 128,
):
  """Get the configuration for running an agent on a substrate using RLLib.

  We need the following 2 pieces to run the training:

  Args:
    substrate_name: The name of the MeltingPot substrate, coming from
      `substrate.AVAILABLE_SUBSTRATES`.
    num_rollout_workers: The number of workers for playing games.
    rollout_fragment_length: Unroll time for learning.
    train_batch_size: Batch size (batch * rollout_fragment_length)
    fcnet_hiddens: Fully connected layers.
    post_fcnet_hiddens: Layer sizes after the fully connected torso.
    lstm_cell_size: Size of the LSTM.
    sgd_minibatch_size: Size of the mini-batch for learning.

  Returns:
    The configuration for running the experiment.
  """
  # Gets the default training configuration
  config = ppo.PPOConfig()
  # Number of arenas.
  config.num_rollout_workers = num_rollout_workers
  # This is to match our unroll lengths.
  config.rollout_fragment_length = rollout_fragment_length
  # Total (time x batch) timesteps on the learning update.
  config.train_batch_size = train_batch_size
  # Mini-batch size.
  config.sgd_minibatch_size = sgd_minibatch_size
  # Observations are already flat Box (via MeltingPotObsWrapper).
  # Use PyTorch as the tensor framework (GPU-compatible).
  config = config.framework("torch")
  # Use 1 GPU for the learner, rollout workers stay on CPU.
  config.num_gpus = 1
  config.log_level = "INFO"

  # 2. Set environment config. This will be passed to
  # the env_creator function via the register env lambda below.
  player_roles = substrate.get_config(substrate_name).default_player_roles
  config.env_config = {"substrate": substrate_name, "roles": player_roles}

  config.env = "meltingpot"

  # 4. Extract space dimensions
  test_env = utils.env_creator(config.env_config)

  # Setup PPO with policies, one per entry in default player roles.
  policies = {}
  player_to_agent = {}
  for i in range(len(player_roles)):
    # Get RGB shape from the underlying MeltingPotEnv (before flattening)
    rgb_shape = test_env._rgb_shape[f"player_{i}"]  # (H, W, C)
    sprite_x = rgb_shape[0] // 8
    sprite_y = rgb_shape[1] // 8

    policies[f"agent_{i}"] = policy.PolicySpec(
        policy_class=None,  # use default policy
        observation_space=test_env.observation_space[f"player_{i}"],
        action_space=test_env.action_space[f"player_{i}"],
        config={
            "model": {
                "custom_model": "meltingpot_model",
                "custom_model_config": {
                    "conv_filters": [[16, [8, 8], 8],
                                     [128, [sprite_x, sprite_y], 1]],
                    "rgb_shape": list(rgb_shape),
                },
            },
        })
    player_to_agent[f"player_{i}"] = f"agent_{i}"

  def policy_mapping_fn(agent_id, episode=None, worker=None, **kwargs):
    return player_to_agent[agent_id]

  # 5. Configuration for multi-agent setup with one policy per role:
  config.multi_agent(policies=policies, policy_mapping_fn=policy_mapping_fn)

  # 6. Set the agent architecture — handled by custom model (MeltingPotModel).
  # The custom model internally builds CNN + FC + LSTM, so we disable
  # RLlib's built-in LSTM wrapper. Config values are passed through to the
  # custom model via model_config.
  config.model["fcnet_hiddens"] = fcnet_hiddens
  config.model["fcnet_activation"] = "relu"
  config.model["conv_activation"] = "relu"
  config.model["post_fcnet_hiddens"] = post_fcnet_hiddens
  config.model["post_fcnet_activation"] = "relu"
  # LSTM is handled inside the custom model — do NOT set use_lstm=True
  # to avoid RLlib's buggy LSTMWrapper with Dict observations.
  config.model["use_lstm"] = False
  config.model["max_seq_len"] = 20
  config.model["lstm_use_prev_action"] = True
  config.model["lstm_use_prev_reward"] = False
  config.model["lstm_cell_size"] = lstm_cell_size

  return config


def train(config, num_iterations=1, output_dir=None):
  """Trains a model.

  Args:
    config: model config
    num_iterations: number of iterations to train for.
    output_dir: directory to save checkpoints. Defaults to ~/ray_results.

  Returns:
    The trained algorithm instance.
  """
  from tqdm import tqdm

  tune.register_env("meltingpot", utils.env_creator)
  ModelCatalog.register_custom_model("meltingpot_model", MeltingPotModel)
  ray.init()

  algo = config.build()
  pbar = tqdm(range(1, num_iterations + 1), desc="Training", unit="iter")
  for i in pbar:
    result = algo.train()
    reward = result.get("episode_reward_mean", float("nan"))
    pbar.set_postfix(reward=f"{reward:.2f}",
                     steps=result.get("num_env_steps_sampled", 0))
    if i % 5 == 0:
      checkpoint = algo.save(output_dir)
      tqdm.write(f"  Checkpoint saved: {checkpoint}")

  # Save final checkpoint
  checkpoint = algo.save(output_dir)
  tqdm.write(f"Final checkpoint: {checkpoint}")
  algo.stop()
  return result


def main():
  parser = argparse.ArgumentParser(description="MeltingPot self-play training")
  parser.add_argument("--substrate", type=str,
                       default="commons_harvest__open",
                       help="MeltingPot substrate name")
  parser.add_argument("--num-iterations", type=int, default=20,
                       help="Number of training iterations")
  parser.add_argument("--num-rollout-workers", type=int, default=2,
                       help="Number of rollout workers")
  parser.add_argument("--train-batch-size", type=int, default=6400,
                       help="Training batch size")
  parser.add_argument("--sgd-minibatch-size", type=int, default=128,
                       help="SGD mini-batch size")
  parser.add_argument("--rollout-fragment-length", type=int, default=100,
                       help="Rollout fragment length")
  parser.add_argument("--num-gpus", type=int, default=1,
                       help="Number of GPUs for the learner")
  parser.add_argument("--output-dir", type=str, default=None,
                       help="Directory to save checkpoints (default: ~/ray_results)")
  args = parser.parse_args()

  config = get_config(
      substrate_name=args.substrate,
      num_rollout_workers=args.num_rollout_workers,
      rollout_fragment_length=args.rollout_fragment_length,
      train_batch_size=args.train_batch_size,
      sgd_minibatch_size=args.sgd_minibatch_size,
  )
  config.num_gpus = args.num_gpus

  # Auto-generate a unique run directory: run_checkpoint/<substrate>_<YYYYMMDD_HHMMSS>
  if args.output_dir is None:
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_name = f'{args.substrate}_{timestamp}'
    project_root = os.path.normpath(
        os.path.join(os.path.dirname(__file__), '..', '..'))
    output_dir = os.path.join(project_root, 'run_checkpoint', run_name)
  else:
    output_dir = args.output_dir
  os.makedirs(output_dir, exist_ok=True)
  print(f'Run directory: {output_dir}')

  results = train(config, num_iterations=args.num_iterations,
                  output_dir=output_dir)
  print("Training complete!")
  print(f"Final reward: {results.get('episode_reward_mean', 'N/A')}")
  print(f"Checkpoints saved to: {output_dir}")


if __name__ == "__main__":
  main()
