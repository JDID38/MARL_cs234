"""Play commons_harvest__open using trained RLlib agent checkpoints and record video."""

import argparse
import os
import sys

import cv2
import numpy as np
import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPO
from ray.rllib.models import ModelCatalog

from meltingpot import substrate

from . import utils
from .custom_model import MeltingPotModel

# Default checkpoint path (relative to this file -> ../../run_checkpoint)
_DEFAULT_CHECKPOINT = os.path.normpath(
    os.path.join(os.path.dirname(__file__), '..', '..', 'run_checkpoint'))


def main():
    parser = argparse.ArgumentParser(
        description='Run trained agents on commons_harvest__open')
    parser.add_argument('--checkpoint', type=str, default=_DEFAULT_CHECKPOINT,
                        help='Path to the RLlib Algorithm checkpoint directory')
    parser.add_argument('--substrate', type=str, default='commons_harvest__open',
                        help='MeltingPot substrate name')
    parser.add_argument('--steps', type=int, default=1000,
                        help='Number of environment steps to run')
    parser.add_argument('--output-dir', type=str, default='trained_gameplay_videos',
                        help='Directory to save the output video')
    parser.add_argument('--fps', type=int, default=12,
                        help='Output video FPS')
    parser.add_argument('--resolution', type=int, default=512,
                        help='Output video resolution (square)')
    args = parser.parse_args()

    # ---- Initialise Ray and register env / model ----
    ray.init(ignore_reinit_error=True)
    tune.register_env('meltingpot', utils.env_creator)
    ModelCatalog.register_custom_model('meltingpot_model', MeltingPotModel)

    # ---- Restore trained algorithm from checkpoint ----
    checkpoint_path = os.path.abspath(args.checkpoint)
    print(f'Loading checkpoint from {checkpoint_path} ...')
    algo = PPO.from_checkpoint(checkpoint_path)
    print('Checkpoint restored successfully.')

    # ---- Build the environment (same wrappers as training) ----
    substrate_name = args.substrate
    player_roles = substrate.get_config(substrate_name).default_player_roles
    num_agents = len(player_roles)

    env_config = {'substrate': substrate_name, 'roles': player_roles}
    env = utils.env_creator(env_config)  # MeltingPotObsWrapper(MeltingPotEnv(...))

    agent_ids = [f'agent_{i}' for i in range(num_agents)]
    player_ids = [f'player_{i}' for i in range(num_agents)]
    player_to_agent = {p: a for p, a in zip(player_ids, agent_ids)}

    # ---- Initialise per-agent LSTM states / prev action / prev reward ----
    states = {}
    prev_actions = {}
    prev_rewards = {}
    for agent_id in agent_ids:
        pol = algo.get_policy(agent_id)
        states[agent_id] = pol.get_initial_state()
        prev_actions[agent_id] = 0
        prev_rewards[agent_id] = 0.0

    # ---- Video setup ----
    os.makedirs(args.output_dir, exist_ok=True)
    output_video = os.path.join(args.output_dir, f'{substrate_name}_trained.mp4')
    output_size = (args.resolution, args.resolution)

    # ---- Run simulation ----
    obs, info = env.reset()

    writer = None
    frames_written = 0
    total_rewards = {aid: 0.0 for aid in agent_ids}

    print(f'Running {num_agents} trained agents for {args.steps} steps ...')

    try:
        for step in range(args.steps):
            # Compute actions for every agent
            actions = {}
            for player_id in player_ids:
                agent_id = player_to_agent[player_id]
                action, state, _ = algo.compute_single_action(
                    obs[player_id],
                    state=states[agent_id],
                    policy_id=agent_id,
                    prev_action=prev_actions[agent_id],
                    prev_reward=prev_rewards[agent_id],
                )
                actions[player_id] = action
                states[agent_id] = state
                prev_actions[agent_id] = action

            # Step environment
            obs, rewards, terminated, truncated, info = env.step(actions)

            # Update prev rewards and accumulators
            for player_id in player_ids:
                agent_id = player_to_agent[player_id]
                r = rewards.get(player_id, 0.0)
                prev_rewards[agent_id] = r
                total_rewards[agent_id] += r

            # Capture WORLD.RGB frame via render()
            frame = env.render()
            if frame is not None:
                frame = frame.astype(np.uint8)
                frame = cv2.resize(frame, output_size,
                                   interpolation=cv2.INTER_NEAREST)
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                if writer is None:
                    writer = cv2.VideoWriter(
                        output_video,
                        cv2.VideoWriter_fourcc(*'mp4v'),
                        args.fps,
                        output_size,
                    )
                writer.write(frame)
                frames_written += 1

            # Check episode end
            if terminated.get('__all__', False) or truncated.get('__all__', False):
                print(f'Episode ended at step {step + 1}')
                break

            # Progress update every 100 steps
            if (step + 1) % 100 == 0:
                print(f'  Step {step + 1}/{args.steps}')

        # ---- Finish ----
        if writer is not None:
            writer.release()
            writer = None

        print(f'\nDone! Wrote {frames_written} frames to {output_video}')
        print('Cumulative rewards per agent:')
        for agent_id in agent_ids:
            print(f'  {agent_id}: {total_rewards[agent_id]:.1f}')

    finally:
        if writer is not None:
            writer.release()
        env.close()
        algo.stop()
        ray.shutdown()


if __name__ == '__main__':
    main()
