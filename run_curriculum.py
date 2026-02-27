"""Run a simple phi curriculum with random agents.

Checkpoint 3 goal:
  - Run an episode loop (default: 100 episodes).
  - Increase phi from 0 to 45 degrees over episodes.
  - Save a plot that shows phi rising smoothly to 45.

Example:
  python run_curriculum.py
  python run_curriculum.py --episodes 100 --num_players 10 --substrate commons_harvest__closed
"""

import argparse
import csv
import math
import os
from typing import Tuple

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
from meltingpot import substrate
from meltingpot.utils.substrates.wrappers import phi_reward_wrapper


DEFAULT_SUBSTRATE = "commons_harvest__closed"
DEFAULT_NUM_PLAYERS = 7
DEFAULT_EPISODES = 100
DEFAULT_START_PHI_DEG = 0.0
DEFAULT_END_PHI_DEG = 45.0
DEFAULT_MAX_STEPS_PER_EPISODE = 2000
DEFAULT_PLOT_PATH = "curriculum_phi_progress.png"
DEFAULT_CSV_PATH = "curriculum_phi_progress.csv"


def linear_phi_deg(
    episode_idx: int,
    total_episodes: int,
    start_phi_deg: float,
    end_phi_deg: float,
) -> float:
    """Linearly interpolate phi in degrees for a given episode index."""
    if total_episodes <= 1:
        return start_phi_deg
    t = episode_idx / float(total_episodes - 1)
    return start_phi_deg + t * (end_phi_deg - start_phi_deg)


def run_random_episode(env, max_steps: int) -> Tuple[int, bool]:
    """Run one random-action episode.

    Returns:
      (steps_run, reached_terminal_state)
    """
    timestep = env.reset()
    action_specs = env.action_spec()

    if timestep.last():
        return 0, True

    for step in range(1, max_steps + 1):
        actions = [int(np.random.randint(0, spec.num_values)) for spec in action_specs]
        timestep = env.step(actions)
        if timestep.last():
            return step, True
    return max_steps, False


def save_phi_plot(episodes, phi_degs, plot_path: str) -> bool:
    """Save a simple episode-vs-phi plot. Returns True if saved."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    plt.figure(figsize=(8, 4.5))
    plt.plot(episodes, phi_degs, linewidth=2.0, label="phi (deg)")
    plt.xlabel("Episode")
    plt.ylabel("Phi (degrees)")
    plt.title("Curriculum Progress: Phi vs Episode")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=140)
    plt.close()
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--substrate", type=str, default=DEFAULT_SUBSTRATE, help="Substrate name")
    parser.add_argument("--num_players", type=int, default=DEFAULT_NUM_PLAYERS, help="Number of players")
    parser.add_argument("--episodes", type=int, default=DEFAULT_EPISODES, help="Number of episodes")
    parser.add_argument(
        "--start_phi_deg",
        type=float,
        default=DEFAULT_START_PHI_DEG,
        help="Starting phi in degrees",
    )
    parser.add_argument(
        "--end_phi_deg",
        type=float,
        default=DEFAULT_END_PHI_DEG,
        help="Final phi in degrees",
    )
    parser.add_argument(
        "--max_steps_per_episode",
        type=int,
        default=DEFAULT_MAX_STEPS_PER_EPISODE,
        help="Safety cap for steps per episode",
    )
    parser.add_argument("--plot_path", type=str, default=DEFAULT_PLOT_PATH, help="Output path for phi plot")
    parser.add_argument("--csv_path", type=str, default=DEFAULT_CSV_PATH, help="Output path for CSV log")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility")
    args = parser.parse_args()

    np.random.seed(args.seed)
    roles = ("default",) * args.num_players

    print(
        f"Running curriculum: episodes={args.episodes}, phi={args.start_phi_deg}..{args.end_phi_deg} deg, "
        f"players={args.num_players}, substrate={args.substrate}",
        flush=True,
    )

    episode_indices = []
    phi_degs = []
    rows = []

    for episode_idx in range(args.episodes):
        phi_deg = linear_phi_deg(
            episode_idx=episode_idx,
            total_episodes=args.episodes,
            start_phi_deg=args.start_phi_deg,
            end_phi_deg=args.end_phi_deg,
        )
        phi_rad = math.radians(phi_deg)
        phis = [phi_rad] * args.num_players

        env = substrate.build(args.substrate, roles=roles)
        env = phi_reward_wrapper.PhiRewardWrapper(env, phis)
        steps_run, reached_terminal = run_random_episode(env, args.max_steps_per_episode)
        env.close()

        episode_indices.append(episode_idx)
        phi_degs.append(phi_deg)
        rows.append(
            {
                "episode": episode_idx,
                "phi_deg": phi_deg,
                "phi_rad": phi_rad,
                "steps": steps_run,
                "terminated": int(reached_terminal),
            }
        )

        if (episode_idx + 1) % 10 == 0 or episode_idx == args.episodes - 1:
            print(
                f"Episode {episode_idx + 1:4d}/{args.episodes}: phi={phi_deg:6.2f} deg "
                f"({phi_rad:0.4f} rad), steps={steps_run}, terminated={reached_terminal}",
                flush=True,
            )

    with open(args.csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["episode", "phi_deg", "phi_rad", "steps", "terminated"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved curriculum log CSV to: {args.csv_path}", flush=True)

    if save_phi_plot(episode_indices, phi_degs, args.plot_path):
        print(f"Saved phi progression plot to: {args.plot_path}", flush=True)
    else:
        print(
            "Matplotlib is not installed, so no plot was generated. "
            f"CSV log is still available at: {args.csv_path}",
            flush=True,
        )

    print("Checkpoint 3 verification complete.", flush=True)


if __name__ == "__main__":
    main()
