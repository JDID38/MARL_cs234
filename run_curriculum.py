"""Run a simple phi curriculum with random agents.

Checkpoint 3 goal:
  - Run an episode loop (default: 100 episodes).
  - Increase phi from 0 to 45 degrees over episodes.
  - Save a plot that shows phi rising smoothly to 45.

Example:
  python run_curriculum.py
  python run_curriculum.py --episodes 100 --num_players 10 --substrate commons_harvest__closed
  python run_curriculum.py --schedule sigmoid
  python run_curriculum.py --schedule log --schedule_log_scale 20
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

# Schedule-specific defaults
DEFAULT_SIGMOID_STEEPNESS = 10.0
DEFAULT_LOG_SCALE = 10.0
DEFAULT_NUM_STEPS = 5


def _normalized_progress(episode_idx: int, total_episodes: int) -> float:
    """Map episode index to [0, 1] (0 at first episode, 1 at last)."""
    if total_episodes <= 1:
        return 1.0
    return episode_idx / float(total_episodes - 1)


def linear_phi_deg(
    episode_idx: int,
    total_episodes: int,
    start_phi_deg: float,
    end_phi_deg: float,
    **kwargs,
) -> float:
    """Linearly interpolate phi in degrees for a given episode index."""
    if total_episodes <= 1:
        return start_phi_deg
    t = _normalized_progress(episode_idx, total_episodes)
    return start_phi_deg + t * (end_phi_deg - start_phi_deg)


def sigmoid_phi_deg(
    episode_idx: int,
    total_episodes: int,
    start_phi_deg: float,
    end_phi_deg: float,
    steepness: float = DEFAULT_SIGMOID_STEEPNESS,
    **kwargs,
) -> float:
    """Sigmoid (S-curve) schedule: slow at start and end, steep in the middle."""
    if total_episodes <= 1:
        return start_phi_deg
    x = _normalized_progress(episode_idx, total_episodes)
    t = 1.0 / (1.0 + math.exp(-steepness * (x - 0.5)))
    return start_phi_deg + t * (end_phi_deg - start_phi_deg)


def log_phi_deg(
    episode_idx: int,
    total_episodes: int,
    start_phi_deg: float,
    end_phi_deg: float,
    log_scale: float = DEFAULT_LOG_SCALE,
    **kwargs,
) -> float:
    """Logarithmic increase: fast rise at the beginning, then flattens."""
    if total_episodes <= 1:
        return start_phi_deg
    x = _normalized_progress(episode_idx, total_episodes)
    t = math.log(1.0 + log_scale * x) / math.log(1.0 + log_scale)
    return start_phi_deg + t * (end_phi_deg - start_phi_deg)


def sqrt_phi_deg(
    episode_idx: int,
    total_episodes: int,
    start_phi_deg: float,
    end_phi_deg: float,
    **kwargs,
) -> float:
    """Square-root schedule: slow at start, accelerates toward the end."""
    if total_episodes <= 1:
        return start_phi_deg
    t = _normalized_progress(episode_idx, total_episodes)
    t = math.sqrt(t)
    return start_phi_deg + t * (end_phi_deg - start_phi_deg)


def step_phi_deg(
    episode_idx: int,
    total_episodes: int,
    start_phi_deg: float,
    end_phi_deg: float,
    num_steps: int = DEFAULT_NUM_STEPS,
    **kwargs,
) -> float:
    """Step schedule: phi jumps at discrete step boundaries (piecewise constant)."""
    if total_episodes <= 1 or num_steps <= 1:
        return start_phi_deg if episode_idx == 0 else end_phi_deg
    x = _normalized_progress(episode_idx, total_episodes)
    step_index = min(int(x * num_steps), num_steps - 1)
    t = step_index / (num_steps - 1)
    return start_phi_deg + t * (end_phi_deg - start_phi_deg)


def cosine_phi_deg(
    episode_idx: int,
    total_episodes: int,
    start_phi_deg: float,
    end_phi_deg: float,
    **kwargs,
) -> float:
    """Cosine schedule: smooth start and end (1 - cos(pi*t))/2."""
    if total_episodes <= 1:
        return start_phi_deg
    t = _normalized_progress(episode_idx, total_episodes)
    t = 0.5 * (1.0 - math.cos(math.pi * t))
    return start_phi_deg + t * (end_phi_deg - start_phi_deg)


def exponential_phi_deg(
    episode_idx: int,
    total_episodes: int,
    start_phi_deg: float,
    end_phi_deg: float,
    **kwargs,
) -> float:
    """Exponential schedule: very slow at start, rapid increase near the end."""
    if total_episodes <= 1:
        return start_phi_deg
    x = _normalized_progress(episode_idx, total_episodes)
    t = (math.exp(x) - 1.0) / (math.e - 1.0)
    return start_phi_deg + t * (end_phi_deg - start_phi_deg)


SCHEDULES = {
    "linear": linear_phi_deg,
    "sigmoid": sigmoid_phi_deg,
    "log": log_phi_deg,
    "sqrt": sqrt_phi_deg,
    "step": step_phi_deg,
    "cosine": cosine_phi_deg,
    "exponential": exponential_phi_deg,
}


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


def save_phi_plot(episodes, phi_degs, plot_path: str, schedule_name: str = "linear") -> bool:
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
    plt.title(f"Curriculum Progress: Phi vs Episode (schedule={schedule_name})")
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
    parser.add_argument(
        "--schedule",
        type=str,
        default="linear",
        choices=list(SCHEDULES.keys()),
        help="Phi curriculum schedule: linear, sigmoid, log, sqrt, step, cosine, exponential",
    )
    parser.add_argument(
        "--schedule_sigmoid_steepness",
        type=float,
        default=DEFAULT_SIGMOID_STEEPNESS,
        help="Steepness of sigmoid schedule (higher = sharper S-curve)",
    )
    parser.add_argument(
        "--schedule_log_scale",
        type=float,
        default=DEFAULT_LOG_SCALE,
        help="Scale for log schedule (higher = more weight on early episodes)",
    )
    parser.add_argument(
        "--schedule_num_steps",
        type=int,
        default=DEFAULT_NUM_STEPS,
        help="Number of discrete steps for step schedule",
    )
    args = parser.parse_args()

    np.random.seed(args.seed)
    roles = ("default",) * args.num_players

    print(
        f"Running curriculum: episodes={args.episodes}, schedule={args.schedule}, "
        f"phi={args.start_phi_deg}..{args.end_phi_deg} deg, "
        f"players={args.num_players}, substrate={args.substrate}",
        flush=True,
    )

    schedule_fn = SCHEDULES[args.schedule]
    schedule_kwargs = {
        "steepness": args.schedule_sigmoid_steepness,
        "log_scale": args.schedule_log_scale,
        "num_steps": args.schedule_num_steps,
    }

    episode_indices = []
    phi_degs = []
    rows = []

    for episode_idx in range(args.episodes):
        phi_deg = schedule_fn(
            episode_idx=episode_idx,
            total_episodes=args.episodes,
            start_phi_deg=args.start_phi_deg,
            end_phi_deg=args.end_phi_deg,
            **schedule_kwargs,
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

    if save_phi_plot(episode_indices, phi_degs, args.plot_path, schedule_name=args.schedule):
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
