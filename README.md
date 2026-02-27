## Phi reward + curriculum + metrics: how to use them in training

This guide explains how to use:

- **`PhiRewardWrapper`** (`meltingpot/utils/substrates/wrappers/phi_reward_wrapper.py`)
- **`run_curriculum.py`** (for a phi curriculum over episodes)
- **`checkpoint4_metrics.py`** (for logging U/S/E metrics to CSV)

in a **normal RL training loop**.

---

## 1. PhiRewardWrapper – what it does and how to use it

### 1.1. Concept

For each agent \(i\) at time \(t\), the wrapper replaces the raw environment reward \(r_i(t)\) with:

\[
u_i(t) \;=\; \cos(\phi_i)\, r_i(t) \;+\; \sin(\phi_i)\, \bar{r}_{-i}(t)
\]

Where:

- \(r_i(t)\) = the original reward for agent \(i\)
- \(\bar{r}_{-i}(t)\) = mean reward of all **other** agents at that step
- \(\phi_i\) = angle (in **radians**) for agent \(i\)

This is implemented in `PhiRewardWrapper`:

- You pass in a list/sequence of `phis` (one value per player, in radians).
- The wrapper intercepts `reset()` and `step()` and returns a new `TimeStep` whose `.reward` field is **already transformed**.

### 1.2. Minimal usage example

```python
from meltingpot import substrate
from meltingpot.utils.substrates.wrappers.phi_reward_wrapper import PhiRewardWrapper
import math

substrate_name = "commons_harvest__closed"
num_players = 7
roles = ("default",) * num_players

# Single phi value for all players, e.g. 30 degrees
phi_deg = 30.0
phi_rad = math.radians(phi_deg)
phis = [phi_rad] * num_players

base_env = substrate.build(substrate_name, roles=roles)
env = PhiRewardWrapper(base_env, phis=phis)

timestep = env.reset()
while not timestep.last():
    # timestep.reward is already u_i(t), not the raw r_i(t)
    actions = ...  # your policy here
    timestep = env.step(actions)

env.close()
```

**Key points:**

- **You do not call `_transform_rewards` yourself** – the wrapper handles it internally.
- Always pass **radians**, not degrees, to `PhiRewardWrapper`.
- If you use the wrapper, **timestep.reward everywhere in your training code is the phi-transformed reward**.

---

## 2. Using a phi curriculum (run_curriculum.py logic) in training

`run_curriculum.py` shows a simple curriculum:

- Over `episodes` episodes, \(\phi\) increases **linearly** from `start_phi_deg` to `end_phi_deg`.
- In the script itself, random actions are used, but you can reuse the **schedule** for your RL agents.

### 2.1. The curriculum function

`run_curriculum.py` defines:

```python
def linear_phi_deg(episode_idx: int,
                   total_episodes: int,
                   start_phi_deg: float,
                   end_phi_deg: float) -> float:
    """Linearly interpolate phi in degrees for a given episode index."""
    ...
```

You can import this and use it inside your training loop.

### 2.2. Example: training loop with a global phi curriculum

In this setup:

- All agents share the same \(\phi\) in a given episode.
- \(\phi\) increases **linearly over episodes**.

```python
import math
from run_curriculum import linear_phi_deg
from meltingpot import substrate
from meltingpot.utils.substrates.wrappers.phi_reward_wrapper import PhiRewardWrapper

substrate_name = "commons_harvest__closed"
num_players = 7
num_episodes = 100
roles = ("default",) * num_players

for episode_idx in range(num_episodes):
    # 1) Compute phi in degrees for this episode
    phi_deg = linear_phi_deg(
        episode_idx=episode_idx,
        total_episodes=num_episodes,
        start_phi_deg=0.0,
        end_phi_deg=45.0,
    )

    # 2) Convert to radians and broadcast to all players
    phi_rad = math.radians(phi_deg)
    phis = [phi_rad] * num_players

    # 3) Build environment and wrap with PhiRewardWrapper
    base_env = substrate.build(substrate_name, roles=roles)
    env = PhiRewardWrapper(base_env, phis=phis)

    # 4) Training episode using your agent(s)
    timestep = env.reset()
    agent.reset_episode()  # if needed by your implementation

    while not timestep.last():
        actions = agent.act(timestep.observation)
        timestep = env.step(actions)

        # timestep.reward is the phi-transformed reward u_i(t)
        agent.learn(timestep)  # or whatever training update you use

    env.close()
```

**Behavior:**

- Episode 0: \(\phi \approx 0^\circ\) (almost pure self-reward).
- Last episode: \(\phi \approx 45^\circ\) (stronger weighting of others’ rewards).
- Within a single episode, \(\phi\) is fixed; only the **next** episode gets a larger \(\phi\).

If you want **different phis per agent**, you can build `phis` however you like (e.g. a list of different angles instead of `[phi_rad] * num_players`).

---

## 3. Logging metrics with checkpoint4_metrics.py during training

`checkpoint4_metrics.py` provides:

- **`gini_coefficient(values)`** – helper function.
- **`MetricsSnapshot` dataclass** – holds:
  - `utilitarian` (U): sum of cumulative returns across players.
  - `sustainability` (S): U divided by episode steps so far.
  - `gini` (E): Gini over cumulative returns.
- **`MetricsPipeline` class** – handles **streaming CSV logging** of these metrics.

### 3.1. What MetricsPipeline expects

```python
from checkpoint4_metrics import MetricsPipeline
```

Constructor:

```python
metrics = MetricsPipeline(csv_path="checkpoint4_metrics.csv")
```

- Opens `checkpoint4_metrics.csv`.
- Writes header:
  - `global_step`, `episode`, `episode_step`, `U`, `S`, `E`.

Main callbacks:

- `on_step(global_step, rewards, cumulative_rewards)`:
  - `rewards`: list/array of per-agent rewards **for this step**  
    (if you are using `PhiRewardWrapper`, this should be the **phi-transformed** rewards).
  - `cumulative_rewards`: list of **per-agent cumulative returns within the current episode**.
  - Returns a `MetricsSnapshot` and writes **one row** into the CSV.
- `on_episode_reset()`:
  - Call once at the **start of each episode** (before the first step).
- `close()`:
  - Call once at the end of training to flush and close the CSV file.

### 3.2. Example: adding metrics to the curriculum training loop

Below is a sketch that combines:

- The phi curriculum (Section 2).
- Phi-wrapped env.
- Streaming metrics logging to `checkpoint4_metrics.csv`.

```python
import math
import numpy as np
from run_curriculum import linear_phi_deg
from meltingpot import substrate
from meltingpot.utils.substrates.wrappers.phi_reward_wrapper import PhiRewardWrapper
from checkpoint4_metrics import MetricsPipeline

substrate_name = "commons_harvest__closed"
num_players = 7
num_episodes = 100
roles = ("default",) * num_players

metrics = MetricsPipeline(csv_path="checkpoint4_metrics.csv")
global_step = 0

for episode_idx in range(num_episodes):
    # 1) Curriculum over phi
    phi_deg = linear_phi_deg(
        episode_idx=episode_idx,
        total_episodes=num_episodes,
        start_phi_deg=0.0,
        end_phi_deg=45.0,
    )
    phi_rad = math.radians(phi_deg)
    phis = [phi_rad] * num_players

    # 2) Env + wrapper
    base_env = substrate.build(substrate_name, roles=roles)
    env = PhiRewardWrapper(base_env, phis=phis)

    # 3) Per-episode book-keeping
    metrics.on_episode_reset()
    cumulative_rewards = np.zeros(num_players, dtype=np.float64)

    timestep = env.reset()
    agent.reset_episode()

    while not timestep.last():
        actions = agent.act(timestep.observation)
        timestep = env.step(actions)
        global_step += 1

        if timestep.reward is not None:
            # timestep.reward is already phi-transformed per agent
            cumulative_rewards += np.asarray(timestep.reward, dtype=np.float64)

            snapshot = metrics.on_step(
                global_step=global_step,
                rewards=timestep.reward,
                cumulative_rewards=cumulative_rewards,
            )

            # Optionally log or print live metrics
            # print("U:", snapshot.utilitarian, "S:", snapshot.sustainability, "E:", snapshot.gini)

        agent.learn(timestep)

    env.close()

# 4) End of training
metrics.close()
```

**Important:**

- `cumulative_rewards` should be **reset to zeros at the start of each episode**.
- You can choose to call `metrics.on_step`:
  - Every step (as above), or
  - Less frequently (e.g. every N steps) – just make sure `cumulative_rewards` is consistent with how often you log.
- The CSV produced by `MetricsPipeline` is separate from any curriculum CSV or plots produced by `run_curriculum.py`.

---

## 4. Recommended usage pattern summary

- **To use phi-shaped rewards in training:**
  - Build env with `substrate.build(...)`.
  - Wrap it with `PhiRewardWrapper(env, phis)` using **radians**.
  - Use `timestep.reward` everywhere as your training signal (it is already transformed).

- **To schedule phi over episodes (curriculum):**
  - Import `linear_phi_deg` from `run_curriculum.py`.
  - Each episode, compute `phi_deg` from `episode_idx`, convert to radians, and rebuild `PhiRewardWrapper` with the new `phis`.

- **To log checkpoint-4 metrics:**
  - Instantiate `MetricsPipeline` once before training.
  - Call `on_episode_reset()` at the start of each episode.
  - Maintain `cumulative_rewards` per agent and call `on_step(...)` during the episode.
  - Call `close()` at the end of training to flush the CSV.

Following these steps, your teammate can drop the examples almost directly into their training loop and have:

- Phi-shaped rewards,
- A phi curriculum over episodes,
- And a CSV of U/S/E metrics tracked throughout training.

