# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

SustainLC: a benchmark for RL-based control of liquid cooling in HPC data centers, built around a
Modelica-derived FMU digital twin of ORNL's Frontier supercomputer (5-cabinet reduced model). Part of the
ExaDigiT digital twin effort (ORNL/HPE). RL agents control CDU pump speed, coolant temperature setpoints,
and blade-group valves, plus a cooling tower setpoint, to trade off server temperature against cooling
energy use.

## Environment setup

```bash
conda env create -f environment.yml
conda activate sustain-lc
```

Requires `pyfmi` (conda-forge, needs a working FMI co-simulation runtime) and PyTorch with CUDA 11.8. There
is no pip-only path — `pyfmi` is conda-only. `requirements.txt` is a partial/secondary list (used mainly for
docs building via Sphinx) and is not sufficient on its own to run the environment or training scripts.

There are no lint/test/build tooling configs in this repo (no pytest, no linter config, no CI). Validate
changes by actually running training briefly or the evaluation notebooks — there is no automated test suite.

## Running things

Train (hardcoded hyperparameters inside each script's `train()` function — there is no argparse/CLI flag
interface despite what the root [readme.md](readme.md) quickstart table implies; edit the constants at the
top of `train()` directly to change hyperparameters):

```bash
python train_multiagent_ca_ppo.py    # per-cabinet + cooling-tower agents, single-headed continuous/discrete PPO
python train_mh_ma_ca_ppo.py         # same, but CDU-cabinet agent uses a multi-head (top-level + valve) policy
tensorboard --logdir MA_CA_PPO_logs      # or MH_MA_CA_PPO_logs, per script
```

Evaluate: run `evaluate_ma_ca_ppo.ipynb` (CA_PPO) or `evaluate_mh_ma_ca_ppo.ipynb` (MultiHead_CA_PPO) in
Jupyter. Policy distillation into decision trees (VIPER, Bastani et al. 2018) is done via
`policy_distillation.ipynb`.

Checkpoints are read/written under `MA_CA_PPO_preTrained/<env_name>/` and
`MH_MA_CA_PPO_preTrained/<env_name>/`.

## Architecture

### FMU digital twin (`frontier_env.py`, `mh_frontier_env.py`)

`LC_Frontier_5Cabinet_4_17_25.fmu` is a compiled Modelica co-simulation model (FMI 2.0) of the liquid
cooling loop: CDUs, blade-group cabinets, cooling tower. It is loaded via `pyfmi` and stepped with
`do_step()`; agent actions are pushed in via `fmu_action_vars`, observations pulled out via
`fmu_observation_vars`. One agent `step()` advances the FMU multiple internal `sim_time_step`s to cover the
configured `step_size`.

- **`SmallFrontierModel`** (`frontier_env.py`) — the base `gymnasium.Env`. Dict observation/action spaces
  keyed per cabinet (`cdu-cabinet-1..5`) and per cooling tower (`cooling-tower-1`). Continuous actions
  (pump speed, temp setpoint, valve openings) are scaled to `[-1, 1]`; the cooling tower action is
  `Discrete`, decoded through `cooling_tower_action_decoding` into a setpoint delta.
- **`MH_SmallFrontierModel`** (`mh_frontier_env.py`) — wraps `SmallFrontierModel`, splitting each CDU
  cabinet's action into a `top-level` (pump/temp, `[-1,1]`) and `valve-level` (`[0,1]`, softmax-able) head,
  for use with the multi-head policy.
- **`exogenous_variable_generator` / `_2`** (`frontier_env.py`) — read `input_04-07-24.csv` (15s-interval
  historical telemetry) and produce the exogenous power-draw and outside-wetbulb-temperature trajectory fed
  into the FMU each episode. `_2` (`exogen_gen_v=2`) is the version used by both training scripts; supports
  more cabinets, smoothing, and optional heat-reuse-unit (HRU) energy diversion (`use_hru`).
- Reward = negative aggregate blade-group temperature (`R_blade`) plus negative total cooling-tower power
  (`R_coolingtower`); reward shaping variants are selected via `use_reward_shaping='reward_shaping_v0|1|2'`.

### PPO agents (`ca_ppo.py`, `multihead_ca_ppo.py`, `multiagent_ca_ppo.py`)

- **`CA_PPO`** (`ca_ppo.py`) — single-head PPO (`ActorCritic`: MLP actor + critic, Tanh for continuous /
  Softmax for discrete action heads). "CA" = Centralized Action: one shared policy instance is
  vmapped/looped across `num_centralized_actions` homogeneous agents (e.g. one policy shared by all 5
  cabinets, keeping a separate `RolloutBuffer` per action index in `buffer_dict`).
  Vanilla PPO-clip implementation (this repo's `ca_ppo.py` is the reference PPO — not a stub).
- **`MultiHead_CA_PPO`** (`multihead_ca_ppo.py`) — same centralized-action idea, but
  `MultiHeadActorCritic` has a shared backbone with two action heads (`top-level` Gaussian, `valve-level`
  Dirichlet/softplus-based) matching `MH_SmallFrontierModel`'s split action space.
- **`multiagent_ppo_dtde`** (`multiagent_ca_ppo.py`) — decentralized-training-decentralized-execution
  wrapper holding one agent per logical role: `'CDUCAB'` (CA_PPO or MultiHead_CA_PPO, depending on
  `agent_type`) and `'CT'` (always CA_PPO, discrete). This is the top-level object the training scripts
  drive; `select_action`/`update`/`save`/`load` all take an `agent_id` (`'CDUCAB'` or `'CT'`).

### Training scripts (`train_multiagent_ca_ppo.py`, `train_mh_ma_ca_ppo.py`)

Both scripts: instantiate the env, build an `agent_mdp_dict` describing per-role state/action dims and PPO
hyperparameters, wrap it in `multiagent_ppo_dtde`, then run a manual rollout/update loop (not using
gymnasium's `VectorEnv` or any RL library — this is a from-scratch PPO loop). `batchify_observations`
splits the env's per-cabinet/per-tower obs dict into the `{'CDUCAB': [...], 'CT': [...]}` batches the agents
expect; `categorize_actions` does the inverse for actions before they're passed to `env.step()`. Logging
goes to CSV (`MA_CA_PPO_logs/` / `MH_MA_CA_PPO_logs/`) and TensorBoard.

### Model building (AutoCSM, out of this repo)

Custom FMUs (beyond the bundled `LC_Frontier_5Cabinet_4_17_25.fmu`) are built by a separate toolchain:
Modelica Buildings library + TRANSFORM + `datacenterCoolingModel` (ORNL) libraries, assembled by
`AutoCSM` (`code.ornl.gov/exadigit/AutoCSM`) from a JSON hierarchical description into a Modelica model,
compiled to FMU via Dymola/OpenModelica, then wrapped by a `frontier_env.py`-style Gym env. This repo does
not contain AutoCSM itself — see [readme.md](readme.md) "Advanced AutoCSM usage" for the external repos.

### Docs

Sphinx docs source is in `docs_src/source/` (built output checked into `docs/`, published via GitHub
Pages). `docs_src/source/agentic_ai/` covers the separate LLM-agent explainability layer described in the
readme's "Agentic LLM-Based Digital Twin" section (LLaMA/Qwen-based explainable control agents) — that
system's code is not in this repository.
