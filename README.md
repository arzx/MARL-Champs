# MARL-Champs

A competitive Multi-Agent Reinforcement Learning (MARL) system where multiple agents trade on the FOREX market simultaneously and compete to achieve the best returns. Training is optimized using Population-Based Training (PBT) and results are tracked with MLflow.

## Overview

Each agent independently decides to buy or sell at every market tick, observing the same market data. Agents are rewarded relative to the group mean — beating the average is rewarded, underperforming is penalized. This zero-sum competitive setup encourages agents to discover diverse and profitable trading strategies.

## Architecture

```text
MARL-Champs/
├── main.py       # Training entry point: PBT config, Tune setup, MLflow logging
├── env.py        # CompetitiveTradingEnv — custom RLlib MultiAgentEnv
└── README.md
```

### `env.py` — CompetitiveTradingEnv

A custom `MultiAgentEnv` built on top of `gym_anytrading`'s `forex-v0` environment.

- **Agents:** N independent agents (configurable via `num_agents`, default 4)
- **Observation:** Flattened market window of shape `(window_size * features,)` — shared across all agents
- **Action space:** Discrete — `0` = sell/short, `1` = buy/long
- **Reward:** Zero-sum competitive signal: `reward_i = profit_i - mean(profit_all_agents)`
- **Market data:** EUR/USD 1H ask prices (`FOREX_EURUSD_1H_ASK`)

### `main.py` — Training

- **Algorithm:** PPO (Ray RLlib new API stack)
- **Scheduler:** Population-Based Training (`PopulationBasedTraining`) — periodically copies weights from better-performing trials into worse ones and mutates hyperparameters
- **Population size:** Controlled by `NUM_AGENTS` (default 4 PBT trials)
- **Hyperparameter search:** `lr` ∈ [1e-5, 1e-2], `gamma` ∈ [0.95, 0.99]
- **Logging:** MLflow — each PBT trial is a separate run under the `competitive_trading_pbt` experiment

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install ray[rllib,tune] gymnasium gym-anytrading mlflow tqdm torch
```

## Running

```bash
python main.py
```

Training runs 100 iterations across 2 PBT trials (configurable). Results are saved to `~/ray_results/` and logged to MLflow.

## Monitoring

### MLflow UI

```bash
mlflow ui --backend-store-uri mlruns
# open http://localhost:5000
```

Tracked metrics per iteration:

- `env_runners/episode_return_mean` — mean episode return across agents
- `env_runners/episode_return_min/max`
- `learners/default_policy/total_loss`
- `learners/default_policy/entropy`
- `learners/default_policy/policy_loss`

### TensorBoard

```bash
tensorboard --logdir ~/ray_results
```

## Configuration

| Parameter | Location | Default | Description |
| --- | --- | --- | --- |
| `NUM_AGENTS` | `main.py` | `4` | Number of PBT population members |
| `num_agents` | `env_config` | `4` | Number of competing agents per env |
| `perturbation_interval` | PBT config | `10` | Iterations between PBT perturbations |
| `training_iteration` stop | `RunConfig` | `100` | Total training iterations |
| `window_size` | `env.py` | `10` | Market observation window |
| `frame_bound` | `env.py` | `(10, 300)` | Data slice used for training |

## Notes

- MLflow does not allow overwriting logged parameters. If you restart training, delete the `mlruns/` directory first: `rm -rf mlruns`
- The `import gym_anytrading` inside `make_trading_env` and `CompetitiveTradingEnv.__init__` is intentional — Ray worker subprocesses do not inherit module-level imports from the main process
