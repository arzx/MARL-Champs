import json
import os
import random
import numpy as np
import torch
import shap

from env import CompetitiveTradingEnv
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.callbacks.callbacks import RLlibCallback
from ray import tune
from ray.tune.schedulers import PopulationBasedTraining
from ray.air.integrations.mlflow import MLflowLoggerCallback

NUM_AGENTS = 4

_ADJECTIVES = [
    "Iron", "Fire", "Dark", "Shadow", "Storm", "Thunder", "Frost", "Gold", "Silver", "Crimson",
    "Jade", "Obsidian", "Neon", "Phantom", "Blazing", "Ancient", "Raging", "Swift", "Wild", "Lunar",
    "Solar", "Void", "Cosmic", "Mystic", "Savage", "Emerald", "Crystal", "Steel", "Toxic", "Arcane",
    "Amber", "Azure", "Blaze", "Chrome", "Dusk", "Eternal", "Fallen", "Gilded", "Hollow", "Inferno",
    "Molten", "Noble", "Onyx", "Primal", "Radiant", "Scarlet", "Umbral", "Valiant", "Wicked", "Abyssal",
    "Burning", "Colossal", "Dire", "Dread", "Fierce", "Gloom", "Grim", "Hallow", "Keen", "Lethal",
    "Manic", "Ruin", "Sable", "Titan", "Venom", "Warped", "Wrath", "Zenith", "Ashen", "Bitter",
]
_NOUNS = [
    "Bear", "Dragon", "Wolf", "Eagle", "Phoenix", "Hawk", "Viper", "Panther", "Raven", "Lynx",
    "Cobra", "Falcon", "Tiger", "Shark", "Hydra", "Sphinx", "Gryphon", "Kraken", "Wyvern", "Scorpion",
    "Sentinel", "Wraith", "Specter", "Reaper", "Hunter", "Warrior", "Tempest", "Cipher", "Blade", "Claw",
    "Dagger", "Fang", "Ghost", "Hammer", "Jackal", "Knight", "Lance", "Mage", "Oracle", "Paladin",
    "Ranger", "Saber", "Talon", "Vanguard", "Witch", "Yeti", "Zealot", "Archer", "Bandit", "Crusher",
    "Demon", "Elder", "Titan", "Ravager", "Rogue", "Shaman", "Slayer", "Stalker", "Tyrant", "Warden",
    "Wyrm", "Juggernaut", "Marauder", "Predator", "Revenant", "Berserker", "Colossus", "Invoker", "Duelist", "Omen",
]

def _make_agent_names(n: int) -> dict:
    combos = [f"{a}{b}" for a in _ADJECTIVES for b in _NOUNS]
    chosen = random.sample(combos, n)
    return {f"agent_{i}": chosen[i] for i in range(n)}

WINDOW_SIZE = 10
FEATURE_NAMES = [
    f"t-{WINDOW_SIZE - i}_{feat}"
    for i in range(WINDOW_SIZE)
    for feat in ["bid", "ask"]
]
SHAP_INTERVAL = 5  # compute SHAP every N training iterations
DASHBOARD_STATE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dashboard_state.json")


class DashboardCallback(RLlibCallback):
    def __init__(self):
        super().__init__()
        self._background_obs = None
        self._names = _make_agent_names(NUM_AGENTS)
        self._history = {f"agent_{i}": [] for i in range(NUM_AGENTS)}

    def on_train_result(self, *, algorithm, metrics_logger=None, result, **kwargs):
        iteration = result.get("training_iteration", 0)
        agent_data = {}

        for agent_id in [f"agent_{i}" for i in range(NUM_AGENTS)]:
            env_runners = result.get("env_runners", {})
            episode_return = (
                (env_runners.get("module_episode_returns_mean") or {}).get(agent_id)
                or (env_runners.get("agent_episode_returns_mean") or {}).get(agent_id)
                or env_runners.get("episode_return_mean", 0.0)
            )
            self._history[agent_id].append(float(episode_return))

            top_features = []
            if iteration % SHAP_INTERVAL == 0:
                try:
                    module = algorithm.get_module(agent_id)
                    top_features = self._compute_shap(module)
                except Exception as e:
                    print(f"[SHAP] {agent_id} iteration {iteration} failed: {e}")

            prev_returns = self._history[agent_id]
            agent_data[agent_id] = {
                "name": self._names[agent_id],
                "avatar_url": f"https://api.dicebear.com/9.x/pixel-art/png?seed={self._names[agent_id]}&size=80",
                "episode_return_mean": float(episode_return),
                "prev_episode_return_mean": prev_returns[-2] if len(prev_returns) >= 2 else None,
                "top_features": top_features,
                "history": prev_returns[-50:],
            }

        with open(DASHBOARD_STATE_FILE, "w") as f:
            json.dump({"iteration": iteration, "agents": agent_data}, f)

    def _compute_shap(self, module):
        if self._background_obs is None:
            import gymnasium as gym
            import gym_anytrading
            from gym_anytrading.datasets import FOREX_EURUSD_1H_ASK

            env = gym.wrappers.FlattenObservation(
                gym.make("forex-v0", df=FOREX_EURUSD_1H_ASK, window_size=WINDOW_SIZE,
                         frame_bound=(10, 300), unit_side="right")
            )
            obs_list = []
            obs, _ = env.reset()
            for _ in range(100):
                obs_list.append(obs.astype(np.float32))
                obs, _, term, trunc, _ = env.step(env.action_space.sample())
                if term or trunc:
                    break
            self._background_obs = torch.FloatTensor(np.array(obs_list))

        def _predict(obs_np):
            with torch.no_grad():
                t = torch.FloatTensor(obs_np)
                return module.forward_inference({"obs": t})["action_dist_inputs"].numpy()

        background_np = self._background_obs[:50].numpy()
        test_np = self._background_obs[50:70].numpy()
        # PermutationExplainer is model-agnostic — no gradients needed
        explainer = shap.PermutationExplainer(_predict, background_np)
        sv = explainer(test_np)
        # sv.values: [n_samples, n_features, n_actions] — index 1 = BUY
        mean_abs = np.abs(sv.values[:, :, 1]).mean(axis=0)
        top_idx = np.argsort(mean_abs)[::-1][:8]
        return [[FEATURE_NAMES[i], float(mean_abs[i])] for i in top_idx]

tune.register_env("competitive_trading", lambda cfg: CompetitiveTradingEnv(cfg))

pbt = PopulationBasedTraining(
    time_attr="training_iteration",
    metric="env_runners/episode_return_mean",
    mode="max",
    perturbation_interval=10,
    hyperparam_mutations={
        "lr": tune.loguniform(1e-5, 1e-2),
        "gamma": tune.uniform(0.95, 0.99),
    },
)

config = (
    PPOConfig()
    .environment("competitive_trading", env_config={"num_agents": NUM_AGENTS})
    .multi_agent(
        policies={f"agent_{i}": (None, None, None, {}) for i in range(NUM_AGENTS)},
        policy_mapping_fn=lambda agent_id, *args, **kwargs: agent_id,
    )
    .callbacks(DashboardCallback)
)

tuner = tune.Tuner(
    "PPO",
    param_space=config.to_dict(),
    tune_config=tune.TuneConfig(scheduler=pbt, num_samples=2),
    run_config=tune.RunConfig(
        stop={"training_iteration": 100},
        callbacks=[
            MLflowLoggerCallback(
                experiment_name="competitive_trading_pbt",
                tracking_uri=os.path.abspath("mlruns"),
                tags={"env": "forex-v0", "algo": "PPO", "num_agents": str(NUM_AGENTS)},
                save_artifact=True,
                log_params_on_trial_end={"log_system_metrics": False},
            )
        ],
    ),
)
results = tuner.fit()

best = results.get_best_result(metric="env_runners/episode_return_mean", mode="max")
print("\nBest trial checkpoint:", best.checkpoint.path)
print("Run SHAP analysis with:")
print(f"  python analysis.py {best.checkpoint.path}")
