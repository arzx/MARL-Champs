import numpy as np
import gymnasium as gym
from gym_anytrading.datasets import FOREX_EURUSD_1H_ASK
from ray.rllib.env.multi_agent_env import MultiAgentEnv

class CompetitiveTradingEnv(MultiAgentEnv):
    def __init__(self, config):
        import gym_anytrading
        self._n_agents = config.get("num_agents", 4)
        self.agent_ids = {f"agent_{i}" for i in range(self._n_agents)}
        self._agent_ids = self.agent_ids

        # One shared base env for market data/stepping
        self._base_env = gym.make(
            'forex-v0',
            df=FOREX_EURUSD_1H_ASK,
            window_size=10,
            frame_bound=(10, 300),
            unit_side='right'
        )
        flat_env = gym.wrappers.FlattenObservation(self._base_env)
        _s = flat_env.observation_space
        single_obs_space = gym.spaces.Box(
            low=_s.low.astype(np.float32),
            high=_s.high.astype(np.float32),
            dtype=np.float32,
        )
        single_act_space = self._base_env.action_space
        self.observation_space = gym.spaces.Dict({
            aid: single_obs_space for aid in self.agent_ids
        })
        self.action_space = gym.spaces.Dict({
            aid: single_act_space for aid in self.agent_ids
        })

        # Per-agent state
        self._positions = {}
        self._profits = {}

        super().__init__()

    def reset(self, *, seed=None, options=None):
        obs, info = self._base_env.reset(seed=seed)
        flat_obs = obs.flatten().astype(np.float32)
        self._positions = {aid: 0 for aid in self.agent_ids}
        self._profits = {aid: 0.0 for aid in self.agent_ids}

        return {aid: flat_obs for aid in self.agent_ids}, {}


    def step(self, action_dict):
        # Step each agent's action independently but on the same market tick
        # Use the base env for the shared market step (one step per tick)
        obs, base_reward, terminated, truncated, _ = self._base_env.step(
            action_dict["agent_0"]  # market advances once per tick
        )
        flat_obs = obs.flatten().astype(np.float32)
        done = terminated or truncated

        # Compute per-agent profits using their individual actions.
        # Use base_reward (per-step price change), NOT info["total_reward"]
        # which is cumulative and grows unboundedly, causing NaN gradients.
        agent_profits = {}
        for aid, action in action_dict.items():
            # gym_anytrading: action 0=sell/short, 1=buy/long
            profit = base_reward if action == 1 else -base_reward
            self._profits[aid] += profit
            agent_profits[aid] = profit

        # Competitive reward: relative to group mean
        mean_profit = np.mean(list(agent_profits.values()))
        rewards = {aid: agent_profits[aid] - mean_profit for aid in action_dict}

        terminateds = {aid: done for aid in action_dict}
        terminateds["__all__"] = done
        truncateds = {aid: False for aid in action_dict}
        truncateds["__all__"] = False
        obs_dict = {aid: flat_obs for aid in action_dict}
        infos = {aid: {} for aid in action_dict}

        return obs_dict, rewards, terminateds, truncateds, infos