from __future__ import annotations

from dataclasses import dataclass

import gymnasium as gym


@dataclass
class EpisodeStats:
    steps: int = 0
    total_reward: float = 0.0
    start_equity: float = 0.0
    end_equity: float = 0.0


class EpisodeStatsWrapper(gym.Wrapper):
    """Collects simple episode stats into info['episode'] on termination."""

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        self._stats = EpisodeStats()

    def reset(self, **kwargs):  # type: ignore[override]
        obs, info = self.env.reset(**kwargs)
        self._stats = EpisodeStats(steps=0, total_reward=0.0)
        eq = info.get("equity", None)
        if isinstance(eq, (int, float)):
            self._stats.start_equity = float(eq)
        return obs, info

    def step(self, action):  # type: ignore[override]
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._stats.steps += 1
        self._stats.total_reward += float(reward)
        eq = info.get("equity", None)
        if isinstance(eq, (int, float)):
            self._stats.end_equity = float(eq)

        if terminated or truncated:
            info = dict(info)
            info["episode"] = {
                "steps": int(self._stats.steps),
                "reward": float(self._stats.total_reward),
                "start_equity": float(self._stats.start_equity),
                "end_equity": float(self._stats.end_equity),
            }
        return obs, reward, terminated, truncated, info

