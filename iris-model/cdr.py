from __future__ import annotations

from dataclasses import dataclass, replace
from collections import deque
from typing import Iterable
import random
import numpy as np
import torch


@dataclass(frozen=True)
class CDRConfig:
    """Class-dependent reward configuration.

    Defaults/presets below reproduce Table 5 of Wu et al. (MNRAS 547,
    stag349, 2026). The paper describes an online network, epsilon-style
    exploration decay, experience replay, multiple full episodes, and an
    immediate TP/TN/FP/FN reward. Because the published equation writes a
    reward-weighted log objective rather than an optimizer sign convention,
    :func:`cdr_loss` minimizes the negative reward-weighted log-likelihood.
    """
    tp: float
    tn: float
    fp: float
    fn: float
    batch_size: int
    episodes: int
    lr: float
    exploration_decay: float
    replay_size: int
    epsilon_start: float = 1.0
    epsilon_min: float = 0.01


PAPER_CDR_PRESETS = {
    # Table 5, magnetogram point-in-time model.
    "cdr_cnn": CDRConfig(tp=4, tn=10, fp=-42, fn=-15,
                         batch_size=8, episodes=8, lr=0.003,
                         exploration_decay=0.7, replay_size=10_000),
    # Table 5, magnetogram time-series model.
    "cdr_cnn_bilstm": CDRConfig(tp=7, tn=4, fp=-24, fn=-8,
                                batch_size=8, episodes=10, lr=0.001,
                                exploration_decay=0.7, replay_size=1_000),
    # Table 5, knowledge-informed feature model.
    "cdr_transformer": CDRConfig(tp=10, tn=4, fp=-20, fn=-15,
                                 batch_size=49, episodes=8, lr=0.00015,
                                 exploration_decay=0.99, replay_size=1_000),
}


@dataclass
class Experience:
    index: int
    action: int
    reward: float
    label: int


class IndexReplayBuffer:
    """Replay memory storing dataset indices rather than huge image tensors."""
    def __init__(self, capacity: int, seed: int = 2026):
        self.capacity = int(capacity)
        self.data: deque[Experience] = deque(maxlen=self.capacity)
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.data)

    def append(self, exp: Experience) -> None:
        self.data.append(exp)

    def sample(self, n: int) -> list[Experience]:
        n = min(int(n), len(self.data))
        return self.rng.sample(list(self.data), n)


def epsilon_after_episode(cfg: CDRConfig, episode_index: int) -> float:
    """Exploration schedule using the paper's stated multiplicative decay."""
    return max(cfg.epsilon_min, cfg.epsilon_start * (cfg.exploration_decay ** episode_index))


def choose_actions(logits: torch.Tensor, epsilon: float, rng: np.random.Generator) -> torch.Tensor:
    """Epsilon-greedy binary action selection.

    The paper lists an 'exploration decay' but does not publish an explicit
    action-selection equation. Epsilon-greedy is the standard DQN-compatible
    interpretation. We therefore log this as a transparent implementation
    assumption rather than presenting it as a verbatim paper detail.
    """
    greedy = (torch.sigmoid(logits.detach()) >= 0.5).long().cpu().numpy()
    explore = rng.random(len(greedy)) < float(epsilon)
    random_actions = rng.integers(0, 2, size=len(greedy), dtype=np.int64)
    out = np.where(explore, random_actions, greedy)
    return torch.from_numpy(out).to(logits.device, dtype=torch.long)


def immediate_rewards(labels: torch.Tensor, actions: torch.Tensor, cfg: CDRConfig) -> torch.Tensor:
    y = labels.long()
    a = actions.long()
    r = torch.empty_like(labels, dtype=torch.float32)
    r[(y == 1) & (a == 1)] = cfg.tp
    r[(y == 0) & (a == 0)] = cfg.tn
    r[(y == 0) & (a == 1)] = cfg.fp
    r[(y == 1) & (a == 0)] = cfg.fn
    return r


def cdr_loss(logits: torch.Tensor, actions: torch.Tensor, rewards: torch.Tensor) -> torch.Tensor:
    """Negative reward-weighted selected-action log probability.

    For action 1 the selected probability is sigmoid(logit); for action 0 it is
    1-sigmoid(logit). Positive reward therefore increases the probability of the
    selected action, while negative reward suppresses it. This implements the
    direction implied by equations (4)-(5) of Wu et al. while using the usual
    minimization convention.
    """
    p = torch.sigmoid(logits).clamp(1e-6, 1 - 1e-6)
    pa = torch.where(actions.bool(), p, 1 - p)
    return -(rewards * torch.log(pa)).mean()


def paper_reward_sensitivity() -> dict[str, list[CDRConfig]]:
    """Reward perturbation grid reported for CDR-Transformer-10.

    Base rewards are TP=10, TN=4, FP=-20, FN=-15. The paper perturbs one
    component at a time over TP 5..15, TN 1..7, FP -15..-25 and FN -10..-20.
    """
    base = PAPER_CDR_PRESETS["cdr_transformer"]
    return {
        "tp": [replace(base, tp=float(v)) for v in range(5, 16)],
        "tn": [replace(base, tn=float(v)) for v in range(1, 8)],
        "fp": [replace(base, fp=float(v)) for v in range(-15, -26, -1)],
        "fn": [replace(base, fn=float(v)) for v in range(-10, -21, -1)],
    }


def reward_counts(labels: torch.Tensor, actions: torch.Tensor) -> dict[str, int]:
    y = labels.long(); a = actions.long()
    return {
        "tp": int(((y == 1) & (a == 1)).sum().item()),
        "tn": int(((y == 0) & (a == 0)).sum().item()),
        "fp": int(((y == 0) & (a == 1)).sum().item()),
        "fn": int(((y == 1) & (a == 0)).sum().item()),
    }
