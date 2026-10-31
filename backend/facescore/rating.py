"""Turn pairwise votes into ratings.

Two estimators:
  * `elo_online`  — incremental, good for live updates as votes arrive.
  * `bradley_terry` — batch MLE, the proper way to fit the whole vote table.

A vote is (winner_id, loser_id, voter_group). Group is a demographic bucket
string like "f_25_34_white" or "all". Fitting per group gives the
"how each demographic would rate you" breakdown.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Vote:
    winner: str
    loser: str
    group: str = "all"
    weight: float = 1.0  # human votes can be weighted above AI votes


def elo_online(votes: list[Vote], k: float = 24.0, base: float = 1500.0) -> dict[str, float]:
    r: dict[str, float] = defaultdict(lambda: base)
    for v in votes:
        rw, rl = r[v.winner], r[v.loser]
        exp_w = 1 / (1 + 10 ** ((rl - rw) / 400))
        delta = k * v.weight * (1 - exp_w)
        r[v.winner] = rw + delta
        r[v.loser] = rl - delta
    return dict(r)


def bradley_terry(votes: list[Vote], iters: int = 200, tol: float = 1e-6,
                  prior: float = 0.5) -> dict[str, float]:
    """MM algorithm for Bradley-Terry with a small uniform prior so unseen
    items don't blow up. Returns Elo-scaled ratings (mean 1500, 400/log10 scale)."""
    ids = sorted({v.winner for v in votes} | {v.loser for v in votes})
    idx = {i: n for n, i in enumerate(ids)}
    n = len(ids)
    if n == 0:
        return {}
    wins = np.full(n, prior)
    pair_w = defaultdict(float)  # (i,j) i<j -> total weight of comparisons
    for v in votes:
        a, b = idx[v.winner], idx[v.loser]
        wins[a] += v.weight
        key = (a, b) if a < b else (b, a)
        pair_w[key] += v.weight
    # prior: every item "played" one virtual game vs the mean
    p = np.ones(n)
    keys = np.array(list(pair_w.keys()))
    wts = np.array(list(pair_w.values()))
    for _ in range(iters):
        denom = np.full(n, prior * 2)  # virtual games
        inv = wts / (p[keys[:, 0]] + p[keys[:, 1]])
        np.add.at(denom, keys[:, 0], inv)
        np.add.at(denom, keys[:, 1], inv)
        new_p = wins / denom
        new_p /= np.exp(np.log(new_p).mean())  # geometric-mean normalise
        if np.abs(np.log(new_p) - np.log(p)).max() < tol:
            p = new_p
            break
        p = new_p
    elo = 1500 + 400 * np.log10(p)
    return {ids[i]: float(elo[i]) for i in range(n)}


def by_group(votes: list[Vote], min_votes: int = 200) -> dict[str, dict[str, float]]:
    """Fit a rating table per voter group, plus 'all'."""
    groups: dict[str, list[Vote]] = defaultdict(list)
    for v in votes:
        groups[v.group].append(v)
        groups["all"].append(v)
    return {g: bradley_terry(vs) for g, vs in groups.items() if len(vs) >= min_votes or g == "all"}


def percentile(score: float, population: np.ndarray) -> float:
    return float((population < score).mean() * 100)


def pick_pair(ratings: dict[str, float], uncertainty: dict[str, int] | None = None,
              rng: np.random.Generator | None = None, window: float = 150.0) -> tuple[str, str]:
    """Choose an informative pair: an under-sampled anchor plus an opponent
    within `window` Elo points. Same idea as chess pairing."""
    rng = rng or np.random.default_rng()
    ids = list(ratings)
    if uncertainty:
        counts = np.array([uncertainty.get(i, 0) for i in ids], dtype=float)
        p = 1 / (1 + counts)
        a = ids[rng.choice(len(ids), p=p / p.sum())]
    else:
        a = ids[rng.integers(len(ids))]
    ra = ratings[a]
    cands = [i for i in ids if i != a and abs(ratings[i] - ra) <= window]
    if not cands:
        cands = [i for i in ids if i != a]
    b = cands[rng.integers(len(cands))]
    return (a, b) if rng.random() < 0.5 else (b, a)
