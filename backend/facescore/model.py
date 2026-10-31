"""Learn features -> Elo. One model per voter group.

LightGBM on 22 geometry features. Small, fast, runs on a £5 VPS. Upgrade to
an image-embedding model once you have >1M human votes.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

from .features import FEATURE_NAMES, to_vector
from .store import Store

MODEL_DIR = Path("models")


class GroupModel:
    def __init__(self, group: str):
        self.group = group
        self.model = None
        self.population: np.ndarray = np.array([])  # training Elo distribution for percentiles

    def fit(self, X: np.ndarray, y: np.ndarray):
        import lightgbm as lgb

        self.model = lgb.LGBMRegressor(
            n_estimators=400, learning_rate=0.03, num_leaves=15,
            min_child_samples=10, subsample=0.8, colsample_bytree=0.8,
            reg_lambda=1.0, verbose=-1,
        )
        self.model.fit(X, y)
        self.population = np.sort(y)
        return self

    def predict(self, x: np.ndarray) -> float:
        return float(self.model.predict(x.reshape(1, -1))[0])

    def percentile(self, elo: float) -> float:
        if self.population.size == 0:
            return 50.0
        return float((self.population < elo).mean() * 100)

    def importances(self) -> dict[str, float]:
        imp = self.model.feature_importances_
        return dict(sorted(zip(FEATURE_NAMES, imp.tolist()), key=lambda t: -t[1]))


def train_all(store: Store, min_faces: int = 50) -> dict[str, GroupModel]:
    faces = {f["id"]: f for f in store.faces()}
    models = {}
    for g in store.groups():
        ratings = store.ratings(g)
        ids = [i for i in ratings if i in faces]
        if len(ids) < min_faces:
            continue
        X = np.stack([to_vector(faces[i]["features"]) for i in ids])
        y = np.array([ratings[i] for i in ids])
        models[g] = GroupModel(g).fit(X, y)
    return models


def save(models: dict[str, GroupModel], d: Path = MODEL_DIR):
    d.mkdir(parents=True, exist_ok=True)
    with open(d / "group_models.pkl", "wb") as f:
        pickle.dump(models, f)


def load(d: Path = MODEL_DIR) -> dict[str, GroupModel]:
    p = d / "group_models.pkl"
    if not p.exists():
        return {}
    with open(p, "rb") as f:
        return pickle.load(f)
