"""End-to-end smoke test. Uses one real detected face (skimage sample image) and
jitters its landmarks to fake a pool of 200 faces, then runs votes -> ratings -> model -> score."""
import numpy as np
import pytest

from facescore import features as F
from facescore import model as M
from facescore.judge import judge_geometry
from facescore.landmarks import extract, normalise
from facescore.rating import Vote, bradley_terry, by_group, elo_online, pick_pair
from facescore.store import Store


@pytest.fixture(scope="module")
def base_pts():
    import cv2
    from skimage import data

    img = cv2.cvtColor(data.astronaut(), cv2.COLOR_RGB2BGR)
    pts, _ = extract(img)
    assert pts.shape == (478, 3)
    return pts


def test_features_sane(base_pts):
    f = F.compute(base_pts)
    assert set(f) == set(F.FEATURE_NAMES)
    assert abs(f["upper_third"] + f["middle_third"] + f["lower_third"] - 1) < 1e-6
    assert 90 < f["jaw_angle_deg"] < 170
    assert 0 <= f["symmetry_rms"] < 0.3


def test_normalise_invariant(base_pts):
    a = normalise(base_pts)
    # scale + translate + roll the raw points; normalised result should match
    th = 0.3
    rot = np.array([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1]])
    moved = (base_pts * 2.5) @ rot.T + np.array([100, -40, 7])
    b = normalise(moved)
    assert np.allclose(a, b, atol=1e-4)


def test_bradley_terry_recovers_order():
    rng = np.random.default_rng(0)
    true = {f"f{i}": v for i, v in enumerate(np.linspace(1200, 1800, 30))}
    votes = []
    ids = list(true)
    for _ in range(4000):
        a, b = rng.choice(ids, 2, replace=False)
        pa = 1 / (1 + 10 ** ((true[b] - true[a]) / 400))
        w, l = (a, b) if rng.random() < pa else (b, a)
        votes.append(Vote(w, l))
    est = bradley_terry(votes)
    r = np.corrcoef([true[i] for i in ids], [est[i] for i in ids])[0, 1]
    assert r > 0.95
    est2 = elo_online(votes)
    assert np.corrcoef([true[i] for i in ids], [est2[i] for i in ids])[0, 1] > 0.85


def test_full_pipeline(tmp_path, base_pts):
    rng = np.random.default_rng(1)
    store = Store(tmp_path / "t.db")
    # fake pool: jitter the real landmarks
    for i in range(120):
        pts = base_pts + rng.normal(0, 3.0, base_pts.shape).astype(np.float32)
        store.add_face(f"p{i}", F.compute(pts), gender=rng.choice(["m", "f"]))
    faces = {f["id"]: f for f in store.faces()}
    for _ in range(3000):
        ratings = store.ratings("all") or {k: 1500.0 for k in faces}
        a, b = pick_pair(ratings, store.vote_counts(), rng)
        win = judge_geometry(faces[a]["features"], faces[b]["features"])
        w, l = (a, b) if win == "A" else (b, a)
        store.add_vote(w, l, rng.choice(["f_25_34", "m_25_34"]), "ai:geometry", 0.3)
    table = by_group(store.votes(), min_votes=100)
    assert {"all", "f_25_34", "m_25_34"} <= set(table)
    store.save_ratings(table)
    models = M.train_all(store, min_faces=50)
    assert "all" in models
    x = F.to_vector(F.compute(base_pts))
    elo = models["all"].predict(x)
    assert 1000 < elo < 2000
    assert 0 <= models["all"].percentile(elo) <= 100
