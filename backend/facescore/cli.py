"""CLI: ingest a face pool, run AI voting, refit, score an image.

  python -m facescore.cli ingest ./pool --meta pool.csv
  python -m facescore.cli aivote --n 5000 --backend anthropic --group all
  python -m facescore.cli refit
  python -m facescore.cli score photo.jpg
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

from . import features as F
from . import judge as J
from . import model as M
from .landmarks import NoFaceError, extract_from_path
from .rating import by_group, pick_pair
from .store import Store


def cmd_ingest(a):
    store = Store(a.db)
    meta = {}
    if a.meta:
        with open(a.meta) as f:
            for row in csv.DictReader(f):
                meta[row["id"]] = row
    n = ok = 0
    for p in sorted(Path(a.dir).glob("*")):
        if p.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        n += 1
        try:
            pts, _ = extract_from_path(p)
            feats = F.compute(pts)
        except (NoFaceError, ValueError) as e:
            print(f"skip {p.name}: {e}", file=sys.stderr)
            continue
        m = meta.get(p.stem, {})
        store.add_face(p.stem, feats, str(p), m.get("gender"), m.get("age_band"), m.get("ethnicity"))
        ok += 1
    print(f"ingested {ok}/{n}")


def cmd_aivote(a):
    store = Store(a.db)
    faces = {f["id"]: f for f in store.faces()}
    if len(faces) < 2:
        sys.exit("pool too small")
    rng = np.random.default_rng(a.seed)
    for i in range(a.n):
        ratings = store.ratings("all") or {k: 1500.0 for k in faces}
        x, y = pick_pair(ratings, store.vote_counts(), rng)
        if a.backend == "geometry":
            win = J.judge_geometry(faces[x]["features"], faces[y]["features"], a.group)
        else:
            win = J.BACKENDS[a.backend](Path(faces[x]["path"]), Path(faces[y]["path"]), a.group)
        w, l = (x, y) if win == "A" else (y, x)
        store.add_vote(w, l, a.group, f"ai:{a.backend}", J.AI_VOTE_WEIGHT)
        if (i + 1) % a.refit_every == 0:
            store.save_ratings(by_group(store.votes(), min_votes=1))
            print(f"{i+1} votes", file=sys.stderr)
    store.save_ratings(by_group(store.votes(), min_votes=1))
    print(f"done: {a.n} AI votes")


def cmd_refit(a):
    store = Store(a.db)
    table = by_group(store.votes(), min_votes=a.min_votes)
    store.save_ratings(table)
    models = M.train_all(store, min_faces=a.min_faces)
    M.save(models)
    for g, m in models.items():
        print(f"[{g}] faces={len(m.population)} top features: {list(m.importances())[:5]}")


def cmd_score(a):
    models = M.load()
    pts, _ = extract_from_path(a.image)
    feats = F.compute(pts)
    out = {"measurements": feats, "readout": F.readout(feats), "ratings": {}}
    x = F.to_vector(feats)
    for g, m in models.items():
        elo = m.predict(x)
        out["ratings"][g] = {"elo": round(elo), "percentile": round(m.percentile(elo), 1)}
    print(json.dumps(out, indent=2))


def main(argv=None):
    p = argparse.ArgumentParser(prog="facescore")
    p.add_argument("--db", default="facescore.db")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("ingest"); s.add_argument("dir"); s.add_argument("--meta"); s.set_defaults(fn=cmd_ingest)
    s = sub.add_parser("aivote"); s.add_argument("--n", type=int, default=1000)
    s.add_argument("--backend", choices=["anthropic", "openai", "geometry"], default="anthropic")
    s.add_argument("--group", default="all"); s.add_argument("--seed", type=int, default=0)
    s.add_argument("--refit-every", type=int, default=500); s.set_defaults(fn=cmd_aivote)
    s = sub.add_parser("refit"); s.add_argument("--min-votes", type=int, default=200)
    s.add_argument("--min-faces", type=int, default=50); s.set_defaults(fn=cmd_refit)
    s = sub.add_parser("score"); s.add_argument("image"); s.set_defaults(fn=cmd_score)

    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
