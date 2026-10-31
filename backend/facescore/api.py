"""FastAPI service.

POST /score        image upload OR raw landmarks -> measurements + Elo + percentile + group breakdown
GET  /pair         next pair of pool faces for a human to vote on
POST /vote         record a human vote
POST /admin/refit  recompute ratings from votes and retrain models

Run: uvicorn facescore.api:app --reload
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

from . import features as F
from . import model as M
from .landmarks import NoFaceError, extract
from .rating import by_group, pick_pair
from .store import Store

DB = os.environ.get("FACESCORE_DB", "facescore.db")
app = FastAPI(title="facescore", version="0.1.0")
store = Store(DB)
models: dict[str, M.GroupModel] = M.load()


class LandmarkIn(BaseModel):
    points: list[list[float]]  # 478 x 3, MediaPipe order (ARKit mapping TBD on iOS side)


class VoteIn(BaseModel):
    winner: str
    loser: str
    voter_group: str = "all"
    voter_id: str | None = None


def _report(pts: np.ndarray) -> dict:
    feats = F.compute(pts)
    x = F.to_vector(feats)
    out = {"measurements": feats, "readout": F.readout(feats), "ratings": {}}
    for g, m in models.items():
        elo = m.predict(x)
        out["ratings"][g] = {"elo": round(elo), "percentile": round(m.percentile(elo), 1)}
    if "all" in out["ratings"]:
        out["elo"] = out["ratings"]["all"]["elo"]
        out["percentile"] = out["ratings"]["all"]["percentile"]
    return out


@app.post("/score")
async def score_image(file: UploadFile = File(...)):
    import cv2

    data = np.frombuffer(await file.read(), np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "not an image")
    try:
        pts, _ = extract(img)
    except NoFaceError:
        raise HTTPException(422, "no face detected")
    return _report(pts)


@app.post("/score/landmarks")
def score_landmarks(body: LandmarkIn):
    pts = np.array(body.points, dtype=np.float32)
    if pts.shape != (478, 3):
        raise HTTPException(400, f"expected 478x3, got {pts.shape}")
    return _report(pts)


@app.get("/pair")
def pair():
    ratings = store.ratings("all") or {f["id"]: 1500.0 for f in store.faces()}
    if len(ratings) < 2:
        raise HTTPException(503, "pool too small")
    a, b = pick_pair(ratings, store.vote_counts())
    return {"a": a, "b": b}


@app.get("/face/{fid}/image")
def face_image(fid: str):
    f = store.face(fid)
    if not f or not f["path"] or not Path(f["path"]).exists():
        raise HTTPException(404)
    return FileResponse(f["path"])


@app.post("/vote")
def vote(v: VoteIn):
    if not store.face(v.winner) or not store.face(v.loser):
        raise HTTPException(404, "unknown face")
    store.add_vote(v.winner, v.loser, v.voter_group, "human", 1.0, v.voter_id)
    return {"ok": True}


@app.post("/admin/refit")
def refit():
    global models
    table = by_group(store.votes())
    store.save_ratings(table)
    models = M.train_all(store)
    M.save(models)
    return {"groups": list(models), "faces_rated": len(table.get("all", {}))}


@app.get("/health")
def health():
    return {"ok": True, "faces": len(store.faces()), "votes": len(store.votes()), "models": list(models)}
