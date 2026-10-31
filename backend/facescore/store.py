"""SQLite store for the face pool, features and votes. Swap for Postgres later."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from .rating import Vote

SCHEMA = """
CREATE TABLE IF NOT EXISTS faces (
  id TEXT PRIMARY KEY,
  path TEXT,
  gender TEXT, age_band TEXT, ethnicity TEXT,
  features TEXT NOT NULL,          -- json dict
  created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS votes (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  winner TEXT NOT NULL, loser TEXT NOT NULL,
  voter_group TEXT NOT NULL DEFAULT 'all',
  source TEXT NOT NULL,            -- 'human' | 'ai:<backend>'
  weight REAL NOT NULL DEFAULT 1.0,
  voter_id TEXT,
  created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS votes_group ON votes(voter_group);
CREATE TABLE IF NOT EXISTS ratings (
  face_id TEXT NOT NULL, voter_group TEXT NOT NULL, elo REAL NOT NULL,
  PRIMARY KEY (face_id, voter_group)
);
"""


class Store:
    def __init__(self, path: str | Path = "facescore.db"):
        self.conn = sqlite3.connect(str(path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)

    # faces
    def add_face(self, fid: str, features: dict, path: str | None = None,
                 gender: str | None = None, age_band: str | None = None, ethnicity: str | None = None):
        self.conn.execute(
            "INSERT OR REPLACE INTO faces(id,path,gender,age_band,ethnicity,features) VALUES (?,?,?,?,?,?)",
            (fid, path, gender, age_band, ethnicity, json.dumps(features)))
        self.conn.commit()

    def faces(self) -> list[dict]:
        rows = self.conn.execute("SELECT * FROM faces").fetchall()
        return [{**dict(r), "features": json.loads(r["features"])} for r in rows]

    def face(self, fid: str) -> dict | None:
        r = self.conn.execute("SELECT * FROM faces WHERE id=?", (fid,)).fetchone()
        return {**dict(r), "features": json.loads(r["features"])} if r else None

    # votes
    def add_vote(self, winner: str, loser: str, group: str = "all", source: str = "human",
                 weight: float = 1.0, voter_id: str | None = None):
        self.conn.execute(
            "INSERT INTO votes(winner,loser,voter_group,source,weight,voter_id) VALUES (?,?,?,?,?,?)",
            (winner, loser, group, source, weight, voter_id))
        self.conn.commit()

    def votes(self) -> list[Vote]:
        rows = self.conn.execute("SELECT winner,loser,voter_group,weight FROM votes").fetchall()
        return [Vote(r["winner"], r["loser"], r["voter_group"], r["weight"]) for r in rows]

    def vote_counts(self) -> dict[str, int]:
        rows = self.conn.execute(
            "SELECT f, COUNT(*) c FROM (SELECT winner f FROM votes UNION ALL SELECT loser FROM votes) GROUP BY f"
        ).fetchall()
        return {r["f"]: r["c"] for r in rows}

    # ratings
    def save_ratings(self, table: dict[str, dict[str, float]]):
        self.conn.execute("DELETE FROM ratings")
        self.conn.executemany(
            "INSERT INTO ratings(face_id,voter_group,elo) VALUES (?,?,?)",
            [(fid, g, elo) for g, r in table.items() for fid, elo in r.items()])
        self.conn.commit()

    def ratings(self, group: str = "all") -> dict[str, float]:
        rows = self.conn.execute("SELECT face_id, elo FROM ratings WHERE voter_group=?", (group,)).fetchall()
        return {r["face_id"]: r["elo"] for r in rows}

    def groups(self) -> list[str]:
        return [r[0] for r in self.conn.execute("SELECT DISTINCT voter_group FROM ratings")]
