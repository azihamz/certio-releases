"""Synthetic face pool generator.

Builds a balanced grid of attributes (gender x age band x ethnicity x lighting
x expression), writes one prompt per face, calls an image API, and writes
pool/<id>.png + pool.csv in the format `cli ingest --meta` expects.

Every face is computer-generated. No real people, so no consent problem in
the vote pool.

Backends:
  * replicate — REPLICATE_API_TOKEN, default model black-forest-labs/flux-schnell
  * fal       — FAL_KEY, default model fal-ai/flux/schnell
  * dry       — writes prompts + csv only, no images (for checking the grid/cost)

  python -m facescore.generate --n 2000 --backend replicate --out pool
  python -m facescore.generate --n 2000 --backend dry --out pool
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import os
import random
import sys
import time
from pathlib import Path

import httpx

GENDERS = ["man", "woman"]
AGE_BANDS = {"18_24": "21", "25_34": "29", "35_44": "39", "45_54": "49", "55_64": "59"}
ETHNICITIES = {
    "white": "white European",
    "black": "Black African or Caribbean",
    "south_asian": "South Asian",
    "east_asian": "East Asian",
    "middle_eastern": "Middle Eastern",
    "latino": "Latino",
    "mixed": "mixed-ethnicity",
}
LIGHTING = ["soft daylight", "even studio light", "overcast window light"]
EXPRESSION = ["neutral expression", "slight relaxed smile"]
HAIR = ["short hair", "medium-length hair", "long hair", "hair tied back"]

PROMPT = (
    "Passport-style front-facing headshot photo of a {age}-year-old {eth} {gender}, "
    "{expr}, {hair}, looking directly at the camera, head level and centred, "
    "full face visible including forehead, ears and chin, no glasses, no hat, "
    "plain light grey background, {light}, sharp focus, realistic skin texture, "
    "no makeup filter, no retouching, 85mm lens"
)
NEGATIVE = "tilted head, profile, side view, sunglasses, hands on face, cropped chin, cartoon, illustration, watermark, text"

# rough per-image costs (USD) for budgeting, check provider pricing
COST = {"replicate": 0.003, "fal": 0.003, "dry": 0.0}


def build_grid(n: int, seed: int = 0) -> list[dict]:
    """Balanced across gender x age x ethnicity; lighting/expr/hair random."""
    rng = random.Random(seed)
    cells = list(itertools.product(GENDERS, AGE_BANDS, ETHNICITIES))
    rows = []
    for i in range(n):
        g, a, e = cells[i % len(cells)]
        row = {
            "gender": "m" if g == "man" else "f",
            "age_band": a,
            "ethnicity": e,
            "lighting": rng.choice(LIGHTING),
            "expression": rng.choice(EXPRESSION),
            "hair": rng.choice(HAIR),
            "seed": rng.randrange(2**31),
        }
        row["prompt"] = PROMPT.format(age=AGE_BANDS[a], eth=ETHNICITIES[e], gender=g,
                                      expr=row["expression"], hair=row["hair"], light=row["lighting"])
        row["id"] = "s" + hashlib.sha1(f"{row['prompt']}|{row['seed']}".encode()).hexdigest()[:10]
        rows.append(row)
    rng.shuffle(rows)
    return rows


def _download(url: str, dest: Path):
    r = httpx.get(url, timeout=60, follow_redirects=True)
    r.raise_for_status()
    dest.write_bytes(r.content)


def gen_replicate(row: dict, dest: Path, model: str | None = None):
    token = os.environ["REPLICATE_API_TOKEN"]
    model = model or os.environ.get("REPLICATE_MODEL", "black-forest-labs/flux-schnell")
    r = httpx.post(
        f"https://api.replicate.com/v1/models/{model}/predictions",
        headers={"Authorization": f"Bearer {token}", "Prefer": "wait"},
        json={"input": {"prompt": row["prompt"], "seed": row["seed"], "aspect_ratio": "1:1",
                        "output_format": "png", "num_outputs": 1}},
        timeout=120,
    )
    r.raise_for_status()
    out = r.json().get("output")
    url = out[0] if isinstance(out, list) else out
    if not url:
        raise RuntimeError(f"no output: {r.text[:200]}")
    _download(url, dest)


def gen_fal(row: dict, dest: Path, model: str | None = None):
    key = os.environ["FAL_KEY"]
    model = model or os.environ.get("FAL_MODEL", "fal-ai/flux/schnell")
    r = httpx.post(
        f"https://fal.run/{model}",
        headers={"Authorization": f"Key {key}"},
        json={"prompt": row["prompt"], "seed": row["seed"], "image_size": "square_hd",
              "num_images": 1, "enable_safety_checker": True},
        timeout=120,
    )
    r.raise_for_status()
    _download(r.json()["images"][0]["url"], dest)


BACKENDS = {"replicate": gen_replicate, "fal": gen_fal}
CSV_FIELDS = ["id", "gender", "age_band", "ethnicity", "lighting", "expression", "hair", "seed", "prompt"]


def run(n: int, out: Path, backend: str, seed: int = 0, retries: int = 2) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    rows = build_grid(n, seed)
    meta = out.parent / f"{out.name}.csv"
    with open(meta, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        w.writerows({k: r[k] for k in CSV_FIELDS} for r in rows)
    print(f"{n} faces planned, est cost ${n * COST[backend]:.2f}, meta -> {meta}", file=sys.stderr)
    if backend == "dry":
        return meta
    fn = BACKENDS[backend]
    done = fail = 0
    for r in rows:
        dest = out / f"{r['id']}.png"
        if dest.exists():
            done += 1
            continue
        for attempt in range(retries + 1):
            try:
                fn(r, dest)
                done += 1
                break
            except Exception as e:  # noqa: BLE001
                if attempt == retries:
                    fail += 1
                    print(f"fail {r['id']}: {e}", file=sys.stderr)
                else:
                    time.sleep(2 ** attempt)
        if (done + fail) % 50 == 0:
            print(f"{done} ok / {fail} failed", file=sys.stderr)
    print(f"done: {done} ok, {fail} failed", file=sys.stderr)
    return meta


def main(argv=None):
    p = argparse.ArgumentParser(prog="facescore.generate")
    p.add_argument("--n", type=int, default=700)
    p.add_argument("--out", default="pool")
    p.add_argument("--backend", choices=["replicate", "fal", "dry"], default="dry")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)
    run(a.n, Path(a.out), a.backend, a.seed)


if __name__ == "__main__":
    main()
