"""AI judge: bootstraps the vote table before humans arrive.

Pluggable backends. Each takes two image paths and returns 'A' or 'B'.
Votes from here get weight < 1 so human votes dominate once they exist.

Backends:
  * anthropic  — Claude vision via the Anthropic API (ANTHROPIC_API_KEY)
  * openai     — any OpenAI-compatible chat endpoint (OPENAI_API_KEY, OPENAI_BASE_URL)
  * geometry   — no API; uses a hand-written heuristic over features. Only for
                 smoke-testing the pipeline, not for real data.
"""
from __future__ import annotations

import base64
import os
import random
from pathlib import Path

import httpx

AI_VOTE_WEIGHT = 0.3

PROMPT = (
    "You are helping calibrate a facial-proportions measurement app. "
    "Two synthetic (computer-generated, not real) faces are shown. "
    "Reply with exactly one character, A or B: which face most people in the group "
    "'{group}' would judge as more conventionally attractive? No explanation."
)


def _b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode()


def _mime(path: Path) -> str:
    return "image/png" if path.suffix.lower() == ".png" else "image/jpeg"


def judge_anthropic(a: Path, b: Path, group: str = "all",
                    model: str = "claude-sonnet-4-5") -> str:
    key = os.environ["ANTHROPIC_API_KEY"]
    body = {
        "model": model, "max_tokens": 2,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": "Face A:"},
            {"type": "image", "source": {"type": "base64", "media_type": _mime(a), "data": _b64(a)}},
            {"type": "text", "text": "Face B:"},
            {"type": "image", "source": {"type": "base64", "media_type": _mime(b), "data": _b64(b)}},
            {"type": "text", "text": PROMPT.format(group=group)},
        ]}],
    }
    r = httpx.post("https://api.anthropic.com/v1/messages", json=body, timeout=60,
                   headers={"x-api-key": key, "anthropic-version": "2023-06-01"})
    r.raise_for_status()
    txt = r.json()["content"][0]["text"].strip().upper()
    return _parse(txt)


def judge_openai(a: Path, b: Path, group: str = "all",
                 model: str | None = None) -> str:
    key = os.environ["OPENAI_API_KEY"]
    base = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
    model = model or os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    body = {
        "model": model, "max_tokens": 2,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": "Face A:"},
            {"type": "image_url", "image_url": {"url": f"data:{_mime(a)};base64,{_b64(a)}"}},
            {"type": "text", "text": "Face B:"},
            {"type": "image_url", "image_url": {"url": f"data:{_mime(b)};base64,{_b64(b)}"}},
            {"type": "text", "text": PROMPT.format(group=group)},
        ]}],
    }
    r = httpx.post(f"{base}/chat/completions", json=body, timeout=60,
                   headers={"Authorization": f"Bearer {key}"})
    r.raise_for_status()
    return _parse(r.json()["choices"][0]["message"]["content"].strip().upper())


def judge_geometry(fa: dict, fb: dict, group: str = "all") -> str:
    """Smoke-test judge over feature dicts. Crude: rewards symmetry and
    balanced thirds, with noise. Do NOT ship a model trained on this."""
    def s(f):
        thirds = -abs(f["upper_third"] - 1 / 3) - abs(f["middle_third"] - 1 / 3) - abs(f["lower_third"] - 1 / 3)
        return thirds * 3 - f["symmetry_rms"] * 10 + random.gauss(0, 0.05)
    return "A" if s(fa) > s(fb) else "B"


def _parse(txt: str) -> str:
    if txt.startswith("A"):
        return "A"
    if txt.startswith("B"):
        return "B"
    raise ValueError(f"judge returned {txt!r}")


BACKENDS = {"anthropic": judge_anthropic, "openai": judge_openai}
