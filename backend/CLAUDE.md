# FaceScore backend: context for Claude Code

## Who / what
- Owner: Aziz Hamza. Busy and wants things done, not explained. Keep updates short and plain.
- Product: an iPhone face-measurement and rating app, a rebuild of the *idea* behind Areum (Bounty Studio). It's a legal clean-room rebuild: our own code, name, data and model.
- The existing Expo/React Native app lives in the **parent folder** (`../`, i.e. `~/FaceScore`). This `backend/` folder is the Python scoring service.

## Hard rules (do not break)
1. **Never copy Areum.** No name, logo, screenshots, store text, UI copied pixel-for-pixel, "Seonho-1", decompiling, or scraping their app or API.
2. **Vote pool = synthetic faces only.** Never show one real user's face to another user.
3. **User face = biometric data (UK GDPR special category).** Prefer on-device landmark extraction and `POST /score/landmarks`. If `/score` (image upload) is used, never write the image to disk or DB.
4. Don't claim "trained on N million human votes" in any copy until it's true. AI votes are weighted 0.3 (`judge.AI_VOTE_WEIGHT`).
5. Never commit API keys. Use `.env` (gitignored).

## Layout
```
facescore/
  landmarks.py  MediaPipe 478-pt extraction + normalise()
  features.py   22 geometry metrics + plain-English readout bands
  rating.py     Bradley-Terry / Elo, per-group fit, pair picker
  judge.py      AI judge (anthropic/openai) + geometry smoke judge
  generate.py   synthetic face pool generator (replicate/fal/dry)
  store.py      SQLite: faces, votes, ratings
  model.py      LightGBM features→Elo per voter group
  api.py        FastAPI: /score /score/landmarks /pair /vote /admin/refit /health
  cli.py        ingest / aivote / refit / score
tests/          pytest, must stay green
```

## Commands
```bash
./setup.sh                                          # venv, deps, model, tests
source .venv/bin/activate
pytest -q
python -m facescore.generate --n 700 --backend dry  # plan + cost only
python -m facescore.generate --n 700 --backend replicate --out pool
python -m facescore.cli ingest pool --meta pool.csv
python -m facescore.cli aivote --n 5000 --backend anthropic
python -m facescore.cli refit
python -m facescore.cli score some_photo.jpg
uvicorn facescore.api:app --reload --port 8000
```

## Env vars (.env)
`ANTHROPIC_API_KEY` or `OPENAI_API_KEY` (AI judge), `REPLICATE_API_TOKEN` or `FAL_KEY` (face generation), `FACESCORE_DB` (default facescore.db).

## Status
- Done and tested: landmarks, features, rating, model, API, CLI, generator.
- `geometry` judge is a fake for tests only. Never train a shipped model on it.
- `features.REPORT_BANDS` thresholds are placeholders. Calibrate them on the real pool (use the pool's 15th to 85th percentile per metric).

## Task queue (do in order, check with Aziz before spending money)
1. Read the parent Expo app (`../`). Summarise what's built: screens, any existing scoring, how it scans.
2. Run `./setup.sh` and confirm the tests pass.
3. Wire the Expo scan screen to the backend. Get on-device landmarks (expo-camera + a face-landmark lib, or a native ARKit module). Map them to MediaPipe order and POST them to `/score/landmarks`. If the app already has its own scoring, ask Aziz which to keep.
4. Build a vote screen: GET `/pair`, show both faces, POST `/vote`. Give a free scan after 25 votes.
5. Build a report screen: Elo, percentile, readout labels, per-group breakdown.
6. Add auth and rate limiting on `/vote` and `/admin/*`. Move to Postgres for production.
7. Deploy the backend (Fly.io / Railway / a £5 VPS) and set the app's API base URL.
8. Paywall via RevenueCat. Show *some* result before the paywall, because Areum's reviews are wrecked by "scan then pay".
