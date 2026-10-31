# facescore

Backend for a face-measurement and rating app. Working name. Rebrand before launch.

## What it does

1. **Measures** a face. MediaPipe gives 478 3D landmarks. Those turn into 22 geometry features: facial thirds, jaw angle, canthal tilt, symmetry, FWHR and ratios.
2. **Collects pairwise votes** ("which is more attractive, A or B") from humans in the app, plus AI votes to bootstrap.
3. **Rates** every pool face with Bradley-Terry / Elo. Ratings are fitted per voter group (e.g. `f_25_34`) so you get the demographic breakdown.
4. **Learns** features → Elo with LightGBM. Any new user's face gets an Elo, a percentile and a per-group breakdown.

## Setup

```bash
pip install -r requirements.txt
./scripts/get_models.sh          # downloads MediaPipe face_landmarker.task
pytest -q                        # 4 tests, ~10s
```

## Run

```bash
# 1. put synthetic faces in ./pool (jpg/png). Optional pool.csv: id,gender,age_band,ethnicity
python -m facescore.cli ingest ./pool --meta pool.csv

# 2. bootstrap with AI votes (needs ANTHROPIC_API_KEY or OPENAI_API_KEY)
python -m facescore.cli aivote --n 5000 --backend anthropic --group all
python -m facescore.cli aivote --n 2000 --backend anthropic --group f_25_34

# 3. fit ratings + train models
python -m facescore.cli refit

# 4. score a photo
python -m facescore.cli score me.jpg

# API
uvicorn facescore.api:app --reload
```

`--backend geometry` is a fake judge with no API calls. Use it only to test the pipeline. Never ship a model trained on it.

## API

| Method | Path | |
|---|---|---|
| POST | `/score` | multipart image → report |
| POST | `/score/landmarks` | `{points: 478x3}` → report (for the iOS app to send on-device landmarks, no photo leaves the phone) |
| GET | `/pair` | next pair for a human to vote on |
| GET | `/face/{id}/image` | pool face image |
| POST | `/vote` | `{winner, loser, voter_group, voter_id}` |
| POST | `/admin/refit` | recompute ratings + retrain |
| GET | `/health` | counts |

## Weighting

AI votes go in at weight 0.3 (`judge.AI_VOTE_WEIGHT`) and human votes at 1.0. Humans take over as their votes build up.

## Rules (legal / UK GDPR)

- **Vote pool = synthetic faces only.** Never show one user's face to another.
- **User scans:** prefer `/score/landmarks` from on-device extraction. If you use `/score`, don't store the image.
- Explicit biometric consent screen, DPIA, privacy policy before launch.
- Don't market "trained on N million human votes" until it's true.

## Next

- [x] Synthetic face pool generator (`python -m facescore.generate`)
- [ ] ARKit → MediaPipe landmark index mapping for iOS
- [ ] iOS app (SwiftUI): consent → scan → vote-25-for-free-scan → report → paywall (RevenueCat)
- [ ] Auth + rate limiting on `/vote` and `/admin/*`
- [ ] Postgres instead of SQLite
- [ ] Band thresholds in `features.REPORT_BANDS` are placeholders. Calibrate them on the real pool
