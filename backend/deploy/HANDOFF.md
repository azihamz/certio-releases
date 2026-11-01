# FaceScore: finish deploy + TestFlight (for Claude Code running on Aziz's Mac)

Do all of this yourself. Aziz doesn't want to copy and paste commands. Ask only before spending money.
Keep updates very short.

## 1. Get the backend live on the VPS
- Server: `root@187.124.114.240`, key `~/.ssh/fae_vps`. Other live services run there (Certio, nginx sites, fa-lead-assistant and so on). Don't touch them.
- Run:
  `ssh -i ~/.ssh/fae_vps root@187.124.114.240 "curl -fsSL https://raw.githubusercontent.com/azihamz/certio-releases/claude/vibrant-wozniak-chdvzv/backend/deploy/install.sh | DOMAIN=187.124.114.240 PUBLIC_PORT=8787 bash"`
- Last time the `facescore` systemd service didn't come up on 127.0.0.1:8100, and the tests passed. The script now prints the journal on failure. Read it, fix the cause, and rerun until `curl http://187.124.114.240:8787/health` works from the Mac.
- If port 8787 can't be reached from outside, open it in the Hostinger VPS firewall (hPanel).
- Code is in `/opt/facescore/src/backend`, venv in `/opt/facescore/venv`, data and `.env` in `/opt/facescore/data`.
- Fixes to the backend go to the GitHub repo `azihamz/certio-releases`, branch `claude/vibrant-wozniak-chdvzv`, folder `backend/`.

## 2. Wire the app (this folder, ~/FaceScore)
- Follow `backend/CLAUDE.md` (hard rules + task queue 1, 3, 4, 5). Keep the rules: no Areum copying, synthetic vote pool only, no storing user face images.
- API base URL for now: `http://187.124.114.240:8787`. Put it in one config constant so a domain can replace it later.
- iOS blocks plain HTTP. Add an App Transport Security exception for that IP only (in app.json / app.config `ios.infoPlist`), not a global allow.
- Scan → on-device landmarks → `POST /score/landmarks` (478x3 points, MediaPipe order). Vote screen → `GET /pair`, `GET /face/{id}/image`, `POST /vote`. Report screen → elo, percentile, readout, `ratings` per group.
- The pool is empty right now, so `/pair` returns 503. Show a friendly "no faces yet" state and don't crash.

## 3. TestFlight
- Use EAS (`npx eas-cli build -p ios --profile production` then `npx eas-cli submit -p ios`). Check that eas.json and an Apple account login exist. If anything needs Aziz's Apple login or 2FA, ask him once, clearly.
- Report back: TestFlight build number + what works / what doesn't.
