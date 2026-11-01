#!/usr/bin/env bash
# Install or update the facescore API on an Ubuntu/Debian VPS (run as root).
#
#   curl -fsSL https://raw.githubusercontent.com/azihamz/certio-releases/claude/vibrant-wozniak-chdvzv/backend/deploy/install.sh | bash
#   ... | DOMAIN=api.example.com bash     # also add an nginx site for that domain
#   ... | DOMAIN=1.2.3.4 PUBLIC_PORT=8080 bash   # serve on the bare IP, on its own port
#
# The API listens on 127.0.0.1 only. Nothing is public unless DOMAIN is set.
# Existing nginx sites are never edited. A new site is kept only if `nginx -t` passes.
set -euo pipefail

REPO=${REPO:-https://github.com/azihamz/certio-releases.git}
BRANCH=${BRANCH:-claude/vibrant-wozniak-chdvzv}
APP_DIR=${APP_DIR:-/opt/facescore}
PORT=${PORT:-8100}
DOMAIN=${DOMAIN:-}
PUBLIC_PORT=${PUBLIC_PORT:-80}
SVC=facescore

[ "$(id -u)" = 0 ] || { echo "run as root"; exit 1; }

echo "==> system packages"
export DEBIAN_FRONTEND=noninteractive
apt-get update -q
apt-get install -y -q git curl python3 python3-venv python3-pip libegl1 libgles2 libgl1 libgomp1 iproute2

echo "==> code ($BRANCH)"
if [ -d "$APP_DIR/src/.git" ]; then
  git -C "$APP_DIR/src" fetch -q origin "$BRANCH"
  git -C "$APP_DIR/src" checkout -q -B "$BRANCH" "origin/$BRANCH"
else
  mkdir -p "$APP_DIR"
  git clone -q --depth 1 -b "$BRANCH" "$REPO" "$APP_DIR/src"
fi
BE="$APP_DIR/src/backend"
mkdir -p "$APP_DIR/data" "$BE/models"

echo "==> python env"
[ -d "$APP_DIR/venv" ] || python3 -m venv "$APP_DIR/venv"
"$APP_DIR/venv/bin/pip" install -q --upgrade pip
"$APP_DIR/venv/bin/pip" install -q -r "$BE/requirements.txt"

echo "==> landmark model"
[ -f "$BE/models/face_landmarker.task" ] || (cd "$BE" && ./scripts/get_models.sh)
[ -f "$APP_DIR/data/.env" ] || cp "$BE/.env.example" "$APP_DIR/data/.env"
# .env wins over the unit's Environment=, so pin the DB to the writable data dir here
sed -i "s#^FACESCORE_DB=.*#FACESCORE_DB=$APP_DIR/data/facescore.db#" "$APP_DIR/data/.env"
grep -q '^FACESCORE_DB=' "$APP_DIR/data/.env" || echo "FACESCORE_DB=$APP_DIR/data/facescore.db" >> "$APP_DIR/data/.env"
chmod 600 "$APP_DIR/data/.env"

echo "==> tests"
(cd "$BE" && FACESCORE_DB="$APP_DIR/data/test.db" "$APP_DIR/venv/bin/pytest" -q) && rm -f "$APP_DIR/data/test.db"

echo "==> systemd service on 127.0.0.1:$PORT"
if ss -ltn "sport = :$PORT" | grep -q LISTEN && ! systemctl is-active -q "$SVC"; then
  echo "port $PORT is already used by something else. Rerun with PORT=<free port>"; exit 1
fi
id -u facescore >/dev/null 2>&1 || useradd --system --home "$APP_DIR" --shell /usr/sbin/nologin facescore
chown -R facescore:facescore "$APP_DIR/data" "$BE/models"
cat > /etc/systemd/system/$SVC.service <<UNIT
[Unit]
Description=facescore API
After=network.target

[Service]
User=facescore
WorkingDirectory=$BE
EnvironmentFile=$APP_DIR/data/.env
Environment=FACESCORE_DB=$APP_DIR/data/facescore.db
Environment=HOME=$APP_DIR/data
Environment=MPLCONFIGDIR=$APP_DIR/data/.mpl
ExecStart=$APP_DIR/venv/bin/uvicorn facescore.api:app --host 127.0.0.1 --port $PORT --workers 1
Restart=on-failure

[Install]
WantedBy=multi-user.target
UNIT
systemctl daemon-reload
systemctl enable -q "$SVC"
systemctl reset-failed "$SVC" 2>/dev/null || true
systemctl restart "$SVC"
UP=
for i in $(seq 1 90); do
  curl -fs "http://127.0.0.1:$PORT/health" >/dev/null && { UP=1; break; }
  systemctl is-failed -q "$SVC" && break
  sleep 1
done
if [ -z "$UP" ]; then
  echo "service did not start. Last log lines:"
  journalctl -u "$SVC" -n 40 --no-pager || true
  exit 1
fi
curl -fsS "http://127.0.0.1:$PORT/health"; echo

if [ -n "$DOMAIN" ]; then
  command -v nginx >/dev/null || { echo "nginx not installed; skipping public site"; exit 0; }
  echo "==> nginx site for $DOMAIN:$PUBLIC_PORT"
  if [ "$PUBLIC_PORT" != 80 ] && ss -ltn "sport = :$PUBLIC_PORT" | grep -q LISTEN && [ ! -e /etc/nginx/sites-enabled/facescore ]; then
    echo "port $PUBLIC_PORT is already in use. Rerun with PUBLIC_PORT=<free port>"; exit 1
  fi
  SITE=/etc/nginx/sites-available/facescore
  sed -e "s/__DOMAIN__/$DOMAIN/g" -e "s/__PORT__/$PORT/g" -e "s/__LISTEN__/$PUBLIC_PORT/g" "$BE/deploy/nginx-facescore.conf" > "$SITE"
  ln -sf "$SITE" /etc/nginx/sites-enabled/facescore
  if nginx -t; then
    systemctl reload nginx
    if [ "$PUBLIC_PORT" != 80 ] && command -v ufw >/dev/null && ufw status | grep -q active; then ufw allow "$PUBLIC_PORT/tcp"; fi
    URL="http://$DOMAIN"; [ "$PUBLIC_PORT" = 80 ] || URL="$URL:$PUBLIC_PORT"
    curl -fsS "$URL/health" && echo && echo "live at $URL"
  else
    rm -f /etc/nginx/sites-enabled/facescore
    echo "nginx test failed; site removed, existing sites untouched"; exit 1
  fi
fi
echo "done"
