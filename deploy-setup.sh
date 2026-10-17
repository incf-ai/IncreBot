#!/usr/bin/env bash
# Root setup for the increbot system service, following deploy.md §1.
#
#   sudo ./deploy-setup.sh
#
# Creates the increbot user, the discord-token secret (from .env), builds the
# image in your rootless podman and copies it into root's, and writes the
# Quadlet unit from deploy.md with this repo's path and the real UID/GIDs. Safe
# to re-run, and the way to deploy a new version: the user and secret are kept,
# the image and unit are refreshed, then the service is (re)started. Stop the
# devcontainer bot before the first run (deploy.md §2): two bots on one token
# both answer.
set -euo pipefail
[[ $EUID -eq 0 && -n "${SUDO_USER:-}" ]] || { echo "run as: sudo ./deploy-setup.sh" >&2; exit 1; }
cd "$(dirname "$0")"
REPO=$PWD
HF=/mnt/storage/huggingface
UNIT=/etc/containers/systemd/increbot.container

# 1.1 the user
if ! id increbot >/dev/null 2>&1; then
    useradd --system --no-create-home --home-dir /nonexistent \
        --shell /usr/sbin/nologin --groups render increbot
fi
UID_N=$(id -u increbot); GID_N=$(id -g increbot); RENDER_N=$(getent group render | cut -d: -f3)
echo "increbot uid=$UID_N gid=$GID_N, render gid=$RENDER_N"

# 1.4 the secret, from .env (never printed)
if podman secret inspect discord-token >/dev/null 2>&1; then
    echo "secret discord-token exists; leaving it"
else
    sed -n 's/^DISCORD_TOKEN=//p' .env | tr -d '\r\n' | podman secret create discord-token - >/dev/null
    echo "secret discord-token created"
fi

# The image: built rootless as you (cached layers make this quick when nothing
# changed), then copied into root's store. The image it replaces stays as
# increbot:previous for rollback (deploy.md §3).
as_user() { sudo -u "$SUDO_USER" -H env XDG_RUNTIME_DIR="/run/user/$SUDO_UID" "$@"; }
as_user podman build -f deploy/Containerfile -t increbot:latest \
    --build-arg GIT_COMMIT="$(as_user git rev-parse --short=12 HEAD 2>/dev/null || echo unknown)" .
NEW=$(as_user podman image inspect --format '{{.Id}}' localhost/increbot:latest)
OLD=$(podman image inspect --format '{{.Id}}' localhost/increbot:latest 2>/dev/null || true)
if [[ "$NEW" == "$OLD" ]]; then
    echo "image unchanged"
else
    [[ -n "$OLD" ]] && podman tag localhost/increbot:latest localhost/increbot:previous
    as_user podman save localhost/increbot:latest | podman load
    echo "image updated"
fi

# 1.5 the unit, taken from deploy.md with the real paths and numbers
mkdir -p "$(dirname "$UNIT")"
awk '/^```ini/{f=1;next} /^```/{if(f)exit} f' deploy.md \
  | sed -e "s#/path/to/increbot#$REPO#g" \
        -e "s#/mnt/storage/huggingface#$HF#g" \
        -e "s#^User=.*#User=$UID_N#" \
        -e "s#^Group=.*#Group=$GID_N#" \
        -e "s#--group-add=[0-9]*#--group-add=$RENDER_N#" > "$UNIT"
echo "wrote $UNIT"
systemctl daemon-reload
systemctl cat increbot.service >/dev/null && echo "increbot.service generated"
systemctl restart increbot
echo "increbot restarted; logs: journalctl -u increbot -f"
