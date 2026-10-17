# Deploying floob (increbot) as a service

This runs the Discord bot in its own small container as a **system** service:
rootful podman, managed by systemd through **Quadlet**, separate from the
devcontainer. You keep developing in the devcontainer. The bot only changes when
you rebuild the image and restart it.

What you get:

- floob starts on boot and restarts after a crash, with nobody logged in;
- it runs a fixed, built binary, not whatever `target/release` currently holds;
- it runs in a locked-down container: a dedicated unprivileged user (not root, not
  you), read-only filesystem, no capabilities, model files mounted read-only, the
  Discord token as a podman secret, no dev tools, and no Claude Code;
- logs go to the system journal.

Everything below runs **on the host**, **not** inside the devcontainer. Commands
that need root are shown with `sudo`. Rootful podman has its own image store and
secrets, separate from the rootless podman the devcontainer uses, so the image and
the secret must be created with `sudo podman`.

Written for podman 4.9 (Ubuntu 24.04). Newer podman accepts the same unit.

---

## 0. Variables used below

Set these in your host shell first. All later commands use them.

```sh
REPO=/path/to/increbot              # the folder VS Code opens as /workspace
HF=/mnt/storage/huggingface         # the shared Hugging Face cache (holds hub/)
```

Check that the model files are in the cache:

```sh
ls "$HF"/hub/models--TheBloke--LLaMA2-13B-Tiefighter-GGUF/snapshots/*/llama2-13b-tiefighter.Q4_K_M.gguf
ls "$HF"/hub/models--hf-internal-testing--llama-tokenizer/snapshots/*/tokenizer.json
```

The cache is mounted read-only, so the service never downloads anything. If a file
is missing, the bot exits with an error instead.

---

## 1. One-time setup

[`deploy-setup.sh`](deploy-setup.sh) does everything below for you. Run it with
`sudo ./deploy-setup.sh`. It does 1.1 and 1.4 once, builds the image (1.3) in
your rootless podman and copies it into root's store, and writes the unit (1.5)
with this repo's path and the real numbers, then restarts the service. The sections explain what it does,
and how to do it by hand.

### 1.1 A user for the bot

The container runs as this user. It owns nothing and can't log in. It is in the
`render` group, which owns the GPU render nodes (`/dev/dri/renderD*`).

```sh
sudo useradd --system --no-create-home --home-dir /nonexistent \
  --shell /usr/sbin/nologin --groups render increbot
id -u increbot; id -g increbot; getent group render   # note the three numbers
```

The unit in 1.5 needs these numbers, because the image has no `/etc/passwd` entry
for `increbot` and Quadlet doesn't look names up on the host.

The bot reads its files through the "other" permission bits, so they must stay
world-readable: the HF cache (`hf download` creates it that way),
`config.local.rsn` and `prompts/` (the persona files). Check with `ls -l "$REPO/config.local.rsn"
"$REPO"/prompts`. A home directory that isn't world-readable doesn't matter,
because podman mounts the files directly.

### 1.2 Keep the build context small: `$REPO/.containerignore`

Without this file, `podman build` would send `target/` (tens of GB), `.env` and your
local config into the build.

```gitignore
target
tmp
logs
.git
.env
config.local.rsn
data
*.gguf
```

### 1.3 The image: `$REPO/deploy/Containerfile`

The first stage builds the bot with Rust 1.98. The devcontainer uses
`devcontainers/rust:1`, which follows the latest stable release, so bump this tag
when the devcontainer's `rustc --version` moves on. The repo's `.cargo/config.toml`
(clang, mold, `target-cpu=native`) applies, which is right because the image is
built and run on this machine. The second stage is a slim Debian with only the
binary and the Vulkan driver.

Both stages are Debian 13 (trixie): its Mesa 25.0 is the first with RDNA4
(Radeon AI PRO R9700) support in RADV, which the GPU backend uses
([gpu-plan.md](gpu-plan.md)). `glslc` compiles the GPU shaders at build time;
`INCREBOT_GPU_SHADERS_REQUIRED=1` makes a missing `glslc` a build error instead of
a silently CPU-only binary.

```dockerfile
FROM docker.io/library/rust:1.98-trixie AS build
RUN apt-get update \
 && apt-get install -y --no-install-recommends clang mold glslc \
 && rm -rf /var/lib/apt/lists/*
ENV INCREBOT_GPU_SHADERS_REQUIRED=1
WORKDIR /src
# .git isn't in the build context, so the commit comes in as a build argument
# (shown by /version, /stats and --version).
ARG GIT_COMMIT=unknown
ENV INCREBOT_GIT_COMMIT=$GIT_COMMIT
COPY . .
# Cache mounts keep crates and build output between image builds.
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/src/target \
    cargo build --release --locked -p increbot \
 && cp target/release/increbot /increbot

FROM docker.io/library/debian:trixie-slim
# mesa-vulkan-drivers: RADV, the Vulkan driver for the R9700s; libvulkan1: the
# loader the bot opens at run time.
RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates \
      mesa-vulkan-drivers libvulkan1 \
 && rm -rf /var/lib/apt/lists/*
COPY --from=build /increbot /usr/local/bin/increbot
# config.rsn and prompts/ (the persona files) are mounted here at run time.
WORKDIR /app
ENV HF_HOME=/hf
ENTRYPOINT ["increbot", "serve", "--config", "/app/config.rsn"]
```

### 1.4 The Discord token as a podman secret

Paste the token when prompted, then press Enter and Ctrl-D. It won't appear on
screen or in shell history. (The trailing newline is fine; the bot trims it.)

```sh
sudo podman secret create discord-token -
```

The secret is stored root-only under `/var/lib/containers/storage/secrets`. To
replace it later (for example, after resetting the token in the Developer Portal),
run `sudo podman secret rm discord-token` and create it again.

### 1.5 The service: `/etc/containers/systemd/increbot.container`

Replace `/path/to/increbot` with your `$REPO`, `/mnt/storage/huggingface` with your
`$HF` if it differs, and the `User=`, `Group=` and `--group-add` numbers with the
ones from 1.1. Quadlet doesn't expand shell variables. Create the file with
`sudoedit /etc/containers/systemd/increbot.container`.

```ini
[Unit]
Description=floob (increbot Discord bot)
Wants=network-online.target
After=network-online.target
# Each start that reaches Discord costs a gateway login, and more than ~1,000 a
# day makes Discord reset the bot token. The backoff below is what keeps a crash
# loop under that. This limit is a backstop: systemd counts every start, manual
# ones included, so it is loose enough for deploys. Even if every start counted,
# 10 in 30 min is under 500 logins a day.
StartLimitIntervalSec=1800
StartLimitBurst=10

[Container]
Image=localhost/increbot:latest
ContainerName=increbot
# The increbot user and group from 1.1 (numeric: the image doesn't know the names).
User=999
Group=999
Secret=discord-token,type=env,target=DISCORD_TOKEN
# Model cache, config and persona, all read-only.
Volume=/mnt/storage/huggingface:/hf:ro
Volume=/path/to/increbot/config.local.rsn:/app/config.rsn:ro
Volume=/path/to/increbot/prompts:/app/prompts:ro
# The /opt-out list (discord.optout_file, data/optouts.json), the only thing the
# bot writes. A named volume: podman creates it on first start and keeps it across
# deploys, rollbacks and container removal. U: owned by User/Group above.
Volume=increbot-data:/app/data:U
# 22 threads as in the devcontainer: 8 pinned P-cores for decode, plus siblings
# and E-cores for prefill (design.md §4).
Environment=INCREBOT_THREADS=22
Environment=RUST_LOG=info
# Plain log lines: without this, the journal gets ANSI color codes.
Environment=NO_COLOR=1
# Mesa can't write its shader cache to the read-only root and warns once per GPU
# context. The shaders compile in well under a second anyway.
Environment=MESA_SHADER_CACHE_DISABLE=true
# Read-only root; podman still mounts a writable tmpfs on /tmp, /var/tmp and /run.
ReadOnly=true
# No NoNewPrivileges=true: on Ubuntu 24.04 it keeps crun from switching the
# container to podman's AppArmor profile, and the leftover "crun" profile denies
# network sockets (DNS fails). With every capability dropped, a setuid binary
# couldn't gain anything anyway, and the image has none (deploy/Containerfile).
DropCapability=ALL
# The GPU backend (gpu-plan.md): the GPUs' device nodes (all of /dev/dri, because
# node numbers change between boots; the bot picks cards by PCI address).
AddDevice=/dev/dri
# ~8.5 GB of weights + up to 9 conversation caches (~0.8 GB each) + headroom.
# --group-add: the render group's GID (1.1), which owns /dev/dri/renderD*.
PodmanArgs=--memory=24g --group-add=992

[Service]
Restart=on-failure
# Automatic restarts back off from 10 s to 10 min, so a crash loop settles at
# about 6 logins an hour and never gives up: after an outage outside the bot, it
# comes back by itself. A manual start resets the backoff.
RestartSec=10
RestartSteps=5
RestartMaxDelaySec=600
# Exit code 2 is a configuration error (a chat channel outside the allowlisted
# guilds). Restarting can't fix it and only spends gateway logins.
RestartPreventExitStatus=2
# The bot shuts down cleanly on SIGTERM within ~5 s.
TimeoutStopSec=30

[Install]
WantedBy=multi-user.target
```

There is no `--pid=host`. With it, the bot could name the other programs that use
a GPU in `/stats` and the logs. But a separate user can't read other users'
`/proc/<pid>/fdinfo` anyway, so it would expose the host's process list for
nothing. Those programs are still detected, just unnamed.

**On SELinux systems** (Fedora, RHEL): add `z` to the volume options, e.g.
`…/prompts:/app/prompts:ro,z`, or the container can't read them. Use `z`, not `Z`:
the HF cache is shared with the devcontainer. Ubuntu uses AppArmor and needs
nothing here.

---

## 2. Build and start

**First, stop the bot in the devcontainer** (Ctrl-C in its `run-bot.sh` terminal).
Never run both at once: two connections with the same token both receive every
message, so floob would answer twice.

```sh
cd "$REPO"
sudo ./deploy-setup.sh                  # builds, copies the image, writes the unit, daemon-reload, starts
journalctl -u increbot -f               # wait for "connected to Discord"
```

The first build compiles everything and takes a few minutes. Later builds reuse the
cache. The image is built in your rootless store, not with `sudo podman build`, so
the build cache and the devcontainer's base images stay in one place. Loading the
model takes a few seconds after the container starts. `/version` in
Discord then shows the commit and build time of what's running. Without git, the
commit reads `unknown`.

`daemon-reload` plus the `[Install]` section is all it takes to start on boot. Quadlet
units aren't `systemctl enable`d.

Reading the system journal needs the `adm` or `systemd-journal` group, or `sudo`.

---

## 3. Day to day

| Task | Command |
|---|---|
| Follow the logs | `journalctl -u increbot -f` |
| Status | `systemctl status increbot` |
| Restart (applies `config.local.rsn` changes) | `sudo systemctl restart increbot` |
| Stop / start | `sudo systemctl stop increbot` / `start` |
| See who opted out (`/opt-out`) | `sudo podman volume export increbot-data \| tar -xO --wildcards '*optouts.json'` |
| Back up the opt-out list | `sudo podman volume export increbot-data -o increbot-data.tar` (restore: `sudo podman volume import increbot-data increbot-data.tar`) |
| Back up the novelty key | It's `discord.novelty_key` in `$REPO/config.local.rsn` (gitignored, kept out of the image by `.containerignore`). Copy that file somewhere safe: a new or lost key re-rolls everyone's `/hotdog-rater`, `/kobold-name` and other fixed results. |
| Persona change | Edit the persona file `config.local.rsn` names (`$REPO/prompts/c21.rsn`): prompt, notes, word biases, sampler, reply length. Then `/super-admin persona-reload` in Discord or `sudo podman kill --signal HUP increbot`. No restart; the folder is mounted live. |
| Switch persona | Point `persona_file` in `config.local.rsn` at another file in `$REPO/prompts/` (it must be there: only that folder is mounted), then `sudo systemctl restart increbot`. `/stats` shows the new `persona:` line. |

### Deploying a new version

After changing code in the devcontainer (and running the tests):

```sh
cd "$REPO"
sudo ./deploy-setup.sh                  # rebuilds ("image updated" / "image unchanged"), restarts
```

When the image changed, the script keeps the one it replaces as
`increbot:previous` in root's store. Each deploy leaves the image that was
`previous` untagged. Clear those out now and
then with `sudo podman image prune`. That doesn't touch the build cache.

The opt-out list is in the `increbot-data` volume (1.5), not in the image, so
deploys and rollbacks keep it. Removing the volume erases it and opts everyone
back in: don't run `sudo podman volume rm increbot-data` or
`sudo podman system prune --volumes`.

### Rolling back

```sh
sudo podman tag increbot:previous increbot:latest
sudo systemctl restart increbot
```

### Changing the unit file

Edit the unit in 1.5 of this file, not `/etc/containers/systemd/increbot.container`:
`deploy-setup.sh` rewrites that from here. Then
`sudo ./deploy-setup.sh` (it restarts the service).

---

## 4. Checks and troubleshooting

| Symptom | Likely cause |
|---|---|
| Exits at once: `DISCORD_TOKEN is not set` | The secret is missing or misnamed: `sudo podman secret ls` should list `discord-token`. A secret made without `sudo` is in your rootless store, which the service doesn't see. |
| `image not known` / `localhost/increbot:latest: image not found` | The image is only in your rootless store. `sudo ./deploy-setup.sh` copies it into root's. |
| `not in the HF cache; downloading…`, then a read-only error | A model or tokenizer file is missing from `$HF/hub`. Download it from the devcontainer with `hf download`, then restart. |
| `Permission denied` reading `/app/config.rsn`, `/app/prompts` or `/hf` | The file isn't world-readable (1.1): `chmod o+r` it, or `o+rx` for a directory. On SELinux systems, also add `z` to the volume options (1.5). |
| Exits at once: `writing /app/data/optouts.json.tmp` … `Permission denied` or `Read-only file system` | The `increbot-data` volume isn't mounted, or without `:U` (1.5). The bot refuses to start rather than lose `/opt-out` requests. |
| Exits at once: `invalid opt-out file` | `optouts.json` in the volume was edited by hand and isn't valid JSON. Fix it (see "See who opted out" in 3) rather than deleting it: an empty list opts everyone back in. |
| `prefill pool can't be pinned` warning | Something restricts the container's CPU set. Pinning needs the real CPU IDs; don't add `--cpuset-cpus`. It still runs, only slower to read prompts. |
| `Temporary failure in name resolution`, repeating every 5 s | Look for `apparmor="DENIED" … profile="crun" … family="inet"` in `journalctl -k`: `NoNewPrivileges=true` is back in the unit (1.5). Remove it. Without such lines, check `getent hosts discord.com` on the host. |
| Exits with `disallowed intents` (close code 4014) | Message Content Intent is off in the Developer Portal (Bot tab). |
| Exited with code 2 and wasn't restarted | A configured chat channel is not a guild channel, or not in an allowlisted guild. The journal names it. Fix `config.local.rsn`, then `sudo systemctl restart increbot`. |
| `activating (auto-restart)` in `systemctl status increbot`, and the gap between restarts keeps growing | It crashes. systemd retries with a backoff from 10 s up to 10 min (1.5) and keeps going. Read the first error with `journalctl -u increbot -b`, fix it, then `sudo systemctl restart increbot`. That restarts it at once and resets the backoff. |
| Won't start: `start request repeated too quickly` / `start-limit-hit` | More than 10 starts in 30 min, automatic and manual combined (1.5). Usually that's many deploys or restarts in a row during a crash. `sudo systemctl reset-failed increbot`, then `sudo systemctl restart increbot`. |
| floob answers twice | The devcontainer bot is still running too. Stop `run-bot.sh`. |
| `GPU unavailable; skipping it` at startup | The card isn't visible in the container. Check `AddDevice=/dev/dri`, and that the `--group-add` number (1.5) matches `getent group render`. Check that `runtime.gpus` has the right PCI addresses (`lspci -D \| grep -i radeon`). The bot runs on the remaining GPUs or the CPU. |
| Replies are CPU-slow although the GPUs are idle | `/stats` shows each GPU's state. "in use by another program" with no program running: something holds VRAM (a leftover process; `amd-smi process` on the host). Backed off cards return after `gpu_holdoff_secs` of being free. |

---

## 5. Logs and privacy

- At `RUST_LOG=info` the logs hold only technical details: times, channel IDs, token
  counts, speeds and errors. No message text. That's what `PRIVACY.md` describes.
- **Message text appears in the logs only at trace level.** Don't set that in the unit:
  journald keeps entries until it rotates them, and removing selected ones is
  awkward. To debug, stop the service and run a one-off container with the same
  settings that logs to a file. Then delete the file when you're done, as the
  privacy policy promises. Replace the numbers as in 1.5:

  ```sh
  sudo systemctl stop increbot
  sudo podman run --rm --user 999:999 --group-add 992 \
    --secret discord-token,type=env,target=DISCORD_TOKEN \
    -v "$HF":/hf:ro -v "$REPO/config.local.rsn":/app/config.rsn:ro \
    -v "$REPO/prompts":/app/prompts:ro -v increbot-data:/app/data:U \
    --read-only --cap-drop ALL \
    --device /dev/dri --memory 24g -e INCREBOT_THREADS=22 -e NO_COLOR=1 \
    -e RUST_LOG=info,increbot::discord=trace increbot:latest > /tmp/floob-debug.log 2>&1
  # reproduce the problem, Ctrl-C, read /tmp/floob-debug.log, then:
  rm /tmp/floob-debug.log && sudo systemctl start increbot
  ```

  The redirect runs in your own shell, so the log file belongs to you, not root.
- journald rotates old logs automatically. Tune it with `SystemMaxUse=` /
  `MaxRetentionSec=` in `journald.conf` if you want a firm limit.
- `run-bot.sh` and its `logs/` folder stay useful for quick runs inside the
  devcontainer, but not alongside the service.
