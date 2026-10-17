#!/bin/bash
# One benchmark run (README.md §1), from the repo root inside the devcontainer:
#   bench/persona/run.sh <name> <persona file> [gpu pci]
# Scenario runs (README §1): SCENARIOS=<file.jsonl> [REPEATS=4] bench/persona/run.sh …
# e.g. bench/persona/run.sh 0066-floob-c19 prompts/c19.rsn
# The persona file holds the prompt, notes, word biases, stop strings and
# sampler settings (persona-plan.md). An exported INCREBOT_DISCOURAGE replaces
# its word list ("" = none) and is recorded in env.txt.
#
# The run reads copies of its inputs saved in the run folder: config.rsn (from
# CONFIG, default config.local.rsn), persona.rsn and scenarios.jsonl, plus
# env.txt (the INCREBOT_* overrides) and binary.sha256. Edits to the live files
# after the run starts don't reach it, and a folder reruns as itself (README
# §1). BIN=<binary> skips the build (queue.sh uses it).
# A folder that already has meta.json is refused unless FORCE=1.
#
# Everything is inside main, so bash reads the whole script before running it:
# editing this file during a run can't change or break that run (with
# `main "$@"; exit` on one line, see the end).
set -euo pipefail

main() {
  local name=$1 persona=$2 gpu=${3:-0000:03:00.0}
  local out=bench/persona/runs/$name config=${CONFIG:-config.local.rsn} bin=${BIN:-}
  if [[ -e $out/meta.json && -z ${FORCE:-} ]]; then
    echo "$out already has a run (FORCE=1 to overwrite)" >&2
    exit 1
  fi
  mkdir -p "$out"
  if [[ -z $bin ]]; then
    cargo build --release --example floob_bench
    bin=target/release/examples/floob_bench
  fi
  cp "$config" "$out/config.rsn"
  cp "$persona" "$out/persona.rsn"
  local scen=()
  if [[ -n ${SCENARIOS:-} ]]; then
    cp "$SCENARIOS" "$out/scenarios.jsonl"
    scen=(--scenarios "$out/scenarios.jsonl" --repeats "${REPEATS:-4}")
  fi
  env | grep '^INCREBOT_' | sort > "$out/env.txt" || true
  sha256sum "$bin" | cut -d' ' -f1 > "$out/binary.sha256"
  HF_HUB_OFFLINE=1 INCREBOT_PERSONA_FILE="$out/persona.rsn" \
    "$bin" --gpu "$gpu" --out "$out" \
    --batches 20 --turns 25 --seed 1 --config "$out/config.rsn" --evict \
    "${scen[@]}" 2>&1 | tee "$out/bench.log"
}

# One line, so bash has read the exit before main runs: an edit that shifts
# this file during a run can't make it read a stray line afterwards.
main "$@"; exit
