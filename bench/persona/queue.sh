#!/bin/bash
# Several benchmark runs from one frozen set of inputs (README.md §1):
#   bench/persona/queue.sh <queue file>
# Each non-empty line not starting with # is one run, fields separated by "|":
#   <name> | <persona file> | [scenarios file] | [repeats] | [config]
# At start, the queue builds floob_bench once and copies the binary, CONFIG
# (default config.local.rsn; a line's own config file replaces it for that
# run), every persona, scenario and config file into
# bench/persona/tmp/queue-<time>/. Every run then uses those copies, so editing
# the live config, a persona, the code or these scripts while the queue runs
# changes nothing in it. Runs go in file order; the first failure stops the queue.
# GPU=<pci> runs the queue on that card (default 0000:03:00.0); two queues on
# two cards run in parallel. The live service prefers 0000:06:00.0: a queue
# there evicts it (to the CPU, if its standby card is busy too).
set -euo pipefail

main() {
  local queue=$1 snap
  snap=bench/persona/tmp/queue-$(date +%Y%m%d-%H%M%S)
  mkdir -p "$snap"
  cargo build --release --example floob_bench
  cp target/release/examples/floob_bench "$snap/floob_bench"
  cp bench/persona/run.sh "$snap/run.sh"
  cp "${CONFIG:-config.local.rsn}" "$snap/config.rsn"
  cp "$queue" "$snap/queue.txt"

  # Read every line and snapshot every input before the first run starts.
  local -a names personas scens reps cfgs
  local n=0 name persona scen rep cfg
  while IFS='|' read -r name persona scen rep cfg || [[ -n $name ]]; do
    # Comments are skipped before xargs, which chokes on an apostrophe.
    [[ $name =~ ^[[:space:]]*(#|$) ]] && continue
    name=$(echo "$name" | xargs)
    persona=$(echo "$persona" | xargs)
    scen=$(echo "${scen:-}" | xargs); rep=$(echo "${rep:-}" | xargs); cfg=$(echo "${cfg:-}" | xargs)
    cp "$persona" "$snap/$n-persona.rsn"
    if [[ -n $scen ]]; then cp "$scen" "$snap/$n-scenarios.jsonl"; scen=$snap/$n-scenarios.jsonl; fi
    if [[ -n $cfg ]]; then cp "$cfg" "$snap/$n-config.rsn"; cfg=$snap/$n-config.rsn; else cfg=$snap/config.rsn; fi
    names[n]=$name personas[n]=$snap/$n-persona.rsn scens[n]=$scen reps[n]=${rep:-4} cfgs[n]=$cfg
    n=$((n + 1))
  done < "$queue"
  echo "queue: $n runs, inputs frozen in $snap" >&2

  local i
  for ((i = 0; i < n; i++)); do
    echo "=== ${names[i]} ($((i + 1))/$n)" >&2
    CONFIG=${cfgs[i]} BIN=$snap/floob_bench SCENARIOS=${scens[i]} REPEATS=${reps[i]} \
      bash "$snap/run.sh" "${names[i]}" "${personas[i]}" "${GPU:-0000:03:00.0}"
  done
}

# One line, so bash has read the exit before main runs: an edit that shifts
# this file during a run can't make it read a stray line afterwards.
main "$@"; exit
