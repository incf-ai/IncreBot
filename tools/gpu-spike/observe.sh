#!/bin/bash
# Observer: what can the bot's container learn about other GPU users?
# Usage: observe.sh <pci>
pci=$1
echo "== sysfs vram_used on $pci: $(( $(cat /sys/bus/pci/devices/$pci/mem_info_vram_used) / 1048576 )) MiB"

echo "== amd-smi process (timed x5)"
for i in 1 2 3 4 5; do
  s=$(date +%s%N); out=$(amd-smi process --json 2>&1); e=$(date +%s%N)
  echo "  call $i: $(( (e - s) / 1000000 )) ms"
done
echo "$out" | head -60

echo "== /proc/*/fdinfo scan for drm-pdev $pci"
readable=0; denied=0
for d in /proc/[0-9]*; do
  pid=${d#/proc/}
  if ! ls $d/fdinfo >/dev/null 2>&1; then denied=$((denied + 1)); continue; fi
  readable=$((readable + 1))
  for f in $d/fdinfo/*; do
    grep -q "drm-pdev:.*$pci" $f 2>/dev/null || continue
    v=$(grep -E '^drm-(memory|total)-vram' $f | tr -s ' \t' ' ' | tr '\n' ' ')
    echo "  pid $pid ($(cat $d/comm 2>/dev/null)) fd ${f##*/}: $(grep drm-client-id $f | tr -s ' \t' ' ') $v"
  done
done
echo "  processes: $readable fdinfo readable, $denied denied"

echo "== /sys/class/kfd/kfd/proc (ROCm processes, world-readable)"
for p in /sys/class/kfd/kfd/proc/*; do
  [ -e "$p" ] || continue
  pid=${p##*/}
  echo "  pid $pid ($(cat /proc/$pid/comm 2>/dev/null || echo '?')): $(grep -h . $p/vram_* 2>/dev/null | tr '\n' ' ')"
done
