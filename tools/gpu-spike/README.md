# GPU spike (gpu-plan.md, milestone 1)

Throwaway C programs that answered the milestone 1 questions. The findings are in
[gpu-plan.md](../../gpu-plan.md), "Findings". Kept for re-measuring (e.g. after a
Mesa or kernel upgrade) and as an intruder for milestone 3.

Build in the dev container (needs the devcontainer-gpu.md packages):

```sh
cd tools/gpu-spike
gcc -O2 spike.c -o /tmp/spike -lvulkan
gcc -O2 gputest.c -o /tmp/gputest -lvulkan
glslc -fshader-stage=compute double.comp -o double.spv   # for gputest
```

| Program | What it does |
|---|---|
| `spike measure <pci> [MiB …]` | sysfs read cost, `VkDevice` overhead, and per allocation size: time to show in sysfs, sysfs delta (slack), own fdinfo, time to release after a free. Also 40 × 200 MiB and 7.8 GiB release times, and whether `HOST_VISIBLE` memory counts as VRAM. |
| `spike hold <pci> <MiB> <secs>` | Holds a device-local allocation: a Vulkan "other program" to detect. |
| `spike leak <pci>` | Shows that memory still allocated at `vkDestroyDevice` isn't released until exit. |
| `observe.sh <pci>` | What the bot's container can learn about other GPU users: `amd-smi process` (timed), `/proc/*/fdinfo`, `/sys/class/kfd/kfd/proc`. Needs `--pid=host`. |
| `gputest` | Runs a compute shader on each GPU in `INCREBOT_GPUS` (chosen by PCI address) and checks the result. Run from the folder holding `double.spv`. |

`<pci>` is a full address such as `0000:06:00.0`.
