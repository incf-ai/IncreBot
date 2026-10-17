# Plan: GPU backend that shares nicely

Run floob on one of the host's two Radeon AI PRO R9700s when nobody else needs it,
and get out of the way as soon as another program (ComfyUI, koboldcpp, vLLM, …)
starts using that GPU. Fall back to the other GPU, then to the CPU. Return to the
GPU once it has been free for a while.

Status: **built (2026-10-12)**: milestones 1–4 and most of 5. Not done: tests
against real koboldcpp/ComfyUI/vLLM (milestone 3), and a live Discord run.
Prefill is ~3.4× short of the reference below (see milestone 2). Dev container
changes: [devcontainer-gpu.md](devcontainer-gpu.md).

## Why

| | CPU today (Q4_K_M) | One R9700 (estimated) | One R9700 (measured, milestone 2) |
|---|---|---|---|
| Decode | ~5 tok/s | ~50–65 tok/s (640 GB/s ÷ 7.9 GB ≈ 80 tok/s ceiling) | **58 tok/s** short context, **54** at ~1,050 tokens |
| 75-token reply | ~15 s | ~1.5 s | **~1.3–1.4 s** of writing |
| Prompt read, 1,000 tokens (cold) | ~34 s | well under 1 s | **1.7 s** (590 tok/s) |

**Expected performance from a single card.** This reference run on one R9700 is
the target to match:

```
[21:14:21] CtxLimit:625/16384, Init:0.03s, Processed:598 in 0.30s (2020.27T/s), Generated:27/1000 in 0.54s (50.47T/s), Total:0.86s
```

That is ~2,000 tok/s prompt processing and ~50 tok/s decode, which agrees with the
decode estimate above. **Measured:** decode matches it (54–58 tok/s); prompt
processing reaches ~590 tok/s, about 30% of it (milestone 2 explains why).

## Decisions (2026-10-12)

| Topic | Decision |
|---|---|
| Detection | **Fast path:** read each GPU's total VRAM use from sysfs at **20 Hz**. Running `amd-smi` 20×/s isn't feasible (it's a Python CLI); it's used only to *name* other programs. |
| Threshold | **Cheap rule:** all *other* users of a GPU together above **100 MiB** = contended. "Other" = sysfs total − idle baseline − **our own allocations, tracked exactly**. |
| Programs to expect | ComfyUI, koboldcpp, vLLM, others: ROCm *and* Vulkan users, so detection must not depend on how they allocate (sysfs covers both). |
| GPU use | **One GPU at a time** runs inference. The model is not split across GPUs. |
| Preference | **GPU 1 (`0000:06:00.0`) → GPU 0 (`0000:03:00.0`) → CPU.** |
| Fallback weights | The weights are also loaded on the other free GPU as a **warm standby** (my reading of "load the weights as a fallback"), so moving from GPU 1 to GPU 0 needs no upload. The CPU weights stay loaded too. |
| On contention | Cancel the GPU run quickly, free **all** our VRAM on that GPU, continue the same reply on the next backend in line. |
| Back to the GPU | Free (under the threshold) for **30 s**, doubling each time it flaps. |
| Container access | `/dev/dri` passthrough and `--pid=host` (for naming programs) are accepted. |

## Architecture

```
            ┌──────────────── engine worker (one job at a time) ────────────────┐
 job ──────▶│ backend ladder: [GPU 1] → [GPU 0] → [CPU]                          │
            │   runs on the highest *ready* backend; migrates mid-reply on       │
            │   preemption (cancel → KV to host → continue on the next one)      │
            └──────────▲───────────────────────────────▲─────────────────────────┘
                       │ preempt(gpu)                  │ ready(gpu) / standby
            ┌──────────┴──────────┐         ┌──────────┴───────────┐
            │ VRAM monitor, 20 Hz │         │ GPU manager per card │
            │ sysfs used − base − │────────▶│ state machine, loads │
            │ ours  > 100 MiB ?   │         │ and frees weights    │
            └─────────────────────┘         └──────────────────────┘
```

### 1. GPU backend: Vulkan via `ash`

- **Why Vulkan, not ROCm:** RADV in Mesa supports RDNA4, and the bot's container
  needs no ROCm runtime.
- **Why raw Vulkan, not Burn's wgpu/cubecl:** the sharing policy depends on two
  things those libraries hide: knowing **exactly how many bytes we hold** (they pool
  allocations), and **freeing them at once**. Raw Vulkan also lets us submit work
  in small pieces, which is what makes cancelling fast.
- **One `VkDevice` per R9700**, chosen by PCI address (`VK_EXT_pci_bus_info`); the
  Intel iGPU is ignored.
- **All our allocations go through one tracked allocator** per device: a running
  byte total that the monitor subtracts.
- **Compute shaders** (GLSL → SPIR-V with `glslc` at build time):
  - Q4_K and Q6_K matrix × vector (decode) and matrix × matrix (prefill), reading the
    GGUF blocks as-is. llama.cpp's MIT-licensed Vulkan K-quant shaders are the
    reference.
  - RMSNorm, RoPE (interleaved), attention over an f16 KV cache, SwiGLU, residual add.
  - The embedding table stays on the CPU; one row per token is uploaded.
- **Work is submitted per layer** (40 per token), each followed by a cancel-flag
  check. A decode token is ~15–20 ms, so a cancel takes effect within ~1 ms of the
  current layer finishing; prefill uses ≤ 512-token chunks per layer.
- **The KV cache** lives on the GPU in f16, the same layout as the CPU's f16 cache
  (`kv_cache_f16`), so it copies across without conversion.

### 2. VRAM monitor (20 Hz)

For each R9700, every 50 ms:

```
other = mem_info_vram_used − baseline − ours − slack
contended  ⇔  other > 100 MiB
```

- **`mem_info_vram_used`** (sysfs) counts every allocation on the card: Vulkan,
  ROCm, display. A read takes ~7 µs.
- **`baseline`** is measured at startup, before we allocate (57 MiB today), and
  re-measured whenever we hold nothing on that card and it's been stable for a while.
- **`ours`** comes from our tracked allocator (`DEVICE_LOCAL` allocations only).
  **`slack`** is the driver's own overhead on top of our allocations. The spike
  measured it at ≈ 0 (32 KiB per device), so a 1 MiB constant is enough.
- **Naming the program** (slow path, at most ~1 Hz, only after a detection, and
  only for logs and `/stats`): `/sys/class/kfd/kfd/proc/*` for ROCm programs and
  `/proc/*/fdinfo` (`drm-memory-vram`) for the rest, both with `--pid=host`. Not
  `amd-smi`: it doesn't list Vulkan programs. See the milestone 1 findings.

### 3. GPU manager: a state machine per card

```
        start / free for hold-off
 Free ───────────────────────────▶ Loading ──▶ Ready (active or standby)
  ▲                                   │             │
  │                                   │contended    │contended
  │  hold-off elapsed, still free     ▼             ▼
  └──────────────────────────── Backoff ◀──── Preempting
                                (30 s, ×2 on flaps)    (cancel, KV to host, free all)
```

- **Loading** uploads the weights (~7.9 GB; about 1 s over PCIe) on a background
  thread. The worker keeps serving on whatever is ready meanwhile.
- **Ready** comes in two kinds:
  - **active:** the highest-preference ready GPU, which runs jobs;
  - **standby:** a ready GPU below the active one, holding weights only (no KV).
- **Contended while Loading or Ready:** go to Preempting, then Backoff, freeing
  everything on that card.
- **Contended during Backoff** restarts the timer and doubles the hold-off (30 s,
  60 s, capped there: off the GPU, floob is slow, so it never waits long). The hold-off resets to 30 s after a
  long quiet spell.
- **A pure policy core.** The decisions are plain functions of (monitor readings,
  clock), so they're unit-tested with a fake clock and fake readings, like
  `chat/edits.rs`.

### 4. Preempting a running reply

When the active GPU becomes contended mid-reply:

1. The monitor flags the GPU; the worker sees it at the next layer boundary
   (≤ ~1 ms), stops submitting and waits on the in-flight fence.
2. Copy the KV cache to the host (f16; ≤ ~0.8 GB at 1,000 tokens, ~40 ms).
3. Free every allocation on that card (weights, KV, scratch) and tell the manager
   to go to Backoff.
4. Continue the **same** reply from the next token on the next ready backend: GPU 0
   (standby weights already there; upload the KV) or the CPU (use it directly). The
   Discord message keeps streaming. Nothing restarts, and no prompt is re-read.

**Target:** our VRAM freed **≤ ~150 ms** after the other program's allocation shows
up in sysfs. That's ≤ 50 ms of detection plus ~1 ms to cancel, ~40 ms for the KV,
and the free itself. Milestone 3 measures it.

**Fallback if KV migration is fragile:** free first, then re-read the prompt on the
new backend. That releases sooner but costs up to ~34 s on the CPU.

### 5. Choosing a backend for a job

At job start, use the highest-preference **active** backend: GPU 1, then GPU 0,
then the CPU. A job that started on the CPU stays there even if a GPU frees up;
the next job uses the GPU. Moving a reply back to the GPU mid-way is out of scope.

Parked conversation caches (`kv_cache_slots`) stay in **host RAM**, as today, and
are uploaded to the GPU when a job uses them. That keeps our VRAM footprint
predictable (weights + one active KV) and makes freeing a card trivial.

### 6. Configuration (as built)

```rust
runtime: Runtime {
    gpus:             ["0000:06:00.0", "0000:03:00.0"],  // preference order; [] = CPU only
    gpu_other_mib:    100,    // other users' VRAM above this = contended
    gpu_poll_hz:      20,
    gpu_holdoff_secs: 30,     // doubles on flaps, capped
    gpu_standby:      true,   // keep weights on the next GPU as a warm fallback
},
```

`INCREBOT_GPUS` overrides `gpus` (comma-separated), as with every key. With
`gpus` set, the config must also have `kv_cache_f16: true` and
`linear_impl: Packed`, and `n_ctx <= 8192`; `Config::validate` says so.

### 7. Visibility

- **`/stats`:** which backend served the last reply; each GPU's state
  (active/standby/backoff and time left); the last preemption (when, how long
  release took, who caused it if known).
- **Logs:** one line per state change and per preemption, with the measured
  release time.
- **No Discord-visible change** apart from speed. A reply that migrates
  mid-stream simply slows down.

## Milestones

### 1. Spike (on the host, ~1 day)

After the [dev container changes](devcontainer-gpu.md):

- [x] `vulkaninfo` shows both R9700s through RADV inside the container.
- [x] `amd-smi list` order: is GPU 0 = `03:00.0`?
- [x] **Do Vulkan programs show up in `amd-smi process`?** Run koboldcpp with its
      Vulkan backend (or a small test program allocating 500 MiB through Vulkan)
      and check. Repeat with a ROCm program (vLLM or ComfyUI).
- [x] How long does one `amd-smi process` call take?
- [x] For the same test allocation: the delta in `mem_info_vram_used`, and
      `drm-memory-vram` in the program's `/proc/<pid>/fdinfo/*`.
- [x] **Slack:** allocate N bytes through Vulkan ourselves, read the sysfs delta.
- [x] How long until a fresh allocation shows up in sysfs?

**Exit:** a short findings section added here, and the threshold formula confirmed.

#### Findings (2026-10-11)

Measured in the dev container (rootless podman, `--cap-drop=ALL`, the
devcontainer-gpu.md options). Test programs: a C/Vulkan spike holding device-local
memory, and torch (ROCm, from the vLLM image) holding 300 MiB. Real koboldcpp and
ComfyUI weren't run; milestone 3 does that.

| Question | Answer |
|---|---|
| `vulkaninfo` | Both cards: "AMD Radeon Graphics (RADV GFX1201)", Mesa 25.0.7, Vulkan 1.4.305. RADV warns that GFX12 support is "not conformant, testing use only". A compute shader dispatched on each card by PCI address gave correct results. |
| `amd-smi list` order | GPU 0 = `03:00.0`, GPU 1 = `06:00.0`, as assumed. |
| Vulkan programs in `amd-smi process`? | **No**, neither with the container's 6.1.2 nor the host's 26.2 (ROCm 7.1.1). `amd-smi` lists KFD (ROCm) processes only. |
| ROCm programs in `amd-smi process`? | Container 6.1.2: **no** (lists nothing at all on RDNA4). Host 26.2: yes, with name, pid and VRAM. |
| One `amd-smi process` call | 108–136 ms. |
| sysfs vs. fdinfo, same allocation | Both exact: 500 MiB → +500.000 MiB in `mem_info_vram_used`, and `drm-memory-vram`/`drm-total-vram` = 512,028 KiB. RADV opens two fds with the same `drm-client-id`, so dedupe by client id. |
| **Slack** | **≈ 0.** A `VkDevice` costs 32 KiB; each allocation is exact up to 4 KiB page rounding (checked from 4 KiB to 8 GiB). |
| Allocation → visible in sysfs | **3–54 µs** after `vkAllocateMemory` returns. VRAM is committed at allocation, not first use. |
| One sysfs read | 2.6–3.0 µs. |

**What this changes in the plan:**

1. **The threshold formula holds**, with `slack` = 0 (or a 1 MiB constant for
   safety): `other = vram_used − baseline − ours > 100 MiB`. **`ours` counts only
   `DEVICE_LOCAL` allocations.** Plain `HOST_VISIBLE` memory goes to GTT, not VRAM
   (256 MiB → +0 VRAM, +256 MiB GTT). `DEVICE_LOCAL|HOST_VISIBLE` (BAR) counts as
   VRAM.
2. **Releasing VRAM isn't instant: about 12 ms per GiB** after `vkFreeMemory`
   returns. The kernel clears it in the background, and the call itself returns in
   under 1 ms. Freeing the ~7.8 GiB of weights takes **~97 ms**, whether as one
   allocation or 40 × 200 MiB. Allocations under 2 GiB drop all at once after
   15–30 ms; larger ones start dropping within 0.2 ms and shrink gradually. To stay
   near the ≤150 ms target, preemption should **free the weights first, then copy
   the KV cache to the host** while the kernel releases the weights. Detection
   (≤50 ms) + max(weights 97 ms, KV copy ~40 ms + KV free ~10 ms) ≈ 150 ms.
   Milestone 3 checks this.
3. **Free every allocation explicitly.** Memory still allocated when
   `vkDestroyDevice` runs is **not** released (2 GiB still held 2 s later). It
   comes back only when the process exits. The tracked allocator must free
   everything it holds before the device is destroyed.
4. **Naming other programs: replace `amd-smi` with two direct reads.**
   - **ROCm programs:** `/sys/class/kfd/kfd/proc/<pid>/vram_<gpu_id>`. It's
     world-readable, needs neither `/dev/kfd` nor ptrace, and gives the pid and its
     VRAM per card. The `gpu_id` maps to a PCI address through
     `/sys/class/kfd/kfd/topology/nodes/*/{gpu_id,properties}`, where
     `location_id` = bus << 8 | device << 3 | function (768 = `03:00.0`,
     1536 = `06:00.0`). The name comes from `/proc/<pid>/comm`, which is readable
     for any pid with `--pid=host`.
   - **Vulkan programs:** only `/proc/<pid>/fdinfo`, and only where ptrace-read
     access allows:
     - Same host uid (e.g. another container running as `vscode`): readable.
     - Another rootless container under a different uid: readable only with
       `--cap-add=SYS_PTRACE` (tested).
     - **Processes on the host itself: never readable**, not even as container root
       with `SYS_PTRACE`.

     So a Vulkan program running directly on the host (e.g. koboldcpp outside a
     container) is detected but stays unnamed ("unknown, 1.2 GiB").
   - This drops `amd-smi`/`rocm-smi` (6.1.2 is useless on RDNA4) and `/dev/kfd`
     from what the bot needs. `--pid=host` is still needed for `comm` and fdinfo.

### 2. Single-GPU forward pass

- Vulkan device setup by PCI address; the tracked allocator; the shader build in
  `build.rs`.
- Kernels, each tested against the CPU implementation (as the Q4_K/Q6_K work was):
  Q4_K/Q6_K matvec and matmul, RMSNorm, RoPE, attention, SwiGLU.
- End to end on TinyLlama Q4_K_M: same top token as the CPU path, logits within the
  activation-rounding tolerance (`tests/it/kquant_decode.rs` style).
- `examples/bench_decode.rs --gpu`: decode and prefill on the 13B.

**Exit:** the 13B runs on GPU 1, measured tok/s recorded here.

#### Results (2026-10-12)

Done: `src/gpu/` (`vk.rs`, `model.rs`, `shaders/`), `tests/it/gpu.rs`, kernel tests in
`gpu/model.rs`, and `bench_decode --gpu <pci>` (`INCREBOT_PROFILE=1` adds GPU time
per kernel from timestamp queries).

| 13B Q4_K_M on GPU 1 | Measured |
|---|---|
| Weights upload (from the CPU model's RAM into resizable-BAR VRAM) | 1.9–2.4 s, 7.4 GiB |
| Decode, short context | 58 tok/s |
| Decode, ~1,050 tokens of context | 54 tok/s |
| Cold prefill, 995 tokens | 1.7 s (590 tok/s) |
| Warm prefill, 50 tokens | 0.28 s (0.15 s GPU; the rest is the host KV cache growing) |

What differs from the plan above, and why:

- **No staging buffers.** Both cards have resizable BAR (all VRAM host-visible), so
  weights and KV uploads are plain `memcpy`s into mapped VRAM.
- **The host KV cache stays canonical.** Each forward uploads the positions the
  card lacks (longest common token prefix) and copies each finished chunk's new
  K/V rows back. Preemption therefore copies nothing (§4's steps 2 and the
  "KV to host" are gone), and a reply resumes at the interrupted 512-token chunk.
- **Q6_K is repacked** at upload (ql+qh of every block, then scales and `d`), so
  every bulk load is 8/16-byte aligned. Q4_K stays as in the GGUF.
- **Decode kernels:** one f32 matrix-vector shader for Q, K and V together, one
  for gate+up with SwiGLU fused, split-K attention (flash-decoding) over the f16
  cache. The matvec reads ~545 GB/s (85% of 640). Without split-K, attention at
  1,000 tokens took 10 ms/token (40 workgroups); now 2.8 ms.
- **Prefill kernels:** activations quantized to 8 bits per 256 values (the CPU's
  `Q8_K`), then int8 dot products (`v_dot4`) with sub-block scales applied in
  integers: 2.5× the first f32 tiled matmul. Tiled FlashAttention-style attention:
  4× the first version.
- **Why prefill is short of 2,000 tok/s:** llama.cpp gets there with matrix cores
  (WMMA) through `VK_KHR_cooperative_matrix`. RADV in Mesa 25.0.7 (trixie) doesn't
  expose it on GFX12, even with `RADV_PERFTEST`. The int8 matmul runs at ~26 TOPS
  effective. A newer Mesa with cooperative matrix on RDNA4 is the lever; on this
  Mesa, further tuning of `mmq.comp` (tile shapes, register pressure: 240 VGPRs) is
  what's left.
- **Accuracy:** decode matches the dense CPU model to f16 rounding (max logit
  difference 0.002 at RMS 2.3, TinyLlama); prefill has the CPU packed path's
  8-bit error (~0.35 at RMS 2.3), same top tokens. The int8 matmul matches the
  CPU integer kernel to 5e-4 of the output scale.

### 3. Sharing: monitor, manager, preemption, resume

- The monitor, the per-card state machine (policy core with unit tests), the
  backend ladder in the worker, KV migration GPU→CPU and GPU→GPU.
- An **intruder test**: a small program allocating 200 MiB on GPU 1 while floob is
  mid-reply. Check that the reply finishes correctly on GPU 0 or the CPU, measure
  the release time, then check the return to GPU 1 after the hold-off.
- Run the same with koboldcpp and ComfyUI for real.

**Exit:** release ≤ ~150 ms (measured), no failed replies across repeated intruder
runs.

#### Results (2026-10-12)

Done: `gpu/monitor.rs`, `gpu/policy.rs` (unit tests on a fake clock),
`gpu/manager.rs` (a thread per card), `engine/backend.rs` (the ladder), and
`tests/it/gpu_sharing.rs`: the intruder is a second Vulkan device in the test process
allocating 200 MiB on GPU 1 mid-reply. `GPU_SHARING_13B=1` runs it on the 13B.

- **The reply continues** on GPU 0 (then the CPU when both are taken), with the
  same tokens in the cache and the same top token as a CPU-only run. 8 of 8
  repeated runs passed (5 TinyLlama, 3 13B), plus later runs.
- **Release, 13B:** from the intruder's allocation until all ~8 GB of ours was
  gone from sysfs: **129, 138, 143, 150, 156 ms**. Of that, the forward stopping
  takes ~0.05 ms, freeing every allocation 22–76 ms, and the rest is detection
  (≤ 50 ms at 20 Hz) and the kernel releasing ~12 ms/GiB. TinyLlama: 25–56 ms.
  The target is met on average; to have margin, poll faster (`gpu_poll_hz`) or
  free fewer, larger allocations.
- **Return:** GPU 1 is used again ~0.3 s after the hold-off (the upload of the
  small model plus the self-test), and the hold-off doubles on a flap.
- **Not done:** the same with koboldcpp and ComfyUI for real (not installed in the
  dev container). Expect the same path: their allocations show in sysfs exactly as
  the intruder's do (milestone 1).

### 4. Standby on GPU 0 and the full ladder

- Warm standby weights on GPU 0 while GPU 1 is active.
- GPU 1 contended: move to GPU 0. Both contended: CPU. Each recovers on its own
  hold-off.

**Done (2026-10-12)**, covered by the intruder test. Every upload ends with a
one-token self-test before the card serves replies. Only the active card holds a
KV cache and scratch buffers (~1.8 GB at `n_ctx` 2048); a standby card holds
weights only, and takes over a reply by uploading the host KV cache.

### 5. Deploy and docs

- [deploy.md](deploy.md): the runtime image on trixie with the Mesa runtime,
  `--device=/dev/dri`, `--pid=host`, the `gpus` setting.
- `design.md`: a new section for the GPU backend, plus the updated latency table.
- `/stats` additions; chatsim gains `--gpu` to try it locally.

**Done (2026-10-12)** except a live run: deploy.md (trixie image with `glslc` at
build time, `INCREBOT_GPU_SHADERS_REQUIRED=1`, `AddDevice=/dev/dri`, `--pid=host`,
`--group-add=keep-groups`), design.md §13 and the §3 table, `/stats` ("ran on:"
and a line per GPU plus the last preemption), `chatsim --gpu`, and the `runtime`
GPU keys in `config.rsn` / `config.local.rsn`. The deployed image hasn't been
rebuilt and run yet.

Later: deploy.md moved to a rootful system service running as a
dedicated `increbot` user. `keep-groups` became a numeric `--group-add` for the
render group, and `--pid=host` was dropped: a separate user can't read other
users' `/proc/<pid>/fdinfo`, so GPU users show up in `/stats` unnamed.

## Risks and open questions

- **The other program's first ~150 ms.** We still hold ~8–9 GB on the card when it
  starts. With 32 GB per card it only fails if it immediately wants almost all of it
  (e.g. vLLM's default of reserving ~90% of VRAM). If that happens, vLLM fails at
  startup. Mitigation: start such programs after floob has backed off, or lower
  vLLM's `--gpu-memory-utilization`. Worth testing in milestone 3.
- **Two unrelated programs** each staying under 100 MiB but together above it trip
  the cheap rule. That's accepted (the decision was the cheap rule).
- **Desktop or compositor drift:** the R9700s are headless today. If a display is
  attached later, its framebuffer raises the baseline; re-baselining when we hold
  nothing covers slow drift.
- **Unnamed programs:** a Vulkan program running directly on the host (not in a
  container) can't be named from the bot's container, because host processes'
  fdinfo is unreadable. It's still detected. (`amd-smi` was dropped for naming:
  6.1.2 lists nothing on RDNA4, and no version lists Vulkan programs.)
- **Release time:** the kernel takes ~12 ms per GiB to release freed VRAM, so the
  ≤150 ms target depends on freeing the weights before copying the KV cache.
- **Flapping:** a program that allocates and frees in a loop. The doubling hold-off
  bounds the churn to one reload per hold-off period.
- **CPU fallback load:** inference on the CPU uses 8–22 threads, which slows other
  CPU work on the host while floob runs there. That's unchanged from today.
- **Effort:** milestone 2 (shaders and correctness) is the bulk of the work. The
  sharing logic (3–4) is moderate and mostly testable without a GPU.
