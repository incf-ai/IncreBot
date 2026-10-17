# Dev container changes for GPU development

Changes to `.devcontainer/` so the GPU backend ([gpu-plan.md](gpu-plan.md)) can be
developed and tested inside the devcontainer. **Apply them on the host:**
`devcontainer.json` and `Dockerfile` are mounted read-only inside the container.
Then run **Dev Containers: Rebuild Container**.

## What this host has (measured 2026-10-12)

| PCI address | sysfs | Render node | GPU | VRAM | Role in the plan |
|---|---|---|---|---|---|
| `0000:06:00.0` | `card2` | `renderD129` | Radeon AI PRO R9700 (`0x1002:0x7551`) | 32,624 MiB | **GPU 1**, preferred |
| `0000:03:00.0` | `card1` | `renderD128` | Radeon AI PRO R9700 (`0x1002:0x7551`) | 30,576 MiB | **GPU 0**, fallback |
| `0000:00:02.0` | `card3` | `renderD130` | Intel UHD 770 (iGPU) | – | not used |

- Both R9700s are headless (no displays connected) and idle at **57 MiB** VRAM used.
- `card`/`renderD` numbers can change between boots. The bot identifies GPUs by
  **PCI address**, never by node number.
- "GPU 0" and "GPU 1" follow PCI order, which is also how `amd-smi` usually numbers
  them. Confirm with `amd-smi list` once it runs (see the checks below).
- The container is Debian 13 (trixie): Mesa 25.0.7 (RADV supports RDNA4 from 25.0),
  and `amd-smi`/`rocm-smi` 6.1.2 are all in the distribution.

## 1. Dockerfile

Add these to the existing `apt-get install` list in `.devcontainer/Dockerfile`:

```dockerfile
    # GPU backend (gpu-plan.md): Vulkan driver, loader, headers, tools, shaders.
    mesa-vulkan-drivers \
    libvulkan-dev \
    vulkan-tools \
    vulkan-validationlayers \
    glslc \
    spirv-tools \
    # GPU usage: amd-smi/rocm-smi to name other programs on the GPUs.
    amd-smi \
    rocm-smi \
```

- **`mesa-vulkan-drivers`** provides RADV, the Vulkan driver for the R9700s.
- **`libvulkan-dev`** is the Vulkan loader the `ash` crate loads at run time.
- **`glslc`** compiles the compute shaders to SPIR-V at build time.
- **`vulkan-validationlayers`** is for debug builds (`VK_LAYER_KHRONOS_validation`).
- **`amd-smi` 6.1.2 predates RDNA4 support (ROCm 6.4).** It may not fully recognise
  the R9700. Detection doesn't depend on it (it reads sysfs); only naming the other
  program in logs does. If it can't see the cards, install a newer `amd-smi` from
  AMD's ROCm repository instead.

## 2. devcontainer.json

Add to `runArgs`:

```jsonc
		// GPU backend (devcontainer-gpu.md): the GPU device nodes. All of /dev/dri,
		// because node numbers can change between boots; the bot picks GPUs by PCI
		// address and ignores the Intel iGPU.
		"--device=/dev/dri",
		// KFD: amd-smi and rocm-smi list ROCm processes (vLLM, ComfyUI) through it.
		"--device=/dev/kfd",
		// Rootless podman: keep the host user's video/render groups, which own the
		// device nodes. Needs crun (podman's default runtime).
		"--group-add=keep-groups",
		// Host PID namespace: lets the bot and amd-smi see other containers'
		// processes (/proc/<pid>/fdinfo), to name who is using a GPU.
		"--pid=host",
```

Add to `containerEnv`:

```jsonc
		// The bot's GPU preference: PCI addresses, first = preferred (gpu-plan.md).
		// "GPU 1" (06:00.0) first, then "GPU 0" (03:00.0), then the CPU.
		"INCREBOT_GPUS": "0000:06:00.0,0000:03:00.0",
```

### What the new options cost

- **`--pid=host`** lets this container see **every process on the host**. Together
  with Claude Code's bypass-permissions mode, that means the agent can read other
  processes' command lines and environments. You approved this; it's still worth
  knowing. Detection works without it, so drop it if attribution matters less than
  isolation.
- **`--device=/dev/kfd`** is only for `amd-smi`/`rocm-smi` process lists. The bot's
  Vulkan path doesn't need it.
- **`/dev/dri`** includes the Intel iGPU's render node. That's harmless; it shows up
  in `vulkaninfo` and the bot ignores it.
- `--cap-drop=ALL` can stay: the device nodes are opened through group permissions,
  not capabilities.

## 3. Checks after rebuilding

Run inside the container:

```sh
# Device access: three render nodes, owned by the render group you're in.
ls -l /dev/dri/ /dev/kfd && id

# Vulkan: two "AMD Radeon Graphics (RADV GFX1201)" devices (Mesa 25.0 has no
# marketing name for the R9700), plus the Intel iGPU and llvmpipe.
vulkaninfo --summary | grep -E 'deviceName|driverName|PCI'

# amd-smi: does it see both cards, and in which order (GPU 0 should be 03:00.0)?
amd-smi list
amd-smi process --json | head -40

# The sysfs fast path (works even without device nodes):
grep . /sys/class/drm/card*/device/mem_info_vram_used
```

The **spike in [gpu-plan.md](gpu-plan.md), milestone 1**, continues from here: it
checks whether Vulkan programs appear in `amd-smi process`, how long `amd-smi`
takes per call, and how a test program's allocation shows up in sysfs and fdinfo.

## 4. Later: the deploy container

The bot's service container ([deploy.md](deploy.md)) will need the same
`--device=/dev/dri` plus the Mesa runtime, but **not** the headers, tools or
validation layers. It runs as a rootful system service under a dedicated user, so
it gets the render group by numeric `--group-add` instead of `keep-groups`, and
it has no `--pid=host` (that user couldn't read other users' fdinfo anyway). Its runtime stage moves from
Debian bookworm (Mesa 22, too old for RDNA4) to trixie. `deploy.md` gets updated
when the GPU backend ships (gpu-plan.md, milestone 5).
