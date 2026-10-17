# .devcontainer/Dockerfile
#
# Dev environment for increbot: a Discord bot that runs
# LLaMA2-13B-Tiefighter (GGUF Q8_0) through Burn's CPU backend, with a
# Vulkan GPU backend in development (gpu-plan.md, devcontainer-gpu.md).
#
# No ROCm compute stack: the GPU path is Vulkan (Mesa RADV). /dev/dri and
# /dev/kfd are passed through in devcontainer.json; amd-smi/rocm-smi are
# only used to name other programs using a GPU.
FROM mcr.microsoft.com/devcontainers/rust:1

# pkg-config + libssl-dev: reqwest/hf-hub (model download) and serenity's
#   TLS stack when built with native-tls.
# clang + mold: faster links of the (large) burn dependency tree.
# hwloc + numactl: inspect the P-core/E-core layout when tuning thread
#   pinning (see design.md, "Threading").
# No linux-perf: the host has kernel.perf_event_paranoid=4, so perf events
# are unavailable to an unprivileged rootless-podman container. Profile on
# the host instead.
# GPU backend (gpu-plan.md): mesa-vulkan-drivers (RADV, the R9700 driver),
#   libvulkan-dev (the loader ash opens at run time), vulkan-tools
#   (vulkaninfo), vulkan-validationlayers (debug builds), glslc +
#   spirv-tools (compute shaders to SPIR-V at build time).
# amd-smi + rocm-smi: name other programs using a GPU in logs.
RUN apt-get update && apt-get install -y --no-install-recommends \
    pkg-config \
    libssl-dev \
    clang \
    mold \
    hwloc \
    numactl \
    ripgrep \
    jq \
    mesa-vulkan-drivers \
    libvulkan-dev \
    vulkan-tools \
    vulkan-validationlayers \
    glslc \
    spirv-tools \
    amd-smi \
    rocm-smi \
    && rm -rf /var/lib/apt/lists/*

# uv, used only to provide the `hf` CLI for fetching model files by hand.
# The bot itself downloads the GGUF through the hf-hub crate.
RUN curl -LsSf --retry 3 https://astral.sh/uv/install.sh -o /tmp/uv-install.sh \
    && env UV_INSTALL_DIR=/usr/local/bin sh /tmp/uv-install.sh \
    && rm -f /tmp/uv-install.sh

RUN chown -R vscode:rustlang "$CARGO_HOME" "$RUSTUP_HOME"

USER vscode
# (rustfmt and clippy already ship in the base image.)
RUN uv tool install "huggingface_hub[cli]" \
    && cargo install cargo-nextest --locked
ENV PATH=/home/vscode/.local/bin:${PATH}
