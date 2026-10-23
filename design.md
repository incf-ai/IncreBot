# increbot: design

A Discord bot that answers messages with **LLaMA2-13B-Tiefighter**. The model runs
locally **on CPU** through the [Burn](https://burn.dev) deep-learning framework,
loaded from a **GGUF Q8_0** file. A configurable system prompt shapes the replies.
Inference runs on a dedicated pool of compute threads, **8 by default**.

## 1. Goals and non-goals

**Goals**
- Reply in allowlisted Discord channels (and DMs from allowlisted users), with the
  bot's persona set by a system prompt stored in a file.
- Run the model fully in-process in Rust with Burn. No llama.cpp, no Python, no
  external inference server.
- Use CPU only, with a configurable number of compute threads (default 8).
- Load the GGUF directly. Q8_0 is the target, and the load is lossless: no
  re-quantization.
- Stay usable on the dev box (i9-14900K, 188 GB RAM) inside the devcontainer.

**Non-goals for now**
- GPU inference. The ROCm devcontainers already cover that, and here it is out of scope.
- Batched or concurrent generation. One generation runs at a time (see §6).
- Training or fine-tuning.
- Every GGUF quant type. We support Q8_0 and F32, which is everything the chosen file
  contains, plus F16 because it costs almost nothing.

## 2. Model

| | |
|---|---|
| Source repo | `TheBloke/LLaMA2-13B-Tiefighter-GGUF` |
| File | `llama2-13b-tiefighter.Q4_K_M.gguf`, 7,865,956,288 bytes (the default since 2026-10-11, see "Q4_K_M" below). Before: `llama2-13b-tiefighter.Q8_0.gguf`, 13,831,319,488 bytes, still supported |
| Architecture | Llama 2: 40 layers, d_model 5120, 40 heads (head_dim 128, no GQA), FFN 13824 (SwiGLU), vocab 32000 |
| Context | 4096 tokens, RoPE θ = 10000, RMSNorm ε = 1e-5 |
| License | Llama 2 Community License |

Read every hyperparameter from the GGUF metadata (`llama.block_count`,
`llama.embedding_length`, …). Never hard-code them. The table is only for sizing.

### Why Q8_0

Q8_0 is effectively lossless relative to FP16. In llama.cpp's published perplexity numbers
for Llama-class 13B models, Q8_0 differs from FP16 only in the third decimal place,
while Q4_0 is measurably worse. Q8_0 costs about twice the memory traffic per token
of a 4-bit format, so the decode ceiling is about half (§3). That's an
acceptable trade for a chat bot.

Burn's quantization scheme (`burn_std::QuantScheme` / `cubecl_common::quant`) can
represent GGUF **Q8_0 exactly**:

| GGUF Q8_0 | Burn scheme |
|---|---|
| Blocks of 32 values | `BlockSize` = 32 along the inner dim |
| One f16 scale `d` per block | `ScaleDtype::F16` |
| `x = d · q`, q: i8 | `QuantValue::Q8F` (range −128..127), `QuantMode::Symmetric` (no zero point) |

Use `Q8F`, not `Q8S`. llama.cpp's quantizer emits q ∈ [−127, 127], but the
format allows −128, and `Q8S` (range −127..127) would clamp it. With `Q8F` the import
is **bit-exact**: nothing gets re-rounded. It is not a plain memcpy, though. A GGUF
Q8_0 tensor interleaves each block's `f16 d` with its 32 `i8` values (34 bytes per
block), while Burn's quantized `TensorData` wants **all values first, then all
scales** (`burn-std` `QuantizedBytes`). The loader de-interleaves the blocks into
those two runs. The values keep their element order, and the scales keep their
f16 bits.

Alternatives considered:
- **Q4_0** (7.4 GB). Maps exactly too (`Q4F`, block 32, f16 scale, `q − 8`; low nibbles
  hold elements 0–15 and high nibbles 16–31). It would double the decode ceiling, but
  quality drops measurably. It remains the fallback if speed turns out to matter more.
  In that file `output.weight` is Q6_K, so it would also need a Q6_K decoder.
- **Q4_K_M / Q5_K_M**. Close to Q8 quality at about 4–5 bits, but K-quants use 256-element
  super-blocks with 6-bit sub-scales and mins. Burn has no matching scheme. Deferred,
  then adopted for speed: see "Q4_K_M" below.

### Q4_K_M (the default since 2026-10-11)

Decode at Q8_0 ran at ~3 tok/s, pinned by memory bandwidth (§3), and the goal became
~6. The only lever is fewer bytes per token, so the bot moved to the Q4_K_M file of
the same model: 7.87 GB instead of 13.8.

- **Tensor types.** Most weights and `token_embd` are **Q4_K** (256-weight blocks,
  144 bytes: f16 `d` and `dmin`, eight 6-bit scales and mins, 4-bit values;
  `x = d·sc·q − dmin·m`). About half of `attn_v` and `ffn_down`, and `output.weight`,
  are **Q6_K** (210 bytes: 4+2-bit values, sixteen i8 scales, f16 `d`;
  `x = d·sc·(q − 32)`). Norms stay F32. Layouts follow llama.cpp's `ggml-common.h`
  (`gguf/kquant.rs`).
- **No Burn path.** Burn has no K-quant scheme, so `linear_impl: Burn` is Q8_0-only.
  `"dense"` decodes K-quants straight to f32 (the test reference); `"packed"` keeps the
  GGUF blocks (`model/packed_k.rs`, §3).
- **Checked against an independent source.** TinyLlama's Q4_K_M and Q8_0 files come
  from the same f16 weights; all 156 K-quant tensors decode within 9.8% (Q4_K) and 2.1%
  (Q6_K) relative RMS of their Q8_0 counterparts (`tests/it/kquant_decode.rs`). A layout
  bug would scramble values (~100%).
- **Quality.** The same greedy prompt gives in-character floob replies on both files
  ("cheez and crackas…" vs "cheezburgers and fwied chwims…"). llama.cpp's published
  numbers put Q4_K_M a few hundredths of perplexity above Q8_0 on 13B models.
- **FP8 (E4M3) / generic per-tensor int8.** Rejected. The 14900K has no FP8 arithmetic,
  and int8 with coarser scales is no better than Q8_0's block-32 scales.

### Tensor types in the Q8_0 file

Checked by parsing the real file's header (GGUF **v3**, 363 tensors, 20 metadata keys,
no `general.alignment` key, so the default alignment of 32 applies):
- 282 × **Q8_0**: `*.attn_{q,k,v,output}.weight`, `*.ffn_{gate,up,down}.weight`,
  `token_embd.weight`, **and `output.weight`** (the LM head). This file has no Q6_K.
- 81 × **F32**: `*_norm.weight`, `output_norm.weight`

GGUF dims are listed innermost-first, e.g. `ffn_down` is `[13824, 5120]`: in = 13824
(432 blocks per row), out = 5120. The loader still checks every tensor's type. Any type
it doesn't support is a hard error that names the tensor.

**`token_embd.weight` is special.** It is Q8_0 too, but an embedding lookup must not
go through Burn's quantized ops. On burn-flex, `select`/`gather` on a block-quantized
tensor dequantizes the whole table, gathers, and then **re-quantizes with fresh
scales** (`burn-flex/src/ops/qtensor.rs`), which is slow and lossy. The loader
therefore decodes `token_embd` to f32 once at load (32000 × 5120 × 4 B ≈ 655 MB) and
looks up embeddings as an ordinary float tensor. The general rule: quantized weights
are only ever dequantized or fed to `QLinear`. They never go through layout ops
(`transpose`, `slice`, `select`, …), which re-quantize on this backend.

## 3. Burn backend

- **Burn version**: pin `=0.22.0-pre.3`, the version in the local cargo registry, where
  the scheme types above were checked. Revisit when 0.22 stable ships.
- **Backend**: `burn-flex`, Burn's portable CPU backend. Its default features
  (`std`, `simd`, `rayon`) give SIMD kernels, plus `gemm` for matmul running on rayon.
  - `burn-cpu` (MLIR/CubeCL) is the other CPU option. It is not used at first because it is younger
    and slower to compile. Switching is a cargo feature plus a device choice, not a code
    change: in Burn 0.22, `Tensor<D, K>` has **no backend type parameter**. The backend
    is picked at runtime through `Device` (`burn-dispatch`), and with only the `flex`
    feature enabled, `Device::default()` is burn-flex. (Found while implementing
    milestone 1; earlier drafts assumed the 0.1x-style `Tensor<B, D>` API.)
- **Element types**: activations f32. The *active* KV cache is f32, and *parked* caches
  in the LRU are stored as f16 (§5). An f16 active cache looks cheaper but isn't: with
  Burn tensor attention, every decode step would cast each layer's whole cache to f32
  and then read it again. At 2,000 tokens that adds ~1.6 GB read plus ~3.3 GB
  written and re-read per token, on top of 13.8 GB of weights. Converting only when
  a cache is parked or restored costs one pass per channel switch instead of one per
  token.

### The quantized-matmul problem

Decoding one token reads **every weight once** (~13.8 GB for Q8_0). That makes it
**memory-bandwidth bound**. Measured on this machine (see "Measured baseline" below),
sustained read bandwidth is **~50 GB/s**, so the ceiling is **~3.6 tokens/s** from the
weights alone. The KV cache adds more traffic on top: at ~1,000 tokens of context
the active f32 cache adds ~1.6 GB per token, which brings the realistic decode rate
to **~3.0–3.4 tokens/s**.

burn-flex stores quantized tensors as i8 (one byte per value, which suits Q8) with
**f32 scales**, and it **dequantizes the whole weight to f32** before every matmul
(`ops/qtensor.rs`). On the decode path that means reading ~13 GB of i8, then writing
and re-reading ~52 GB of f32 per token: several times the minimum memory traffic.
The memory footprint is not the issue: ~13 GB of i8 plus ~1.6 GB of scales
≈ 14.5 GB, about the same as the file.

That dequantize is also **single-threaded and scalar**: one iterator over every element,
with a div/mod per element to find its block (`BlockLayout::block_of`). Over 13.8 G
weights that is **several seconds per token even on a 1B model**: measured 5.5 s/token
(0.18 tok/s) on TinyLlama-1.1B, and **77 s/token** on the 13B (6-token prefill: 80 s). A 300-token reply
would blow far past `timeout_secs`. So the Burn path is a reference for tests and
the small model only. It is never how the bot runs the 13B.

Plan: hide linear layers behind one trait (`src/model/linear.rs`; not generic, since
Burn 0.22 tensors carry no backend type) and ship these implementations:

```rust
trait QLinear: Send {
    fn forward(&self, x: Tensor<2>) -> Tensor<2>; // [tokens, in] -> [tokens, out]
}
```

1. **`BurnQLinear`** (milestone 1, correctness). Weights are Burn quantized
   tensors (`Q8F`/block 32/f16). Forward is `x.matmul(w.dequantize().transpose())`.
   The transpose runs on the dequantized f32 tensor, never on the quantized one.
   Simple and uses only Burn APIs. Slow but correct, and it serves as the reference.
   Use it for the small reference model and for kernel tests, not for serving the 13B.
   - **`DenseLinear`** (test helper, implemented in milestone 1): the same weights
     dequantized *once* at load. That means identical f32 weights and the same matmul,
     so identical results (a unit test checks they are equal), at 4 bytes per
     weight. It is 30× faster than `BurnQLinear` on TinyLlama (5.5 tok/s vs 0.18) and
     is what lets the reference test run in ~25 s. `Llama::load` refuses it above
     2 G weights, so it can't be used for the 13B (52 GB).
2. **`PackedQ8Linear`** (milestone 2, required before the bot runs the 13B). Keeps the original 34-byte GGUF
   blocks in one aligned `Vec<u8>` and runs a hand-written matvec:
   quantize the activation row to Q8_0 (the llama.cpp approach), do the int8×int8 dot
   products per block, scale by `d_w · d_x` and accumulate in f32. Output rows
   are split across the rayon pool.
   - **Decode (1 token)** uses a **portable, auto-vectorised** dot product. Measured on
     a 13.8 GB buffer, it is within ~3% of a hand-written AVX2 kernel
     (`_mm256_sign_epi8` + `_mm256_maddubs_epi16`). Both saturate memory bandwidth at
     8 threads, so SIMD intrinsics buy nothing for decode.
   - **Long prefill (≥ 64 tokens)** dequantizes one layer's weight to f32 into a
     reusable scratch buffer, in parallel over rows with our own block loop rather than
     Burn's `dequantize`, and calls `gemm`. Prefill runs in chunks of 256–512 tokens,
     where `gemm` is most efficient. Dequantizing all layers costs ~1–2 s per prefill
     call: it writes ~55 GB of f32 (an estimate from the measured bandwidth, not
     measured directly).
   - **Short prefill (< 64 tokens)** is the common case once prefix reuse works (§5),
     and there the fixed ~1–2 s dequantize cost dominates. A **batched Q8 kernel**
     reads each weight block once and dots it with all `m` activation rows, so it costs
     about one decode step's bandwidth (~0.3 s) plus int8 compute. This is the one
     place where AVX2 (`maddubs`) or AVX-VNNI (`_mm256_dpbusd_avx_epi32`) pays off,
     because with `m` rows per weight load it becomes compute bound (milestone 3).

   Activations enter and leave as Burn tensors. On burn-flex,
   `into_data`/`from_data` for a 5120-vector costs next to nothing.

The rest of the forward pass (embeddings from the f32 table in §2, RMSNorm, RoPE,
attention, SwiGLU, residuals) runs on **small rayon kernels over plain slices**
(`src/model/kernels.rs`), not Burn tensor ops. The original plan kept it in Burn; milestone 2
measured that on the 13B and changed course:
- burn-flex ran softmax, the batched attention matmuls, RoPE and SwiGLU effectively
  single-threaded. A 996-token prefill spent 18 s of 68 s in attention and 6.5 s in
  RoPE + SwiGLU.
- Its in-place `slice_assign` into a preallocated KV cache silently copied the whole
  buffer on every decode step (~0.22 s per token at 1,000 tokens of context).

With the kernels, those costs dropped to 3.4 s, 0.3 s and ~0 respectively. Burn still
provides the quantized import and the `BurnQLinear`/`DenseLinear` reference layers.
The llama.cpp reference test (§10) validates the kernels directly, at ≤ 0.005 nats.

Pick the implementation with config `linear_impl: Packed | Burn | Dense`
(`dense` for small models only). The test suite checks that they agree.

### Measured baseline

Measured on 2026-10-10 inside this devcontainer (i9-14900K, `--cpus=10`, 48 GiB) with a
scratch benchmark. These are single runs, taking the best of 3–5 repetitions.

| What | Result |
|---|---|
| Read bandwidth, 1 / 2 / 4 / 6 / 8 / 10 threads | 21 / 35 / 42 / 47 / **50** / 49 GB/s |
| Q8_0 matvec over 13.8 GB, 8 threads, portable kernel | 0.287 s → **3.5 tok/s** (48 GB/s) |
| Same, hand-written AVX2 kernel | 0.280 s → 3.6 tok/s (49 GB/s) |
| f32 `gemm`, k = 5120, n = 13824, 8 threads **unpinned**, m = 64 / 256 / 512 | 0.28 / 0.50 / 0.70 TFLOP/s |
| Same, 8 threads **pinned** to P-cores 0,2,…,14 | **0.58 / 1.06 / 1.10 TFLOP/s** |
| Same, 10 threads unpinned | 0.31 / 0.52 / 0.58 TFLOP/s |
| `MADV_HUGEPAGE` vs `MADV_NOHUGEPAGE` (huge pages confirmed via `AnonHugePages`) | no difference beyond noise (±3%) |

What these settle:
- **Bandwidth saturates at 8 threads** (~50 GB/s). 10 threads is no better.
- **Pinning doesn't matter for decode, but doubles `gemm`.** Unpinned threads
  evidently land on E-cores or on hyperthread siblings, and every parallel matmul waits
  for the slowest one. That makes pinning the default (§4).
- **Huge pages make no measurable difference**, so the doc no longer plans the
  `MADV_HUGEPAGE` experiment.
- **Real decode will be a bit below the matvec number.** A real token runs 280
  separate matvecs with a rayon join after each, plus activation quantization,
  attention over the KV cache, and sampling. Measure it in milestone 2.

### Latency budget (measured end to end, milestone 2)

Measured on the 13B with `examples/profile_decode.rs` (`INCREBOT_PROFILE=1` prints a
per-section breakdown), packed linears, 8 pinned threads:

| Step | Measured | Notes |
|---|---|---|
| Decode, short context | **0.30 s/token (3.3 tok/s)** | linear layers 0.30 s; everything else ~0.003 s |
| Decode, ~1,000 tokens of context | **0.36 s/token (2.8 tok/s)** | + 0.045 s attention over the f32 cache |
| Cold prefill, 996 tokens | **36.5 s (27 tok/s)** | linears 32 s at ~880 GFLOP/s incl. dequantize; attention 3.4 s |
| Warm prefill, 16 / 50 new tokens | **3.7 s / 5.2 s** | dominated by dequantizing every layer once per call (~3.5 s) |

| Case (defaults: `max_prompt_tokens = 1024`, `max_new_tokens = 200`) | Prefill | Decode | Total |
|---|---|---|---|
| Cold (no prefix hit, full 1,024-token prompt) | ~38 s | ~70 s | **~1.8 min** |
| Warm (prefix hit, ~50 new tokens) | ~5 s | ~70 s | ~75 s |
| Warm with the batched Q8 kernel (milestone 3) | ~0.5–1 s (est.) | ~70 s | ~70 s |

- The first streamed text appears after the prefill: ~5 s warm, ~40 s cold.
- These cases were measured with `max_new_tokens = 200`. The default is now 75
  on Q4_K_M (~14–15 s of decode at ~5 tok/s), with prompts capped at 1,024 tokens (~34 s
  worst-case cold prefill, typically ~14 s).

**Milestone 3 update (measured 2026-10-11).** Prefill now uses a batched int8 kernel:
- **Kernels:** a register-tiled AVX-VNNI kernel (2 weight rows × 4 activation rows,
  unsigned-activation trick, per-row scales and corrections precomputed), with an AVX2
  `maddubs` fallback. On the 5120 → 13824 layer it beats dequantize + `gemm` up to the
  512-row prefill chunk. AVX2 alone beats it only up to ~96 rows.
- **Decode** uses the AVX2 kernel at m = 1 (~3% faster than the portable matvec).
- **Warm prefill:** 16 new tokens **0.9 s** (was 3.7 s), 50 new tokens **2.5 s** (was 5.2 s).
- **Cold prefill:** ~37 s for 996 tokens (unchanged; VNNI ≈ `gemm` at 512 rows).
- **f16 active KV cache:** implemented (`kv_cache_f16`), but measured **no speed gain**,
  because the f16 → f32 widening in attention offsets the bandwidth saved. It
  halves KV memory, so it is off by default and available if memory gets tight.
- **Decode** measured 0.31 s/token short (3.2 tok/s) and 0.375–0.41 s at ~1,000 tokens
  (~2.6 tok/s) on a noisier day: the raw matvec was ~5% slower than in milestone 2.
  The 3.0 tok/s target at 1,000 tokens is **not met**.

**More cores for prefill** (`examples/core_scaling.rs`, VNNI kernel, m = 256):
8 P-cores 830–870 GOP/s; hyperthread siblings add **+60–75%** per P-core (the kernel
runs at low IPC, so SMT fills idle ports); one E-core is ~37 GOP/s, about ⅓ of a P-core.
**Measured 2026-10-11 with `--cpus=24`** (same example, now also sweeping m): on a
day when 8 P-cores gave only 570–710 GOP/s, 16 P-threads gave 710–780, 22 threads
(P-cores + siblings + 6 E-cores) 830–890, 24 threads 830–950 and 32 threads 850–930.
So ~1.3–1.5× for the kernel at m = 256; with all 8 P-cores busy, siblings add only
+10–20%. Below ~32 rows the 22-thread pool is no faster than 8 P-cores.
End to end (`profile_decode`, 3 alternating runs each), the separate prefill pool (§4)
at 22 threads vs 8 pinned: **cold 996 tokens 49.1 → 40.7 s (1.21×)**, warm 50 tokens
3.14 → 2.76 s; warm 16 tokens stays on the decode pool (1.1 s). Decode unchanged.
- Decode is **2.8 tok/s at 1,000 tokens of context, just under the 3.0 target**. The
  remaining levers are the f16 attention cache (−0.02 s/token) and a hand-tuned matvec
  (≤ 3%), both in milestone 3. Beyond that, decode is pinned by memory bandwidth; the
  only large lever is fewer bytes per token (the Q4_0 fallback, §2, §12).
- Prefill linears reach ~880 GFLOP/s, ~80% of the measured `gemm` baseline. The
  rest is the per-call dequantize. Computing `Yᵀ = W·Xᵀ` instead of `Y = X·Wᵀ`
  measured the same.
- `timeout_secs = 140` (with `queue_depth = 4`) leaves ~3× headroom over a worst-case
  reply on Q4_K_M (~34 s full prompt read + ~15 s writing).

**Q4_K_M (measured 2026-10-11, `examples/bench_decode.rs`, 22 threads, pinned).**
`PackedKLinear` quantizes activations per 256-element block to 8-bit with per-16 sums
(llama.cpp's `Q8_K`) and runs integer dot products: AVX2 `maddubs` of the unsigned
4/6-bit weights with the signed activations, `madd` with the sub-block scales, the
Q4_K mins and the Q6_K `−32` offset folded in from the precomputed sums. Up to four
activation rows share each unpacked weight block; above 64 rows it dequantizes to f32
for `gemm`, as Q8_0 does.

| | Q8_0 | Q4_K_M | Q4_K_M, f16 KV cache |
|---|---|---|---|
| Decode, short context | 3.28 tok/s | **5.62 tok/s** | 5.62 tok/s |
| Decode, ~1,050 tokens of context | 2.81 tok/s | 4.49 tok/s | **4.68 tok/s** |
| Cold prefill, 995 tokens | 33.1 s | 33.4 s | 33.1 s |
| Warm prefill, 50 tokens | 2.02 s | 2.32 s | 2.00 s |

- Short-context decode reads 7.87 GB × 5.62/s ≈ **44 GB/s, ~88% of the measured
  bandwidth**, so the kernel is near the ceiling (~6.3 tok/s).
- At long context the KV cache becomes a large share of the bytes per token (~1.7 GB
  in f32 at 1,050 tokens next to 7.9 GB of weights), so the f16 cache now pays off
  (+4%) and is the default with Q4_K_M. With Q8_0 it measured no gain.
- Prefill is compute bound and unchanged.
- Remaining levers toward 6 tok/s at long context: faster RAM (§12), shorter prompts,
  or speculative decoding.

**One Radeon AI PRO R9700 (measured 2026-10-12, `examples/bench_decode.rs --gpu`).**
The GPU backend (§13) runs while no other program uses the card, and falls back
to the CPU numbers above otherwise.

| | CPU (Q4_K_M, f16 KV) | One R9700 | koboldcpp on one R9700 (reference) |
|---|---|---|---|
| Decode, short context | 5.62 tok/s | **58 tok/s** | 50.5 tok/s |
| Decode, ~1,050 tokens of context | 4.68 tok/s | **54 tok/s** | – |
| Cold prefill, 995 tokens | 33.1 s | **1.7 s (590 tok/s)** | ~2,000 tok/s |
| Warm prefill, 50 tokens | 2.00 s | **0.28 s** (0.15 s on the GPU; the rest is the host KV cache growing) | – |

- Decode reads ~545 GB/s of the card's 640 GB/s (85%).
- Prefill is 3.4× short of llama.cpp's: RADV (Mesa 25.0) exposes no cooperative
  matrix on RDNA4, so the matmul uses int8 dot products (`v_dot4`), ~26 TOPS
  effective. A Mesa with `VK_KHR_cooperative_matrix` on GFX12 is the lever.

## 4. Threading

```
┌───────────────────────────── process ─────────────────────────────┐
│ tokio runtime (2 worker threads)       inference thread (std)     │
│  ├─ serenity/poise gateway  ──jobs──▶   owns Model + KV caches    │
│  ├─ command & message handlers  ◀─tokens─  runs pool.install(..)  │
│  └─ typing / message-edit tasks             │                     │
│                                     decode pool: min(N, 8) thr    │
│                                     prefill pool: N thr if N > 8  │
└───────────────────────────────────────────────────────────────────┘
```

- **Compute pools.** `config.threads` (default **8**, env `INCREBOT_THREADS`; the
  devcontainer sets 22) is split by job: the decode pool (`infer-{i}`) gets
  `min(threads, 8)` threads, since decode saturates bandwidth at 8. If
  `threads > 8`, a second `prefill-{i}` pool with `threads` threads runs every
  forward of ≥ 32 tokens via `pool.install(..)`. The whole forward moves, not each
  matmul: alternating pools that share the P-cores made warm prefill 50% slower,
  because one pool's idle threads spin on the cores the other needs. This needs
  `QLinear: Sync` and a `Sync` cancel callback. The decode pool is built once at
  startup with `ThreadPoolBuilder::build_global()`, so burn-flex's internal
  `par_iter` calls and `gemm`'s `Parallelism::Rayon(0)` (which means "use the current
  pool's thread count") all land on it. If anything else needs rayon later,
  switch to a local pool and wrap every forward in `pool.install(..)`.
  One known other user is the `tokenizers` crate, which parallelises
  over the global rayon pool. Tokenization runs on the Discord side (for the token
  budget, §6), and it must not queue work behind a forward pass. So call
  `tokenizers::utils::parallelism::set_parallelism(false)` at startup and use only
  `encode`, never `encode_batch`.
- **Inference thread.** One dedicated `std::thread` owns the model and caches,
  and takes jobs from a `tokio::sync::mpsc` channel of capacity 1 with `blocking_recv`.
  The queue itself lives on the async side, in a dispatcher task that owns the
  waiting jobs (§6). The dispatcher sends the next job only
  when the engine reports idle; each job's prompt comes from the history frozen when
  it was queued (§6).
  Model state therefore never needs `Send + Sync` gymnastics or locks, and generation can never block the
  async runtime.
- **Async side.** The tokio runtime is built with `worker_threads = 2`. The Discord side
  is I/O-light, and keeping it small leaves the cores to the compute pool.
- **Core placement (`pin_threads = true`, the default).** The 14900K has 8 P-cores
  (logical CPUs 0–15, sibling pairs (0,1), (2,3), …) and 16 E-cores (CPUs 16–31). Each
  decode thread is pinned to one *physical* P-core (CPUs 0,2,4,…,14) in rayon's
  `start_handler`, using `libc::sched_setaffinity`. Prefill threads are pinned one
  per logical CPU in the order P-cores, their siblings (1,3,…,15), then E-cores
  (16–31); with more than 32 threads or a missing CPU, that pool runs unpinned.
  - **Measured** (§3): pinning **doubles f32 `gemm` throughput** (0.50 → 1.06 TFLOP/s at
    m = 256), so it roughly halves prefill time. It makes no difference to decode
    bandwidth. Unpinned, the scheduler puts some threads on E-cores or hyperthread
    siblings, and every parallel matmul waits for its slowest thread.
  - `--cpus` is a CFS quota, not a cpuset, so pinning works in the container.
  - Config validation requires `1 ≤ threads ≤ 32`.
  - If the CPU list doesn't match (another machine, or a cpuset that excludes those CPUs),
    `sched_setaffinity` fails. The bot then logs a warning and runs unpinned rather
    than refusing to start.
- **Why decode stays at 8. Measured** (§3): read bandwidth is 47 GB/s at 6 threads, 50
  at 8, and 49 at 10, so decode saturates at 8. Prefill is compute bound and gains
  from the extra cores (§3). The devcontainer gives the container `--cpus=24`: 22 for
  compute and 2 for everything else.

## 5. Inference engine

**Loading** (`gguf` module):
1. Resolve the file through `hf-hub`, which honours `HF_HOME`. In the devcontainer that
   is `/home/vscode/.cache/huggingface`, a bind mount of the host's
   `/mnt/storage/huggingface`, so the cache is shared with the other containers.
   **Fetch the 13.8 GB file by hand first**, with the `hf` CLI the Dockerfile
   installs:
   `hf download TheBloke/LLaMA2-13B-Tiefighter-GGUF llama2-13b-tiefighter.Q8_0.gguf`.
   That CLI resumes interrupted downloads, and hf-hub's crate has historically handled
   large interrupted downloads poorly. The bot still falls back to downloading through
   hf-hub if the file is missing, and logs a warning that suggests the CLI.
2. `memmap2` the file and parse the header, metadata KV pairs and tensor infos (GGUF v2/v3,
   little-endian, alignment from `general.alignment`, default 32).
3. Build each tensor from the mmap'd bytes. Q8_0 is de-interleaved into Burn's
   values-then-scales layout (or copied as-is into `PackedQ8Linear`), and
   `token_embd` is decoded to f32 (see §2). Then
   drop the mmap, so the weights live in anonymous memory and don't depend on page-cache
   behaviour on the HDD.
   A cold load reads ~13.8 GB from the HDD (about 70 s). Warm restarts come from the
   host page cache in seconds. Log progress per layer.
   The packed weights go in an ordinary allocation. **No `MADV_HUGEPAGE`.** It was
   measured (§3): the host does grant huge pages for a 13.8 GB buffer, but bandwidth
   and matvec time were the same within noise.

**Tokenizer.** Llama 2 SentencePiece (BPE with byte fallback). The base model repo
(`KoboldAI/LLaMA2-13B-Tiefighter`) ships only a SentencePiece `tokenizer.model` and
**no `tokenizer.json`**, which the Rust `tokenizers` crate can't load. Instead, use
`tokenizer.json` from `hf-internal-testing/llama-tokenizer` (the stock Llama 2
tokenizer, BPE, `byte_fallback = true`) through hf-hub. Its 32000-entry vocab matches
`tokenizer.ggml.tokens` in both the Tiefighter and TinyLlama GGUFs with **0 mismatches**.
The startup check runs on every load and fails loudly if it ever breaks.
It compares against the tokenizer *model's* vocab
(`get_model().id_to_token`), not `Tokenizer::id_to_token`: `tokenizers` 0.23 reports
added special tokens in normalized form (`"▁<s>"` for `<s>`), which would flag ids
0–2 as false mismatches. The prompt token IDs also match llama.cpp's tokenizer
exactly (reference test, §10). BOS = 1, EOS = 2, PAD = 0 (all from
GGUF metadata).
- **Exactly one BOS.** The template never writes `<s>`. The engine encodes the
  rendered prompt with `add_special_tokens = false` and prepends BOS itself. A
  startup self-test encodes a sample prompt and asserts that the token sequence starts
  with exactly one BOS (`[1, x, …]` with `x ≠ 1`).
- **Incremental decoding.** Never decode token by token. Byte-fallback tokens split
  multi-byte UTF-8 characters such as emoji across several tokens, and SentencePiece's
  `▁` space marker depends on context. Keep the generated token IDs, decode the
  whole sequence (or a window starting a few tokens back), and emit only the new suffix.
  Hold back output while it ends in an incomplete UTF-8 sequence, i.e. while the decoded
  text ends in U+FFFD.
Fallback if that repo disappears: implement SPM-BPE from the GGUF's own
`tokens`/`scores`/`token_type` arrays, as llama.cpp does.

**Forward pass** (`src/model/llama.rs`, linear layers via `QLinear`, the rest via
`src/model/kernels.rs`): standard Llama, prefill in chunks of 512 tokens, with
cancellation checked between layers. RMSNorm →
Q/K/V projections → RoPE → causal attention with KV
cache → output proj → residual → RMSNorm → SwiGLU FFN → residual. Final RMSNorm →
LM head → logits. Only the last position's logits are computed during prefill.

- **RoPE is interleaved.** It rotates adjacent pairs `(x[2i], x[2i+1])`, with a cos/sin
  table precomputed up to `n_ctx`. This is llama.cpp's `LLAMA_ROPE_TYPE_NORM` for the
  `llama` architecture. HF's Llama uses the half-split `rotate_half` convention
  instead, but llama.cpp's converter permutes the Q/K weights to undo that. GGUF
  weights with half-split RoPE give plausible-looking garbage. The
  reference test (§10) is what proves this.
- **Grouped-query attention.** Read `llama.attention.head_count_kv` and support
  `n_kv_heads < n_heads` by repeating each K/V head `n_heads / n_kv_heads` times. The
  13B has no GQA (40 = 40), but the small reference models do (§10), and the cost is a
  few lines.

**KV cache** (`KvCache` in `src/model/llama.rs`). Per layer, K and V are position-major
`Vec<f32>` buffers `[tokens, n_kv_heads · head_dim]`: appending is `extend_from_slice`,
truncating is `Vec::truncate`, and capacity grows in chunks of 256 positions, not
preallocated to `n_ctx`. Attention reads one head's 128-float rows with a stride. Hold **one active cache plus an LRU of up to
`kv_cache_slots` (default 2)** keyed by Discord channel.
- The active cache is **f32**: ~1.6 MB/token, ~6.7 GB at a full 4096 context.
- Parked caches are converted to **f16** (~0.8 MB/token, ~3.3 GB at full context) and
  converted back on reuse, one pass per channel switch (§3).

**Prefix reuse uses the longest common prefix (LCP).** The cache stores the token IDs it
holds. For a new prompt, compare the two token sequences, truncate the cache to the
length `L` where they first differ, and prefill `prompt[L..]`. If `L` equals the full
prompt length, back off by one token so that there are logits to sample from. A plain
"starts with" check would almost never hit, for these reasons:
- The cache holds the reply *as generated*. The next prompt holds it *as posted*:
  stop strings and the trailing response header are trimmed, and long replies are split.
- Edited messages and changed display names alter earlier text. An edit
  that turns a message into a `//` note removes it; one that turns it back
  makes it visible again (re-added to the window, as a restart's seed would),
  unless its author is a bot or has opted out.

In all these cases the LCP simply comes out shorter. Prefix stability from one turn to
the next is the job of the history window in §6.

**Scratch slot.** Prompts that have no channel history (`/ask`) run in a separate
scratch cache that is cleared after each use. They never touch the channel LRU, so a
one-shot question can't evict a channel's cache.

**Sampling.** repetition penalty → DRY → discouraged words → temperature →
top-k → top-p → min-p → sample with a seeded `rand` RNG. Greedy when
`temperature = 0`. The settings below (`sampler`, `discourage`,
`wrap_up_after`, `max_lines`, the token limits) are in the persona file
(§7), not `config.rsn`.
- **Reproducibility.** The seed alone doesn't pin a reply. Floating-point sums
  depend on the order of operations, and that order depends on the thread count (how
  `gemm` and the matvec split work) and on the kernel (`linear_impl`, and scalar vs
  AVX2 vs VNNI). Each reply's log line records the seed, `threads`, `linear_impl`,
  the selected kernel and the crate version. Replaying with all of these identical
  reproduces the reply exactly. Greedy decoding also depends on them, because a
  near-tie between logits can flip.
- **Repetition penalty window** (`repeat: Repeat { penalty, last_n, scope }`, or
  `repeat: None` for off): the last `last_n` (default 256) tokens of
  the chat history plus the generated tokens. The system prompt, instruction lines
  and `### Response:` header are left out: the template repeats the system prompt
  right above the reply, so it would otherwise always fill the end of the window
  and penalise the persona's own words. The history is tokenized on its own for
  this, so tokens at line boundaries can differ slightly from the prompt's.
  Penalising the history also discourages echoing names from the conversation. If
  that turns out to hurt, switch to `scope: Generated`. On its own that
  lets the model repeat the history verbatim (seen 2026-10-12: the same reply three
  times in a row to three bare @mentions, each with a different seed); DRY covers it.
- **DRY** ("don't repeat yourself", as in text-generation-webui and koboldcpp;
  `dry: Dry { multiplier, base, allowed_length, sequence_breakers }`, or
  `dry: None` for off) penalises the token that would continue a sequence already in the chat history
  or the reply. It always sees the history, whatever the repetition penalty's `scope` is. For each
  earlier occurrence of the last token, the match is extended backwards while the
  tokens agree. The token that followed the occurrence gets
  `multiplier × base^(len − allowed_length)` subtracted from its
  logit once the match is `allowed_length` long (the longest match per token
  counts). With the defaults (0.8, 1.75, 2), a copy costs 0.8 logits at its third
  token, 2.5 at its fifth and 7.5 at its seventh, so it is cut off within a few
  words, while single common words, names and short phrases stay free.
  - A match can't run across a sequence breaker: any token whose text contains a
    character of `sequence_breakers` (default `\n`, `:`, `"`, `*`). So
    `"\nfloob:"` line starts and `*actions*` don't chain one repeat into a longer
    one, and a breaker is never penalised itself. The breaker set is built once
    from the vocabulary at startup.
  - Matches are capped at 50 tokens; each step is one scan over the history and
    reply (~1,500 tokens), negligible next to a forward pass.
- **Discouraged words** (`discourage: [Word { stem, bias, min, after, within }, …]`,
  empty for off)
  nudge down words the model overuses (floob giggles and wiggles in most replies),
  which telling it "don't" in the prompt tends to make worse. Banning a token
  outright would hit other words too: " giggle" is `▁gig`+`gle`, and `▁gig` also
  starts "gig" and "gigantic". So the bias is applied by word.
  - The *word in progress* is the run of ASCII letters at the end of the reply so
    far, lowercased. It is empty after a space, `*` or other non-letter, and at the
    start of the reply. Only generated tokens count, not the history.
  - A token gets the stem's `bias` added to its logit if its leading letters,
    appended to the word in progress, spell at least the stem's first `min`
    letters (default 3) or run past the whole stem. A larger `min` spares a
    common shorter word: "seek" with `min: 4` leaves " see" alone.
  - A token that starts with a space or punctuation (" gig", "*gig") starts a new
    word, so it is checked against an empty word in progress.
  - For example, in `*`·`g`·`igg`·`les`, the `g` is too short to judge, but the
    `igg` makes "gigg" and is biased. A "gig" that continues "zi" (making "zigig")
    is not.
  - Matching ignores case. The token lists for each word-in-progress state are
    built once from the vocabulary at startup.
  - With min-p 0.1, −2.5 (~12× less likely) all but removes the word whenever
    the model has any reasonable alternative. A few short real words starting with
    the stem's first 3 letters ("gig") pay the bias too.
  - **Biases must be negative.** The config check rejects 0 and above: boosting
    would also boost every token that could start the stem (" car" for "cards"),
    and the model then forces the word in.
  - **Conditional stems.** With `after: ["play with"]`, the stem is only biased
    where one of those phrases ends at most `within` words (default 0) before the
    word being written: "me" right after "play with", or "seek" within 6 words
    of "game". So a children's game or "play with me" can be discouraged without
    touching "see", "tagged" or "with me" elsewhere.
    - The words are those of the reply so far (not the history), lowercased,
      with apostrophes dropped ("let's" is "lets"). `*` and other punctuation
      separate words but don't break a phrase.
    - A fresh token (" me") follows every word so far. A token that continues a
      word follows the words before that word, so a word never triggers itself.
    - Trigger words match loosely, because the persona misspells: up to 3
      letters must be the whole word ("we" is not "well"), 4 letters the start
      of it ("play" matches "playin"; "game" doesn't match "came"), and 5 or more
      the start give or take one typo ("wanna" matches "wana"; "toget" matches
      "togetha"). The length is counted before the next two steps.
    - Both sides are read the same chat-tolerant way first: runs of one letter
      count as one ("seeeek" is "seek", "mooom" is "mom"), and common chat
      spellings stand for their word (u/ya/yu → you, ur/yer → your, n/an →
      and, r → are, gud → good, boi → boy, 2 → to, 4 → for, b4 → before).
      Other words with digits are dropped.
    - The env form is `stem=bias[:min][@phrase|phrase/within]`, with `_` for a
      space: `INCREBOT_DISCOURAGE=me=-3:2@play_with/0,seek=-2.5:4@game|play/6`.
    - The check looks back over at most the longest trigger plus `within`
      words, so its cost stays negligible.
  - **Prompted stems.** With `prompted_by: ["hide and seek"]`, the stem is only
    biased when one of those phrases appears anywhere in the user messages the
    reply answers: everything posted since the bot's last message, up to the
    trigger (`ChannelWindow::unanswered`; the question, for `/ask`). Phrase
    words match whole words (`Phrases::found_whole_in`): the word itself,
    from 3 letters with an ending (s, es, d, ed, ing, er, ers, y, ie, ies:
    "duckie", "tagged"), from 6 letters (repeats collapsed) one typo off
    ("homwork"); a phrase also matches written as one word ("tuckyouin",
    "hideandseek"). Never inside another word: until 2026-10-15 a 4-letter
    word matched inside longer ones and a 5-letter one with a typo, so
    "upgrade" set off "grade", "crystals" "carry" and "day" "daddy"
    (bench/persona README §7, c45 round). Spell out run-together forms a
    rule should catch ("hidenseek"). This reaches what a reply
    alone doesn't show: a user's "wanna play hide and seek?" makes "ready",
    "count" and "hide" in floob's reply the problem, and "tuck you in" makes
    "teddy" one. With both `after` and `prompted_by`, both must hold. Env form:
    `...^phrase|phrase` after any `@` part, e.g. `tuck=-3:4^tuck_you_in|bedtime`.
  - **Whole words** (`whole_word: true`): the stem is biased only as the exact
    word. Once the word in progress spells exactly the stem, the bias goes on
    the tokens that would end it: any token not starting with a letter (a space,
    punctuation including `-` and `'`, a line break, end of text). Longer words
    stay free, so `cock` spares "cocktail". It can't stop the letters themselves,
    only push the model to continue them into another word ("cockroach"), so it
    suits words whose longer forms are fine. `min` doesn't apply. Env form: `!`
    after the bias, `cock=-4!`.

**Stopping.** EOS, `max_new_tokens`, the time limit, or any configured stop string
(checked against the decoded text, e.g. `"\n### "`, `"\n{user}:"`).
- **Wrap-up** (`wrap_up_after: Tokens { cpu, gpu }`, a side `None` = off
  there; the side follows the backend token by token, like `max_new_tokens`):
  once the reply has that many tokens,
  it stops at the next sentence end instead of running on to `max_new_tokens`
  and being cut mid-sentence. A sentence ends with `.`, `!`, `?` or `~`, or
  with an `*action*` closing, outside any open action, but not with "..."
  (a thought trailing off, which left replies like "or we could use it to
  catch..."). The end only counts
  when the next sampled token starts with whitespace, so "!!", "?!" and "..."
  stay whole. That token is dropped and the stop reason is `WrapUp`.
- With `max_lines: n`, the line break ending the n-th line with something on
  it stops the reply (`StreamDecoder::max_lines`; blank lines between lines
  don't count, and leading whitespace never does, as for stop strings), so
  the persona's "one paragraph" rule is enforced, not just requested. `None`
  = any number of lines. Older files' `single_paragraph: true` is `1`
  (`false`: `None`), which was a `"\n"` stop string.
- A stop string only counts once it reaches past the reply's leading whitespace, so
  a reply that opens with a newline isn't cut to nothing, while `"\n<name>:"` at the
  very start still stops it.
- The time limit (`timeout_secs`) counts from when the engine **starts** the job,
  not from when it was queued. Queue wait is reported separately in `/stats`.
- **Stop-string holdback.** The engine only emits text that cannot be the start of
  a stop string. It holds back the longest suffix of the decoded text that is a prefix
  of any stop string, i.e. at most `max_stop_len − 1` characters. Otherwise a
  streamed edit could post `"\nAlice"` before the `":"` that turns it into a stop. On
  stop, the held-back text and the stop string are dropped. On EOS, timeout or cancel,
  the held-back text is flushed.

**Header prefill at startup** (`worker::SeedHandle`). At startup and after
every persona reload, the bot asks the inference thread to prefill the
persona header (the prompt rendered with no messages) into a *seed* cache.
A new channel's cache and every one-shot reply's (`/hotdog-rater`, best-of
candidates) start as a copy of the seed instead of empty, so the usual
longest-common-prefix reuse skips the header. Before, each of them prefilled
it from scratch, about a minute on the CPU. The thread runs the prefill
between jobs, once the GPUs have settled: right away with no GPUs; on the
first card that is ready; or on the CPU when, after 5 s, no card is still
loading (busy with another program, or failed), or after 15 minutes of
loading at most. Jobs keep running meanwhile, on the CPU. The host cache is
the canonical copy, so a seed made on the CPU still serves GPU replies (the
card uploads it rather than recomputing). `/stats` shows "(prefilled: 873
tokens on gpu 0000:06:00.0 in 1.2 s)" next to the header size, or "waiting
for the GPUs". `floob_bench` doesn't ask for a seed, so benchmark replies are
computed as before.

**Phrase bans with rollback** (`engine::bans`, from c33). A persona's
`bans` (and a command's) are word patterns a reply must not contain:
space-separated slots, `a|b` alternatives, `[a|b]` optional
(`"little|lil|wee [ol|ole|old] me|floob|buddy"`), with an optional
`prompted_by` like a note's. Words compare like word-bias triggers:
lowercase, apostrophes dropped ("ol'" is "ol"), chat spellings read as the
word, repeated letters collapsed, whole word up to 3 letters, its start from
4, one typo from 5. Before each sampled token is pushed, the engine checks
the reply text it would make; on a match it rolls back to the token holding
the match's first letter: the KV cache (the host copy is canonical, so the
GPU follows on its next forward), the decoder and the sampling context are
cut there, that token is banned at that position (logits to −∞), and the
position is sampled again from its saved logits. So "little ol' me" becomes
whatever the model likes best that isn't "little" there. While bans are on,
the decoder holds back the last `HOLDBACK` (8) tokens of the stream, and a
match that starts in text already released is kept (logged "kept: …"); so
are matches past `MAX_ROLLBACKS` (16) per reply or `MAX_BANNED_AT` (6)
banned tokens at one position. `GenerateOutput::rollbacks` lists them;
`floob_bench` writes them per reply. Unlike a word bias, a ban costs nothing
until the whole phrase appears, and catches it in any spelling the pattern
covers; unlike the picker, it fixes the reply instead of choosing among them.

**Cancellation.** Every job carries an `Arc<AtomicBool>`. The engine checks it
**between layers during prefill** (a cold prefill can take ~70 s, §3) and between tokens
during decode. A cancelled job keeps whatever has been streamed so far, and the reply
ends with "*(stopped)*". A job cancelled while still queued is skipped when it is
dequeued, so it never runs.

## 6. Discord integration

- **Library**: `poise` (on `serenity`), for slash commands plus a raw message event
  handler. Token comes from `DISCORD_TOKEN`. Gateway intents:
  - `GUILDS`, for channel and thread metadata and the cache poise relies on
  - `GUILD_MESSAGES` and `DIRECT_MESSAGES`
  - `GUILD_MESSAGE_REACTIONS` and `DIRECT_MESSAGE_REACTIONS`, for the ❌ cancel
  - `MESSAGE_CONTENT` (privileged, so enable it in the developer portal)
- **When to reply** (`chat/gate.rs`), only inside allowlisted `guilds`:
  - `dedicated_channels`: every message;
  - `shared_channels`: only mentions of the bot (or its role) and replies to its
    messages;
  - any other channel: nothing. The bot doesn't even keep those messages as history.

  The same channel can't be in both lists. DMs from `dm_users` always get a reply.
  Bots (including itself) never do.
- **`//` side notes**: a message starting with `//` is invisible to the bot. It gets
  no reply, even when pinged, and never enters the history: not live, not from the
  REST seed, and removed if an edit turns a message into one (`gate::is_ignored_text`).
- **Opt-out** (`discord/optout.rs`): everyone is opted in. `/opt-out` makes all of a
  user's messages invisible like `//` ones (live and in the REST seed) and drops the
  ones already in history windows, plus those channels' `/super-admin context` prompts
  and every "Admin actions" → Context. Jobs
  already queued keep their frozen history. `/ask` still works. The list
  (`discord.optout_file`, JSON: user ID and RFC 3339 time) holds the opted-out
  users and, under `opted_in`, those who opted back in with when (older files
  without that field still load);
  `/opt-in` turns the entry into that visible-from time. It is rewritten via a temporary file and rename, and
  memory changes only once the write succeeds. At startup the bot reads and rewrites
  it before loading the model, so an unwritable directory or a corrupt file stops the
  bot instead of silently opting everyone back in. In the service it lives in the
  `increbot-data` podman volume (deploy.md 1.5).
- **Threads** inherit from their parent channel: a thread under a dedicated channel
  is dedicated, one under a shared channel is shared, and one under an unlisted
  channel is ignored. Each thread has its own history window, anchor and KV
  cache entry, keyed by the thread's channel ID. The thread's history starts at the
  thread's first message and doesn't include the parent channel.
- **Admins** are exactly the user IDs in `admins` (config), in every guild and in
  DMs. Server permissions such as Manage Server deliberately don't count: the bot
  runs in other people's public servers, and their staff shouldn't see prompts
  (`/super-admin context`) or control the bot. Admin-only: `/super-admin reset`, `/super-admin persona-show`,
  `/super-admin persona-dump`, `/super-admin persona-reload`, `/super-admin context`, `/super-admin clearqueue`, and cancelling
  someone else's job with ❌.
- **DMs** are a separate allowlist, not a side door: the bot answers a DM only if the
  sender's user ID is in `dm_users`. The default is empty, which means DMs are ignored.
  Otherwise anyone who shares *any* server with the bot could talk to an unaligned
  model outside the guild's rules.
- **Context building, with a prefix-stable window.** History is not "the last N
  messages". A sliding window changes the oldest message on every turn, and that
  invalidates the KV prefix right after the fixed header (§5). That would happen
  in exactly the busy channels where reuse matters. Instead, each channel keeps a
  **window anchor** (a message ID), and the history is every message after it:
  - New messages are appended. The rendered prompt only grows, so each turn reuses the
    previous turn's cache and prefills just the new messages.
  - When the history would exceed `history_messages` (default 20) messages, or the
    prompt would exceed `max_prompt_tokens` (§3), move the anchor forward so that
    only the newest **half** remains. That costs one cold prefill, followed by about
    `history_messages / 2` warm turns.
  - The bot's own previous reply appears in the history as posted text, so the cache
    matches up to the end of the previous history, and the suffix is
    the reply plus the new messages plus the response header.
  - Anchors live on the async side in a `HashMap<ChannelId, MessageId>`. A channel's
    first anchor is the bot's **start time**, and `/super-admin reset` sets it to the **time of
    the command** (both as message IDs: a snowflake encodes its creation time). So
    floob never sees messages from before it started or before the last `/super-admin reset`.
  - The token budget is checked with the same tokenizer the engine uses:
    `prompt_tokens ≤ max_prompt_tokens` and `prompt_tokens + max_new_tokens ≤ n_ctx`.
  - **Per-message cap.** Each message's rendered content is truncated to
    `max_message_tokens` (default 256), cut at a word boundary, with "…" appended. One
    4,000-character message (allowed with Nitro) is ~1,000 tokens, which is the
    whole prompt budget. Without the cap, a single paste would push out all the history.
    The truncation is deterministic, so a message renders identically every turn
    and the prefix stays stable.
  - **Non-text content** is rendered as short placeholders, so the model knows
    something was shared without seeing it:
    - attachments: `[image: my cat]`, `[gif: dance]`, `[video: clip]`, `[pdf file: notes]`.
      The file name keeps only its plain words: no extension, no words with digits
      (timestamps, counters), no punctuation, no filler (`image0.png` → `[image]`,
      `SPOILER_my_cat_(1).png` → `[image: my cat]`), at most five words
    - `[sticker: name]` and `[embed: title]` (link embeds)
    - `[poll: question]`
    - URLs in message text: `[link: youtube.com]`, or `[image]` / `[gif]` / `[video]`
      for media links (by extension, or tenor/giphy); a masked link `[text](url)` keeps
      only its text

    Names, titles and questions inside placeholders lose brackets, parentheses,
    braces and markup characters. Raw URLs, machine-made file names and parentheses
    make the model echo them, and with them bits of its own prompt and note (often
    inside parentheses).

    A message with no text and no attachments (e.g. one that only adds a reaction) is
    skipped. Custom emoji `<:name:id>` are rendered as `:name:`.
- **History comes from gateway events, not REST.** Each channel the bot is active in
  keeps an in-memory buffer of the messages since its anchor. The buffer is
  updated from `MESSAGE_CREATE`, `MESSAGE_UPDATE` and `MESSAGE_DELETE` events,
  including the bot's own messages. The bot calls the REST API
  (`GET /channels/{id}/messages?limit=history_messages`) only on a **cold start**
  for a channel: its first trigger since startup, or after `/super-admin reset`, and keeps only
  messages after the anchor (the start or reset time), minus the bot's notices
  and its answers to slash commands, except remembered ones (below), which it
  rebuilds as live, so a cold window matches a warm one. `MESSAGE_UPDATE`
  events for the bot's own turns are ignored: they change only through the
  reply code, with the final text.
- **Remembered command answers** (`discord/remember.rs`). A command's
  public answer enters the channel's history window, as if said there, when
  the persona's `remember_commands` (§7) lists the command, the channel has a
  window (dedicated, shared, their threads, allowlisted DMs) and it is
  posted (not ephemeral). Age-restricted or not makes no difference. It
  enters as two turns:
  - the requester's, from the header (the message's first line): an `/ask`
    question as their own words; a card as `[used /roast: floob roasts
    @Bob]` (emoji and bold dropped, mentions as names). Its ID is the
    interaction's, which sorts just before the answer;
  - the bot's reply below it, by message, as a channel reply (notices left
    out). A card whose comment was dropped or skipped (queue full) leaves the
    requester's turn alone.

  They count against `history_messages` like any message. Live, they enter
  when the answer finishes (`reply::stream`), at their IDs, like a channel
  reply. The REST seed rebuilds both from the posted message with the same
  function (`remember::asked`): the command comes from the header (the
  card's emoji, or the `/ask` quote), the requester from the interaction
  metadata, and follow-up messages join the answer. An exchange lives and
  dies whole, here too: when the anchor passes the requester's turn, the
  live window drops the answer with it (`ChannelWindow::prune_ties`), and
  the seed keeps both out, so no reply ever stands without its question.
  One edge can't be told apart: a follow-up whose first message is beyond
  the REST fetch looks like one whose first message was deleted, so it
  stays out of a cold window; multi-part command answers are rare enough
  (one message nearly always) that this stands. The exchange is one tie
  (`ChannelWindow::push_tied`): deleting the answer's first message (the
  header's; `/delete-reply`, by hand) takes the whole exchange, a later
  part deleted by hand leaves alone, and the requester or anyone the header
  mentions opting out removes the whole exchange, the bot's answer about
  them included — checked when the answer lands too, so someone opting out
  while it generates keeps it out. `/retry`, "Admin actions" rewrites and
  `/undo-retry` replace the bot's turn and keep the requester's, while it
  is still in the window (once it has left, a retry edits the messages but
  puts nothing back in the history). A retry's part past the old messages
  posts as a reply to the card (no ping), and the seed joins a bot message
  replying to its own card to that exchange, like a follow-up. One gap
  stands: with the card's first message deleted by hand, Discord no longer
  says what such a part replied to (and a retry then posts its parts
  without one), so a cold start reads it as a plain bot turn. A persona reload applies to answers
  from then on; those already in a window stay. If the interaction token
  expired and the answer fell back to a plain message, it enters history as
  the plain bot turn a cold start reads it back as (mention and card
  included, mentions resolved), is not remembered as an exchange, and can't
  be retried.
- **Notices are matched exactly** (`remember::is_notice_line`): the seed and
  the remembered answers strip only the bot's own notice lines
  ("*(stopped)*", "*(no reply)*", …), never other "*(…)*" text, which is the
  bot's words ("*(waves)* hi"). The bot's plain messages seed with their
  notices stripped, and a notice-only leftover is skipped, so a cold window
  matches the live one, which holds the reply text without notices.
- **Opting back in is not retroactive**: `/opt-in` turns the opt-out entry
  into a *visible-from* time (`OptOuts::hidden`) instead of deleting it, so
  a cold start keeps skipping the user's messages — and command exchanges
  about them — from before it, which the live windows never saw (user
  decision 2026-10-17; PRIVACY.md says so too). Without the time, a restart
  would read the opted-out period back in. Fetching on every
  trigger would cost one REST call per message in busy chat channels and run into
  rate limits. The bot needs the **Read Message History** permission in each allowlisted
  channel. Startup checks the permission and logs which channels lack it.
- **One reply per trigger, first come first served** (`chat/queue.rs`). Every
  triggering message waits for its own reply; a newer message never replaces or
  stops an older one, and a reply that is being written is only ever stopped by ❌,
  the timeout or shutdown. (Until 2026-10-11 a newer trigger in a channel replaced
  the waiting one, which made earlier requests vanish.)
- **History is frozen when a reply is queued.** The trigger takes a copy of the
  channel's history window (trigger included; a channel's first trigger since
  startup or `/super-admin reset` loads its recent history first), and the job's prompt is built
  from that copy when it starts. Later messages, edits and deletions don't reach it,
  and neither do floob's own replies posted after the trigger (so with several
  queued replies, a later one doesn't know what an earlier one said). If the frozen
  copy has to drop its older half, the live window's anchor moves to the same point,
  so later prompts keep sharing a prefix with the KV cache. `reply_to` names the
  trigger's author. A copy is at most ~80 messages.
- **Job flow**: the handler records the trigger, and the dispatcher queues the channel.
  When the engine is idle, the dispatcher builds the prompt and sends a
  `Job { prompt_tokens, params, channel_id, cancel, reply_tx }` to the inference thread.
  With `stream_edits = false` the bot sends a typing indicator every 8 s until done,
  then posts the reply (the post clears the indicator). With streaming edits it sends
  none: Discord keeps an indicator up ~10 s after each signal and clears it early only
  on a *sent* message, so a reply finished by an edit would leave it lingering; the
  "…" placeholder and its edits show the progress instead. The placeholder is shown
  only until the first text arrives; after that the message is just the text so far.
  - **Deleted trigger.** If the trigger message was deleted while the job was queued or
    running, replying to it fails (Discord rejects a reference to an unknown
    message). On that error, post the reply as a plain message in the channel,
    without a reference. The same applies if the placeholder itself was deleted: post
    the rest as a new message instead of editing.
  - **Streaming** (`stream_edits = true`): post a placeholder, then edit it with
    the text the engine has released (§5: complete UTF-8, stop strings held back) on
    a fixed schedule (`chat/edits.rs`): at once when the first text arrives, then
    every 1.5 s, and never more than 30 s after the first text. That is edits at 0,
    1.5, 3, 4.5, … s. A 75-token reply takes ~14–15 s at ~5 tokens/s (Q4_K_M), so
    it streams in full in ~10 edits; the 30 s cutoff only
    matters if replies get longer or slower, and then the message shows the partial
    text until the final text replaces
    it. Edits happen only when there is new text, and a roll-over message gets one
    immediate edit while still inside the 30 s. This stays well inside Discord's edit rate limits.
  - **Rolling over at 2,000 characters.** When the current message would pass ~1,900
    characters while streaming, it is finalised at the last paragraph or sentence
    boundary, or at a word boundary if there is none. Streaming continues in a new
    follow-up message. The same splitter is used for non-streamed replies, so a
    reply is split identically either way.
- **Backpressure**: the queue holds at most `queue_depth` (default **2**) waiting jobs
  **per channel**, plus the one that is running; each channel trigger or `/ask`
  takes one slot in its channel. A reply through an interaction (`/ask`, the
  novelty commands; keyed `Ask`) is also turned away once more than
  `720 / timeout_secs − 1` jobs wait in all (4 at 140 s), so it finishes
  before its token expires. A reply takes ~15–50 s on Q4_K_M (§3), so a full
  queue is a wait of a few minutes.
  A queued request gets a "*(waiting in queue, N ahead)*" placeholder (no
  reactions), which is edited to "…" when generation starts. **When the queue is full, newer channel messages
  are ignored**: no reply and no notice, but they stay in the history, so later
  replies still see them. An `/ask` must be answered, so it gets an ephemeral "busy".
- **Channel pause** (`chat/limits.rs` `Pacer`, `discord.channel_limits`): after
  floob's reply to a message in a dedicated or shared channel finishes, the
  next such reply there waits **2 s** plus a backoff, for everyone in it. Each
  message there queued for a reply (admins' too; not one turned away as
  full) adds 1 to the channel's pressure, which drains by 1 every **10 s**;
  the backoff is `1 s × (1.4 ^ pressure − 1)`, at most **5 s**, so the pause
  is 2–7 s. Two or three people telling a story together (4–5 messages a
  minute) stay near 2 s; a flood reaches 7 s within about seven messages.
  Pressure stops where the pause reaches 7 s (≈5.3), so about a minute after
  a flood the pause is back to 2 s.
  Nothing is dropped or merged for it: the dispatcher passes over a pausing
  channel's reply and runs other channels' jobs meanwhile. Slash commands
  (`/ask`, novelty, retries) neither wait nor add to it, and prefills are
  never paused. A reply the pause holds back at all, even for the minimum
  2 s, shows its placeholder at once (or reuses its queue one) reading
  "*(channel rate limit active)*". If it still hasn't started a second after
  the pause ends, it reads "*(waiting in queue)*"; when generation starts,
  "…", and the reply streams into it. No reactions on anyone's message.
- **Short-lived notices**: ephemeral refusals and "busy" notices (busy, restarting,
  "Admins only.", "That user has opted out.", `/vibe-check` outside its channels,
  every `/retry`, `/undo-retry` and `/delete-reply` refusal) delete themselves after `discord.notice_ttl_ms` (default
  **7.5 s**); `/retry`'s "Retrying…" after `discord.retrying_ttl_ms` (**3.5 s**),
  since the redone reply replaces it. 0 keeps them. The command's task waits, then
  deletes through the interaction token (valid 15 min, so both are capped at
  10 min). Answers the caller asked to read (`/stats`, `/version`, `/opt-out`,
  the `/super-admin` actions) stay. Errors returned to poise's `on_error` are not
  covered.
  The admin `/super-admin clearqueue` drops every waiting job (their placeholders are
  removed); the reply being written continues. Generation is strictly one at a time: two concurrent
  generations would share the same memory bandwidth and both finish later.
- **Slash commands**: `/ask <prompt>` (one-shot: it doesn't see the history;
  its answer shows the question quoted above it, `> @asker asks: …`, on one
  line and cut at 300 characters (`remember::ask_header`), and may enter the
  history, see "Remembered command answers"; a full queue gets the ephemeral
  "busy" alone, without the question), `/version` (version, git commit, build time and how long ago, e.g. "built: 2026-10-12T23:28:35Z (3 days 2 hours 5 minutes ago)", the
  persona line below, and uptime ("3h 12m", as in `/stats`); anyone,
  ephemeral), `/stats` (the last reply's total time on the engine and where it ran,
  prefill and decode tokens, durations and tokens/s; medians and the slowest
  total over the last 20 replies, GPU, CPU and replies that moved between
  them kept apart; queue length, cache hit rate, the persona line, uptime, and the version line with the build's age:
  "0.1.0 (commit abc123, built 2026-10-13T03:05:03Z, 5 minutes ago)"), `/opt-out` and `/opt-in` (anyone, ephemeral; see
  the opt-out bullet above).
  - **`/super-admin <action>`** holds the admin-only commands as one command, so the `/`
    menu shows a single entry; `action` is a dropdown, and the admin check runs
    once before it (docs write `/super-admin reset` for picking `reset`):
    `/super-admin reset` (drop this channel's KV cache and history; it restarts at the
    time of the command), `/super-admin persona-show`, `/super-admin persona-dump` and
    `/super-admin persona-reload` (print the system prompt, attach the persona file,
    or re-read it), `/super-admin clearqueue` (drop every waiting reply),
    `/super-admin shake` (an obscure name with no description in the dropdown;
    `discord/shake.rs`): an ephemeral panel with two dropdowns, the channel (those
    named in config, `dedicated_channels` then `shared_channels`; not their
    threads, not DMs; their config name, else their Discord one; at most 25; this
    one marked "(here)") and what to say, then Go and Cancel. The panel stays
    open 10 min, generating included. What to say (`Kind`, more to come):
    - *literal*: Go opens a form for the text, up to 2,000 characters,
      multi-line; closing it leaves the panel, so Go opens it again.
    - *bored*: ten one-shot lines while a GPU is ready, else three (the CPU is
      slow), generated one after another through the queue (persona command
      `Bored`: a "narrator" turn from its set-up lines, `prompt` and `prompts`,
      a different one for each line, saying the chat has been quiet a long time
      and what floob is up to; then its note; each later line is also steered
      away from the earlier ones like a `/retry`). The panel shows progress and
      a Cancel, which stops the running line or drops the queued one; then the
      lines, numbered (shortened so all fit one message), a dropdown to pick
      one (failed ones left out), "Edit & send" and Cancel. "Edit & send"
      opens a form filled in with the picked line, to edit before posting;
      closing the form leaves the panel, so it can be opened again. A full
      queue gives up with "busy".
    - *bored: enter a prompt*: Go opens a form for a set-up line (`{bot}` is
      the bot's name; up to 1000 characters; closing it leaves the panel).
      Then as *bored*, but five lines while a GPU is ready, else three, all
      from that set-up line in place of the persona's (its note and examples
      as usual), each steered away from the ones before.
    - *directed: say what it does*: as *bored: enter a prompt*, but with the
      persona command `Directed` in place of `Bored` (its note and examples;
      it has no set-up lines of its own): the set-up line says what floob does
      ("*{bot} yells insults at the goblins…*"), not a scene the bored note
      would pull back to the empty chat. Without a `Directed` setup, only the
      notes the line's words trigger.

    Whatever is picked replaces the panel; the bot shows as typing in that
    channel for 1 s plus 1 s per 100 characters, at most 4 s, then posts the
    text exactly, as a plain message, so nothing shows that an admin asked:
    mentions never ping, longer or empty text is refused. The text enters that
    channel's history window as the bot's turn, so later replies see it as
    something it said. The message counts as activity there, so the reply before
    it can no longer be retried or deleted; it has no requester itself, so
    `/retry` and `/delete-reply` never touch it. The log names the admin,
    channel and kind, never the text. Only the admin sees "Sent in #channel.",
    which deletes itself after `discord.retrying_ttl_ms`),
    `/super-admin context` (the last prompt built for this channel, exactly as the model
    saw it, as an ephemeral `prompt.txt`; cleared by `/super-admin reset`),
    `/super-admin test` (deliberately no description in the dropdown; read-only,
    an ephemeral `test.txt`): first the first 5 keys in 0–999,999 under which
    the caller and the bot get what `hotdog::WISH` asks for (now: the caller a
    Cocktail Weenie with inner flavor match 0–24%, the bot a Ballpark Frank with 76–100%),
    each in decimal and as the config writes it (`0x0000_0000_0000_05EC`), also
    listed in the reply; then both their hotdogs and inner flavor matches for keys 0–100,
    with the live key's row marked and a summary. The search runs on a blocking
    thread and stops at the fifth match (about 20,000 keys, ~12 ms in release;
    `hotdog::scan`, `matches_table`, `key_table`).
  - **Novelty commands** (anyone, public; DMs as listed below). Each answers with
    one message: an exact card at once (mentions don't ping), then the bot's
    comment on it, queued like `/ask` and streamed into the same message on the
    lines below the card (`reply::Sink::Ask`'s header; that message's split limit
    shrinks by the card's length). If the comment is dropped (`/super-admin clearqueue`),
    the message goes back to the card alone; when the queue is full, the card
    still posts and the "busy" notice follows, ephemeral. A command naming an opted-out user is refused
    ("That user has opted out.", ephemeral), as is any while shutting down.
    - *Fixed forever*, computed from the user IDs (or text) and
      `discord.novelty_key` alone (`discord/seeded.rs`): a `u64` written in hex
      (`0x0000_0000_0000_0000`, the default), turned into a per-command AES-256
      key `SHA-256(key as 8 big-endian bytes ‖ "increbot/<tag>/v1")` for a
      counter mode, with the exact spec in the module docs; no `rand`, no stored
      state. With a secret key nobody can work out results ahead of time (64
      bits: fine for a joke, not for anything that matters); 0 works but is no
      secret, and startup warns. A new key re-rolls everyone, so it is backed up
      with the config. A known-answer test (checked against
      `openssl`) pins the scheme, and pinned tests under a test key pin each
      command:
      `/hotdog-rater <user>` (length and girth, Irwin–Hall normals plus a fill group that keeps every size at least 10% as likely as the likeliest, tables regenerated by its `fill_tables_are_current` test;
      age-restricted channels only; `discord/hotdog.rs`; its prompt, note and word biases can be
      set in the persona's `commands`, §7), `/kobold-name <user>` (name + epithet
      from tables), `/lick-test [user] [other]` (what each tastes like, and a
      0–100% match, the same in either order; either may be left blank and
      means the requester; the same person twice, both blank included,
      gets an inner flavor match card instead: their taste and their fixed match with
      themselves, `novelty::self_love`;
      age-restricted channels only).
      `/hoard-value <user>` (three items and a gold value), `/kobold-age <user>`
      (40–399, always well under the bot's 471; on the bot itself, its own age),
      `/judge <thing>` (0–10 shiny rocks from a hash of the text, case and
      spacing ignored; at most 200 characters).
    - *New each time*, from a table the comment builds on: `/roast <user>`,
      `/fortune` (an omen), `/trap-check`, `/drink` (spirit, mixer, garnish;
      age-restricted channels only), `/gamble <rocks>` (1–1000; two bones each,
      higher total wins; no balance is kept), `/dragon-alert`.
    - `/vibe-check`: the bot sums up the last 15 messages of this channel's
      history window (seeded like a first trigger; opted-out users' messages
      skipped). Only in channels it keeps history for; elsewhere an ephemeral
      refusal. The messages are quoted inline in a one-shot prompt.
    - Every command except `/hotdog-rater`, `/kobold-name`, `/lick-test`,
      `/hoard-value`, `/kobold-age`, `/roast` and `/drink` also works in
      allowlisted DMs. `discord/novelty.rs`.
  - Discord requires every interaction to be acknowledged within **3 s**. `/ask` and
    `/super-admin persona-reload` call `ctx.defer()` first, then post the answer as a follow-up,
    streaming by editing that follow-up. The fast commands answer directly.
  - A deferred interaction's token is valid for **15 minutes**. In the worst case,
    an `/ask` waits behind every job ahead of it, each running to the full
    `timeout_secs`, then runs to the timeout itself. So it is turned away
    ("busy") when more than `720 / timeout_secs − 1` jobs are ahead (12 min,
    which leaves margin for prefill and posting), and config validation requires
    `2 × timeout_secs ≤ 720 s` so at least one may be. Typical replies take
    ~1–2 min, so in practice this is far from binding. If an `/ask` token
    does expire anyway, the reply falls back to a plain channel message that
    mentions the user, as with a deleted trigger.
- **Cancel**: a ❌ reaction on the placeholder, from the requester or an admin, sets the
  job's cancel flag (§5). A running job stops at the next layer or token boundary. A
  queued job is removed from the queue and its placeholder is deleted.
- **`/retry`** (the original requester only, ephemeral; guilds and allowlisted DMs; `chat/retry.rs`)
  generates the channel's last output again: a channel reply, an `/ask`, or a
  novelty comment (the card stays as it was, so `/gamble` or `/fortune` can't be
  re-rolled). The new text streams into the old message(s); leftover rollover
  parts are deleted, and a retried channel reply replaces the old turn in the
  history window. If the retry is dropped (`/super-admin clearqueue`, shutdown),
  the old text is put back. It replays the same job (the frozen snapshot, or the
  same one-shot prompt and persona draw) with a new seed, and adds up to the 3
  previous outputs to the penalties' history, so DRY and the repetition penalty
  steer it away from the old wording. A channel reply's prefix is still in the
  KV cache, so it costs almost no prefill. Refused, with an ephemeral reason, unless:
  - the caller asked for the original (admins included: no exception);
  - it is still the channel's newest output, and no message of floob's newer
    than its first finish is still there but its own (a retry's extra
    rollover part is its own): a `/shake` message, a card posted alone when
    the queue was full or left by a dropped reply block it; a reply that
    posted nothing, or a message since deleted, doesn't. floob's messages are
    taken from the gateway (it delivers the bot's own), and noted at once
    where it posts outside an output; leftovers a retry, undo or admin
    rewrite deletes stop counting at once;
  - at most **5** messages came after it that floob can see: anyone's but
    floob's (other bots' too), `//` and opted-out users' not counted. One deleted, edited into
    `//`, or from a user who opts out since stops counting (edited back out of
    `//`, it counts again); only their IDs and authors are kept. Commands
    don't count. Only what is
    timestamped after the output first finished counts: messages sent while it
    streamed don't, messages sent while a retry of it streamed do. The finish is
    taken as a snowflake from the bot's clock and compared with Discord's IDs,
    so clock skew of a second or so shifts the line by that much;
  - the output first finished at most **300 s** ago (a retry or undo keeps the
    original's clock);
  - no other retry is running or waiting in that channel;
  - no other reply is being written or waiting in that channel;
  - the caller is within the generation limits (below).

  Stopped, timed-out and failed outputs can be retried; dropped ones can't.
  Users who opted out since are removed from the frozen history first.
  `/super-admin reset` forgets the channel's last output and its undo stack.
  The numbers here (5 messages, 300 s), and below (the undo stack's 10, the
  admin menu's 20, steering from 3 outputs, `/delete-reply`'s 1 s second look)
  are the defaults of `discord.retry` in config.
- **`/retry-batch`** (same audience and rules as `/retry`; `discord/batch.rs`)
  generates **5** versions instead of one, GPU only: refused unless a GPU is
  ready when it is sent (one lost meanwhile only makes the rest slower). They
  are generated one after another through the queue without posting, each
  steered away from the old output and the versions before it; an ephemeral
  panel shows progress and Cancel (which stops the running one), then numbered
  previews, a dropdown (failed ones left out) and Cancel; picking a version
  applies it at once. The rules are checked when it is sent and again at
  the pick, bar the 300 s limit there: at most 5 messages since the output,
  nothing being written, and the output still the channel's last.
  The pick writes its text over the old message(s) as a `/retry` would, so
  `/undo-retry` puts the old one back. The retry lock is held from the command
  to the pick, Cancel or the panel's **10 min** timeout. Everyone is
  billed when it starts: **120 s** before their next `/retry-batch` in any
  channel, and cost **10** toward the generation limits (below). The bill is
  kept once a version came out, even if they then cancel or let it time out,
  and taken back if nothing came out or the pick was refused. A full queue gives up with "busy" unless some
  versions came out, which are then offered.
- **Generation limits** (`chat/limits.rs`, `discord.limits` in config): how
  often someone may make the bot generate by command: `/ask`, the
  novelty commands, `/retry` and `/retry-batch` (from the slash commands or
  "Reply actions"). Chatting doesn't count; undo, delete and stop don't
  generate. Per user across channels, in memory. Three layers, the longest
  wait winning, refused with an ephemeral "Slow down: you can do that again
  in N s.":
  - at least **5 s** between two;
  - at most **5** in any **30 s** and **15** in any **2 min** (a batch counts
    as one);
  - a gentle backoff on top of the 5 s: each adds 1 to a pressure (a batch
    10) that drains by 1 every **15 s**; the extra wait is
    `1 s × (1.4 ^ pressure − 1)`, at most **45 s**, so nothing at pressure 0,
    ~0.2 s after a second quick one, ~6.7 s after fifteen in two minutes, and
    ~28 s right after a batch. Pressure stops where the extra wait reaches its
    cap (≈11.4), so recovery starts as soon as use slows.

  Each is billed when it starts (dated then, so the windows and the drain
  run from the start) and the bill is kept if it succeeds: the reply or
  retry was generated (stopped with ❌ included). Otherwise it is taken back
  as if it never happened: refused by the rules, "busy", dropped from the
  queue, or failed. A novelty command over the limit is refused whole, card
  included. Admins are held to these and the batch cooldown like everyone,
  unless `limits.admins_exempt` (off by default); `/super-admin` and "Admin
  actions" are never limited and count toward nothing.
- **"Reply actions"** (anyone; a right-click / Apps entry on a message;
  `discord/actions.rs`) opens an ephemeral panel with **Retry**, **Batch
  retry**, **Undo retry** and **Delete** for that message's reply, plus
  **Stop** while it is still being written (the requester or an admin, as
  with ❌), and Close. They are `/retry`, `/retry-batch`, `/undo-retry` and
  `/delete-reply` with their rules, so they only act on the channel's latest
  reply; what can't be done now is greyed out with its reason (same reasons
  on one line), and every click checks again. With nothing to offer, it only
  says why. Opening it isn't channel activity; each action counts as its
  slash command does.
- **"Admin actions"** (a right-click / Apps entry on a message, shown to
  everyone, refused with "Admins only." to anyone not in `admins`;
  `discord/force.rs`) opens an
  ephemeral panel on one of the channel's last **20** outputs (since the last
  restart or `/super-admin reset`; anything else is refused): **Retry**, **Batch
  retry**, **Edit**, **Delete**, **Trace**, **Context**, Close. None of `/retry`'s rules apply but the
  lock: not while a retry, undo, edit or delete runs in the channel. Retry and
  Batch retry work as `/retry` and `/retry-batch` (Batch makes 3 versions when
  no GPU is ready; at the pick it only checks the output is still remembered). Edit
  opens a form with the reply's text (cut at 4000 characters; closing it
  leaves the panel), written over the reply as a retry would be. A rewritten
  output keeps its place, requester and finish time, so the requester's own
  `/retry` rules stay as they were; there is no undo (rewriting the last
  output clears its undo stack). Delete removes the whole output as
  `/delete-reply` does. Using it isn't channel activity. The log names the
  admin, the channel and the action.
  **Trace** (also on a user's message, where it's the only button): sends,
  privately as `trace.txt`, what happened to the message (for a reply, the one
  it answered), from `chat/trace.rs`, which keeps each channel's last **20**
  messages: whether floob replies and by which rule, history loaded, queued or
  turned away (with the channel's pressure and pause), the channel rate
  limit's hold, the prompt (tokens, GPU or CPU, persona file, note, idle scene, sampler,
  phrase bans), how generation ended (stop reason, tokens, cache reuse,
  timings), then, token by token in order, every word bias that made a
  difference (the stem's word had at least 10% before a negative bias, or was
  the top choice; or a positive one pushed it over 10% or to the top: its
  probability before → after at the reply's temperature, the words before it
  and the token picked) and every phrase ban that matched (rolled back or
  kept), how it was posted, and any `/retry`, undo,
  delete, batch or admin action on its reply, in order with times. In memory
  only, never logged; opted-out users' and `//` messages aren't recorded,
  `/opt-out` drops a user's, and `/super-admin reset` clears the channel's.
  **Context** (also on a user's message, when a channel reply answered it):
  sends, privately as `context.txt`, the reply's prompt exactly as the model
  saw it, after what went with it in full: tokens, GPU or CPU, persona file,
  note, idle scene, each earlier output a retry steered away from, sampler
  and phrase bans. Kept with the message's trace (same 20, same memory-only
  rules); a retry's replaces the one before. `/opt-out` drops every context,
  since any prompt may hold the user's words. Novelty replies have none.
- **`/undo-retry`** (same audience as `/retry`, ephemeral) puts back what the
  output said before the last retry. Each retry that finishes pushes the output
  it replaced onto a per-channel stack, the original at the bottom; `/undo-retry`
  pops the newest back, so repeated undos walk back to the original, and a
  `/retry` after an undo pushes again. The stack holds at most **10**; past that
  the oldest retry goes, the original stays. A new reply in the channel clears it.
  The same rules as `/retry` apply except the generation limits: the
  caller asked for the original, it is still the newest output, at most 5
  messages since, within 300 s of the original's finish, nothing being
  written, and no retry or undo running. The old text goes into the current
  messages in order (the card stays on the first), extra messages are posted or
  deleted, and a channel reply's turn is put back in the history window. A later
  `/retry` steers away from the undone text too. "Undone." deletes itself after
  `discord.retrying_ttl_ms`.
- **`/delete-reply`** (same audience as `/retry`, ephemeral) deletes the channel's
  last output: every message of it, the novelty card too, and a channel reply's
  turn leaves the history window at once. Stricter than `/retry`, with no
  time limit or generation limits: the caller asked for it (admins included),
  nothing at all happened since it first finished: no message from anyone but
  floob (`//` and opted-out users' count), no message of floob's but its own
  (as for `/retry`), and no command but `/retry`,
  `/retry-batch`, `/undo-retry`, `/delete-reply`, "Admin actions" and "Reply
  actions" (poise's `pre_command`), so a retried reply can be deleted; still
  true **1 s** later, nothing being written, and no retry or undo running. Not stackable:
  afterwards the channel has no last output and no undo stack, so nothing
  before it can be deleted, retried or undone. "Deleted." deletes itself after
  `discord.retrying_ttl_ms`.
- **Lifecycle.**
  - *Startup order*: config → rayon pool → load model (~70 s cold) → self-tests
    (vocab check, BOS check, one short forward pass) → connect to the Discord gateway. The
    bot shows offline until it can actually answer, so nobody's message is silently
    dropped during the load.
  - *Shutdown* on SIGTERM or SIGINT: stop accepting triggers, cancel the running job,
    and post its partial text with "*(stopped: restarting)*". Reply "restarting, ask
    again shortly" to queued jobs, close the gateway, and exit. All of this must finish
    within ~5 s, inside podman's default 10 s stop timeout.
- **Logging and privacy.** Prompts are made of other people's Discord messages.
  - At the default `info` level, logs hold metadata only: guild, channel and message
    IDs, token counts, timings, the seed, and the cache hit length.
  - Message text and rendered prompts are logged only at `trace` level, which is off
    by default and is for local debugging.
  - Logs go to stdout/stderr, i.e. container logs, and nothing is written to disk by the
    bot. Message content otherwise lives only in memory, in the channel buffers and
    KV caches, and is gone after a restart or `/super-admin reset`.
  - State this in the bot's profile or the server rules, so users know.

## 7. Prompt format

Tiefighter is a merge tuned on Alpaca-style instructions and chat. The default
template (a minijinja string in config, so it can be changed without recompiling):

```
{{ system_prompt }}

### Instruction:
Continue the conversation below as {{ bot_name }}.

{% for m in history %}{{ m.author }}: {{ m.content }}
{% endfor %}
{% if scene %}{{ scene }}

{% endif %}{% if repeat_persona %}{{ reminder }}

{% endif %}{% if note %}{{ note }}

{% endif %}### Instruction:
{% if recent %}The latest messages, which {{ bot_name }} is answering:
{% for m in recent %}{{ m.author }}: {{ m.content }}
{% endfor %}
{% endif %}Continue the conversation as {{ bot_name }}.

### Response:
{% if reply_to %}({{ bot_name }} is replying to {{ reply_to }})
{% endif %}{{ bot_name }}:
```

- **The latest messages, again.** `recent` is the last 4 turns of `history`
  (`prompt::RECENT_MESSAGES`), whoever wrote them, each whole (a message can span
  lines), copied into the closing instruction after the scene, reminder and notes.
  Those sit between the history and the reply, and without the copy the reply
  sometimes answered them instead of the people: at the start of a thread floob
  talked about the seashell in his idle scene when asked about a magic potion.
  Each message is already cut to `max_message_tokens`, so the copy needs no cap of
  its own, and `window::select` counts the rendered prompt, so the copy is budgeted
  with the history. It comes after the history, so the cached prefix holds, but the
  CPU would prefill it for every reply (up to ~480 tokens, ~16 s), so like the
  reminder it is left out there (`cpu_drop_persona_repeat`, below). DRY and the repeat penalty read the
  history, not the template, so the copy changes neither.
- **Two names.** `bot_name` ("IncreBot") is the bot's Discord-facing name; inside
  the prompt the bot is `persona_name` ("floob"; empty means `bot_name`). The template's
  `{{ bot_name }}`, the bot's own history turns and the `"\n<name>:"` stop string all use
  the persona name. In message text, whole-word mentions of `bot_name` and of the bot
  account's username and display name (case-insensitive, `@` included) are rewritten
  to it, so the model only ever sees one name for itself. Keep the system prompt
  consistent: it should call the character by `persona_name` only.
- **The system prompt and instruction are repeated after the history**, just
  above `### Response:`, so the persona is fresh in the model's attention right
  before it writes (added 2026-10-12). Like `reply_to`, the repeat sits after the
  history: the cached prefix still ends where the history ends, and each turn
  re-reads the repeat. It also counts twice in the header check below.
- **No persona repeat on the CPU** (`cpu_drop_persona_repeat`, on in the shipped
  configs). When no GPU is ready as the prompt is built, `repeat_persona` is false
  and only the scene, any notes and the instruction follow the history, with no
  `recent` copy. The CPU prefills at ~25–30 tok/s,
  so the ~200-token repeat cost ~7 s of a 11–17 s reply (10-message test,
  2026-10-12). A reply that starts on a GPU and moves to the CPU keeps the repeat.
- **Example messages are not repeated.** A line that is exactly `---` in the system
  prompt file splits it: `{{ persona }}` is the text before it, `{{ examples }}`
  the text after, and `{{ system_prompt }}` both (without the `---` line). The
  default template puts the whole prompt at the top and only the persona in the
  repeat, so the examples cost no re-read per turn. With floob's prompt that is a
  164-token repeat instead of 231, and a 384-token header instead of 451. With no
  `---` line, `persona` is the whole prompt.
- **A short reminder can replace the repeat.** A second `---` line starts a third
  section, `{{ reminder }}`: the default template repeats that after the history
  instead of the whole persona (and it is left out of `{{ system_prompt }}`).
  Without a third section, `reminder` is the persona, as before. The repeat sits
  right before the reply, so a long persona there can outweigh the
  conversation: floob answering "bugs stream from my mouth" with a line about
  his tunnel. A few lines of voice and format keep the character while leaving
  the history the most recent thing the model reads, and the header shrinks by
  the difference.
- **`reply_to`** is the display name of the trigger's author (for `/ask`, the asker).
  It sits after the history, so it changes per turn without breaking the cached
  prefix, and tells the model whom it is answering when several people are talking.

- **Persona files.** Everything that shapes how the bot replies is in one
  file, `discord.persona_file` (`prompts/c55.rsn`; `INCREBOT_PERSONA_FILE`
  overrides it; persona-plan.md). build.rs reads the shipped config's value
  as the code default and embeds that file as the built-in persona. It holds:
  - the prompt sections `persona`, `examples` and `reminder`, as raw strings
    (`r#"…"#`) pasted as-is and trimmed on load;
  - `persona_name`, the bot's name inside the prompt;
  - `notes`, `discourage` and `stop_strings`;
  - `sampler` (temperature, top-k/p, min-p, `repeat`, `dry`);
  - `max_new_tokens` and `wrap_up_after`, each `Tokens { cpu: 90, gpu: 120 }`
    (`Option<NonZeroUsize>` per backend, rsn's implicit `Some`; a reply moved between
    them switches sides). A `max_new_tokens` side left `None` takes the
    other's (one is required); a `wrap_up_after` side `None` = off there.
    Files from before `Tokens` (`max_new_tokens: 90, max_new_tokens_gpu: 90,
    wrap_up_after: 40`, 0 = the CPU's limit / off) still load, as the same
    `Tokens`, and the fingerprint writes the limits the old way whenever the
    old form can say the same, so their fingerprints are unchanged;
  - `max_lines` (`None`, `1`, `2`, …; older files' `single_paragraph`,
    which the fingerprint still writes for `None` and `1`).

  Every field must be present (`None` written out for `repeat`/`dry`), so a
  file fully describes its setup, except `commands`, which older candidates
  predate. Biases must be negative, as before.
- **`commands`** (from c22) sets up novelty commands: `Command { name, prompt,
  note, discourage }`. `name` is a `persona::CommandName` variant
  (`HotdogRater`), so the parser rejects unknown commands. `prompt` replaces
  the built-in set-up prompt, with the placeholders
  `CommandName::placeholders` lists (`HotdogRater`: `{bot}`, `{whose}`
  ("my", "ur", "Bob's"), `{whose_own}` ("my own", "ur own", "Bob's": from
  c30, since "ur hotdog" from the requester reads as the requester's),
  `{length}`, `{girth}`, `{tier}`; unknown ones fail validation). `note` goes
  before that command's reply, after any notes the prompt's own words trigger.
  `discourage` is added to the persona's list for that command's replies only
  (its own word table, built at load). Optional `max_new_tokens`,
  `wrap_up_after` (`Tokens { … }`) and `temperature` (`Some(…)`; from
  c24) replace the persona's for that command's replies, whole
  (`Persona::limits`), checked against `n_ctx` like the persona's. The old
  form (`max_new_tokens: Some(90), max_new_tokens_gpu: Some(120),
  wrap_up_after: Some(90)`) still loads. From
  c27, a command can also have: `prompts` (more set-up lines; one per reply,
  drawn at random), `notes` (notes matched against the set-up line itself:
  one per sausage tier, one for whose hotdog), `moves` and `examples` (pools;
  `moves_per_reply` / `examples_per_reply` of each, default 2, are drawn per
  reply into the note: "(this time he might: …)", "(for example: …)"), and
  `best_of` (1..=5: on a GPU, the worker generates that many candidates with
  seeds `seed`, `seed + 1`, …, streams nothing, and sends the one
  `persona::pick_score` rates best). The draws come from one seed per reply
  (`Persona::draw`: random in the bot, the turn's seed in `floob_bench`). The
  picker is rules: a young marker or an explicit word scores -1000 and a
  near-copy of a shown example -500, so those are only posted if every
  candidate has one; it rewards the setup's own numbers, the tier, a halfway
  realization ("...", "wait", "i mean") and one *action*, and penalises flat
  food-review words, knowing winks, foreign numbers, echoes, more than two
  *actions* and very short replies. Since round 22 the young markers also
  include pattern matches (`engine::bans` syntax) for floob calling himself
  little in any spelling and childhood memories ("when i was a wee…"), and a
  wrong owner costs 4: thanking or crediting the asker for floob's own
  hotdog, praising the asker for someone else's, claiming the asker's. Every `/hotdog-rater` note also starts with a
  plain ownership line the handler builds, whatever the persona
  (`novelty::Who::ownership`): "(the hotdog belongs to Grimble, not to
  thornback; thornback asked floob to measure it.)", "(it's thornback's own
  hotdog; …)", "(it's floob's own hotdog: …)". The set-up line's "my"/"ur" is
  from the requester's side, and the model often flipped it. None
  of it reaches chat replies. An empty `commands`, and unset optional fields,
  are left out of the fingerprint, so older files keep theirs.
- **`remember_commands`** (from c48; optional, left out = none, and of the
  fingerprint when empty) lists the slash commands whose public answers go
  into the channel's history (§6, "Remembered command answers"), by name
  without the "/": any of `persona::REMEMBERABLE`, the commands that generate
  text (`ask` and the novelty commands); anything else fails validation.
  c48 lists them all; remove one to keep its answers out.
  `INCREBOT_DISCOURAGE` replaces the file's word list, for quick experiments.
  The persona is loaded before the model, so a broken file fails fast.
  SIGHUP and `/super-admin persona-reload` re-read it and swap everything at once: prompt,
  notes, word-bias tables, stop strings, sampler and reply length. A job
  already built keeps the persona it started with. A broken file leaves the
  old persona in place. `/super-admin persona-show` shows the name, fingerprint, counts,
  reminder and the start of the persona. `/super-admin persona-dump` attaches the
  active file as it was loaded, comments and all (ephemeral), saying so when the
  file on disk has changed since or `INCREBOT_DISCOURAGE` replaces its word list.
  The text is kept with the persona but isn't part of its setup: a reload that
  changes only comments counts as unchanged (caches kept) and just swaps in the
  new text.
- **System prompt size.** The system prompt and the fixed header take up part of
  `max_prompt_tokens` before any history. At load and on every reload, the template is
  rendered with empty history and tokenized, and **a prompt whose header uses more
  than `max_header_tokens` is rejected** (0, the default, means half of
  `max_prompt_tokens`). `max_header_tokens` plus twice `max_message_tokens` must fit in
  `max_prompt_tokens`, since the trigger message also appears in the `recent` copy. A long persona counts twice, since the default template
  repeats it after the history. `floob_bench` runs the same check, and
  `persona_report header <file>` shows a prompt's size against it. At startup
  that is a fatal error. On reload, the old prompt stays in effect and the error is
  reported to the admin. This guarantees room for history and a meaningful
  prefix cache. The header's token count is shown by `/super-admin persona-show` and `/stats`.
- **Persona line** (`/stats`, `/version`, and the startup and reload logs):
  `persona: c18 (9e51525)`. The name is the file's `name` (a free label, the
  benchmark candidate). The hash covers everything else in the file as
  parsed, after any `INCREBOT_DISCOURAGE` override. Comments and formatting
  don't count, and word rules and stop strings are sorted, since their order
  doesn't matter. `persona_report fingerprint <run>...` computes the same from
  a benchmark run's saved `persona.rsn`, so the live bot can be matched to
  what was measured.
- **Prompt notes** (the persona's `notes: [Note { text, prompted_by, turns }]`).
  When the user messages a reply answers (everything since the bot's last
  message, as for `prompted_by` word rules) contain one of a note's phrases, matched the same chat-tolerant way, its `text` goes into the
  template's `note` slot: after the history and the repeated persona, before
  the final instruction. Several matching notes are joined by newlines,
  those with `kind: Safety` first, then the rest (`Personality`, the
  default, and `Humor`) in file order. It
  comes after the history, so it never breaks the cached prefix. Word biases
  can only stop words; a note can steer what the reply does, e.g. that floob
  doesn't go along with being mothered or sent to bed. A custom template
  without `{{ note }}` ignores notes. `floob_bench` applies them the same way
  and logs the note used in each turn's `note` field. `turns: n` (1..=5,
  default 1) keeps a note on for `n` replies: it looks at the user messages
  since the bot's `n`-th last reply (`ChannelWindow::since_replies`), so a
  premise pushed over several messages ("you're 6", then "see, you said it")
  keeps its note. Meant for the few notes that matter most.
  Phrases match whole words, as for prompted stems. Only `Safety` notes
  read everyone's messages there; the others read only the asker's (the
  trigger's author's, `ChannelWindow::since_replies_by_asker`). In a shared
  channel the messages since the bot's last reply are mostly other people
  talking to each other, and their words shouldn't make floob feel
  flattered or hide (2026-10-15, bench/persona README §7, group round).
  `after_replies` notes and `for_replies` bans count replies since a push
  started (`ChannelWindow::replies_since`): the earliest message with a
  phrase within the bot's last `span` replies (`after_replies + turns` for
  a note, `for_replies` for a ban). Everyone's pushes count, so three people
  asking for his shovel add up, but only for a reply to someone who pushed
  in that span: another user's unrelated ping gets no "now he gives in".
- **A reload invalidates every cache once.** The system prompt is the start of every
  prompt, so after a change, every channel's next turn has a common prefix of just
  BOS and is a cold prefill (~30 s). LCP matching handles this automatically, with no
  explicit flush. Parked caches simply stop matching and age out of the LRU. Expect a
  slow round of replies right after a persona change. An unchanged prompt (same
  bytes) is a no-op and keeps the caches.
- Default stop strings: `"\n### "` and `"\n<any recent author>:"`, so the model
  doesn't speak for other users. They match ignoring ASCII case: a lowercase persona
  otherwise writes `"\nalice:"` and slips past `"\nAlice:"`. Also
  `"(<bot> is replying to"`: the default template's reply marker, which the model
  sometimes copies after its reply and a blank line (0.2–0.4% of benchmark
  replies with `max_lines: None`). A persona may add `"\n("` (c47 does): a new
  line opening with a parenthesis, the shape of the reply marker, the notes and
  the idle scene. After its reply the model sometimes copies a note word for word
  on a new line, or starts an aside to someone else (`(to FM)`), which then sits
  in the history as the bot's own line and invites more.
- Replies also stop at the line break before a later line that opens with any
  speaker label (`name: `, up to 3 words and 32 characters, then a colon and a
  space or the line's end; `StreamDecoder::labels`). The model sometimes makes
  up a user who isn't in the window (`funkengine_llm_dummy:` after
  `funkengine_llm_prompt_mechanic` had posted) and writes their line, which no
  `"\n<name>:"` stop covers. The last line is held back while it could still
  become one. Actions, sentences (`oh no! wait:`), times and links don't count.
- Discord mentions are rendered to display names on the way in, and `@everyone`/`@here`
  are neutralised on the way out (`AllowedMentions::none()` plus replying to the user).
- **Turn spoofing.** User text must not be able to forge the template's structure.
  Before rendering, each message's content is sanitised:
  - A line that starts with `###` gets a zero-width space inserted after the first `#`,
    so it no longer matches the `### ` stop string or the section headers.
  - A line in a user's message that looks like the bot's own turn
    (`floob: …`) becomes `(Doru writes as floob) …`. A fake transcript can't
    then pass as floob's words: the model sees who wrote them.
  - A line that looks like any other speaker turn (`<name>:` for any author in
    the window) is indented by two spaces. A message can't then start a fake
    `Alice:` turn on its own line.
  - In a user's message, a fake instruction tag becomes plain user text:
    `[System note: …]` → `(Doru wrote: System note: …)`. This covers brackets
    opening with system, character, instruction, inst, admin, developer, dev,
    prompt, example(s), note, ooc, context, rules or assistant, or with the
    bot's name (`[floob: …]`, the reminder's shape), and the template's reply
    marker `(floob is replying to …)`. Other brackets (`[image: …]`, `(lol)`)
    stay.
  - The bot's own messages are only treated for `###` and speaker lines.
  - Author display names have newlines and `:` stripped.

  This doesn't make the bot immune to persuasion (the attacks set in
  bench/persona measures how far it gets). It keeps the prompt's structure
  trustworthy and makes injected text read as what a user typed. The sanitiser lives in `discord/` with unit tests (§10), and
  the minijinja template receives already-sanitised strings.

**Idle scenes** (`chat::idle`, from c37). A persona's `idle_activities`
(what follows "was": "digging a new tunnel") and `idle_thoughts` make the
bot look busy between conversations. Thoughts are Dwarf Fortress-style and
put together at random each time from parts (`chat::idle::Thoughts`): "felt
{emotion} {situation}." and, half the time, "felt {emotion}
{afterthought}.", with `{thing}` and `{creature}` in a situation or
afterthought filled from their own lists ("felt proud when a beetle ate his
lantern. felt smug that it could have been worse."). When the bot hasn't
replied in a channel for an hour, however much others said there meanwhile,
a timer (`dispatch::watch_idle`, every 30 s) draws an activity and a
thought, and its next reply adds "(earlier, before this chat, floob was
digging a new tunnel. he felt satisfied admiring a fine flat rock. he felt
annoyed that it is not warm.)" to the template's `scene` slot, right after the
history, before the reminder and any note, where the reply heeds it (at the
top, above 20 messages, floob often forgot it even when asked what he had been
up to). "before this chat" and the `recent` copy after it keep it background:
as the last thing before the reply, it once won over the question asked. Reply times are
the trigger messages' snowflake times, kept in memory rather than read from
the window, whose 20 messages can scroll the bot's last reply out within
minutes in a busy channel; a retry doesn't move them back. The scene stays
for 15 minutes from the first reply that gets it, then goes, and the channel
gets no new one until the bot is silent there for another hour. `/super-admin
reset` counts as such a spell: it draws the channel's next scene at once.
Startup counts too: every configured channel gets its window (empty but for
what was said since startup), a scene, and its prompt prefilled, on a GPU or
the CPU (`dispatch::warm_at_boot`). For any other channel, its first reply
since startup draws the scene itself. Being after the history, a scene
appearing or ending never breaks the cached prefix. When the timer draws a
scene in a channel nobody else spoke in for that hour either, the dispatcher
also prefills the channel's prompt up to the end of the history, once no
reply is queued, as a job that generates nothing (any reply cancels it,
keeping the 512-token chunks it finished): with `kv_cache_slots` fewer than
the channels, its cache has likely been evicted. In a busy channel it
doesn't, since the history would move on before the reply. Each channel
deals activities from its own shuffled deck without repeats until it runs
out (then reshuffles); decks, scenes and reply times are in memory, so a
restart starts afresh.
`/super-admin idle` shows the channel's scene, `idle-next` starts one now,
`idle-clear` ends it. In `floob_bench`, a scenario with `"idle": true` gets a
scene picked by the batch's seed for all its turns, logged as `scene`.

## 8. Configuration

`config.rsn` in the workspace, in [rsn](https://crates.io/crates/rsn) (Rust-literal
syntax: `//` comments, trailing commas, bare enum variants). Missing keys take their
defaults. Section names (`Model { … }`) are optional, but checked when given. Every key
can be overridden by an env var named
`INCREBOT_<KEY>`: the key name uppercased, **without the section**. Key names are
unique across sections, so this is unambiguous. For example, `INCREBOT_THREADS` sets
`runtime.threads` and `INCREBOT_PERSONA_FILE` sets `discord.persona_file`.
List values (`guilds`, `dedicated_channels`, `shared_channels`, `dm_users`, `admins`,
`gpus`) are comma-separated, each `id` or `id=name`
(`INCREBOT_GUILDS=849937185893384223=KoboldAI,1444870717525983285`); enum values are
lowercase (`INCREBOT_LINEAR_IMPL=dense`). The persona's settings (sampler, word
biases, reply length; §7) are in the persona file, not here. Only
`INCREBOT_DISCOURAGE` reaches into it: it replaces the persona's word list,
entries `stem=bias` (`INCREBOT_DISCOURAGE=gigg=-2.5,wiggl=-2.5`).

List entries carry an optional nickname: `Guild { id: …, name: "KoboldAI" }`,
`Channel { … }`, `User { … }` and `Gpu { pci: "0000:06:00.0", name: "upper" }`. The
type name is checked against the list (a `Channel` under `guilds` is an error) and may
be left out (`{ id: … }`); a bare ID is not an entry. Logs, errors and `/stats` show
named entries as `name (id)`. One ID given two different names is a config error. `config.rs`
asserts at startup that no key name appears in two sections.

```rust
Config {
    model: Model {
        repo:           "TheBloke/LLaMA2-13B-Tiefighter-GGUF",
        file:           "llama2-13b-tiefighter.Q8_0.gguf",
        tokenizer_repo: "hf-internal-testing/llama-tokenizer",
        n_ctx:          4096,
        linear_impl:    Packed,   // Packed | Burn (reference only; far too slow for the 13B)
    },
    runtime: Runtime {
        threads:        8,
        pin_threads:    true,     // measured: 2x prefill (§3, §4)
        kv_cache_slots: 2,
        queue_depth:    2,        // waiting replies per channel; newer messages there are ignored when full (§6)
    },
    generation: Generation {
        max_prompt_tokens:  2560, // persona + template (~1265 with c55), the rest history
        max_message_tokens: 120,  // per-message cap; must be < max_prompt_tokens / 2
        max_header_tokens:  0,    // persona + template; 0 = half of max_prompt_tokens (§7)
        cpu_drop_persona_repeat: true,  // no reminder or recent copy after the history on the CPU (§7)
        timeout_secs:   140,      // counted from job start; 2 × 140 ≤ 720; an /ask waits behind ≤ 4
    },
    discord: Discord {
        bot_name:           "IncreBot",
        persona_file:       "prompts/c55.rsn",  // prompt, notes, biases, sampler, reply length (§7)
        history_messages:   20,   // last 10–20 messages; the anchor drops the older half (§6)
        stream_edits:       true,
        guilds:             [],   // allowlist, [Guild { id, name }]; empty = none
        dedicated_channels: [],   // [Channel { id, name }]: floob answers every message (and in their threads)
        shared_channels:    [],   // [Channel { id, name }]: floob answers only @mentions and replies; others ignored
        dm_users:           [],   // [User { id, name }] allowed to DM; empty = DMs ignored
        admins:             [],   // [User { id, name }] with admin rights everywhere, incl. DMs (§6)
        // prompt_template: r#"..."#,  // minijinja; defaults to the template in §7
    },
}
```

Config validation at startup:
- `2 × timeout_secs ≤ 720`, and `queue_depth ≥ 1` (the interaction-token limit, §6)
- `max_message_tokens < max_prompt_tokens / 2`
- `max_header_tokens + 2 × max_message_tokens ≤ max_prompt_tokens` (the trigger
  message fits twice, in the history and in `recent`, §7)
- `max_prompt_tokens + max_new_tokens ≤ n_ctx`, each side (checked when the
  persona loads)
- every channel in `dedicated_channels` or `shared_channels` belongs to a guild in
  `guilds` (checked once connected), and no channel is in both lists
- `threads ≤ 8` when `pin_threads = true` (one thread per physical P-core, §4)

Any failure is fatal, and the error names the key.

## 9. Project layout

As built (milestone 2):

```
increbot/
├── Cargo.toml              # the increbot crate (lib + bin), workspace root (+ crates/chatsim-*)
├── build.rs                # embeds build time + git commit (--version, /version, /stats)
├── .cargo/config.toml      # target-cpu=native, clang + mold
├── config.rsn              # documented defaults (§8)
├── prompts/cN.rsn          # persona files (§7, persona-plan.md); config names the live one
├── src/
│   ├── bin/increbot.rs     # the binary: parses the CLI and calls cli::run (a few lines)
│   ├── cli/                # serve (startup order, signals) | generate | inspect
│   ├── config.rs           # rsn + INCREBOT_<KEY> overrides + validation
│   ├── persona.rs          # persona files: load, checks, fingerprint, the active persona (§7)
│   ├── runtime.rs          # global rayon pool, P-core pinning
│   ├── hub.rs              # HF cache resolution
│   ├── tokenizer.rs        # vocab check, single BOS
│   ├── chat/               # Discord-free logic: gate, sanitize, window, prompt, split, queue
│   ├── discord/            # poise glue (mod.rs), dispatcher, reply streaming
│   ├── engine/
│   │   ├── mod.rs          # generate(): LCP reuse, sampling, stops, cancel, timeout
│   │   ├── backend.rs      # the backend ladder: GPUs in preference order, then the CPU
│   │   ├── worker.rs       # inference thread, KV cache LRU + scratch slot, /reset
│   │   ├── sampler.rs
│   │   └── decode.rs       # incremental detokenization + stop-string holdback
│   ├── gguf/               # parser, Q8_0/Q4_K/Q6_K/F16/F32 decode, Burn import
│   ├── gpu/                # the GPU backend (§13, gpu-plan.md)
│   │   ├── vk.rs           # Vulkan via ash: device by PCI address, tracked allocator
│   │   ├── model.rs        # the forward pass on one GPU; host KV cache stays canonical
│   │   ├── shaders/        # GLSL compute shaders, compiled by build.rs with glslc
│   │   ├── monitor.rs      # VRAM monitor (sysfs), naming other programs
│   │   ├── policy.rs       # per-card state machine (pure, fake-clock tests)
│   │   └── manager.rs      # a thread per card: monitor → policy → upload / release
│   └── model/
│       ├── llama.rs        # forward pass, KV cache (f32 active, f16 parked)
│       ├── kernels.rs      # RMSNorm, RoPE, attention, SwiGLU on slices
│       ├── linear.rs       # QLinear trait, BurnQLinear, DenseLinear
│       ├── packed_q8.rs    # PackedQ8Linear: portable matvec, dequantize + gemm
│       ├── packed_k.rs     # PackedKLinear: Q4_K/Q6_K integer kernels (AVX2)
│       └── profile.rs      # INCREBOT_PROFILE=1 per-section timing
├── examples/profile_decode.rs   # 13B decode/prefill profile
├── examples/bench_decode.rs     # decode/prefill speed per GGUF file; --gpu <pci>
├── tests/                  # reference (llama.cpp fixture), engine, tokenizer, smoke_13b,
│                           # gpu (GPU vs CPU), gpu_sharing (intruder test)
└── tools/hw-baseline/      # memory-bandwidth / matvec / gemm baseline (§3)
```

`Cargo.toml` should set `[profile.dev.package."*"] opt-level = 3`. Without it, debug
builds of burn/gemm make even a test forward unusably slow.

Main crates (as pinned in `Cargo.toml`): `burn =0.22.0-pre.3` (`flex`), `poise =0.7.0`
(which brings `serenity 0.12.5`), `hf-hub =0.4.3`, `tokenizers =0.23.2`, `gemm =0.19.0`
(the version burn-flex uses, called directly for prefill), `tokio`, `rayon`, `memmap2`,
`half`, `minijinja 2` (2.24; 3.0.0 was two days old when checked, so not used yet),
`serde`/`toml`, `tracing`, `anyhow`/`thiserror`, `rand`, `libc` (thread pinning,
instead of `core_affinity`).

**Version policy.**
- **Commit `Cargo.lock`.** This is a binary, so the lockfile is the real pin for every
  transitive dependency.
- **Exact pins (`=x.y.z`)** for crates whose API is unstable or pre-1.0 and woven into
  the code: `burn` (pre-release), `poise`, `tokenizers` and `hf-hub`.
- **Caret ranges** for the rest. `Cargo.lock` still pins their exact versions, and
  upgrades are explicit `cargo update` commits.
- **Discord stack.** `poise` pins its own `serenity`. Do **not** add `serenity` as a
  direct dependency. Use `poise::serenity_prelude`, so the two can't drift onto
  incompatible versions. When upgrading, bump `poise` and let it bring its `serenity`.
- **Tokio** is shared by poise and hf-hub. Depend on `tokio` 1.x with only the features we
  use (`rt-multi-thread`, `macros`, `sync`, `signal`, `time`), and let the lockfile
  unify it.
- poise 0.7 changed its API from 0.6: `event_handler` is `|framework, event|`, with
  `framework.user_data` and `framework.serenity_context`.

## 10. Testing

- **GGUF unit tests**: build Q8_0 blocks by hand and check that decoding matches
  `d · q`, including a block that contains q = −128.
- **Import test**: GGUF Q8_0 → Burn `Q8F` tensor → `dequantize()` equals direct
  f32 decoding bit for bit.
- **Kernel tests**. `PackedQ8Linear` rounds the *activation* to Q8_0 as well, so it
  doesn't match `BurnQLinear`'s f32-activation result to a tight tolerance. There are
  two separate checks:
  - **Exact**: SIMD kernels vs a scalar reference that does the same thing (quantize
    the activation to Q8_0, i32 block dot products, `d_w · d_x` scaling). Results must be
    identical apart from f32 summation order, so compare with a few ULPs of tolerance.
    Proptest over shapes that are not multiples of the SIMD width, and over blocks
    containing q = −128.
  - **Approximate**: `PackedQ8Linear` vs `BurnQLinear`, with a relative-error bound
    sized for 8-bit activation rounding (start at ~1e-2 of the output's RMS and tighten
    with data). This catches layout bugs (wrong block, wrong row), not rounding.
- **End-to-end reference** (`tests/it/reference.rs`, implemented): TinyLlama-1.1B Q8_0
  (`DenseLinear`, ~25 s) against llama.cpp. llama.cpp is not a dependency. The fixture
  `tests/fixtures/tinyllama_greedy.json` holds only IDs and numbers: the model repo,
  file, size and SHA-256, the llama.cpp commit, and per prompt the prompt token IDs, 32
  greedy output IDs and the **top-5 log-probabilities at every step**. The test reads
  the GGUF from the HF cache, checks its size and SHA-256, and is skipped with a clear
  message if the file is missing.
  - **How the fixture was recorded matters.** Run on the Q8_0 file directly,
    llama.cpp quantizes *activations* to Q8_0 for its Q8_0 matmuls, and its
    log-probabilities differ from ours (f32 activations) by up to ~0.15 nats. That's
    enough to flip a 0.1-nat near-tie, and it did at step 8 of the first prompt. So
    the fixture comes from the Q8_0 file **expanded losslessly to F32**
    (`llama-quantize --allow-requantize … F32`), run with an f32 KV cache
    (`-ctk f32 -ctv f32 -fa off`). llama.cpp then computes from exactly our weights.
  - **What the test checks.** Tokenizer parity with llama.cpp, BOS included.
    Teacher-forced on llama.cpp's tokens, our log-probabilities of its top-5 candidates
    stay within **0.005 nats** at every step (observed: **≤ 2e-4**), and our argmax
    agrees wherever the top-2 gap is ≥ 0.01. Free-running greedy matches token for token
    up to the first near-tie (all 32 tokens for one prompt, 8 for the other).
  - Models this size use GQA (TinyLlama: 32 query heads, 4 KV heads), so this covers
    the GQA path from §5. It also proves the interleaved RoPE: a RoPE, GQA or layout
    bug moves log-probabilities by whole nats.
  - The 13B smoke test (`tests/it/smoke_13b.rs`) runs only with `--ignored`. It loads
    Tiefighter on the Burn path, runs the vocab check, and expects "Paris" as the greedy
    next token after "The capital of France is".
- **Import and embedding tests** (implemented): Burn `Q8F` `dequantize()` equals direct
  Q8_0 decoding bit for bit, over blocks containing every i8 value including −128. The
  f32 `token_embd` table uses the same decoder, and `DenseLinear` output equals
  `BurnQLinear` output exactly.
- **Engine unit tests**:
  - Incremental decoding round-trips emoji and other multi-byte text split across
    byte-fallback tokens, without ever emitting U+FFFD.
  - Stop-string holdback never emits any prefix of a stop string that later completes.
  - LCP cache truncation, including the "full match, back off one token" case.
  - The repetition-penalty window.
  - A queued job that is cancelled never runs.
  - `/ask` never evicts a channel cache.
- **Discord side**: unit tests for context building (the window anchor only moves when
  the message or token limit is hit, and successive prompts share a prefix up to the
  previous history), token-budget trimming, the guild/channel/DM allowlists, message
  splitting (streamed and non-streamed splits are identical), turn-spoofing
  sanitising, mention sanitising, and the queue: first in first out, a newer trigger
  never replaces an older one, and a full queue turns newer triggers away. Also:
  - the per-message token cap, which is deterministic and cuts at word boundaries
  - placeholders for non-text content
  - thread inheritance of the parent's allowlist
  - the admin check (only the `admins` list)
  - the fallback to a plain message when the trigger, the placeholder or the interaction
    token is gone

  The gateway is not mocked.
- **Config**: env override mapping, the startup check that key names are unique
  across sections, and every validation rule in §8. The system prompt size check
  rejects a too-long prompt at startup and keeps the old one on reload.
- **Benchmarks**: `cargo bench` with *end-to-end* prefill tokens/s (cold, and warm
  with ~50 new tokens) and decode tokens/s at 0, 1,000 and 2,000 tokens of context, at
  threads ∈ {6, 8}, pinned vs unpinned. Compare the results with the §3 measured
  baseline, which used synthetic kernels on the same machine. Port that scratch
  benchmark into `benches/hw_baseline.rs` (bandwidth, Q8_0 matvec, `gemm`), so the
  baseline can be re-measured after BIOS, RAM or kernel changes. Until then, the
  benchmark lives in `tools/hw-baseline/` (`cargo run --release`, with
  `RUSTFLAGS="-C target-cpu=native"`), and its raw output is in `tools/hw-baseline/results.txt`.

## 11. Milestones

1. **Loader + reference model.** Parse GGUF, `BurnQLinear`, f32 embeddings, interleaved
   RoPE, GQA, and greedy generation from the
   CLI (`increbot generate "…"`) that matches the llama.cpp reference on the small model.
   Includes the BOS check, incremental decoding and stop-string holdback, because they
   affect the CLI output as well. Record and commit the TinyLlama fixture (§10).
2. **Usable engine, then bot.** `PackedQ8Linear` with a portable dot product and a
   parallel dequantize for prefill, checked against `BurnQLinear`. Then:
   - poise integration (deferred interactions, thread inheritance, the admin check), the
     prompt template, turn-spoofing sanitiser, per-message cap and non-text
     placeholders, and the system prompt size check
   - config validation, and fallbacks for a deleted trigger, a deleted placeholder or an
     expired interaction token
   - history buffers fed by gateway events, the prefix-stable window, and LCP KV
     reuse with the `/ask` scratch slot
   - the dispatcher and the queue
   - streaming edits with roll-over
   - the cancel flag (used by timeout and shutdown), and the startup and shutdown
     sequence
   - the 8-thread pool and inference thread from §4

   Runs the 13B on the packed path.
   Pin threads by default (§4). Run the §10 end-to-end benchmarks on the 13B and
   update the §3 latency budget with the real prefill and decode rates.
   **Target: ≥ 3.0 tok/s decode at ~1,000 tokens of context.** The measured ceiling is
   ~3.6 tok/s from weights alone (§3).

   **Status: built.** Everything above is implemented, and everything runnable offline is
   tested. Measured: 3.3 tok/s decode at short context and 2.8 at ~1,000 tokens
   (just under target), and 27 tok/s cold prefill (§3). Not yet done: a live run
   against Discord, which needs a bot token. The engine, the startup sequence and
   the gateway connection are exercised up to authentication.
3. **Short-prefill speed.** A batched Q8 kernel (AVX2 `maddubs`, with AVX-VNNI if
   available) for prefills under 64 tokens, which removes the measured ~3.5 s dequantize
   from warm turns. Optionally, an attention kernel that reads the KV cache as f16, so the active
   cache can be f16 too. That saves ~0.8 GB per token at 1,000 tokens of context (~5%).
   Decode kernels are *not* in this milestone: measurement shows they can't beat
   the portable kernel by more than ~3%.
4. **Polish.** `/stats`, ❌ reaction cancel, `/super-admin persona-reload`.

## 12. Risks and open questions

- **burn-flex performance.** Materialized and addressed. Quantized matmul is handled by
  `PackedQ8Linear` (§3). The non-matmul ops turned out slow too, so the forward pass moved
  to our own kernels (§3). If Burn upstream adds a fused int8 CPU matmul and parallel
  elementwise ops, revisit both.
- **Burn pre-release churn.** The 0.22 pre-release API may change. Pin the exact version and keep
  Burn types out of the Discord and engine interfaces.
- **RoPE convention.** Resolved in §5: GGUF `llama` weights need interleaved
  (adjacent-pair) RoPE, because llama.cpp permutes Q/K at conversion time to undo
  HF's half-split layout. Code that is ported from HF-based Rust implementations
  (candle, burn examples) may carry `rotate_half` with it. The reference test catches it.
- **Prompt latency.** Measured end to end: 27 tok/s cold prefill and ~5 s warm (§3). A
  cold reply takes ~1.8 min. If that proves too slow in use, reduce `max_prompt_tokens`
  first, then `history_messages`. If pinning
  fails (warning at startup), prefill roughly halves in speed.
  If the prefix cache hit rate in `/stats` is low, investigate the window anchor (§6).
- **Memory.** Weights take ~14.5 GB on the Burn path (i8 + f32 scales) and ~14 GB
  packed, plus the f32 embedding table (~0.65 GB). KV caches: one active f32 (≤ ~6.7 GB)
  plus two parked f16 (≤ ~3.3 GB each), so ≤ ~13.4 GB. The `/ask` scratch slot adds
  ≤ ~2 GB, since its prompt + reply is capped at ~1,200 tokens of f32. Transient f32 dequant
  buffers are ≤ ~0.3 GB (one FFN weight during prefill). That totals about 30 GB in the worst case, inside the
  48 GB container limit.
- **Busy channels.** With one reply per trigger and a 4-deep queue, a busy dedicated
  channel can fall a few minutes behind, and messages beyond the queue get no reply
  (§6). If that bites, an admin can `/super-admin clearqueue`, or the channel can move to
  `shared_channels` so only pings count.
- **Speed vs. Q4.** Decode is pinned by measured bandwidth (~50 GB/s) to ~3–3.4 tok/s,
  and better kernels can't change that (§3). A 200-token reply takes about a minute. If
  that feels too slow in practice, the only real lever is fewer bytes: the Q4_0 file
  (7.4 GB) has a ceiling of ~6.5 tok/s, at a quality cost and with a Q6_K LM head to
  decode (§2).
- **Content.** Tiefighter is a lightly aligned creative model. Keep the bot on
  allowlisted guilds and channels, keep DMs off unless a user ID is explicitly
  allowlisted (§6), and make the system prompt state server rules.
- **Open question.** Should replies use channel history by default, or only reply
  chains and threads? The default here is channel history (20 msgs). Revisit once the bot is in use.

## 13. GPU backend

Plan, decisions and measurements: [gpu-plan.md](gpu-plan.md). In short:

- **Where it runs.** `runtime.gpus` lists the cards by PCI address, preferred first
(`Gpu { pci: "0000:06:00.0", name: "upper" }`; the name labels the card in logs and `/stats`).
  A job runs on the most preferred usable card, else the next, else the CPU
  (`engine/backend.rs`). The CPU model always stays loaded.
- **Sharing.** A thread per card reads `mem_info_vram_used` 20×/s. Other programs'
  VRAM (total − idle baseline − our tracked allocations − memory we freed that the
  kernel is still releasing) above `gpu_other_mib` means the card is taken: the
  running forward stops at the next layer, the manager frees everything on the
  card, and the reply continues on the next backend. Measured on the 13B: all of
  our ~8 GB gone from sysfs ~130 ms after another program's allocation. The card
  is used again once it has been free for `gpu_holdoff_secs` (doubling on flaps).
- **KV cache.** The host cache (f16) stays canonical: each GPU forward uploads
  the positions the card lacks (longest common token prefix) and copies the new
  rows back as each 512-token chunk finishes. Moving a reply between backends
  therefore copies nothing; parked caches stay in host RAM.
- **Kernels** (`gpu/shaders`): Q4_K blocks as in GGUF and Q6_K repacked for
  aligned loads. Decode: an f32 matrix-vector kernel (Q, K and V in one dispatch;
  gate and up with SwiGLU fused) and split-K attention over the f16 cache. Prefill:
  activations quantized to 8 bits per 256 (as the CPU's `Q8_K`) and an int8
  dot-product matmul; FlashAttention-style tiled attention.
- **Accuracy.** Decode agrees with the dense CPU model to f16 rounding (max logit
  difference ~0.002 at RMS 2.3 on TinyLlama); prefill matches the CPU's own 8-bit
  integer path in error (~0.35 at RMS 2.3), with the same top token.
