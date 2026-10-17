# Panics plan

Status: **5 more sites fixed, 2026-10-17** (round 5, on top of four earlier
rounds the same day — most of them from reconsidering the `vk.rs` decline
from round 3 and finding a real Result-propagation path that doesn't need
the restructure that made the typestate option too risky). 27 of the
original 32 non-test call sites are now gone for real — **5 remain**
(confirmed by re-scanning the tree): 3 in the live bot, 2 harmless sim-tool
sites. All deliberately kept, with reasons recorded below. Details below,
under "What's done".

## Goal

Make every panic-capable call in production code (`.unwrap()`, `.expect()`,
`panic!`, `unreachable!`, `todo!`, `unimplemented!`) either removed in favor
of a real error path, or deliberately kept with a comment explaining why it
can't fire. Test code is out of scope: `#[cfg(test)]` modules and functions
are expected to use `.unwrap()` freely.

## Why this matters here specifically

A panic's blast radius depends on which thread it happens on, and floob has
two different outcomes:

- **Main thread panic → process exits.** systemd has `Restart=on-failure`
  with backoff from 10 s to 10 min (`deploy.md`), so this is self-healing:
  the bot goes offline for ~10 s and comes back. Not free, but bounded and
  visible (Discord shows the bot as offline).
- **Background thread panic → silent hang, no restart.** The inference
  thread (`src/engine/worker.rs`) and the GPU manager/upload threads
  (`src/gpu/manager.rs`) aren't joined, there's no `catch_unwind` or
  `panic::set_hook` anywhere in the codebase, and the binary doesn't set
  `panic = "abort"` in `Cargo.toml`. If one of these threads panics, it dies,
  its side of the job channel drops, and the main thread's send/recv calls
  on that channel start failing or blocking. systemd sees the process as
  still running — no restart, no backoff, nothing in the journal past the
  panic message. The bot looks alive in Discord (or goes quiet) with no
  automatic recovery. **This is the failure mode worth hardening against**,
  not main-thread panics, which already self-heal.

So the priority order is: background-thread panics in the live bot binary
first, main-thread panics second, and the standalone sim/TUI tools
(`crates/chatsim-core`, `crates/chatsim-tui` — not linked into the bot
binary; see `Cargo.toml` workspace members) last, if at all.

## What's there today

Scanning `src/` and `crates/` for the panic-capable macros/methods above,
excluding `tests/`, `examples/`, `benches/`, and inline `#[cfg(test)]`
modules/functions, finds 32 call sites. None are bare `.unwrap()` on
unvalidated Discord/network input; all carry an `.expect()` message or are
`unreachable!()` on an enum arm ruled out just above. Grouped by where they'd
actually hurt:

### In the live bot's background threads (highest priority)

| File | Lines | What |
|---|---|---|
| `src/engine/worker.rs` | 238, 282, 407, 506 | timer runtime, thread spawn, "at least one candidate", "index from position" |
| `src/gpu/manager.rs` | 170, 325, 360 | GPU manager/upload thread spawn, "checked above" |
| `src/gpu/vk.rs` | 690, 695, 839 | "not recording" — command-buffer state invariant |
| `src/gpu/model.rs` | 195 | "seven linears" — fixed tensor count |
| `src/model/linear.rs` | 28 | "f32 tensor" |
| `src/model/llama.rs` | 401, 495 | `unreachable!()` on enum arm, "never cancelled" |
| `src/model/packed_q8.rs` | 178 | `unreachable!()` on kernel dispatch |
| `src/gguf/mod.rs` | 270, 316 | `unreachable!()`, fixed-size array `try_into()` |

### In the live bot, main/request-handling path

| File | Lines | What |
|---|---|---|
| `src/chat/sanitize.rs` | 132, 198 | char-boundary / just-matched invariants |
| `src/discord/seeded.rs` | 46, 80, 81 | fixed-size slice of a SHA-256 digest |
| `src/persona.rs` | 583 | parsing hardcoded ban-pattern strings, once, via `OnceLock` |

### In the sim/TUI tools only (not the bot binary)

| File | Lines | What |
|---|---|---|
| `crates/chatsim-core/src/room.rs` | 39, 602, 650, 657, 671 | built-in persona parse, state invariants |
| `crates/chatsim-core/src/scripted.rs` | 36, 176, 208, 210 | mutex lock, "still active" |

## Risk tiers and what to do about each

**Tier 1 — genuinely can't fire, leave as-is.** Fixed-size conversions
(`seeded.rs`, `gguf/mod.rs:316`), enum arms ruled out one match-arm earlier
(`llama.rs:401`, `packed_q8.rs:178`, `gguf/mod.rs:270`), and char-boundary
invariants right after the code that establishes them (`sanitize.rs`). These
are documentation of an invariant, not a risk. No change.

**Tier 2 — spawn/runtime-creation failures.** `worker.rs:238,282` and
`manager.rs:170,360` all `.expect()` on `thread::Builder::spawn` or a tokio
runtime build. These only fail under OS resource exhaustion. Checked where
each one actually runs (the open question this tier started with): `worker.rs:282`
and `manager.rs:170` are on the *main* thread (the `.spawn()` call itself, in
`serve()`'s synchronous startup), so they're fixed below by propagating a
`Result` instead of panicking. `worker.rs:238` (the inference thread's own
tokio runtime) and `manager.rs:360` (a per-card upload thread, spawned from
inside that card's manager thread) are *not* — they run on background
threads and are covered by Tier 3's fix instead.

**Tier 3 — state invariants inside the worker/GPU threads themselves.**
`worker.rs:407,506`, `manager.rs:325`, `vk.rs:690,695,839`, `model.rs:195`,
`linear.rs:28`, `llama.rs:495`. These run *inside* the background thread, so
if the invariant is ever wrong, the failure mode is the silent hang
described above, not a clean restart. This is the tier worth actually
changing:

- Where the invariant really is guaranteed by surrounding code (e.g.
  `model.rs:195` "seven linears" from a fixed model architecture), keep the
  panic but make sure it happens *before* the thread starts doing real work,
  so a wrong build fails fast and loud on startup instead of mid-inference.
- Where it's cheap, wrap the thread's main loop body in
  `std::panic::catch_unwind` (or a small helper) so a panic inside becomes a
  logged error and a clean shutdown of that thread's channels — enough for
  the main thread to detect the disconnect and either exit (triggering the
  systemd restart) or report the failure visibly, instead of the current
  silent half-alive state.
- At minimum, have the main thread treat a closed job/result channel from
  the inference or GPU thread as fatal and exit, so *any* panic in those
  threads converts into the main-thread-panic failure mode (bounded, visible,
  self-healing) rather than the silent one. This is probably the single
  highest-value change here, and doesn't require touching the 32 call sites
  at all.

**Tier 4 — sim/TUI tools.** Not linked into the bot binary; no user-facing
availability impact. Leave as-is unless someone's actively using these tools
and finds the panics annoying.

## Non-goals

- Replacing every `.expect("msg")` with `Result` plumbing. Most of these
  messages already say why the call can't fail; turning them into
  `Result<_, _>` just moves the "this should never happen" branch to a
  caller who can do even less about it.
- Touching `crates/chatsim-core` / `crates/chatsim-tui` unless the channel
  draining they're missing becomes an actual problem for whoever runs them.
- Adding `panic = "abort"` to `Cargo.toml`. That would turn the Tier-3
  silent-hang cases into process exits too, which sounds good, but it also
  removes the ability to `catch_unwind` at all, closing off the Tier-3 fix
  above. Decide one or the other, not both.

## What's done

Both the main-thread Tier 2 fixes and the cheap Tier 3 win (treating a dead
inference thread as fatal) are in. No `.expect()` was removed outright —
these are the two cases where propagating up actually has somewhere useful
to go. Verified with `cargo build --workspace --all-targets`,
`cargo clippy --workspace --all-targets` (no new warnings), and
`cargo test --workspace` (296 unit + 20 integration tests passing).

- **`engine::worker::spawn`** (`src/engine/worker.rs`) now returns
  `anyhow::Result<EngineHandle>` instead of panicking if the OS won't hand
  out the inference thread. `cli/serve.rs` and the two other callers
  (`crates/chatsim-core/src/engine.rs`, `bench/persona/floob_bench.rs`)
  propagate it with `?`; `serve()` already returns `Result`, so this now
  prints a normal chained anyhow error and exits 1 instead of a panic
  backtrace.
- **`gpu::manager::Gpus::start`**'s per-card manager-thread spawn
  (`src/gpu/manager.rs:172`) no longer panics the whole process over one
  card. It now matches the existing "a card that can't be opened is logged
  and skipped" pattern a few lines above: on spawn failure it logs an error
  and moves on, leaving that one card in its default `preempted` (unusable)
  state forever instead of taking every other card and the Discord
  connection down with it.
- **The per-card upload-thread spawn** inside `manage()`
  (`src/gpu/manager.rs:360`) no longer panics that card's manager thread
  (which would've frozen the card in `Loading` forever, invisibly — the
  exact silent-hang pattern this plan is about). It now matches the
  handling of a *failed* upload a few lines below: log the error and call
  `policy.loaded(now, false)`, letting the policy's normal backoff/retry
  take over.
- **The dispatcher's "inference thread is gone" paths**
  (`src/discord/dispatch.rs`, both `engine.jobs.send(job).await.is_err()`
  sites) no longer just `warn!` and quietly return — that was the silent
  half-alive state this whole plan exists to fix: the dispatcher task would
  exit, but the Discord gateway connection stayed up, so the bot looked
  online while never replying again. They now call a new
  `inference_thread_gone()` helper that logs a clear `error!` ("the
  inference thread's job channel is closed … exiting so the service
  restarts with a fresh model load") and calls `std::process::exit(1)`,
  converting *any* inference-thread panic into the bounded, self-healing,
  systemd-restarts-it failure mode instead of the silent one. This is the
  fix that covers `worker.rs:238`'s in-thread `.expect()` too, and would
  cover a future one added inside the inference thread — without touching
  that call site at all.

### Round 2: 9 of the remaining 29 eliminated or made safer

These fall into two kinds: three were already sitting inside functions that
return `Result` and simply never had the macro swapped (free — no new
plumbing); the rest needed a real, if small, structural change. All verified
the same way as above (clean build, no new clippy warnings, 296 unit + 20
integration tests passing).

- **Already-free `Result` conversions** — the enclosing function already
  returned `Result` and already used `?`/`bail!` a few lines above; swapping
  the macro was the whole fix:
  - `model/llama.rs`: `LinearImpl::Packed => unreachable!()` → `bail!(...)`
    (unreachable only because an earlier branch already handles Packed).
  - `model/llama.rs`: `forward()`'s `None => unreachable!("never cancelled")`
    → `bail!(...)`.
  - `gguf/mod.rs`: `to_f32()`'s `GgmlType::Other(_) => unreachable!(...)` →
    `bail!(...)` (the line above already rejects it via `bytes()`; this arm
    is reachable only by a future bug, not by any input).
- **`gpu/model.rs` "seven linears"** — replaced 7 manual
  `lins.next().expect("seven linears")` calls with collecting into a `Vec`
  and one `lins.try_into::<[Linear; 7]>()`, erroring with the actual count
  on mismatch. This isn't just cleaner: the old code would have *silently
  dropped* an 8th entry if `named` (a `[_; 7]` literal a few lines up) ever
  grew without this code being updated to match — `try_into` catches a
  length change in either direction instead of only panicking on "too few."
- **`model/packed_q8.rs` kernel dispatch** — `batched_with` matched the
  3-variant `Kernel` twice: once to branch to the tiled VNNI path, then
  again in the per-row loop, where `AvxVnni` was `unreachable!()` because
  the first match already returned for it. Added a 2-variant `ScalarKernel`
  (`Portable`, `Avx2`) and restructured the first match so each arm either
  diverges (`AvxVnni`'s tiled path ends in `return y`) or produces a
  `ScalarKernel` — exhaustive over all 3 `Kernel` variants, no catch-all.
  The per-row loop now matches `ScalarKernel`, which is *also* exhaustive
  with no catch-all. Genuinely illegal-states-unrepresentable, not just a
  relocated check.
- **`Option::take_if` instead of filter-then-take-then-expect** — two call
  sites did `x.as_ref().filter(pred)` to decide, then separately
  `x.take().expect("checked above")` to act, with nothing stopping the two
  steps from disagreeing in theory. `Option::take_if` (stable since Rust
  1.80) does both atomically in one call, so there's no second step to
  `.expect()`:
  - `gpu/manager.rs`: the finished-upload check in `manage()`.
  - `crates/chatsim-core/src/room.rs`: `on_event` used to check
    `self.running` matched the job, then `finish()` separately re-took it
    with `.expect("on_event checked the job")`. Restructured so
    `Finished`/`Failed` events call `self.running.take_if(...)` directly and
    hand the extracted `Running` to a renamed `finish_with(r, result)`;
    `Started`/`Text` keep the `&mut` lookup (they don't need ownership).
- **`engine/worker.rs` "index from position"** — `Caches::checkout` did
  `self.parked.iter().position(...)` then `self.parked.remove(i).expect(...)`
  separately. `VecDeque` has no `take_if`-equivalent, so added a small
  `remove_where` helper that chains `position` into `remove` and lets the
  two `Option`s compose — `d.remove(i)` can't actually return `None` given
  `i` just came from `position` on the same, untouched deque, but there's
  nothing to `.expect()` either way since both ends are already `Option`.
- **`chat/sanitize.rs` "found"** — `defuse_tags` found a `[`/`(` at a byte
  offset with `rest.find(['[', '('])`, then re-derived "there's a char
  there" via `tail.chars().next().expect("found")`. Replaced with
  `if tail.as_bytes()[0] == b'[' { '[' } else { '(' }` — an `if`/`else` over
  the two bytes `find` could have matched has no third branch to fail into,
  so this is a real elimination, not a relocation.

### Round 3: 7 more sites, plus one investigated and declined

Verified the same way as rounds 1 and 2 (clean build, no new clippy
warnings, full `cargo test --workspace` green — 296 unit + 20 integration +
the rest of the workspace's own test crates, no failures anywhere).

- **`worker.rs:242`** (the inference thread's own tokio runtime build) no
  longer panics at all. Changed the `.expect()` into a `match` that logs and
  `return`s from the thread closure on failure — dropping `jobs_rx`/`idle_tx`
  the same way a panic would, which `inference_thread_gone()` (round 1) already
  treats as fatal. Same outcome, no unwind.
- **`scripted.rs:36,176`** — two plain `.lock().unwrap()` on a
  `std::sync::Mutex` that would propagate a panic from a *poisoned* lock.
  Changed to `.lock().unwrap_or_else(|e| e.into_inner())`, matching the
  convention already used everywhere else in the codebase (`worker.rs`,
  `manager.rs`) for exactly this. Not a judgment call — just a missed case
  of an existing pattern.
- **`seeded.rs:46,80,81`** — these are fixed-size *array* conversions (the
  SHA-256 digest and the AES block are already `[u8; N]`-shaped, not a
  runtime-length buffer), so going through `TryFrom<&[u8]>` was a choice,
  not a necessity. Replaced with array-pattern destructuring
  (`let [a0, a1, ..., ..]: [u8; 32] = digest.into();`), which the compiler
  checks at compile time — genuinely zero runtime fallibility, not "trust."
  Verified against the pinned known-answer test (`seeded::tests::known_answers`)
  and the novelty commands' pinned output tests: byte-for-byte unchanged.
- **`worker.rs:411`** ("at least one candidate") — traced `best_of` end to
  end and changed its type from `usize` to `std::num::NonZeroUsize`, in
  `Command::best_of` (persona.rs), `Persona::best_of()`'s return type, and
  `Job::best_of` (worker.rs). "Can't be zero" is now a compile-time fact
  instead of the `.max(1)` that used to re-assert it at the point of use.
  Then restructured `run_job`'s best-of-N loop to run candidate 0
  unconditionally (seeding `best_score`/`text`/`result` as plain values)
  and loop over the rest comparing against it, instead of accumulating into
  an `Option` and unwrapping after. **Caveat:** no existing test exercises
  `best_of > 1` through the engine (only `best_of == 1`, via the GPU-sharing
  integration tests) — this was true before the change too, not something
  introduced by it, but the restructure's correctness for that path rests on
  manual trace-through (documented in the commit/conversation), not test
  coverage. Worth a real test if best-of-N commands are ever run with a GPU
  attached in practice.
- **`vk.rs`'s "not recording" (3 sites) — investigated, declined.** The
  typestate design (a `Recording<'_>` guard returned by `begin()`, holding
  `&mut Commands`) is sound in isolation, but tracing where `dispatch`/`copy`
  are actually called from (`model.rs`'s `record_layer`/`matvec`/`quantize`/
  `matmul`) found that those functions interleave recording calls with reads
  and writes of ~15-20 *other* fields of the same `Active` struct that
  `Commands` lives inside. A `Recording` borrowing `Commands` would have to
  stay alive across that whole call chain, which means `record_layer` and
  friends could no longer take `a: &mut Active` as one parameter — they'd
  need rewriting to destructure `Active` into disjoint field borrows
  throughout the forward pass. That's a restructuring of GPU forward-pass
  field access, not an API change, in unsafe Vulkan FFI code — for a benefit
  that hasn't changed: protecting 3 `.expect()` calls in a begin/submit
  discipline that reading all of `forward()` end to end confirms is already
  airtight (every `begin()` has its `submit()` in the same loop iteration,
  no interleaving, no path to a stale recording state). Not worth it; see
  "Next steps" if this changes.

### Round 4: the sim-tool sites, finished off — including one real bug

Revisited `scripted.rs`'s and `room.rs`'s remaining sites instead of leaving
them as "lower priority, didn't get to it."

- **`scripted.rs:208,210`** — the thing that made this not a drop-in
  `take_if` last round was `poll()` calling `self.take_due(...)` (needing
  `&mut self`) between the two `self.active` derivations. Checked
  `take_due`'s body: it only mutates the `Active` value in place via
  `self.active.as_mut()`, never the `Option` wrapper itself, so nothing
  between the checks can actually clear `self.active`. `take_if` applies
  cleanly after all — one call replaces both `.expect()`s:
  `self.active.take_if(|a| cancelled || a.chunks.is_empty())`.
- **`room.rs:669,683` — this one was an actual reachable bug, not just an
  unproven invariant.** Checked whether `self.messages` can be mutated
  between a reply starting and `finish_with` looking its placeholder back
  up, since the two aren't in the same function this time. They can:
  `Command::Clear` (`/clear`) calls `self.messages.clear()` unconditionally
  and never touches `self.running`. Run `/clear` while a reply is streaming,
  and `finish_with`'s `self.msg_mut(r.msg).expect("placeholder exists")`
  would panic the whole sim tool on a normal user action, not an edge case
  requiring a future bug. Fixed by changing both to
  `let Some(m) = self.msg_mut(r.msg) else { return };` — on a now-missing
  placeholder, the stats (`Ok` path, recorded earlier in the function) still
  land, and the function just skips updating a message and pushing history
  that no longer have anywhere to go, instead of crashing.
- **`room.rs:39,603` stay.** `39` runs once at construction, parsing a
  built-in constant. `603` reads the message it just posted, in the same
  statement sequence, in the same single-threaded call — nothing can run
  `/clear` *inside* that one function call. Both genuinely safe, unlike
  `669`/`683` turned out to be.

All verified the same way as every other round: clean build, no new clippy
warnings, full `cargo test --workspace` green.

### Round 5: the real `gguf/mod.rs` miss, and `vk.rs` done properly

- **`gguf/mod.rs:318` — a genuine oversight from round 2, fixed.** `Reader::array::<N>()`
  already returned `Result<[u8; N]>` (it calls `self.take(N)?` on the line
  above); the `.unwrap()` on the subsequent `.try_into()` was a style choice,
  not a structural necessity, and I'd wrongly filed the whole call under
  "already the floor for a runtime-length buffer" without separating *can
  the invariant be eliminated* (no, that part was right) from *can it at
  least not panic given Result is already flowing right here* (yes, trivially).
  Now `.map_err(|_| anyhow!("internal error: take({N}) returned the wrong
  length"))` instead of `.unwrap()`. Zero architectural cost — it was just
  sitting there unconverted.
- **`vk.rs`'s 3 "not recording" sites — gone, via the other half of the
  question the typestate decline didn't fully answer.** Round 3 declined a
  typestate guard because `Recording` borrowing `Commands` would conflict
  with `model.rs`'s `Active`-field access throughout the forward pass. That
  ruled out *eliminating* the invariant via types — it didn't rule out
  *propagating* it as `Result` instead of panicking, which needs no such
  borrow. Made `Commands::cb/stamp/dispatch/copy/barrier_all` return `Result`
  (`cb`'s `self.cur.expect("not recording")` became
  `self.cur.map(|s| self.bufs[s]).context("not recording")`), then threaded
  `?` through every caller the compiler pointed at: `model.rs`'s `matvec`,
  `quantize`, `matmul`, `record_layer` (and its `norm` closure) all changed
  from `()` to `Result<()>` (`quantize`: `Quantized` to `Result<Quantized>`),
  and `forward()` picked up `?` on its own direct `dispatch`/`copy` calls and
  on calling `record_layer`/`matvec`. ~30 call sites touched in total, but
  every one was compiler-enforced (a missed `?` is a type error, not a
  silent bug) and the exact path is covered by real GPU hardware in this
  environment (`INCREBOT_GPUS` set): `gpu::model::tests::matvec_modes_match_f32_reference`,
  `integer_matmul_matches_cpu_integer_kernel`, and all 4 of
  `tests/it/gpu.rs`'s and `gpu_sharing.rs`'s integration tests ran against
  it (not skipped) and passed. A future bug in this invariant now fails one
  job with a clear error instead of panicking the inference thread.

## Next steps

1. Decide whether the GPU manager thread itself (`manage()`'s main loop, not
   just the two spawns inside it) deserves the same "closed channel is
   fatal" treatment if it panics outright — right now a panic there just
   silently freezes that one card (same shape of bug as the dispatcher one,
   smaller blast radius since the bot still answers on CPU or another card).
2. Revisit whether any Tier 3 call sites are worth a `catch_unwind` wrapper
   instead, now that it's clear how often (if ever) this actually fires in
   practice.
3. `vk.rs`'s typestate option (eliminating the invariant at compile time,
   rather than propagating it as `Result` the way round 5 did) is still on
   the table if `model.rs`'s `Active` struct ever gets restructured for
   other reasons — revisit then; the cost was specifically "this requires
   splitting `Active`'s field borrows throughout the forward pass," not
   "this is a bad idea on its own."
4. The remaining 5 of the original 32 stay as `.expect()`, deliberately:
   `linear.rs:28` (would mean adding `Result` to the `QLinear` trait and
   every implementor, for a conversion already guaranteed by an explicit
   `.convert::<f32>()` one line above); `sanitize.rs`'s char-boundary
   `.expect()` (it's actually guarding "the loop counter is below the
   string's length," already proven by the `while` condition — not
   expressible as a compile-time fact in stable Rust without dependent
   types, and it doubles as a regression guard for an intricate scanner);
   `persona.rs`'s `pick_patterns()` (the real fix is a `#[test]` that calls
   it, not a code-structure change); and `room.rs`'s 2 sim-tool sites
   (genuinely safe: one runs once at construction, the other reads a
   message in the same function call that just posted it).

## Guardrails: workspace-wide lints

`[workspace.lints.rust]`/`[workspace.lints.clippy]` in the root
`Cargo.toml`, with `[lints] workspace = true` in every member crate
(`increbot`, `chatsim-core`, `chatsim-tui`). This covers each crate's lib,
every binary, every example and every test/bench target automatically — no
per-file `#![deny(...)]` needed, though two of the lints below needed a
per-file `#![allow(...)]` back out for files whose whole job conflicts with
the rule. **Caveat worth knowing:** the `clippy::*` lints below only fire
under `cargo clippy`; plain `cargo build`/`cargo test` use rustc directly
and never run clippy's lint passes at all, `[lints.clippy]` config in
Cargo.toml or not. Only `unused_must_use`, `unreachable_patterns` and
`unsafe_op_in_unsafe_fn` (real rustc lints) are enforced by an ordinary
build. There's no CI in this repo, so right now `cargo clippy
--workspace --all-targets` run by hand is the only thing that checks the
clippy-tier rules.

- **`unused_must_use = "deny"`** (rustc). A dropped `Result` (exactly the
  shape of bug round 5's `vk.rs`/`model.rs` changes introduced 17 fresh
  opportunities for) now fails even a plain build. Verified: the rustc
  invocation for every package/target kind carries `--deny=unused_must_use`;
  a deliberately introduced dropped `Result` in a scratch file failed the
  build with the expected error, then was reverted.
- **`unreachable_patterns = "deny"`** (rustc). A match arm the compiler can
  prove is dead — e.g. the `Kernel`/`ScalarKernel` split in round 5's
  `packed_q8.rs` fix. Zero pre-existing violations.
- **`unsafe_op_in_unsafe_fn = "deny"`** (rustc). There are 9 `unsafe fn`s in
  `gpu/vk.rs`/`gpu/model.rs`; the code already wraps every unsafe op in its
  own block with a `// SAFETY:` comment even inside them, so this makes
  that habit compiler-enforced instead of just a habit someone could quietly
  stop following. Zero pre-existing violations.
- **`clippy::dbg_macro = "deny"`, `clippy::unimplemented = "deny"`,
  `clippy::let_underscore_lock = "deny"`.** Pure insurance — zero
  pre-existing violations, nothing to lose. `let_underscore_lock` matters
  given how much `Mutex` this codebase uses: it catches `let _ =
  mutex.lock();`, which drops the guard immediately and silently defeats
  the lock.
- **`clippy::print_stdout = "deny"`, `clippy::print_stderr = "deny"`.**
  `cli/mod.rs` has a deliberate stdout/stderr split (`serve`'s WARN+ to
  stderr so podman's journal doesn't redden every line; everything else to
  stdout); a stray `println!`/`eprintln!` in the live bot's own code would
  silently bypass it. The actual scope, once checked for real (not just the
  `src`/`crates` grep from the original discussion), turned out much larger
  than estimated — **~70 hits across 13 files**, not the "10 hits, 3 files"
  first thought: every file under `examples/` (`core_scaling.rs`,
  `bench_decode.rs`, `profile_decode.rs` — auto-discovered by Cargo, missed
  by the original `src`/`crates`-only grep), `bench/persona/*.rs`,
  `cli/generate.rs`, `cli/inspect.rs`, `chatsim-tui/src/main.rs`,
  `chatsim-core/tests/engine_tinyllama.rs`, `gpu/model.rs`'s test module,
  and every submodule of `tests/it`. All confirmed legitimate: benchmark/
  report tools whose whole job is printing, the two CLI subcommands'
  intentional output, or test-only diagnostics (`SKIPPED: ...` messages,
  measured-result prints) — none in the live bot's `discord`/`engine`/
  non-test `gpu` path. Allowed back in with one `#![allow(...)]` per
  compilation root (one `tests/it/main.rs` attribute covers all 7 of its
  submodules at once, rather than touching each file) instead of per call
  site. Verified clean `cargo clippy --workspace --all-targets` (zero
  `print_stdout`/`print_stderr` hits anywhere) and full `cargo test
  --workspace` passing.
- **`thiserror` removed from `Cargo.toml`.** Found via
  `RUSTFLAGS="-W unused_crate_dependencies" cargo build -p increbot --lib`:
  declared as a dependency, imported nowhere in the entire workspace.
  **Tried enabling `unused_crate_dependencies = "deny"` workspace-wide and
  reverted it** — it evaluates per compilation unit, and this package has
  many example/test/bench targets that each use only a slice of the full
  `[dependencies]` list (e.g. `bench/persona/floob_bench.rs` alone would
  need ~18 `#[allow]`s to stop flagging dependencies it simply doesn't
  need). It did its one real job already by finding `thiserror`; keeping it
  on permanently would mean allowing it back almost everywhere for no
  ongoing benefit. Left as an occasional manual check, not a standing lint.

### Checked and declined (not added)

`clippy::exit` would flag 7 call sites — almost certainly including
`dispatch.rs`'s `inference_thread_gone()` and `serve.rs`'s signal-handling
`std::process::exit` calls, which are this session's own deliberate,
load-bearing design. `clippy::unwrap_used`/`clippy::expect_used` would
require `#[allow]`-ing every one of the 5 `.expect()` sites this plan spent
five rounds individually justifying, plus all test code — fighting the lint
rather than being helped by it. `clippy::indexing_slicing` and
`clippy::arithmetic_side_effects` would be extremely noisy against the hot
numeric/GPU kernels (`model.rs`, `packed_q8.rs`), which index and do raw
arithmetic constantly and intentionally. `clippy::panic` found 1 hit in
`build.rs` (a build script, not shipped code) and stopped the build before
reaching the rest of the workspace, so there may be more (at least one
test-only `panic!` in `engine/bans.rs`); also squarely a "restriction"-group
lint meant for a different project philosophy than this one. `unused_results`
(the unconditional version of `unused_must_use`, firing on *any* discarded
non-`()` value, `#[must_use]` or not) would be far too noisy for ordinary
code.
