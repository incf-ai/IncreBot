# Persona files plan

Status: **implemented 2026-10-13** (`src/persona.rs`). Differences from the
plan below:

- Persona files live in `prompts/` (`prompts/c17.rsn`, `prompts/c18.rsn`),
  not `personas/`: the service mounts only `prompts/` into the container, so
  live edits and `/persona reload` keep working without changing the unit.
- `max_new_tokens_gpu` moved into the persona file too, with `max_new_tokens`.
- `Repeat` and `Dry` fields are all required as well (no partial structs).
- The `chatsim` crates (`crates/chatsim-*`) were already broken before this
  change: they used sampler fields the engine had dropped. They're fixed and
  use persona files now. `RoomConfig` carries a `Persona` (default: the
  built-in `prompts/c18.rsn`), and the room applies its notes and stop
  strings. `EngineBackend` takes its sampler, word biases and reply length
  from the persona. The TUI loads `persona_file` and falls back to the
  built-in one. The room's header check now uses `max_header_tokens`, like
  the bot.
- Fingerprints: c17 `9f77f89`, c18 `9e51525`. A conversion check confirmed
  that both files match the old prompt files and configs field by field, and
  both still render to 803 header tokens.

## Goal

One `.rsn` file per persona (a benchmark candidate or the live setup) that
holds everything that shapes how floob replies: the prompt sections, the prompt
notes, the word biases, the stop strings, and the sampler and length settings.
The prompt sections are raw strings, so prompt text can be pasted in as-is,
without indenting or escaping.

## Why

A candidate is currently spread over four places:

- the prompt `.txt` (`prompts/system.txt`, `prompts/legacy/cN.txt`);
- the prompt notes, in `config.local.rsn` or `bench/persona/tmp/config-cN.rsn`;
- the word biases, in the config or a `tmp/*-discourage.txt` passed as
  `INCREBOT_DISCOURAGE`;
- the sampler and length settings in `config.local.rsn`.

The persona fingerprint in `/stats` exists because of that spread. With one
file:

- **A candidate is one file.** `diff personas/c17.rsn personas/c18.rsn` shows
  exactly what changed.
- **Deploying is one line** in the config: `persona_file: "personas/c18.rsn"`.
- **Queue lines get shorter:** `name | persona file | scenarios | repeats`,
  with no `@discourage` list and no per-run config.
- **The prompt sections are named fields**, so the `---` splitting goes away.
  Today a stray `---` line in the examples would silently shift the reminder.
- **Comments are allowed**, so the file can say why each note or bias exists.

## Raw strings: checked

rsn 0.2.0 (the version the bot pins) parses multi-line `r#"…"#` strings with
quotes and backslashes untouched, and the text can start at column 0. Checked
on 2026-10-13 with a test parse.

- The newline right after `r#"` and the one before `"#` end up in the string;
  the loader trims each section, as the prompt code already does.
- The only text a section can't contain is `"#`. If it ever comes up,
  `r##"…"##` handles it.

## File sketch

````
// floob c18: c17 with "much older than other kobolds"
Persona {
    name: "c18",
    persona_name: "floob",

    persona: r#"
[Character: floob; species: floob is a small and stubby half-kobold; age: adult, 471 kobold years old; ...]
"#,

    examples: r#"
[Example messages:
pip: hey floob
floob: oh! hi pip!! um, u know how to open these? ...]
"#,

    reminder: r#"
[Character note: floob always speaks for himself in first person ...]
"#,

    notes: [
        Note { text: "(floob is a grown kobold who runs his own burrow. ...)",
               prompted_by: ["mommy", "mama", "go to your room", ...] },
        ...
    ],

    discourage: [
        Word { stem: "gigg", bias: -2.5 },
        Word { stem: "now", bias: -3, after: ["grown up", "grownup", "grown", "big", "adult"], within: 1 },
        ...
    ],
    stop_strings: ["[Character", "(floob is replying to", "[floob:"],

    sampler: Sampler { temperature: 0.8, top_k: 40, top_p: 0.95, min_p: 0.05,
                       repeat: Some(Repeat { ... }), dry: Some(Dry { ... }) },
    max_new_tokens: 75,
    wrap_up_after: 40,
    single_paragraph: false,
}
````

## What goes where

The rule: anything the benchmark varies, or might vary, goes in the persona
file. Settings tied to the machine or the Discord deployment stay in the
config.

| persona file (how floob behaves) | `config.local.rsn` (how this machine runs the bot) |
|---|---|
| `persona`, `examples`, `reminder`, `persona_name` | model, GPUs, threads, `n_ctx`, `max_prompt_tokens`, header limit |
| `notes`, `discourage`, `stop_strings` | Discord: `bot_name`, guilds, channels, admins, opt-outs |
| temperature, top-k, top-p, min-p, repeat penalty, DRY | prompt template, `history_messages`, queue depth, timeouts |
| `max_new_tokens`, `wrap_up_after`, `single_paragraph` | `persona_file: "personas/c18.rsn"` |

Context size is borderline: it changes how much history floob sees, but it is
tied to GPU memory, so it stays in the config.

## Design decisions

1. **Self-contained files.** Every behavior field must be
   present, and a missing field is an error. Each file then fully describes its
   candidate, and a change to the shipped defaults can't change an old
   candidate without anyone noticing. The alternative is optional fields that
   override the config's values.
2. **Hot reload** `/persona reload` and SIGHUP currently reload only the
   prompt text. With biases and sampler settings in the file, a reload has to
   rebuild the word-bias token tables and the sampler parameters too, which
   means putting them behind the same lock as the prompt.
   - reload everything. A half-applied persona (new text, old
     notes) is the mix the benchmark never measured.
   - Alternative: reload the text live, need a restart for the rest, and have
     the reload say so.
3. **Fingerprint.** One hash of the parsed file, excluding `name` and
   comments: `persona: c18 (a1b2c3d)`. A comment-only edit doesn't change it.
   `persona_report fingerprint` hashes a run's saved persona file the same way.
4. **Env overrides.** `INCREBOT_SYSTEM_PROMPT_FILE` is replaced by
   `INCREBOT_PERSONA_FILE`. `INCREBOT_DISCOURAGE` stays as an override on top
   of the file, for quick bias experiments; the run's `env.txt` records it, and
   the fingerprint covers the result after the override.
5. **Bias list format.** Structured `Word { … }` entries, one per line, as in
   `config.local.rsn` today: clearer and easier to diff than the compact
   `gigg=-2.5,…` string. The compact form stays only for the env override.
6. **`.txt` prompts.** drop support and keep one path. The
   shipped `config.rsn` points at an example persona file. The alternative is
   accepting a `.txt` prompt for one more release.
7. **Sampler and length settings in the file**  yes (the
   table above). The alternative is a prompt-only persona file, with the
   sampler staying in the config.

## Migration

- New candidates from c19 on are `personas/cN.rsn` (or
  `bench/persona/personas/cN.rsn`).
- c17 and c18 are converted, so the live persona is a file. Each converted
  file must reproduce the same rendered prompt and the same rules as the old
  files. The current fingerprint (`prompt …, rules …`) checks this before the
  new single hash replaces it.
- Older candidates (c0–c15) stay as they are: their run folders already hold
  snapshots of the prompt, config and bias list.
- The README's candidate table gets a "persona file" column from c19 on.

## Work

1. **Types and loading**: a `Persona` type in `src/config.rs` (or a new
   `src/persona.rs`) with validation:
   - biases below 0 only;
   - `within` requires `after`;
   - stop strings not empty;
   - note phrases valid;
   - the header-limit check, run on the rendered persona.

   Then remove the moved fields from `GenerationConfig` and `DiscordConfig`,
   and add `persona_file` with its env key.
2. **The bot**: load the persona at startup. Rebuild the prompt, word biases
   and sampler on reload (decision 2). Show the persona line in `/stats`,
   `/version` and the logs with the new hash. `/persona show` shows the file's
   sections.
3. **The prompt code**: take the three sections as fields and drop the `---`
   splitting. The template is unchanged: `{{ reminder }}` and `{{ note }}`
   keep their places.
4. **The benchmark**:
   - `floob_bench` takes `--persona`;
   - `run.sh` snapshots `persona.rsn` instead of `system.txt`, `discourage.txt`
     and the config's persona fields;
   - `queue.sh` lines become `name | persona file | scenarios | repeats`;
   - `persona_report header`, `discourage` and `fingerprint` read persona files.
5. **Docs**: design.md §5 and §7, deploy.md (the persona change row), the
   README (§1 usage, Naming, the candidate table), and `config.rsn` comments.
6. **Tests**:
   - parsing raw strings;
   - rejecting a missing field and a positive bias;
   - the reload swapping all parts;
   - the fingerprint ignoring comments and `name`;
   - the converted c17 rendering the same prompt as `/prompts/legacy/c17.txt` plus
     `config-c17.rsn`.

Timing: none of this touches a benchmark queue already running, because
queues use a frozen binary. Build it while a queue runs, and deploy it with
the next `deploy-setup.sh`. The old binary can't read a config with
`persona_file`, so the config and the binary have to change together.
