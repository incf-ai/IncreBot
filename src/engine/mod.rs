pub mod backend;
pub mod bans;
pub mod decode;
pub mod sampler;
pub mod worker;

use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use anyhow::Result;

use crate::model::KvCache;
use crate::tokenizer::Tokenizer;
pub use backend::{Backends, Forward};
use decode::{Step, StreamDecoder};
use sampler::Sampler;

/// Why generation ended.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StopReason {
    Eos,
    MaxTokens,
    StopString,
    /// Past `wrap_up_after` tokens, at the end of a sentence.
    WrapUp,
    Timeout,
    Cancelled,
}

#[derive(Clone, Debug)]
pub struct GenerateOutput {
    pub tokens: Vec<u32>,
    pub stop: StopReason,
    /// Prompt positions reused from the cache.
    pub reused_tokens: usize,
    pub prefill_tokens: usize,
    pub prefill: Duration,
    pub decode: Duration,
    /// Phrase bans that rolled the reply back (`bans`), as matched; "kept: …"
    /// for one left in (released already, or out of retries).
    pub rollbacks: Vec<String>,
    /// With `SamplerParams::observe`: the word biases that made a difference
    /// and the phrase-ban rollbacks, in the order generation met them.
    pub steered: Vec<Steered>,
}

/// One step where the reply was steered, for its trace.
#[derive(Clone, Debug)]
pub enum Steered {
    /// At token `at` (0 = the reply's first), after `before` (the reply's
    /// last words so far), a word bias made a difference; `picked` was drawn.
    Bias { at: usize, before: String, picked: String, hit: sampler::BiasHit },
    /// A phrase ban matched from token `at`: rolled back there, or `kept`.
    Ban { at: usize, phrase: String, kept: bool },
    /// Token `at` started a word and was drawn again under the word-start
    /// settings (`sampler.word_start`): `normal` was the first draw (`changed`
    /// if `picked` isn't it), with their probabilities at the normal
    /// temperature. `word` is the whole word it began, once the reply is done
    /// (`picked` until then). Dropped when a rollback discards the token.
    /// `odds`: the likeliest word starts (with the two draws) and how the
    /// redraw moved them: (text, chance under the normal settings, under the
    /// word-start ones), see `sampler::Redrawn::odds`.
    WordStart {
        at: usize,
        normal: String,
        picked: String,
        word: String,
        changed: bool,
        normal_prob: f32,
        picked_prob: f32,
        odds: Vec<(String, f32, f32)>,
    },
}

pub struct GenerateRequest<'a> {
    pub prompt: &'a [u32],
    pub max_new_tokens: usize,
    /// The limit while the reply runs on a GPU (gpu-plan.md). The limit follows
    /// the backend: a reply moved to the CPU past `max_new_tokens` stops there.
    pub max_new_tokens_gpu: usize,
    /// Once the reply has this many tokens, stop at the next sentence end
    /// (design.md §5); 0 = off.
    pub wrap_up_after: usize,
    /// The same while the reply runs on a GPU, following the backend like
    /// `max_new_tokens_gpu`.
    pub wrap_up_after_gpu: usize,
    pub stops: Vec<String>,
    /// Stop at the line break ending this many non-blank lines (`StreamDecoder::max_lines`).
    pub max_lines: Option<std::num::NonZeroUsize>,
    /// Stop at a later line opening with any speaker label (`StreamDecoder::labels`).
    pub labels: bool,
    /// Counted from the start of this call, i.e. from job start (design.md §5).
    pub timeout: Option<Duration>,
    /// The chat history without the template, tokenized: what DRY (and the
    /// repetition penalty, with `RepeatParams::history`) sees before the generated tokens.
    pub history: &'a [u32],
    /// Checked between layers during prefill and between tokens during decode.
    pub cancel: Option<&'a AtomicBool>,
    /// Phrases the reply must not contain (`bans`); `None` or empty = off.
    pub bans: Option<&'a bans::Bans>,
}

/// Prefills the part of `prompt` not already in `cache` (longest common prefix,
/// design.md §5), then decodes. `on_text` receives released text as it settles.
/// `model` is the CPU model (`&Llama`) or the backend ladder (`&mut Backends`).
pub fn generate(
    mut model: impl Forward,
    tok: &Tokenizer,
    cache: &mut KvCache,
    sampler: &mut Sampler,
    req: GenerateRequest<'_>,
    mut on_text: impl FnMut(&str),
) -> Result<GenerateOutput> {
    let start = Instant::now();
    let cancelled = || req.cancel.is_some_and(|c| c.load(Ordering::Relaxed));
    let reuse = reusable_prefix(&cache.tokens, req.prompt);
    cache.truncate(reuse);
    let suffix = &req.prompt[reuse..];
    let mut out = GenerateOutput {
        tokens: Vec::new(),
        stop: StopReason::MaxTokens,
        reused_tokens: reuse,
        prefill_tokens: suffix.len(),
        prefill: Duration::ZERO,
        decode: Duration::ZERO,
        rollbacks: Vec::new(),
        steered: Vec::new(),
    };

    let Some(mut logits) = model.forward(suffix, cache, &cancelled)? else {
        out.stop = StopReason::Cancelled;
        out.prefill = start.elapsed();
        return Ok(out);
    };
    out.prefill = start.elapsed();
    let decode_start = Instant::now();

    let mut context = req.history.to_vec();
    let first_chars = req.bans.and_then(|b| b.first_chars.as_deref());
    let bans = req.bans.filter(|b| !b.is_empty());
    let mut dec = StreamDecoder::new(req.stops).max_lines(req.max_lines).labels(req.labels).hold_tokens(if bans.is_some() { bans::HOLDBACK } else { 0 });
    // For rollback (`bans`): the raw logits each position was sampled from,
    // the tokens banned at each position, and where the cache's reply starts.
    let base = cache.len();
    let mut saved: Vec<Vec<f32>> = Vec::new();
    let mut banned: Vec<Vec<u32>> = Vec::new();
    // Matches starting before this byte were kept (released, or out of tries).
    let mut kept_until = 0;
    let decode = |ids: &[u32]| tok.decode(ids);
    let limit = |model: &dyn Forward| {
        if model.on_gpu() {
            req.max_new_tokens_gpu
        } else {
            req.max_new_tokens
        }
    };
    let wrap_up = |model: &dyn Forward| {
        if model.on_gpu() {
            req.wrap_up_after_gpu
        } else {
            req.wrap_up_after
        }
    };
    while dec.ids().len() < limit(&model) {
        let pos = dec.ids().len();
        if bans.is_some() {
            saved.truncate(pos);
            saved.push(logits.clone());
            banned.resize_with(pos + 1, Vec::new);
            for &id in &banned[pos] {
                logits[id as usize] = f32::NEG_INFINITY;
            }
        }
        // Nothing visible yet: no banned first character.
        if let Some(ids) = first_chars
            && decode(dec.ids())?.trim().is_empty()
        {
            for &id in ids {
                if let Some(l) = logits.get_mut(id as usize) {
                    *l = f32::NEG_INFINITY;
                }
            }
        }
        let next = sampler.sample(&mut logits, &context, req.history.len());
        if let Some(r) = sampler.take_redrawn() {
            let picked = decode(&[r.picked])?;
            out.steered.push(Steered::WordStart {
                at: pos,
                normal: decode(&[r.normal])?,
                word: picked.clone(),
                picked,
                changed: r.normal != r.picked,
                normal_prob: r.normal_prob,
                picked_prob: r.picked_prob,
                odds: r.odds.iter().map(|&(id, n, h)| Ok((decode(&[id])?, n, h))).collect::<Result<_>>()?,
            });
        }
        let hits = sampler.take_hits();
        if !hits.is_empty() {
            let so_far = decode(dec.ids())?;
            let tail: String = so_far.chars().rev().take(40).collect::<Vec<_>>().into_iter().rev().collect();
            let picked = if next == tok.eos { "(end)".to_string() } else { decode(&[next])? };
            for hit in hits {
                out.steered.push(Steered::Bias { at: pos, before: tail.clone(), picked: picked.clone(), hit });
            }
        }
        let eos = next == tok.eos;
        let wrap = wrap_up(&model);
        if wrap > 0 && dec.ids().len() >= wrap && sentence_ends_before(tok, dec.ids(), next)? {
            out.stop = StopReason::WrapUp;
            break;
        }
        if let Some(b) = bans {
            let mut ids = dec.ids().to_vec();
            // At the end, the last word is finished: check it as such.
            let text = if eos {
                decode(&ids)? + " "
            } else {
                ids.push(next);
                decode(&ids)?
            };
            if let Some((start, hit)) = b.find(&text, kept_until) {
                // The token whose text holds the match's first letter.
                let mut p = ids.len().saturating_sub(1);
                while p > 0 && decode(&ids[..p])?.len() > start {
                    p -= 1;
                }
                let released = decode(&ids[..p])?.len() < dec.emitted();
                if released || out.rollbacks.len() >= bans::MAX_ROLLBACKS || banned[p].len() >= bans::MAX_BANNED_AT {
                    out.steered.push(Steered::Ban { at: p, phrase: hit.clone(), kept: true });
                    out.rollbacks.push(format!("kept: {hit}"));
                    kept_until = start + 1;
                } else {
                    tracing::debug!(%hit, at = p, "phrase ban: rolling back");
                    out.steered.push(Steered::Ban { at: p, phrase: hit.clone(), kept: false });
                    out.rollbacks.push(hit);
                    banned[p].push(*ids.get(p).unwrap_or(&next));
                    banned.truncate(p + 1);
                    // Redraws in the discarded text didn't happen.
                    out.steered.retain(|s| !matches!(s, Steered::WordStart { at, .. } if *at >= p));
                    if p < pos {
                        dec.truncate(p);
                        context.truncate(req.history.len() + p);
                        cache.truncate(base + p);
                    }
                    logits = saved[p].clone();
                    continue;
                }
            }
        }
        if eos {
            out.stop = StopReason::Eos;
            break;
        }
        context.push(next);
        match dec.push(next, decode)? {
            Step::Text(t) => on_text(&t),
            Step::Stop(t) => {
                on_text(&t);
                out.stop = StopReason::StopString;
                break;
            }
        }
        if req.timeout.is_some_and(|limit| start.elapsed() >= limit) {
            out.stop = StopReason::Timeout;
            break;
        }
        if cancelled() {
            out.stop = StopReason::Cancelled;
            break;
        }
        if dec.ids().len() >= limit(&model) {
            break;
        }
        match model.forward(&[next], cache, &cancelled)? {
            Some(l) => logits = l,
            None => {
                out.stop = StopReason::Cancelled;
                break;
            }
        }
    }
    if out.stop != StopReason::StopString {
        on_text(&dec.finish(decode)?);
    }
    out.tokens = dec.ids().to_vec();
    // Each redrawn word start's whole word, as the reply has it.
    let text = decode(&out.tokens)?;
    for s in &mut out.steered {
        if let Steered::WordStart { at, word, .. } = s
            && *at < out.tokens.len()
        {
            let from = decode(&out.tokens[..*at])?.len();
            if let Some(w) = word_from(&text, from) {
                *word = w;
            }
        }
    }
    out.decode = decode_start.elapsed();
    Ok(out)
}

/// The word in `text` starting at byte `from` (leading whitespace skipped):
/// letters, digits and apostrophes. `None` if there is none there.
fn word_from(text: &str, from: usize) -> Option<String> {
    let rest = text.get(from..)?.trim_start();
    let end = rest.find(|c: char| !(c.is_alphanumeric() || c == '\'' || c == '\u{2019}')).unwrap_or(rest.len());
    (end > 0).then(|| rest[..end].to_string())
}

/// Whether the reply `ids` ends a sentence that `next` doesn't continue: the
/// text ends in `.`, `!`, `?` or `~`, or closes an `*action*`, outside any open
/// action, and `next` starts with whitespace. Waiting for the space keeps "!!",
/// "?!" and "..." whole.
fn sentence_ends_before(tok: &Tokenizer, ids: &[u32], next: u32) -> Result<bool> {
    let text = tok.decode(ids)?;
    if !ends_sentence(&text) {
        return Ok(false);
    }
    let mut with_next = ids.to_vec();
    with_next.push(next);
    let longer = tok.decode(&with_next)?;
    Ok(longer.get(text.len()..).is_some_and(|rest| rest.starts_with(char::is_whitespace)))
}

/// Whether `text` ends a sentence or an `*action*`, outside any open action. A
/// trailing "..." is a thought trailing off, not an end: wrap-up
/// stopping there leaves "or we could use it to catch...".
fn ends_sentence(text: &str) -> bool {
    text.ends_with(['.', '!', '?', '~', '*'])
        && !text.ends_with("..")
        && text.matches('*').count().is_multiple_of(2)
}

/// How many cached positions can be kept for `prompt`: the longest common prefix,
/// backed off by one when it covers the whole prompt so the last prompt token is
/// re-run and yields logits.
pub fn reusable_prefix(cached: &[u32], prompt: &[u32]) -> usize {
    let lcp = cached
        .iter()
        .zip(prompt)
        .take_while(|(a, b)| a == b)
        .count();
    if lcp == prompt.len() {
        lcp.saturating_sub(1)
    } else {
        lcp
    }
}

#[cfg(test)]
mod tests {
    use super::reusable_prefix;

    #[test]
    fn lcp_reuse() {
        assert_eq!(reusable_prefix(&[], &[1, 2, 3]), 0);
        assert_eq!(reusable_prefix(&[1, 2, 9, 9], &[1, 2, 3]), 2);
        // Cache holds the generated reply; the next prompt holds the posted one.
        assert_eq!(reusable_prefix(&[1, 2, 3, 7, 8], &[1, 2, 3, 7, 5, 6]), 4);
        // Full match: back off one so there are logits to sample from.
        assert_eq!(reusable_prefix(&[1, 2, 3], &[1, 2, 3]), 2);
        assert_eq!(reusable_prefix(&[1, 2, 3, 4], &[1, 2, 3]), 2);
    }
}

    #[test]
    fn sentence_ends_outside_actions() {
        assert!(ends_sentence("hi pip!! u came?"));
        assert!(ends_sentence("ok... *flops on rock*"));
        assert!(ends_sentence("uh... ok."));
        // Trailing off is not an end.
        assert!(!ends_sentence("or we could use it to catch..."));
        assert!(!ends_sentence("hmm.."));
        // Inside an open action, or mid-word.
        assert!(!ends_sentence("hi! *flops on rock."));
        assert!(!ends_sentence("hi pip"));
        assert!(!ends_sentence("hi pip! "));
    }

#[cfg(test)]
mod word_tests {
    use super::*;

    #[test]
    fn redrawn_words_are_read_whole() {
        let text = "i found a shiny rock, don't tell";
        assert_eq!(word_from(text, 9).as_deref(), Some("shiny"));
        assert_eq!(word_from(text, 21).as_deref(), Some("don't"));
        assert_eq!(word_from(text, 20), None);
        assert_eq!(word_from(text, 99), None);
    }
}
