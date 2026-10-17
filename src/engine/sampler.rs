//! Sampling (design.md §5): repetition penalty → DRY → discouraged words →
//! temperature → top-k → top-p → min-p → seeded draw. Greedy when
//! `temperature == 0`.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

#[derive(Clone, Debug)]
pub struct SamplerParams {
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub min_p: f32,
    /// The repetition penalty; `None` = off.
    pub repeat: Option<RepeatParams>,
    /// DRY ("don't repeat yourself"); `None` = off. See [`apply_dry`].
    pub dry: Option<DryParams>,
    /// Discouraged word stems; `None` = off. See [`WordBias`].
    pub words: Option<Arc<WordBias>>,
    /// The user messages this reply answers, as [`normalized_words`]: what
    /// `prompted_by` rules look for.
    pub message: Arc<[String]>,
    /// Record the word biases that made a difference ([`BiasHit`]), for a
    /// reply's trace; off otherwise (it costs a pass over the vocabulary on
    /// each biased step).
    pub observe: bool,
    /// A second draw for some word starts; `None` = off. See [`WordStartParams`].
    pub word_start: Option<WordStartParams>,
}

/// Draws a word start again, now and then, with other settings (the persona's
/// `sampler.word_start`): when the normal draw starts a new word, with
/// `chance` that draw is dropped and the token is drawn again from the same
/// logits under these settings, among the tokens that start a new word. Where
/// words break stays as the normal settings have it; only which word comes
/// next is drawn differently. Nothing is rolled back: the redraw happens
/// before the token is used.
#[derive(Clone, Debug)]
pub struct WordStartParams {
    pub chance: f32,
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub min_p: f32,
    /// Per vocabulary id, from [`TokenShape::of`].
    pub shapes: Arc<[TokenShape]>,
}

/// How a token's text starts and ends, for [`WordStartParams`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TokenShape {
    /// Its first visible character is a letter or digit (" snacks", "s").
    pub word: bool,
    /// It starts with whitespace (" snacks").
    pub spaced: bool,
    /// It ends inside a word: a letter, digit or apostrophe ("don'"), so a
    /// letter after it continues that word.
    pub open: bool,
}

impl TokenShape {
    /// The shape of a token's text as it reads mid-stream (`Tokenizer::pieces`).
    pub fn of(piece: &str) -> Self {
        let in_word = |c: char| c.is_alphanumeric() || c == '\'' || c == '\u{2019}';
        Self {
            word: piece.trim_start().starts_with(char::is_alphanumeric),
            spaced: piece.starts_with(char::is_whitespace),
            open: piece.ends_with(in_word),
        }
    }
}

impl WordStartParams {
    /// Whether token `id` starts a new word after a token ending `open` (in a
    /// word): a word with a space before it, or one right after a space or
    /// punctuation (or at the reply's start).
    fn starts_word(&self, id: u32, open: bool) -> bool {
        self.shapes.get(id as usize).is_some_and(|s| s.word && (s.spaced || !open))
    }
}

/// A word start [`Sampler::sample`] drew again ([`WordStartParams`]).
#[derive(Clone, Debug, PartialEq)]
pub struct Redrawn {
    /// The normal draw, dropped.
    pub normal: u32,
    /// The token returned (may be the same one).
    pub picked: u32,
    /// Their probabilities at the normal temperature, over the whole
    /// vocabulary (before top-k / top-p / min-p).
    pub normal_prob: f32,
    pub picked_prob: f32,
    /// How the redraw moved the odds: the likeliest word starts
    /// ([`REDRAW_CANDIDATES`]), plus `normal` and `picked` if they aren't
    /// among them, most likely first: (token, its probability under the
    /// normal settings, under the word-start ones). Both before top-k /
    /// top-p / min-p; the second among word starts only, as the redraw draws.
    pub odds: Vec<(u32, f32, f32)>,
}

/// How many of the likeliest word starts [`Redrawn::odds`] lists.
pub const REDRAW_CANDIDATES: usize = 4;

/// The settings one draw uses.
#[derive(Clone, Copy)]
struct Draw {
    temperature: f32,
    top_k: usize,
    top_p: f32,
    min_p: f32,
}

/// A word bias that made a difference at one step: the stem's tokens had at
/// least [`HIT_PROB`] (or were the top choice) before a negative bias, or got
/// there through a positive one. Probabilities at the reply's temperature,
/// before top-k / top-p / min-p.
#[derive(Clone, Debug)]
pub struct BiasHit {
    pub stem: String,
    pub bias: f32,
    pub before: f32,
    pub after: f32,
    /// The stem's token was the most likely one before the bias.
    pub top_before: bool,
    /// … and after it.
    pub top_after: bool,
}

/// The probability from which a biased word counts as likely.
pub const HIT_PROB: f32 = 0.10;

#[derive(Clone, Debug)]
pub struct RepeatParams {
    pub penalty: f32,
    /// Window over the most recent tokens the penalty sees.
    pub last_n: usize,
    /// The penalty also sees the history before the generated tokens
    /// (`RepeatScope::Context`); otherwise only the generated ones. DRY always
    /// sees both.
    pub history: bool,
}

#[derive(Clone, Debug)]
pub struct DryParams {
    pub multiplier: f32,
    pub base: f32,
    pub allowed_length: usize,
    /// Tokens a match can't extend across (their text contains a sequence
    /// breaker such as `"\n"` or `":"`).
    pub breakers: Arc<HashSet<u32>>,
}

impl Default for SamplerParams {
    fn default() -> Self {
        Self {
            temperature: 0.8,
            top_k: 40,
            top_p: 0.95,
            min_p: 0.05,
            repeat: Some(RepeatParams {
                penalty: 1.1,
                last_n: 256,
                history: true,
            }),
            dry: None,
            words: None,
            message: Arc::from([]),
            observe: false,
            word_start: None,
        }
    }
}

impl SamplerParams {
    pub fn greedy() -> Self {
        Self {
            temperature: 0.0,
            repeat: None,
            ..Self::default()
        }
    }
}

pub struct Sampler {
    params: SamplerParams,
    rng: StdRng,
    /// Since the last [`take_hits`](Self::take_hits), when observing.
    hits: Vec<BiasHit>,
    /// The last word start drawn again.
    redrawn: Option<Redrawn>,
}

impl Sampler {
    pub fn new(params: SamplerParams, seed: u64) -> Self {
        Self {
            params,
            rng: StdRng::seed_from_u64(seed),
            hits: Vec::new(),
            redrawn: None,
        }
    }

    /// The word start the last [`sample`](Self::sample) drew again, if it did.
    pub fn take_redrawn(&mut self) -> Option<Redrawn> {
        self.redrawn.take()
    }

    /// The word biases that made a difference since the last call (only
    /// with [`SamplerParams::observe`]).
    pub fn take_hits(&mut self) -> Vec<BiasHit> {
        std::mem::take(&mut self.hits)
    }

    /// Picks the next token. `context` is the history followed by the tokens
    /// generated so far; its first `history_len` tokens are the history.
    pub fn sample(&mut self, logits: &mut [f32], context: &[u32], history_len: usize) -> u32 {
        let p = &self.params;
        if let Some(r) = &p.repeat {
            let penalised = if r.history {
                context
            } else {
                &context[history_len.min(context.len())..]
            };
            apply_repeat_penalty(logits, penalised, r.penalty, r.last_n);
        }
        if let Some(d) = &p.dry {
            apply_dry(logits, context, d);
        }
        if let Some(w) = &p.words {
            let generated = &context[history_len.min(context.len())..];
            if p.observe {
                let before = logits.to_vec();
                let applied = w.apply_recording(logits, generated, &p.message, true);
                if !applied.is_empty() {
                    self.hits.extend(bias_hits(w, &applied, &before, logits, p.temperature));
                }
            } else {
                w.apply(logits, generated, &p.message);
            }
        }
        if p.temperature <= 0.0 {
            return argmax(logits);
        }
        let normal = Draw { temperature: p.temperature, top_k: p.top_k, top_p: p.top_p, min_p: p.min_p };
        let first = draw(&mut self.rng, logits, normal);
        let Some(w) = &self.params.word_start else { return first };
        let generated = &context[history_len.min(context.len())..];
        let open = generated.last().is_some_and(|&t| w.shapes.get(t as usize).is_some_and(|s| s.open));
        if !w.starts_word(first, open) || self.rng.random::<f32>() >= w.chance {
            return first;
        }
        let words: Vec<f32> = logits
            .iter()
            .enumerate()
            .map(|(i, &l)| if w.starts_word(i as u32, open) { l } else { f32::NEG_INFINITY })
            .collect();
        let hot = Draw { temperature: w.temperature, top_k: w.top_k, top_p: w.top_p, min_p: w.min_p };
        let second = draw(&mut self.rng, &words, hot);
        // How likely each was under the normal temperature (over the whole
        // vocabulary, before top-k / top-p / min-p).
        let t = self.params.temperature;
        let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let z: f32 = logits.iter().map(|&l| ((l - max) / t).exp()).sum();
        let prob = |id: u32| logits.get(id as usize).map_or(0.0, |&l| ((l - max) / t).exp() / z);
        let ht = if w.temperature > 0.0 { w.temperature } else { 1e-3 };
        let wmax = words.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let wz: f32 = words.iter().map(|&l| ((l - wmax) / ht).exp()).sum();
        let hot_prob = |id: u32| words.get(id as usize).map_or(0.0, |&l| ((l - wmax) / ht).exp() / wz);
        let mut ids: Vec<u32> = (0..words.len() as u32).filter(|&i| words[i as usize].is_finite()).collect();
        ids.sort_unstable_by(|&a, &b| words[b as usize].total_cmp(&words[a as usize]));
        ids.truncate(REDRAW_CANDIDATES);
        for id in [first, second] {
            if !ids.contains(&id) {
                ids.push(id);
            }
        }
        ids.sort_by(|&a, &b| prob(b).total_cmp(&prob(a)));
        let odds = ids.into_iter().map(|id| (id, prob(id), hot_prob(id))).collect();
        self.redrawn =
            Some(Redrawn { normal: first, picked: second, normal_prob: prob(first), picked_prob: prob(second), odds });
        second
    }
}

/// One draw from `logits` under `d`: temperature → top-k → top-p → min-p →
/// a draw from `rng` (greedy at temperature 0).
fn draw(rng: &mut StdRng, logits: &[f32], d: Draw) -> u32 {
    if d.temperature <= 0.0 {
        return argmax(logits);
    }

    let mut cands: Vec<(u32, f32)> = logits
        .iter()
        .enumerate()
        .map(|(i, &l)| (i as u32, l / d.temperature))
        .collect();
    cands.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
    if d.top_k > 0 {
        cands.truncate(d.top_k);
    }

    // Softmax over the survivors (sorted, so the first is the max).
    let max = cands[0].1;
    let mut probs: Vec<f32> = cands.iter().map(|&(_, l)| (l - max).exp()).collect();
    let sum: f32 = probs.iter().sum();
    probs.iter_mut().for_each(|x| *x /= sum);

    // top-p: smallest prefix whose mass reaches p (always at least one token).
    let mut keep = probs.len();
    if d.top_p < 1.0 {
        let mut acc = 0.0;
        for (i, &x) in probs.iter().enumerate() {
            acc += x;
            if acc >= d.top_p {
                keep = i + 1;
                break;
            }
        }
    }
    // min-p: drop tokens below min_p × the top probability.
    if d.min_p > 0.0 {
        let floor = d.min_p * probs[0];
        keep = keep.min(probs.iter().take_while(|&&x| x >= floor).count().max(1));
    }
    probs.truncate(keep);

    let total: f32 = probs.iter().sum();
    let mut r = rng.random::<f32>() * total;
    for (i, &x) in probs.iter().enumerate() {
        if r < x {
            return cands[i].0;
        }
        r -= x;
    }
    cands[keep - 1].0
}

pub fn argmax(logits: &[f32]) -> u32 {
    let mut best = 0;
    for (i, &l) in logits.iter().enumerate() {
        if l > logits[best] {
            best = i;
        }
    }
    best as u32
}

/// Of the stems `applied` (with the tokens each biased), those that made a
/// difference ([`BiasHit`]), comparing the logits `before` and `after`.
fn bias_hits(w: &WordBias, applied: &[(usize, Vec<u32>)], before: &[f32], after: &[f32], temperature: f32) -> Vec<BiasHit> {
    let t = if temperature > 0.0 { temperature } else { 1.0 };
    // Softmax at `t`: the max and the normaliser; and the top token.
    let dist = |l: &[f32]| {
        let m = l.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let z: f32 = l.iter().map(|&x| ((x - m) / t).exp()).sum();
        (m, z, argmax(l))
    };
    let (mb, zb, tb) = dist(before);
    let (ma, za, ta) = dist(after);
    let mut out = Vec::new();
    for (i, toks) in applied {
        let s = &w.stems[*i];
        let mut toks = toks.clone();
        toks.sort_unstable();
        toks.dedup();
        let prob = |l: &[f32], m: f32, z: f32| {
            toks.iter().filter_map(|&k| l.get(k as usize)).map(|&x| ((x - m) / t).exp() / z).sum::<f32>()
        };
        let (pb, pa) = (prob(before, mb, zb), prob(after, ma, za));
        let (top_before, top_after) = (toks.contains(&tb), toks.contains(&ta));
        let hit = if s.bias < 0.0 {
            pb >= HIT_PROB || top_before
        } else {
            (pa >= HIT_PROB && pb < HIT_PROB) || (top_after && !top_before)
        };
        if hit {
            out.push(BiasHit { stem: s.stem.clone(), bias: s.bias, before: pb, after: pa, top_before, top_after });
        }
    }
    out
}

/// llama.cpp-style: each distinct token in the window has its logit divided by
/// `penalty` if positive, multiplied if negative.
fn apply_repeat_penalty(logits: &mut [f32], context: &[u32], penalty: f32, last_n: usize) {
    if penalty == 1.0 || last_n == 0 {
        return;
    }
    let window = &context[context.len().saturating_sub(last_n)..];
    let mut seen = std::collections::HashSet::with_capacity(window.len());
    for &t in window {
        if seen.insert(t)
            && let Some(l) = logits.get_mut(t as usize)
        {
            *l = if *l > 0.0 { *l / penalty } else { *l * penalty };
        }
    }
}

/// Longest DRY match counted; the penalty is already prohibitive long before.
const DRY_MAX_MATCH: usize = 50;

/// DRY (as in text-generation-webui and koboldcpp): penalises the tokens that
/// would continue a sequence already in `context`. For each earlier occurrence
/// of the last token, the match is extended backwards while the tokens before it
/// agree with the tokens before the end; the token that followed it gets
/// `multiplier × base^(len − allowed_length)` subtracted from its logit once the
/// match is `allowed_length` long. Matches stop at breaker tokens, so names,
/// `"name:"` prefixes and line starts aren't penalised; a breaker is never
/// penalised either.
fn apply_dry(logits: &mut [f32], context: &[u32], p: &DryParams) {
    let Some((&last, before)) = context.split_last() else {
        return;
    };
    if p.multiplier <= 0.0 || p.breakers.contains(&last) {
        return;
    }
    let n = context.len();
    let mut longest: HashMap<u32, usize> = HashMap::new();
    for (i, _) in before.iter().enumerate().filter(|&(_, &t)| t == last) {
        let next = context[i + 1];
        if p.breakers.contains(&next) {
            continue;
        }
        // context[i + 1 - len..=i] == context[n - len..]; i < n - 1, so the two
        // windows are distinct sequences even when they overlap.
        let mut len = 1;
        while len < DRY_MAX_MATCH && len <= i {
            let t = context[i - len];
            if t != context[n - 1 - len] || p.breakers.contains(&t) {
                break;
            }
            len += 1;
        }
        let e = longest.entry(next).or_default();
        *e = (*e).max(len);
    }
    for (token, len) in longest {
        if len >= p.allowed_length
            && let Some(l) = logits.get_mut(token as usize)
        {
            *l -= p.multiplier * p.base.powi((len - p.allowed_length) as i32);
        }
    }
}

/// By default, a word in progress spelling this many letters of a stem gets its
/// bias.
pub const STEM_MIN_LETTERS: usize = 3;

/// Discouraged word stems (design.md §5): a token gets a stem's bias (added to
/// its logit) when it would make the word being written spell at least the
/// first `min` letters of the stem ([`STEM_MIN_LETTERS`] by default; more keeps
/// "seek" off " see"), or run past it. So " giggle",
/// "*gig" and the "igg" of "*|g|igg|les" are caught, but not "g" alone, " gr", or
/// "gig" continuing "zi". Letters are ASCII, matched case-insensitively; the
/// start of the reply counts as a word start.
///
/// A stem with `after` triggers is only biased where one of them ends at most
/// `within` words before the word being written ("me" after "play with"). The
/// words come from the reply so far, apostrophes dropped ("let's" is "lets").
/// Trigger words match loosely, since the persona misspells: up to 3 letters
/// must be the whole word, 4 letters the start of it ("play" in "playin"), and
/// 5 or more the start give or take one typo ("wanna" in "wana").
///
/// A stem with `prompted_by` phrases is only biased when one of them appears
/// anywhere in the user messages the reply answers ("seek" while someone has
/// just asked for hide and seek), matched the same way.
pub struct WordBias {
    /// Each token's text (`Tokenizer::pieces`), to find the word in progress.
    pieces: Vec<String>,
    stems: Vec<Stem>,
    /// The longest stem: a longer word in progress matches none.
    longest: usize,
    /// Finished words a trigger check needs (longest trigger + `within`); 0
    /// when no stem has triggers.
    lookback: usize,
    /// Tokens that end a word (not starting with an ASCII letter), for
    /// `whole_word` stems; empty when there are none.
    enders: Vec<u32>,
}

/// One discouraged stem for [`WordBias::new`].
#[derive(Clone, Debug, PartialEq)]
pub struct StemRule {
    /// ASCII letters.
    pub stem: String,
    pub bias: f32,
    /// Letters of the stem a word must spell before it's biased.
    pub min: usize,
    /// Trigger phrases ("play with"); empty = biased everywhere.
    pub after: Vec<String>,
    /// Words allowed between the end of a trigger and the biased word.
    pub within: usize,
    /// Phrases in the user messages the reply answers ("hide and seek"); empty
    /// = no condition on the message.
    pub prompted_by: Vec<String>,
    /// Bias only the word exactly: once the word in progress is the stem, the
    /// tokens that would end it (any not starting with a letter: a space,
    /// punctuation, a line break, end of text), so longer words stay free
    /// ("cock" but not "cocktail"). `min` doesn't apply.
    pub whole_word: bool,
}

impl StemRule {
    /// A stem biased everywhere.
    pub fn new(stem: &str, bias: f32, min: usize) -> Self {
        Self {
            stem: stem.into(),
            bias,
            min,
            after: Vec::new(),
            within: 0,
            prompted_by: Vec::new(),
            whole_word: false,
        }
    }
}

struct Stem {
    /// Lowercase.
    stem: String,
    bias: f32,
    /// Tokens starting a new word with the stem (" gig", "*gig"): biased anywhere.
    fresh: Vec<u32>,
    /// `cont[k]`: tokens starting with letters that continue a word spelling the
    /// stem's first `k` letters ("igg" after "g"; with `k = 0`, "gig" after "*").
    cont: Vec<Vec<u32>>,
    /// Trigger phrases; empty = always on.
    after: Vec<Vec<TriggerWord>>,
    within: usize,
    /// `prompted_by` phrases; empty = always on.
    prompted_by: Phrases,
    /// Bias [`WordBias::enders`] once the word in progress is exactly the stem,
    /// instead of `fresh` and `cont` (which stay empty).
    whole_word: bool,
}

/// Phrases to look for in a user message (`prompted_by` word rules, prompt
/// notes), matched more loosely than `after` triggers: a missed match lets the
/// reply through, a false one only nudges it.
#[derive(Clone, Debug, Default)]
pub struct Phrases(Vec<(Vec<TriggerWord>, TriggerWord)>);

impl Phrases {
    pub fn new(phrases: &[String]) -> Self {
        Self(
            phrases
                .iter()
                .map(|t| trigger_words(t))
                .filter(|t| !t.is_empty())
                .map(|t| {
                    // The words run together, for "tuckyouin".
                    let joined = TriggerWord {
                        len: t.iter().map(|w| w.len).sum(),
                        word: collapse_repeats(&t.iter().map(|w| w.word.as_str()).collect::<String>()),
                    };
                    (t, joined)
                })
                .collect(),
        )
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Whether one of the phrases appears anywhere in `message` (as
    /// [`normalized_words`]), loosely: from 4 letters also inside a longer
    /// word. Rules on user messages use [`found_whole_in`](Self::found_whole_in);
    /// this is for the reply scorer (`persona::pick_score`) and
    /// `floob_bench`'s legacy baseline.
    pub fn found_in(&self, message: &[String]) -> bool {
        self.found(message, TriggerWord::matches_in_message)
    }

    /// [`found_in`](Self::found_in) for whole words only
    /// ([`TriggerWord::matches_whole`]: "deal" in "deals", not "ideal"; "coal"
    /// not in "coalition"). What every rule on the user's messages uses:
    /// notes, `prompted_by` word biases and bans (2026-10-15; matching inside
    /// words fired notes on "upgrade", "systemd" and "crystals").
    pub fn found_whole_in(&self, message: &[String]) -> bool {
        self.found(message, TriggerWord::matches_whole)
    }

    fn found(&self, message: &[String], hit: fn(&TriggerWord, &str) -> bool) -> bool {
        self.0.iter().any(|(p, joined)| {
            (p.len() <= message.len() && message.windows(p.len()).any(|w| p.iter().zip(w).all(|(t, m)| hit(t, m))))
                || (p.len() > 1 && message.iter().any(|m| hit(joined, m)))
        })
    }
}

impl Stem {
    fn prompted(&self, message: &[String]) -> bool {
        self.prompted_by.is_empty() || self.prompted_by.found_whole_in(message)
    }

    /// Whether a trigger ends within `within` words of the end of `words`.
    fn triggered(&self, words: &[String]) -> bool {
        self.after.is_empty()
            || self.after.iter().any(|t| {
                (words.len().saturating_sub(self.within)..=words.len())
                    .filter(|&end| end >= t.len())
                    .any(|end| t.iter().zip(&words[end - t.len()..end]).all(|(t, w)| t.matches(w)))
            })
    }
}

impl std::fmt::Debug for WordBias {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_map()
            .entries(self.stems.iter().map(|s| (&s.stem, s.bias)))
            .finish()
    }
}

impl WordBias {
    /// How many rules there are, and how many of them `message` (the user
    /// messages a reply answers, as [`normalized_words`]) switches on: those
    /// without `prompted_by`, or with one of its phrases in the message.
    pub fn counts(&self, message: &[String]) -> (usize, usize) {
        (self.stems.len(), self.stems.iter().filter(|s| s.prompted(message)).count())
    }

    pub fn new(pieces: Vec<String>, rules: &[StemRule]) -> Self {
        let stems: Vec<Stem> = rules
            .iter()
            .map(|r| {
                let stem = r.stem.to_ascii_lowercase();
                let min = r.min.clamp(1, stem.len().max(1));
                let hits = |word: &str| word.len() >= min && (stem.starts_with(word) || word.starts_with(&stem));
                let mut fresh = Vec::new();
                let mut cont = vec![Vec::new(); stem.len()];
                for (id, piece) in pieces.iter().enumerate().filter(|_| !r.whole_word) {
                    let Some(start) = piece.find(|c: char| c.is_ascii_alphabetic()) else {
                        continue;
                    };
                    let letters = leading_letters(&piece[start..]).to_ascii_lowercase();
                    if start > 0 {
                        if hits(&letters) {
                            fresh.push(id as u32);
                        }
                        continue;
                    }
                    for (k, list) in cont.iter_mut().enumerate() {
                        if hits(&format!("{}{letters}", &stem[..k])) {
                            list.push(id as u32);
                        }
                    }
                }
                Stem {
                    stem,
                    bias: r.bias,
                    fresh,
                    cont,
                    after: r.after.iter().map(|t| trigger_words(t)).filter(|t| !t.is_empty()).collect(),
                    within: r.within,
                    prompted_by: Phrases::new(&r.prompted_by),
                    whole_word: r.whole_word,
                }
            })
            .collect();
        let enders = if rules.iter().any(|r| r.whole_word) {
            (0..pieces.len() as u32)
                .filter(|&id| !pieces[id as usize].starts_with(|c: char| c.is_ascii_alphabetic()))
                .collect()
        } else {
            Vec::new()
        };
        let longest = stems.iter().map(|s| s.stem.len()).max().unwrap_or(0);
        let lookback = stems
            .iter()
            .flat_map(|s| s.after.iter().map(|t| t.len() + s.within))
            .max()
            .unwrap_or(0);
        Self {
            pieces,
            stems,
            longest,
            lookback,
            enders,
        }
    }

    /// Adds the biases for the next token after `generated` (the reply so far),
    /// answering `message` (the user messages, as [`normalized_words`]).
    fn apply(&self, logits: &mut [f32], generated: &[u32], message: &[String]) {
        self.apply_recording(logits, generated, message, false);
    }

    /// [`apply`](Self::apply); with `record`, which stems it applied (by
    /// index) and to which tokens.
    fn apply_recording(&self, logits: &mut [f32], generated: &[u32], message: &[String], record: bool) -> Vec<(usize, Vec<u32>)> {
        let mut applied = Vec::new();
        let word = self.word_in_progress(generated);
        // A fresh token starts a new word, so every word so far precedes it; a
        // continuing token extends the last word, which then doesn't count.
        let (before_fresh, before_cont) = if self.lookback > 0 {
            let (words, open) = self.recent_words(generated);
            let cut = words.len().saturating_sub(usize::from(open));
            (words.clone(), words[..cut].to_vec())
        } else {
            (Vec::new(), Vec::new())
        };
        for (i, s) in self.stems.iter().enumerate().filter(|(_, s)| s.prompted(message)) {
            if s.whole_word {
                if word.as_deref() == Some(s.stem.as_str()) && s.triggered(&before_cont) {
                    for &t in &self.enders {
                        if let Some(l) = logits.get_mut(t as usize) {
                            *l += s.bias;
                        }
                    }
                    if record {
                        applied.push((i, self.enders.clone()));
                    }
                }
                continue;
            }
            let fresh: &[u32] = if s.triggered(&before_fresh) { &s.fresh } else { &[] };
            let cont = word
                .as_deref()
                .filter(|w| s.stem.starts_with(w))
                .filter(|_| s.triggered(&before_cont))
                .and_then(|w| s.cont.get(w.len()));
            for &t in fresh.iter().chain(cont.into_iter().flatten()) {
                if let Some(l) = logits.get_mut(t as usize) {
                    *l += s.bias;
                }
            }
            if record && (!fresh.is_empty() || cont.is_some_and(|c| !c.is_empty())) {
                applied.push((i, fresh.iter().chain(cont.into_iter().flatten()).copied().collect()));
            }
        }
        applied
    }

    /// The last words of `generated` (normalized, at least `lookback` + 1 of
    /// them when the reply is that long), and whether the last one is still
    /// open (the text ends in a letter).
    fn recent_words(&self, generated: &[u32]) -> (Vec<String>, bool) {
        let mut tail = String::new();
        for &t in generated.iter().rev() {
            let piece = self.pieces.get(t as usize).map_or("", String::as_str);
            if piece.is_empty() {
                break;
            }
            tail.insert_str(0, piece);
            // One more than needed: the first may be cut off mid-word.
            if normalized_words(&tail).len() > self.lookback + 2 {
                break;
            }
        }
        // The last word is still open if the reply ends in letters, unless
        // `normalized_words` drops that word (it has digits: "13.3cm").
        let last = tail.rsplit(char::is_whitespace).next().unwrap_or("");
        let open = tail.ends_with(|c: char| c.is_ascii_alphabetic()) && !last.chars().any(|c| c.is_ascii_digit());
        (normalized_words(&tail), open)
    }

    /// The letters ending `generated`, lowercased ("" after a space or `*`), or
    /// `None` once they are longer than every stem.
    fn word_in_progress(&self, generated: &[u32]) -> Option<String> {
        let mut parts = Vec::new();
        let mut len = 0;
        for &t in generated.iter().rev() {
            let piece = self.pieces.get(t as usize).map_or("", String::as_str);
            let letters = trailing_letters(piece);
            len += letters.len();
            if len > self.longest {
                return None;
            }
            parts.push(letters);
            if piece.is_empty() || letters.len() < piece.len() {
                break;
            }
        }
        Some(parts.iter().rev().map(|p| p.to_ascii_lowercase()).collect())
    }
}

/// Chat spellings read as the word they stand for, so "tuck u in" matches
/// "tuck you in" and "gud boi" matches "good boy".
pub(crate) const CHAT_SPELLINGS: &[(&str, &str)] = &[
    ("u", "you"),
    ("ya", "you"),
    ("yu", "you"),
    ("ur", "your"),
    ("yer", "your"),
    ("n", "and"),
    ("an", "and"),
    ("r", "are"),
    ("gud", "good"),
    ("boi", "boy"),
    ("2", "to"),
    ("4", "for"),
    ("b4", "before"),
];

/// Lowercase ASCII words with apostrophes dropped ("let's" is "lets"), chat
/// spellings replaced ([`CHAT_SPELLINGS`]), and other words with digits left
/// out.
fn chat_words(s: &str) -> Vec<String> {
    s.chars()
        .filter(|&c| c != '\'' && c != '\u{2019}')
        .collect::<String>()
        .to_ascii_lowercase()
        .split(|c: char| !c.is_ascii_alphanumeric())
        .filter(|w| !w.is_empty())
        .filter_map(|w| match CHAT_SPELLINGS.iter().find(|(from, _)| *from == w) {
            Some((_, to)) => Some(to.to_string()),
            None => w.chars().all(|c| c.is_ascii_alphabetic()).then(|| w.to_string()),
        })
        .collect()
}

/// A run of the same letter as one ("seeeek" and "seek" both "sek", "mooom"
/// "mom"), so stretched spellings match.
pub(crate) fn collapse_repeats(w: &str) -> String {
    let mut out = String::with_capacity(w.len());
    for c in w.chars() {
        if !out.ends_with(c) {
            out.push(c);
        }
    }
    out
}

/// The words of `s` as triggers compare them: [`chat_words`] with repeated
/// letters collapsed.
pub fn normalized_words(s: &str) -> Vec<String> {
    chat_words(s).iter().map(|w| collapse_repeats(w)).collect()
}

/// One word of a trigger phrase.
#[derive(Clone, Debug)]
pub(crate) struct TriggerWord {
    /// Collapsed, like the words it is compared with.
    pub(crate) word: String,
    /// Letters before collapsing, which decide how loosely it matches.
    pub(crate) len: usize,
}

pub(crate) fn trigger_words(phrase: &str) -> Vec<TriggerWord> {
    chat_words(phrase)
        .into_iter()
        .map(|w| TriggerWord {
            len: w.len(),
            word: collapse_repeats(&w),
        })
        .collect()
}

impl TriggerWord {
    /// Whether `word` (normalized) matches: the whole word for triggers up to
    /// 3 letters, its start at 4, and its start give or take one typo from 5
    /// ("wana" for "wanna", "togetha" for "toget").
    pub(crate) fn matches(&self, word: &str) -> bool {
        let t = self.word.as_str();
        match self.len {
            0..=3 => word == t,
            4 => word.starts_with(t),
            _ => {
                let n = t.len();
                word.starts_with(t)
                    || (n.saturating_sub(1)..=n + 1)
                        .any(|k| k <= word.len() && edit_distance(t.as_bytes(), &word.as_bytes()[..k]) <= 1)
            }
        }
    }

    /// The word itself, from 3 letters with an ending ("deals", "offered",
    /// "duckie", "smellyyy"; "ai" isn't "aid"), or from 6 letters (repeats collapsed) one typo off
    /// ("filthey"); never inside or at the start of another word ("ideal",
    /// "coalition", "firewall"), and no typo for shorter ones ("day" isn't
    /// "daddy", collapsed "dady"; "trade" isn't "grade").
    pub(crate) fn matches_whole(&self, word: &str) -> bool {
        const ENDINGS: [&str; 10] = ["s", "es", "d", "ed", "ing", "er", "ers", "y", "ie", "ies"];
        let t = self.word.as_str();
        word == t
            || (t.len() >= 3 && word.strip_prefix(t).is_some_and(|rest| ENDINGS.contains(&rest)))
            || (t.len() >= 6 && edit_distance(t.as_bytes(), word.as_bytes()) <= 1)
    }

    /// In a user message, also anywhere inside a word from 4 letters ("seek"
    /// in "hidenseek", "duck" in "duckie"); shorter ones stay whole words, so
    /// "tag" doesn't fire on "stage".
    fn matches_in_message(&self, word: &str) -> bool {
        self.matches(word) || (self.len >= 4 && word.contains(self.word.as_str()))
    }
}

/// Levenshtein distance.
pub(crate) fn edit_distance(a: &[u8], b: &[u8]) -> usize {
    let mut prev: Vec<usize> = (0..=b.len()).collect();
    for (i, &x) in a.iter().enumerate() {
        let mut cur = vec![i + 1; b.len() + 1];
        for (j, &y) in b.iter().enumerate() {
            cur[j + 1] = (prev[j] + usize::from(x != y)).min(prev[j + 1] + 1).min(cur[j] + 1);
        }
        prev = cur;
    }
    prev[b.len()]
}

fn leading_letters(s: &str) -> &str {
    &s[..s.len() - s.trim_start_matches(|c: char| c.is_ascii_alphabetic()).len()]
}

fn trailing_letters(s: &str) -> &str {
    &s[s.trim_end_matches(|c: char| c.is_ascii_alphabetic()).len()..]
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Pieces: 0 "s" (continues a word), 1 " snack", 2 " zebra" (new words),
    /// 3 "!" (punctuation), 4 " like" (ends in a word), 5 "," (doesn't).
    fn word_start(chance: f32, temperature: f32) -> WordStartParams {
        let pieces = ["s", " snack", " zebra", "!", " like", ","];
        WordStartParams {
            chance,
            temperature,
            top_k: 0,
            top_p: 1.0,
            min_p: 0.0,
            shapes: pieces.iter().map(|p| TokenShape::of(p)).collect(),
        }
    }

    #[test]
    fn word_bias_counts_rules_and_those_the_message_switches_on() {
        let pieces = vec![" gig".to_string(), " hide".to_string()];
        let mut seek = StemRule::new("hide", -2.0, 2);
        seek.prompted_by = vec!["hide and seek".into()];
        let w = WordBias::new(pieces, &[StemRule::new("gig", -2.0, 2), seek]);
        assert_eq!(w.counts(&normalized_words("hi there")), (2, 1));
        assert_eq!(w.counts(&normalized_words("lets play hide and seek")), (2, 2));
    }

    #[test]
    fn token_shapes() {
        let s = TokenShape::of;
        assert_eq!(s(" snack"), TokenShape { word: true, spaced: true, open: true });
        assert_eq!(s("s"), TokenShape { word: true, spaced: false, open: true });
        assert_eq!(s("!"), TokenShape { word: false, spaced: false, open: false });
        assert_eq!(s(" !"), TokenShape { word: false, spaced: true, open: false });
        assert!(s("don'").open && !s("\n").word && !s("").word);
    }

    #[test]
    fn word_starts_are_drawn_again_without_moving_word_breaks() {
        let w = word_start(1.0, 0.0);
        // After " like": "s" continues it, " snack" and " zebra" start words.
        assert!(!w.starts_word(0, true) && w.starts_word(1, true) && w.starts_word(2, true) && !w.starts_word(3, true));
        // After "," (or at the start): "s" starts a word too.
        assert!(w.starts_word(0, false));
        let logits = [5.0, 3.0, 2.9, -10.0, -10.0, -10.0];
        let base = SamplerParams { temperature: 1.0, top_k: 0, top_p: 1.0, min_p: 0.0, repeat: None, ..SamplerParams::default() };
        let (mut continued, mut zebras) = (0, 0);
        for seed in 0..300 {
            let plain = Sampler::new(base.clone(), seed).sample(&mut logits.clone(), &[4], 0);
            let mut s = Sampler::new(SamplerParams { word_start: Some(w.clone()), ..base.clone() }, seed);
            let got = s.sample(&mut logits.clone(), &[4], 0);
            // The same first draw: a continuation stays one, untouched.
            assert_eq!(plain == 0, got == 0, "seed {seed}");
            // A word start is drawn again (greedy here): always " snack".
            if got != 0 {
                assert_eq!(got, 1, "seed {seed}");
                let r = s.take_redrawn().unwrap();
                assert_eq!((r.normal, r.picked), (plain, 1));
                // e^3 / (e^5 + e^3 + e^2.9) at temperature 1.
                assert!((r.picked_prob - 0.1077).abs() < 1e-3 && r.normal_prob > 0.0, "{r:?}");
                // The word starts " snack", " zebra" and " like", likeliest
                // first; greedy here, so " snack" gets it all.
                let ids: Vec<u32> = r.odds.iter().map(|o| o.0).collect();
                assert_eq!(ids, [1, 2, 4], "{r:?}");
                assert!((r.odds[0].2 - 1.0).abs() < 1e-3 && r.odds[1].2 < 1e-3, "{r:?}");
            } else {
                assert_eq!(s.take_redrawn(), None);
                continued += 1;
            }
            zebras += usize::from(plain == 2);
        }
        assert!(continued > 150 && zebras > 0, "{continued} {zebras}");
        // chance 0: the plain draw.
        let none = SamplerParams { word_start: Some(word_start(0.0, 0.0)), ..base.clone() };
        for seed in 0..50 {
            let plain = Sampler::new(base.clone(), seed).sample(&mut logits.clone(), &[4], 0);
            assert_eq!(Sampler::new(none.clone(), seed).sample(&mut logits.clone(), &[4], 0), plain);
        }
    }

    #[test]
    fn whole_word_phrases_skip_longer_words() {
        let p = Phrases::new(&["deal".into(), "coal".into(), "give me".into(), "filthy".into()]);
        let has = |s: &str| p.found_whole_in(&normalized_words(s));
        for yes in ["deal?", "deals r deals", "coal!", "GIVE ME that", "u filthey thing"] {
            assert!(has(yes), "{yes}");
        }
        for no in ["my ideal snack", "the coalition", "coalesce", "forgive me", "dealership"] {
            assert!(!has(no), "{no}");
        }
    }

    /// A tiny vocabulary: ids 0.. are these pieces.
    fn words() -> WordBias {
        let pieces = [" gig", "gle", "*", "g", "igg", "les", " gr", "in", " Gig", "zi", "gig", " ", "", "s"];
        WordBias::new(
            pieces.iter().map(|s| s.to_string()).collect(),
            &[StemRule::new("gigg", -2.0, STEM_MIN_LETTERS)],
        )
    }

    /// The ids `words()` biases after `generated`.
    fn biased(w: &WordBias, generated: &[u32]) -> Vec<u32> {
        let mut logits = vec![0.0; w.pieces.len()];
        w.apply(&mut logits, generated, &[]);
        (0..logits.len() as u32).filter(|&i| logits[i as usize] != 0.0).collect()
    }

    #[test]
    fn word_bias_applies_at_word_starts_and_inside_matching_words() {
        let w = words();
        // Start of the reply: " gig", " Gig" (any case) and "gig" (a word start).
        assert_eq!(biased(&w, &[]), [0, 8, 10]);
        // After "*": the same; "g" alone is too short to tell.
        assert_eq!(biased(&w, &[2]), [0, 8, 10]);
        // "*g": "igg" makes "gigg"; "gig" makes "ggig", no match.
        assert_eq!(biased(&w, &[2, 3]), [0, 4, 8]);
        // " gig": "gle" makes "giggle", which runs past the stem; "g" and "gig"
        // make "gigg" and "giggig"; "igg" makes "gigigg", no match.
        assert_eq!(biased(&w, &[0]), [0, 1, 3, 8, 10]);
        // "zi": "gig" makes "zigig", no match; space tokens still apply.
        assert_eq!(biased(&w, &[9]), [0, 8]);
        // After "giggle" the word is past every stem: only fresh words.
        assert_eq!(biased(&w, &[0, 1]), [0, 8]);
        // " gr" + "in": no match.
        assert_eq!(biased(&w, &[6, 7]), [0, 8]);
        // A space token ends the word; an empty (special) piece does too.
        assert_eq!(biased(&w, &[9, 11]), [0, 8, 10]);
        assert_eq!(biased(&w, &[9, 12]), [0, 8, 10]);
    }

    #[test]
    fn whole_word_bias_ends_only_the_exact_word() {
        let pieces = [" cock", "tail", " the", ",", "s", "-", "'s", "", "roach", " c", "ock", " Cock", "\n"];
        let rule = StemRule { whole_word: true, ..StemRule::new("cock", -4.0, STEM_MIN_LETTERS) };
        let w = WordBias::new(pieces.iter().map(|s| s.to_string()).collect(), &[rule]);
        let enders = [0, 2, 3, 5, 6, 7, 9, 11, 12];
        // Nothing until the word is exactly "cock"; then every ending token
        // (space, punctuation, "-", "'s", empty, line break), never "tail",
        // "s" or "roach".
        assert_eq!(biased(&w, &[]), [] as [u32; 0]);
        assert_eq!(biased(&w, &[0]), enders);
        assert_eq!(biased(&w, &[9, 10]), enders);
        assert_eq!(biased(&w, &[11]), enders);
        assert_eq!(biased(&w, &[0, 1]), [] as [u32; 0]);
        assert_eq!(biased(&w, &[0, 4]), [] as [u32; 0]);
        assert_eq!(biased(&w, &[9]), [] as [u32; 0]);
        // Rules without whole_word don't build the list.
        assert!(words().enders.is_empty());
    }

    #[test]
    fn word_bias_min_letters_spares_shorter_words() {
        let pieces = [" see", " seek", " se", "ek", "e"];
        let w = |min| WordBias::new(pieces.iter().map(|s| s.to_string()).collect(), &[StemRule::new("seek", -2.0, min)]);
        // Default 3: " see" is collateral.
        assert_eq!(biased(&w(3), &[]), [0, 1]);
        // 4: only " seek", and " se" + "ek"; " se" + "e" spells just "see".
        assert_eq!(biased(&w(4), &[]), [1]);
        assert_eq!(biased(&w(4), &[2]), [1, 3]);
    }

    /// A vocabulary for trigger tests, and a rule on `stem` after `after`.
    fn triggered(stem: &str, min: usize, after: &[&str], within: usize) -> WordBias {
        let pieces = [
            " play", " with", " me", " the", " ball", " m", "e", " game", " tag", " came", " wana", " lets", "'s",
            " let", " playin", " togetha", "*", " we",
        ];
        let rule = StemRule {
            after: after.iter().map(|s| s.to_string()).collect(),
            within,
            ..StemRule::new(stem, -2.0, min)
        };
        WordBias::new(pieces.iter().map(|s| s.to_string()).collect(), &[rule])
    }

    #[test]
    fn triggered_bias_needs_the_phrase_just_before() {
        // 0 " play", 1 " with", 2 " me", 3 " the", 5 " m", 6 "e", 17 " we"
        let w = triggered("me", 2, &["play with"], 0);
        assert_eq!(biased(&w, &[]), [] as [u32; 0]);
        assert_eq!(biased(&w, &[1]), [] as [u32; 0]);
        assert_eq!(biased(&w, &[0, 1]), [2]);
        // " m" is too short to judge; "e" after it makes "me", still right after
        // "play with" (a fresh " me" would now follow "m").
        assert_eq!(biased(&w, &[0, 1, 5]), [6]);
        // One word in between is too many for `within: 0`.
        assert_eq!(biased(&w, &[0, 1, 3]), [] as [u32; 0]);
        // An action marker doesn't end the phrase: "play with *me".
        assert_eq!(biased(&w, &[0, 1, 16]), [2]);
    }

    #[test]
    fn a_reply_ending_in_a_word_with_digits_is_fine() {
        // "5cm" ends in letters but is dropped as a word (digits): it must not
        // count as the open word (this underflowed and panicked, 2026-10-14).
        let pieces = [" play", " with", " me", " 5", "cm", " 13", ".", "3", "cm"];
        let rule = StemRule {
            after: vec!["play with".into()],
            ..StemRule::new("me", -2.0, 2)
        };
        let w = WordBias::new(pieces.iter().map(|s| s.to_string()).collect(), &[rule]);
        let mut logits = vec![0.0; pieces.len()];
        w.apply(&mut logits, &[3, 4], &[]);
        w.apply(&mut logits, &[5, 6, 7, 8], &[]);
        assert_eq!(logits[2], 0.0, "no trigger, no bias, no panic");
        // Words with digits are invisible to triggers: " me" after "play with
        // 5cm" still follows "play with".
        w.apply(&mut logits, &[0, 1, 3, 4], &[]);
        assert_eq!(logits[2], -2.0);
    }

    #[test]
    fn triggered_bias_window_counts_words_between() {
        // 7 " game", 8 " tag", 3 " the", 4 " ball"
        let w = triggered("tag", 3, &["game"], 2);
        assert_eq!(biased(&w, &[7]), [8]);
        assert_eq!(biased(&w, &[7, 3, 4]), [8]);
        assert_eq!(biased(&w, &[7, 3, 4, 3]), [] as [u32; 0]);
    }

    #[test]
    fn prompted_bias_needs_the_phrase_in_the_message() {
        // 8 " tag"; the rule fires anywhere in the reply, but only when the
        // user's message has "hide and seek" (or a near spelling) in it.
        let pieces = [" play", " with", " me", " the", " ball", " m", "e", " game", " tag"];
        let rule = StemRule {
            prompted_by: vec!["hide and seek".into(), "tag".into()],
            ..StemRule::new("tag", -2.0, 3)
        };
        let w = WordBias::new(pieces.iter().map(|s| s.to_string()).collect(), &[rule]);
        let msg = |s: &str| normalized_words(s);
        let mut logits = vec![0.0; pieces.len()];
        w.apply(&mut logits, &[3], &msg("wanna play hide or seek?"));
        assert_eq!(logits[8], 0.0, "\"hide or seek\" isn't \"hide and seek\"");
        w.apply(&mut logits, &[3], &msg("floob wanna play Hide n Seeeek?"));
        assert_eq!(logits[8], -2.0, "n is and, repeated letters collapse");
        let mut logits = vec![0.0; pieces.len()];
        w.apply(&mut logits, &[], &msg("Tag, you're it!"));
        assert_eq!(logits[8], -2.0);
        let mut logits = vec![0.0; pieces.len()];
        w.apply(&mut logits, &[], &msg("tagged you"));
        assert_eq!(logits[8], -2.0, "a word with an ending");
        let mut logits = vec![0.0; pieces.len()];
        w.apply(&mut logits, &[], &msg("on stage"));
        assert_eq!(logits[8], 0.0, "whole words only");
    }

    #[test]
    fn message_phrases_match_chat_spellings() {
        let hit = |phrase: &str, message: &str| {
            let rule = StemRule {
                prompted_by: vec![phrase.into()],
                ..StemRule::new("tag", -2.0, 3)
            };
            let w = WordBias::new(vec![" tag".into()], &[rule]);
            let mut logits = vec![0.0];
            w.apply(&mut logits, &[], &normalized_words(message));
            logits[0] != 0.0
        };
        assert!(!hit("seek", "hidenseek time!!"), "whole words only");
        assert!(hit("hide and seek", "hideandseek time!!"), "run together");
        assert!(hit("seek", "hide & seeeek uwu"));
        assert!(hit("tuck you in", "want me 2 tuck u in?"));
        assert!(hit("tuck you in", "tuckyouin"));
        assert!(hit("go to your room", "go 2 ur room!!"));
        assert!(hit("good boy", "if ur a gud boi"));
        assert!(hit("mom", "mooom says no"));
        assert!(hit("duck", "a squeaky duckie"));
        assert!(!hit("tag", "on stage"), "short words stay whole");
        assert!(!hit("mom", "one moment"));
        assert!(!hit("daddy", "how's ur day"));
        assert!(!hit("grade", "wanna trade?"));
        assert!(!hit("carry", "crystals obviously"));
        assert!(!hit("ai", "first aid kit"));
        assert!(hit("homework", "did ur homwork?"), "a typo from 6 letters");
    }

    #[test]
    fn trigger_words_match_loosely_by_length() {
        // 4 letters: the start of a word ("playin"), not a typo ("came" for "game").
        assert_eq!(biased(&triggered("tag", 3, &["play"], 0), &[14]), [8]);
        assert_eq!(biased(&triggered("tag", 3, &["game"], 0), &[9]), [] as [u32; 0]);
        // 5+: one typo ("wana" for "wanna"); misspelt endings ("togetha").
        assert_eq!(biased(&triggered("tag", 3, &["wanna"], 0), &[10]), [8]);
        assert_eq!(biased(&triggered("tag", 3, &["toget"], 0), &[15]), [8]);
        // Up to 3: the whole word; apostrophes dropped ("let" + "'s" is "lets").
        assert_eq!(biased(&triggered("tag", 3, &["let's"], 0), &[13, 12]), [8]);
        assert_eq!(biased(&triggered("tag", 3, &["let's"], 0), &[11]), [8]);
        assert_eq!(biased(&triggered("tag", 3, &["we"], 0), &[17]), [8]);
        assert_eq!(biased(&triggered("tag", 3, &["we"], 0), &[10]), [] as [u32; 0]);
    }

    #[test]
    fn observing_reports_only_biases_that_mattered() {
        let p = SamplerParams {
            temperature: 1.0,
            repeat: None,
            words: Some(Arc::new(words())),
            observe: true,
            ..SamplerParams::default()
        };
        // " gig" (0) and " Gig" (8) the likeliest: the bias pushes them down.
        let mut s = Sampler::new(p.clone(), 0);
        let mut logits = vec![0.0; 14];
        logits[0] = 4.0;
        s.sample(&mut logits, &[], 0);
        let hits = s.take_hits();
        assert_eq!(hits.len(), 1, "{hits:?}");
        let h = &hits[0];
        assert_eq!((h.stem.as_str(), h.bias, h.top_before), ("gigg", -2.0, true));
        assert!(h.before > 0.7 && h.after < h.before, "{h:?}");
        assert!(s.take_hits().is_empty(), "taken");
        // Unlikely anyway (well under 10%, not the top): nothing reported.
        let mut logits = vec![0.0; 14];
        logits[0] = -5.0;
        logits[8] = -5.0;
        logits[6] = 3.0;
        s.sample(&mut logits, &[], 0);
        assert!(s.take_hits().is_empty());
        // Not observing: nothing recorded.
        let mut quiet = Sampler::new(SamplerParams { observe: false, ..p }, 0);
        let mut logits = vec![0.0; 14];
        logits[0] = 4.0;
        quiet.sample(&mut logits, &[], 0);
        assert!(quiet.take_hits().is_empty());
    }

    #[test]
    fn word_bias_adds_to_logits_through_the_sampler() {
        let p = SamplerParams {
            temperature: 0.0,
            repeat: None,
            words: Some(Arc::new(words())),
            ..SamplerParams::default()
        };
        let mut logits = vec![0.0; 14];
        logits[0] = 1.0; // " gig"
        logits[6] = 0.5; // " gr"
        // Only the generated tokens count: " gig" in the history doesn't matter.
        assert_eq!(Sampler::new(p, 0).sample(&mut logits, &[0, 2], 1), 6);
        assert_eq!(logits[0], -1.0);
    }

    #[test]
    fn greedy_picks_argmax() {
        let mut s = Sampler::new(SamplerParams::greedy(), 0);
        assert_eq!(s.sample(&mut [0.1, 3.0, -1.0, 2.9], &[], 0), 1);
    }

    #[test]
    fn repeat_penalty_window_covers_prompt_and_applies_once_per_token() {
        let mut logits = [2.0, -2.0, 1.0];
        apply_repeat_penalty(&mut logits, &[0, 0, 1, 2], 2.0, 3);
        // Window is [0, 1, 2]; token 0 penalised once even though it appears twice overall.
        assert_eq!(logits, [1.0, -4.0, 0.5]);
        let mut logits = [2.0, -2.0, 1.0];
        apply_repeat_penalty(&mut logits, &[0, 0, 1, 2], 2.0, 1);
        assert_eq!(logits, [2.0, -2.0, 0.5]);
    }

    #[test]
    fn generated_scope_skips_the_history() {
        let p = SamplerParams {
            temperature: 0.0,
            repeat: Some(RepeatParams {
                penalty: 100.0,
                last_n: 256,
                history: false,
            }),
            ..SamplerParams::default()
        };
        // Token 1 is only in the history, so it still wins.
        assert_eq!(Sampler::new(p.clone(), 0).sample(&mut [1.0, 2.0], &[1, 0], 1), 1);
        // Once generated, it is penalised.
        assert_eq!(Sampler::new(p, 0).sample(&mut [1.0, 2.0], &[0, 1], 1), 0);
    }

    fn dry(breakers: &[u32]) -> DryParams {
        DryParams {
            multiplier: 1.0,
            base: 2.0,
            allowed_length: 2,
            breakers: Arc::new(breakers.iter().copied().collect()),
        }
    }

    #[test]
    fn dry_penalises_continuing_a_repeated_sequence() {
        // 1 2 3 4 … 1 2 3: the match "1 2 3" (length 3) would continue with 4.
        let mut logits = [0.0; 6];
        apply_dry(&mut logits, &[1, 2, 3, 4, 5, 1, 2, 3], &dry(&[]));
        assert_eq!(logits, [0.0, 0.0, 0.0, 0.0, -2.0, 0.0]);
        // A match shorter than allowed_length isn't penalised: "3" alone → 4.
        let mut logits = [0.0; 6];
        apply_dry(&mut logits, &[3, 4, 5, 3], &dry(&[]));
        assert_eq!(logits, [0.0; 6]);
        // The longest match per token counts: "2 3" → 4 (len 2) and "1 2 3" → 5 (len 3).
        let mut logits = [0.0; 6];
        apply_dry(&mut logits, &[2, 3, 4, 1, 2, 3, 5, 1, 2, 3], &dry(&[]));
        assert_eq!(logits, [0.0, 0.0, 0.0, 0.0, -1.0, -2.0]);
    }

    #[test]
    fn dry_matches_stop_at_breakers() {
        // 0 is a breaker: "0 2 3" only matches as "2 3" (len 2).
        let mut logits = [0.0; 6];
        apply_dry(&mut logits, &[0, 2, 3, 4, 0, 2, 3], &dry(&[0]));
        assert_eq!(logits, [0.0, 0.0, 0.0, 0.0, -1.0, 0.0]);
        // A breaker as the next token, or as the last token, isn't penalised.
        let mut logits = [0.0; 6];
        apply_dry(&mut logits, &[2, 3, 0, 2, 3], &dry(&[0]));
        assert_eq!(logits, [0.0; 6]);
        let mut logits = [0.0; 6];
        apply_dry(&mut logits, &[2, 0, 4, 2, 0], &dry(&[0]));
        assert_eq!(logits, [0.0; 6]);
        // Off: nothing changes.
        let p = SamplerParams {
            temperature: 0.0,
            repeat: None,
            ..SamplerParams::default()
        };
        let mut s = Sampler::new(p.clone(), 0);
        assert_eq!(s.sample(&mut [0.0, 0.0, 0.0, 0.0, 1.0], &[1, 2, 3, 4, 1, 2, 3], 0), 4);
        let mut s = Sampler::new(SamplerParams { dry: Some(dry(&[])), ..p }, 0);
        assert_eq!(s.sample(&mut [0.0, 0.0, 0.0, 0.0, 1.0], &[1, 2, 3, 4, 1, 2, 3], 0), 0);
    }

    #[test]
    fn same_seed_same_draws() {
        let p = SamplerParams {
            repeat: None,
            ..SamplerParams::default()
        };
        let logits: Vec<f32> = (0..100).map(|i| (i as f32 * 0.37).sin()).collect();
        let draw = |seed| {
            let mut s = Sampler::new(p.clone(), seed);
            (0..20)
                .map(|_| s.sample(&mut logits.clone(), &[], 0))
                .collect::<Vec<_>>()
        };
        assert_eq!(draw(7), draw(7));
    }

    #[test]
    fn min_p_and_top_k_restrict_candidates() {
        let p = SamplerParams {
            temperature: 1.0,
            top_k: 2,
            top_p: 1.0,
            min_p: 0.0,
            repeat: None,
            ..SamplerParams::default()
        };
        let mut s = Sampler::new(p, 1);
        for _ in 0..200 {
            let t = s.sample(&mut [5.0, 4.9, 4.8, -10.0], &[], 0);
            assert!(t == 0 || t == 1);
        }
        let p = SamplerParams {
            temperature: 1.0,
            top_k: 0,
            top_p: 1.0,
            min_p: 0.5,
            repeat: None,
            ..SamplerParams::default()
        };
        let mut s = Sampler::new(p, 1);
        for _ in 0..200 {
            // p(1)/p(0) = e^-2 < 0.5, so only token 0 survives.
            assert_eq!(s.sample(&mut [3.0, 1.0, 0.0], &[], 0), 0);
        }
    }
}
