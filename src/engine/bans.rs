//! Phrase bans with rollback (design.md §5): word patterns the reply must not
//! contain. While a reply is generated, `generate` checks the text after each
//! sampled token; when a pattern matches, it rolls the reply (KV cache,
//! decoder, sampling context) back to the token that started the match, bans
//! that token at that position and samples again. Released text is never
//! taken back: replies with bans hold back their last [`HOLDBACK`] tokens.
//!
//! A pattern is space-separated word slots; `a|b|c` is one slot with
//! alternatives, `[a|b]` an optional slot:
//! `"little|lil|wee [ol|ole|old] me|floob|feller|buddy"`. Words are compared
//! lowercase, apostrophes dropped ("ol'" is "ol"), chat spellings read as the
//! word ("u" is "you"). Stricter than word-bias triggers, since a ban changes
//! the reply: up to 4 letters the whole word (repeated letters may be
//! stretched, "weee" for "wee", but "we" isn't "wee"); from 5 letters its
//! start ("floobs" for "floob"); from 6 letters also the whole word with one
//! typo ("grownup" for "grownups", "kobbold" for "kobold"; not "moment" for
//! "momma"). A number matches only itself. A word still being written never
//! matches; the hold-back leaves room to roll back once it's finished.

use std::collections::HashSet;
use std::sync::Arc;

use anyhow::{Result, bail, ensure};

use super::sampler::{CHAT_SPELLINGS, collapse_repeats, edit_distance, trigger_words};

/// Tokens held back from the stream while bans are active: the longest
/// rollback. A match that starts in text already released is left alone.
pub const HOLDBACK: usize = 8;
/// Rollbacks per reply before the rest is kept as it comes.
pub const MAX_ROLLBACKS: usize = 16;
/// Tokens banned at one position before that position is kept as it comes.
pub const MAX_BANNED_AT: usize = 6;

#[derive(Clone, Debug)]
struct Slot {
    alts: Vec<Alt>,
    optional: bool,
}

/// One alternative word: as written (lowercase) and with repeats collapsed.
#[derive(Clone, Debug)]
struct Alt {
    raw: String,
    norm: String,
}

impl Alt {
    fn matches(&self, w: &Word) -> bool {
        if !w.complete {
            return false;
        }
        if self.raw.len() <= 4 || self.raw.chars().all(|c| c.is_ascii_digit()) {
            return w.raw == self.raw || (w.norm == self.norm && w.raw.len() >= self.raw.len());
        }
        // From 5 letters its start ("floobs", "mommy's"); one typo only from 6
        // letters and against the whole word ("feller" for "fella" would need
        // its own entry): "moment" isn't "momma", "mossy" isn't "mommy".
        w.norm.starts_with(&self.norm)
            || (self.raw.len() >= 6
                && w.norm.len() + 1 >= self.norm.len()
                && edit_distance(self.norm.as_bytes(), w.norm.as_bytes()) <= 1)
    }
}

/// One compiled pattern.
#[derive(Clone, Debug)]
pub struct Ban {
    pub pattern: String,
    slots: Vec<Slot>,
}

/// A word of the reply: normalized, and where it starts in the text.
struct Word {
    /// Lowercase, apostrophes dropped, chat spellings replaced.
    raw: String,
    norm: String,
    start: usize,
    end: usize,
    /// Followed by something other than a letter: the word is finished. The
    /// last word of a reply still being generated may not be.
    complete: bool,
    /// What separates it from the word before crosses a sentence or an
    /// *action* ("!", "?", "*", brackets, a line break): a match can't span it.
    after_break: bool,
}

/// Separators a match can't cross.
const BREAKS: [char; 6] = ['!', '?', '*', '(', ')', '\n'];

impl Ban {
    pub fn parse(pattern: &str) -> Result<Self> {
        let mut slots = Vec::new();
        for raw in pattern.split_whitespace() {
            let (optional, body) = match raw.strip_prefix('[') {
                Some(r) => match r.strip_suffix(']') {
                    Some(b) => (true, b),
                    None => bail!("ban {pattern:?}: unclosed [ in {raw:?}"),
                },
                None => (false, raw),
            };
            ensure!(!body.contains(['[', ']']), "ban {pattern:?}: stray bracket in {raw:?}");
            let mut alts = Vec::new();
            for alt in body.split('|') {
                ensure!(
                    !alt.is_empty()
                        && (alt.chars().all(|c| c.is_ascii_alphabetic() || c == '\'')
                            || alt.chars().all(|c| c.is_ascii_digit())),
                    "ban {pattern:?}: words are ASCII letters and apostrophes, or a number, got {alt:?}"
                );
                // A number matches only itself ("5", not "15"): it's short.
                ensure!(trigger_words(alt).len() <= 1, "ban {pattern:?}: {alt:?} must be one word");
                // A number stays a number ("2" isn't the chat spelling of "to" here).
                let lower = alt.replace('\'', "").to_ascii_lowercase();
                let raw = if lower.chars().all(|c| c.is_ascii_digit()) { lower } else { spelled(&lower) };
                alts.push(Alt { norm: collapse_repeats(&raw), raw });
            }
            slots.push(Slot { alts, optional });
        }
        ensure!(
            slots.iter().any(|s| !s.optional),
            "ban {pattern:?}: needs at least one required word"
        );
        Ok(Self { pattern: pattern.to_string(), slots })
    }

    /// The first match in `words` starting at or after word `from`: (first
    /// word, one past the last).
    fn find_in(&self, words: &[Word], from: usize) -> Option<(usize, usize)> {
        (from..words.len()).find_map(|i| self.match_at(words, i, 0).map(|end| (i, end)))
    }

    /// Matches slots `s..` against words from `i`; the end of the match.
    fn match_at(&self, words: &[Word], i: usize, s: usize) -> Option<usize> {
        let Some(slot) = self.slots.get(s) else {
            return Some(i);
        };
        // Take the word, or skip an optional slot. Skipping doesn't move `i`,
        // so a match always begins on word `i` itself.
        words
            .get(i)
            .filter(|w| !(w.after_break && s > 0 && i > 0))
            .filter(|w| slot.alts.iter().any(|a| a.matches(w)))
            .and_then(|_| self.match_at(words, i + 1, s + 1))
            .or_else(|| slot.optional.then(|| self.match_at(words, i, s + 1)).flatten())
    }
}

/// The bans one reply is generated under: the persona's patterns, plus two
/// checks from the chat itself.
#[derive(Clone, Debug, Default)]
pub struct Bans {
    pub patterns: Vec<Ban>,
    /// Names the reply mustn't start with as "name:" (the bot's own and the
    /// chat's authors): a speaker label is the model writing a transcript.
    prefixes: Vec<String>,
    /// The first words of the bot's previous reply (normalized): a reply
    /// opening with all of them repeats it.
    echo: Vec<String>,
    /// Tokens the reply can't start with (the persona's
    /// `banned_first_chars`): masked while it is only whitespace, so no
    /// rollback is needed (nor counted by [`is_empty`](Self::is_empty)).
    pub first_chars: Option<Arc<HashSet<u32>>>,
    /// The reply's first visible word mustn't be Capitalized ("Oh", "I'm");
    /// all-caps of two letters or more ("NASA") is fine (the persona's
    /// `lowercase_start`).
    capitalized_start: bool,
}

/// How many opening words of the previous reply make a repeat.
const ECHO_WORDS: usize = 6;

impl Bans {
    pub fn parse(patterns: &[String]) -> Result<Self> {
        let patterns = patterns.iter().map(|p| Ban::parse(p)).collect::<Result<_>>()?;
        Ok(Self { patterns, ..Default::default() })
    }

    /// Also rolls back a reply that starts "name:" for any of `names`.
    pub fn with_prefixes(mut self, names: impl IntoIterator<Item = String>) -> Self {
        self.prefixes = names.into_iter().filter(|n| !n.trim().is_empty()).map(|n| n.trim().to_lowercase()).collect();
        self
    }

    /// Also rolls back a reply that opens with the first [`ECHO_WORDS`] words
    /// of `previous` (the bot's last reply here); off when it's shorter.
    pub fn with_echo(mut self, previous: &str) -> Self {
        let w: Vec<String> = words(previous).into_iter().take(ECHO_WORDS).map(|w| w.norm).collect();
        self.echo = if w.len() == ECHO_WORDS { w } else { Vec::new() };
        self
    }

    /// Also masks `ids` at the reply's start ([`first_chars`](Self::first_chars)).
    pub fn with_first_chars(mut self, ids: Arc<HashSet<u32>>) -> Self {
        self.first_chars = Some(ids);
        self
    }

    /// Also rolls back a reply whose first visible word is Capitalized.
    pub fn with_capitalized_start(mut self, on: bool) -> Self {
        self.capitalized_start = on;
        self
    }

    pub fn is_empty(&self) -> bool {
        self.patterns.is_empty() && self.prefixes.is_empty() && self.echo.is_empty() && !self.capitalized_start
    }

    /// The earliest match in `text` that starts at or after byte `from`: its
    /// start byte and the matched text.
    pub fn find(&self, text: &str, from: usize) -> Option<(usize, String)> {
        let lead = text.len() - text.trim_start().len();
        if from <= lead {
            let rest = text[lead..].to_lowercase();
            if let Some(n) = self.prefixes.iter().find(|n| rest.starts_with(&format!("{n}:"))) {
                return Some((lead, text[lead..lead + n.len() + 1].to_string()));
            }
        }
        let words = words(text);
        if from <= lead
            && self.capitalized_start
            && let Some(w) = words.first().filter(|w| w.start == lead && w.complete)
        {
            let word = &text[w.start..w.end];
            let letters: Vec<char> = word.chars().filter(|c| c.is_alphabetic()).collect();
            let shout = letters.len() >= 2 && letters.iter().all(|c| c.is_uppercase());
            if letters.first().is_some_and(|c| c.is_uppercase()) && !shout {
                return Some((w.start, format!("capitalized: {word}")));
            }
        }
        if from <= lead
            && !self.echo.is_empty()
            && words.len() >= self.echo.len()
            && words[self.echo.len() - 1].complete
            && words.iter().zip(&self.echo).all(|(w, e)| &w.norm == e)
        {
            let end = words[self.echo.len() - 1].end;
            return Some((words[0].start, format!("echo: {}", &text[words[0].start..end])));
        }
        let first = words.iter().position(|w| w.start >= from)?;
        self.patterns
            .iter()
            .filter_map(|b| b.find_in(&words, first))
            .min()
            .map(|(i, end)| {
                let start = words[i].start;
                (start, text[start..words[end - 1].end].to_string())
            })
    }
}

/// A chat spelling read as the word it stands for ("u" is "you").
fn spelled(w: &str) -> String {
    CHAT_SPELLINGS.iter().find(|(from, _)| *from == w).map_or(w.to_string(), |(_, to)| to.to_string())
}

/// The text's words with their start bytes: runs of ASCII letters, digits and
/// apostrophes, normalized like `sampler::chat_words` (apostrophes dropped,
/// lowercase, chat spellings replaced) and collapsed like trigger words.
fn words(text: &str) -> Vec<Word> {
    let mut out: Vec<Word> = Vec::new();
    let mut start = None;
    let is_word = |c: char| c.is_ascii_alphanumeric() || c == '\'' || c == '\u{2019}';
    for (i, c) in text.char_indices().chain(std::iter::once((text.len(), ' '))) {
        match (is_word(c), start) {
            (true, None) => start = Some(i),
            (false, Some(s)) => {
                let raw: String = text[s..i]
                    .chars()
                    .filter(|&c| c != '\'' && c != '\u{2019}')
                    .collect::<String>()
                    .to_ascii_lowercase();
                start = None;
                if raw.is_empty() {
                    continue;
                }
                let spelled = if raw.chars().all(|c| c.is_ascii_digit()) { raw.clone() } else { spelled(&raw) };
                let prev_end = out.last().map_or(0, |w| w.end);
                out.push(Word {
                    norm: collapse_repeats(&spelled),
                    raw: spelled,
                    start: s,
                    end: i,
                    complete: i < text.len(),
                    after_break: text[prev_end..s].contains(BREAKS),
                });
            }
            _ => {}
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bans(p: &[&str]) -> Bans {
        Bans::parse(&p.iter().map(|s| s.to_string()).collect::<Vec<_>>()).unwrap()
    }

    const SELF_LITTLE: &str = "little|lil|wee|teeny|tiny [ol|ole|old] me|floob|feller|fella|guy|buddy";

    #[test]
    fn a_capitalized_first_word_matches_but_shouting_does_not() {
        let b = Bans::default().with_capitalized_start(true);
        for (text, hit) in [("Oh hi", "Oh"), (" I'm fine", "I'm"), ("I think", "I"), ("Doru! hi", "Doru")] {
            assert_eq!(b.find(text, 0).map(|(_, m)| m), Some(format!("capitalized: {hit}")), "{text:?}");
        }
        for text in ["oh hi", "AAAA!!", "NASA is", "OMG hi", "*Wiggles* hi", "\"Paris\"", "Oh"] {
            assert_eq!(b.find(text, 0), None, "{text:?}");
        }
        // Later words are fine.
        assert_eq!(b.find("oh Hi there", 0), None);
    }

    #[test]
    fn rounds_20_to_22_young_lines_match() {
        let b = bans(&[SELF_LITTLE]);
        for (text, hit) in [
            ("u could share some with little ol' me while we wait", "little ol' me"),
            ("what brings you ta show such a prize t' this lil ole floob?", "lil ole floob"),
            ("u could even share it with ur little buddy floob.", "little buddy"),
            ("those numbers look so BIG to a little feller like me", "little feller"),
            ("too big for lil' ol' me!", "lil' ol' me"),
        ] {
            let (start, m) = b.find(text, 0).unwrap_or_else(|| panic!("no match in {text:?}"));
            assert_eq!(m, hit, "{text:?}");
            assert_eq!(&text[start..start + hit.len()], hit);
        }
    }

    #[test]
    fn small_sausages_and_bits_are_left_alone() {
        let b = bans(&[SELF_LITTLE]);
        for text in [
            "a tiny weenie! *squints*",
            "a little bit small for a frank",
            "it's a little old, maybe from last week",
            "lil sausage, big heart",
        ] {
            assert_eq!(b.find(text, 0), None, "{text:?}");
        }
    }

    #[test]
    fn optional_slots_and_from() {
        let b = bans(&["when i was [a] [but] [just] [a] little|lil|wee|young|kitten"]);
        assert_eq!(b.find("reminds me of when i was a wee kobold", 0).unwrap().1, "when i was a wee");
        assert_eq!(b.find("back when i was but a wee lil bobbit", 0).unwrap().1, "when i was but a wee");
        assert_eq!(b.find("when i was young... or maybe not", 0).unwrap().1, "when i was young");
        assert_eq!(b.find("when i was hungry", 0), None);
        let text = "when i was little, and when i was young!";
        let (first, _) = b.find(text, 0).unwrap();
        let (second, m) = b.find(text, first + 1).unwrap();
        assert_eq!((first, m.as_str(), &text[second..second + 4]), (0, "when i was young", "when"));
    }

    #[test]
    fn chat_spellings_and_typos_count() {
        let b = bans(&["can i keep it"]);
        assert!(b.find("ooh can i keeep it?!", 0).is_some());
        let b = bans(&["thank you"]);
        assert!(b.find("thank u smol bean!!", 0).is_some());
    }

    #[test]
    fn matches_dont_cross_sentences_and_short_words_wait() {
        let b = bans(&[SELF_LITTLE]);
        // c33's one false positive: "teensy!! *me…" while "*measures" was being written.
        assert_eq!(b.find("hmm, teensy!! *me", 0), None);
        assert_eq!(b.find("hmm, teensy!! *measures it*", 0), None);
        assert_eq!(b.find("a teeny one. me? i like it", 0).map(|m| m.1), None);
        assert_eq!(b.find("it's so tiny me", 0), None, "\"me\" may still become \"meat\"");
        assert_eq!(b.find("it's so tiny me ", 0).unwrap().1, "tiny me");
        // A word still being written waits ("flo" may become "flowers").
        assert_eq!(b.find("you sure you want to give this to little flo", 0), None);
        assert_eq!(b.find("you sure you want to give this to little floob?", 0).unwrap().1, "little floob");
        // Commas and "..." don't break a match.
        assert_eq!(b.find("too big for lil... ol' me!", 0).unwrap().1, "lil... ol' me");
    }

    #[test]
    fn round_24_young_lines_match() {
        let b = bans(&[
            "my [own] little|lil|liddle|wittle|widdle|teensy|teeny|tiny|wee|smol weenie|wienie|weeny|wiener|willie|willy|sausage|hotdog|frank",
            "they|theyre|they're|are [the|real] grownups|grownup|grwoedups|growups|growedups",
        ]);
        for (text, hit) in [
            ("smol bean!! oh, u mean my wittle wienie? i dunno", "my wittle wienie"),
            ("you should ask pip or dori about that, they're grwoedups and know", "they're grwoedups"),
            ("it's my own lil weenie!", "my own lil weenie"),
            ("they are the grownups here", "are the grownups"),
        ] {
            assert_eq!(b.find(text, 0).map(|m| m.1), Some(hit.to_string()), "{text:?}");
        }
        for text in ["my sausage is bigger", "a wittle weenie, how cute", "grownups only!", "they are grown in caves"] {
            assert_eq!(b.find(text, 0), None, "{text:?}");
        }
    }

    #[test]
    fn chat_false_positives_of_c34_c35() {
        // "we feel" ("we" isn't "wee", "feel" isn't "fella") and "am the"
        // ("the" isn't "three").
        let b = bans(&[
            SELF_LITTLE,
            "im|am [just|only|still|really|actually|now] 1|2|3|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve [years|yrs|year|kobold]",
        ]);
        for text in ["oh no, we feel bad!", "i am the best trap maker", "a little feel of it", "im ten feet down"] {
            assert_eq!(b.find(text, 0).map(|m| m.1), if text.contains("ten") { Some("im ten".into()) } else { None }, "{text:?}");
        }
        assert_eq!(b.find("i am three years old!", 0).unwrap().1, "am three years");
        assert_eq!(b.find("such a weee little feller here", 0).unwrap().1, "little feller");
    }

    #[test]
    fn rounds_26_27_young_lines_match() {
        let b = bans(&[
            "little|lil|liddle|widdle|wee|teeny|tiny|smol [ol|ole|old] me|floob|feller|fella|guy|buddy|kobold|kob|kobbie",
            "little|lil|tiny|young|younger|small|smol|baby|kid|hatchling [kobold] again",
            "made|turned|making me [a|an|into] [a] baby|kid|child|hatchling|toddler|kit|little|younger|young",
            "big sis|sissy|sister|bro|brother|brudder|bruder",
            "lil|little|baby|bby sis|brother|brudder|bro",
            "nap time",
            "naptime|nappytime|nappytimes",
            "mommy|mama|mom|momma",
            "as a wee|lil|little one",
            "when i was [just] your size",
        ]);
        for (text, hit) in [
            ("yay! i'm lil again!", "lil again"),
            ("ugh... mossbyte made me a baby again...", "made me a baby"),
            ("now i'm a tiny baby kobold again?!", "baby kobold again"),
            ("hi big sis!!", "big sis"),
            ("u my big sis, and i ur lil bby brudder!", "big sis"),
            ("ok, nap time if u say so.", "nap time"),
            ("is it naptime already?!", "naptime"),
            ("i'm not a baby sophie... ok, mommy...", "mommy"),
            ("then remembers hes just a little kob.", "little kob"),
            ("when i was just your size as a wee one and used to", "when i was just your size"),
        ] {
            assert_eq!(b.find(text, 0).map(|m| m.1), Some(hit.to_string()), "{text:?}");
        }
        for text in [
            "i'm big and old again!",
            "a big sister rock",
            "i took a nap on my rock",
            "the kobold is little but strong",
            "made me a sandwich",
        ] {
            assert_eq!(b.find(text, 0).map(|m| m.1), if text.contains("sister") { Some("big sister".into()) } else { None }, "{text:?}");
        }
    }

    #[test]
    fn c36_quality_false_positives() {
        let b = bans(&[
            "mommy|mama|mom|momma",
            "im|am [just|only|still|really|actually|now] 1|2|3|4|5|6|7|8|9|10|11|12|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve [years|yrs|year|kobold]",
        ]);
        for text in ["wait a moment!", "hi mossy!", "i'm right here", "i'm too slow", "im to blame"] {
            assert_eq!(b.find(text, 0), None, "{text:?}");
        }
        assert_eq!(b.find("ok mommy!", 0).unwrap().1, "mommy");
        assert_eq!(b.find("i'm 2!", 0).unwrap().1, "i'm 2");
    }

    #[test]
    fn speaker_labels_and_repeats_of_the_last_reply() {
        let b = Bans::default().with_prefixes(["floob".to_string(), "thornback".to_string()]);
        assert_eq!(b.find("floob: hi!!", 0).unwrap().1, "floob:");
        assert_eq!(b.find("  Thornback: you want help?", 0).unwrap(), (2, "Thornback:".to_string()));
        assert_eq!(b.find("floob is here: hi", 0), None);
        assert_eq!(b.find("hi thornback: what", 0), None);
        let prev = "oh no!! not cricket!!! his name was timothy";
        let b = Bans::default().with_echo(prev);
        assert!(b.find("oh no!! not cricket!!! his name was", 0).is_some());
        assert_eq!(b.find("oh no!! not cricket!!! *sniffs*", 0), None, "five words is not a repeat");
        assert_eq!(b.find("oh no!! poor cricket!!! his name was", 0), None);
        assert!(Bans::default().with_echo("hi!").is_empty(), "short replies don't count");
    }

    #[test]
    fn numbers_match_whole() {
        let b = bans(&["im|am [only|just] 5|five [years|yrs|year]"]);
        assert_eq!(b.find("ok ok im 5 now!", 0).unwrap().1, "im 5");
        assert_eq!(b.find("i am only five years old", 0).unwrap().1, "am only five years");
        assert_eq!(b.find("im 15 meters down", 0), None);
        assert_eq!(b.find("i'm 471!", 0), None);
    }

    #[test]
    fn bad_patterns_fail() {
        for p in ["", "[maybe]", "lit[tle", "two words|x y]", "13x7", "a|"] {
            assert!(Bans::parse(&[p.to_string()]).is_err(), "{p:?}");
        }
    }
}
