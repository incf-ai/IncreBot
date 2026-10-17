//! Incremental detokenization with stop-string holdback (design.md §5).
//!
//! Text is always decoded from the whole generated sequence and only the new,
//! settled suffix is released, so byte-fallback tokens that split a UTF-8
//! character never surface as U+FFFD, and no prefix of a stop string is released
//! before it is known not to complete.
//!
//! Stop strings match ignoring ASCII case, so a model writing in lowercase can't
//! slip past `"\nAlice:"` with `"\nalice:"`. ASCII case folding keeps byte offsets
//! identical, so positions found in the folded text index the original.
//!
//! A stop string only counts if it reaches past the reply's leading whitespace,
//! so a `"\n"` stop doesn't end a reply that starts with a newline.
//!
//! With a line limit (`max_lines`), the line break that ends the last allowed
//! line is a stop too. Only lines with something on them count, so a blank
//! line between two doesn't use one up.

use std::num::NonZeroUsize;

/// What a decoding step produced.
#[derive(Debug, PartialEq, Eq)]
pub enum Step {
    /// New text safe to show (may be empty).
    Text(String),
    /// A stop string was found. Carries the last text to show; the stop string and
    /// everything after it are dropped.
    Stop(String),
}

pub struct StreamDecoder {
    stops: Vec<String>,
    ids: Vec<u32>,
    /// Byte length of the text already released.
    emitted: usize,
    /// The last this-many tokens are never released before `finish` (phrase
    /// bans may still roll them back, `engine::bans`).
    hold_tokens: usize,
    max_lines: Option<NonZeroUsize>,
}

impl StreamDecoder {
    pub fn new(stops: Vec<String>) -> Self {
        Self {
            stops: stops
                .into_iter()
                .filter(|s| !s.is_empty())
                .map(|s| s.to_ascii_lowercase())
                .collect(),
            ids: Vec::new(),
            emitted: 0,
            hold_tokens: 0,
            max_lines: None,
        }
    }

    /// Stop at the line break ending the `n`-th non-blank line.
    pub fn max_lines(mut self, n: Option<NonZeroUsize>) -> Self {
        self.max_lines = n;
        self
    }

    /// Never release the last `n` tokens before `finish` (or a stop string).
    pub fn hold_tokens(mut self, n: usize) -> Self {
        self.hold_tokens = n;
        self
    }

    /// Byte length of the text released so far.
    pub fn emitted(&self) -> usize {
        self.emitted
    }

    /// Drops the tokens from `n` on. The caller checks that none of their text
    /// was released (`decode(&ids[..n]).len() >= emitted()`).
    pub fn truncate(&mut self, n: usize) {
        self.ids.truncate(n);
    }

    /// Adds one token and returns what can be released. `decode` turns the full id
    /// sequence into text.
    pub fn push<E>(
        &mut self,
        id: u32,
        decode: impl Fn(&[u32]) -> Result<String, E>,
    ) -> Result<Step, E> {
        self.ids.push(id);
        let text = decode(&self.ids)?;
        let held = match self.hold_tokens {
            0 => None,
            n => Some(decode(&self.ids[..self.ids.len().saturating_sub(n)])?.len()),
        };
        Ok(self.advance(&text, held))
    }

    /// Releases whatever is still held back (on EOS, the token limit, timeout or
    /// cancel). Never call after `Step::Stop`.
    pub fn finish<E>(&mut self, decode: impl Fn(&[u32]) -> Result<String, E>) -> Result<String, E> {
        let text = decode(&self.ids)?;
        let rest = text.get(self.emitted..).unwrap_or_default().to_string();
        self.emitted = text.len();
        Ok(rest)
    }

    pub fn ids(&self) -> &[u32] {
        &self.ids
    }

    /// `held`: release nothing past this byte (the text before the held-back
    /// tokens), except up to a stop string.
    fn advance(&mut self, text: &str, held: Option<usize>) -> Step {
        // Already-released text must still be a prefix; if a decoder ever revises it,
        // release nothing until it settles rather than emit something inconsistent.
        if text.len() < self.emitted || !text.is_char_boundary(self.emitted) {
            return Step::Text(String::new());
        }

        // A stop string can start inside the held-back tail, so search from the
        // earliest position it could start at.
        let max_stop = self.stops.iter().map(String::len).max().unwrap_or(0);
        let mut search_from = self.emitted.saturating_sub(max_stop.saturating_sub(1));
        while !text.is_char_boundary(search_from) {
            search_from -= 1;
        }
        let folded = text.to_ascii_lowercase();
        let lead = text.len() - text.trim_start().len();
        let first_stop = self
            .stops
            .iter()
            .filter_map(|s| {
                folded[search_from..]
                    .match_indices(s.as_str())
                    .map(|(i, _)| i)
                    .find(|&i| search_from + i + s.len() > lead)
            })
            .map(|rel| search_from + rel)
            .chain(self.max_lines.and_then(|n| line_end(text, lead, n.get())))
            .min();
        if let Some(at) = first_stop {
            let end = at.max(self.emitted);
            let out = text[self.emitted..end].to_string();
            self.emitted = end;
            return Step::Stop(out);
        }

        // Hold back an incomplete UTF-8 sequence (decoded as U+FFFD) ...
        let mut safe = text.len();
        let trimmed = text.trim_end_matches('\u{FFFD}');
        if trimmed.len() < text.len() {
            safe = trimmed.len();
        }
        // ... and the longest tail that could still become a stop string.
        let hold = self
            .stops
            .iter()
            .map(|s| longest_suffix_prefix(&folded[..safe], s))
            .max()
            .unwrap_or(0);
        safe -= hold;
        if let Some(h) = held {
            safe = safe.min(h);
            while !text.is_char_boundary(safe) {
                safe -= 1;
            }
        }

        if safe <= self.emitted {
            return Step::Text(String::new());
        }
        let out = text[self.emitted..safe].to_string();
        self.emitted = safe;
        Step::Text(out)
    }
}

/// Where the line break ending the `n`-th line with something on it is, after
/// the leading whitespace `lead`.
fn line_end(text: &str, lead: usize, n: usize) -> Option<usize> {
    let (mut ended, mut filled) = (0, false);
    for (i, c) in text[lead..].char_indices() {
        if c == '\n' {
            if filled {
                ended += 1;
                if ended == n {
                    return Some(lead + i);
                }
            }
            filled = false;
        } else if !c.is_whitespace() {
            filled = true;
        }
    }
    None
}

/// Length of the longest proper prefix of `stop` that `text` ends with, on char
/// boundaries.
fn longest_suffix_prefix(text: &str, stop: &str) -> usize {
    stop.char_indices()
        .map(|(i, _)| i)
        .skip(1)
        .filter(|&len| text.ends_with(&stop[..len]))
        .max()
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::convert::Infallible;

    /// Decodes ids as indices into a list of text pieces.
    fn run(pieces: &[&str], stops: &[&str]) -> (Vec<Step>, String) {
        let mut d = StreamDecoder::new(stops.iter().map(|s| s.to_string()).collect());
        let decode = |ids: &[u32]| {
            Ok::<_, Infallible>(ids.iter().map(|&i| pieces[i as usize]).collect::<String>())
        };
        let mut steps = Vec::new();
        for i in 0..pieces.len() as u32 {
            let s = d.push(i, decode).unwrap();
            let stop = matches!(s, Step::Stop(_));
            steps.push(s);
            if stop {
                return (steps, String::new());
            }
        }
        let rest = d.finish(decode).unwrap();
        (steps, rest)
    }

    fn released(steps: &[Step]) -> String {
        steps
            .iter()
            .map(|s| match s {
                Step::Text(t) | Step::Stop(t) => t.as_str(),
            })
            .collect()
    }

    #[test]
    fn holds_back_partial_stop_string_until_it_completes() {
        let (steps, _) = run(
            &["Hi there", "\n", "Ali", "ce", ":", " more"],
            &["\nAlice:"],
        );
        // "\n", "\nAli", "\nAlice" are all prefixes of the stop string: nothing of them is shown.
        assert_eq!(steps[0], Step::Text("Hi there".into()));
        assert_eq!(steps[1], Step::Text(String::new()));
        assert_eq!(steps[2], Step::Text(String::new()));
        assert_eq!(steps[3], Step::Text(String::new()));
        assert_eq!(steps[4], Step::Stop(String::new()));
        assert_eq!(released(&steps), "Hi there");
    }

    #[test]
    fn releases_held_text_when_it_diverges_from_the_stop_string() {
        let (steps, rest) = run(&["Hi", "\n", "Alfred", " says"], &["\nAlice:"]);
        assert_eq!(released(&steps) + &rest, "Hi\nAlfred says");
        assert_eq!(steps[1], Step::Text(String::new()));
    }

    #[test]
    fn stop_strings_ignore_ascii_case() {
        let (steps, _) = run(&["hewwo", "\n\n", "ali", "ce", ":", " hi"], &["\nAlice:"]);
        assert_eq!(steps[4], Step::Stop(String::new()));
        assert_eq!(released(&steps), "hewwo\n");
        let (steps, _) = run(&["ok\nFLOOB: x"], &["\nfloob:"]);
        assert_eq!(steps[0], Step::Stop("ok".into()));
        // A lowercase prefix of a stop string is held back too.
        let (steps, rest) = run(&["ok", "\nal", "fred"], &["\nAlice:"]);
        assert_eq!(steps[1], Step::Text(String::new()));
        assert_eq!(released(&steps) + &rest, "ok\nalfred");
    }

    #[test]
    fn non_ascii_text_keeps_its_offsets() {
        // Folding only touches ASCII, so released text is the original, not folded.
        let (steps, _) = run(&["Ünïcode ÉÉ", "\nBOB: hi"], &["\nbob:"]);
        assert_eq!(released(&steps), "Ünïcode ÉÉ");
        // Non-ASCII letters must match exactly.
        let (steps, rest) = run(&["x\némile: y"], &["\nÉmile:"]);
        assert_eq!(released(&steps) + &rest, "x\némile: y");
    }

    #[test]
    fn finish_flushes_held_back_tail() {
        let (steps, rest) = run(&["ok", "\n###"], &["\n### "]);
        assert_eq!(released(&steps), "ok");
        assert_eq!(rest, "\n###");
    }

    #[test]
    fn utf8_pieces_resolve_into_the_full_character() {
        // Real decoders revise the tail: "a\u{FFFD}" becomes "a😀" once all bytes are in.
        let mut d = StreamDecoder::new(vec![]);
        let texts = ["a", "a\u{FFFD}", "a\u{FFFD}", "a😀", "a😀b"];
        let mut out = String::new();
        for (i, t) in texts.iter().enumerate() {
            let decode = |_: &[u32]| Ok::<_, Infallible>(t.to_string());
            if let Step::Text(s) = d.push(i as u32, decode).unwrap() {
                out += &s;
            }
        }
        assert_eq!(out, "a😀b");
    }

    #[test]
    fn stop_string_spanning_already_released_text_cuts_at_release_point() {
        // "x" is held back as a possible start of "x\n"; when the stop completes, nothing of it shows.
        let (steps, _) = run(&["abc x", "\nrest"], &["x\n"]);
        assert_eq!(released(&steps), "abc ");
    }

    #[test]
    fn leading_whitespace_does_not_trigger_a_newline_stop() {
        let (steps, _) = run(&["\n", " \nhi", " there", "\nmore"], &["\n"]);
        assert_eq!(released(&steps), "\n \nhi there");
        assert!(matches!(steps.last(), Some(Step::Stop(_))));
        // A longer stop at the very start still counts.
        let (steps, _) = run(&["\nAlice:", " hi"], &["\n", "\nalice:"]);
        assert_eq!(released(&steps), "");
        assert!(matches!(steps.last(), Some(Step::Stop(_))));
    }

    fn run_lines(pieces: &[&str], n: usize) -> (Vec<Step>, String) {
        let mut d = StreamDecoder::new(vec![]).max_lines(NonZeroUsize::new(n));
        let decode = |ids: &[u32]| Ok::<_, Infallible>(ids.iter().map(|&i| pieces[i as usize]).collect::<String>());
        let mut steps = Vec::new();
        for i in 0..pieces.len() as u32 {
            let s = d.push(i, decode).unwrap();
            let stop = matches!(s, Step::Stop(_));
            steps.push(s);
            if stop {
                return (steps, String::new());
            }
        }
        let rest = d.finish(decode).unwrap();
        (steps, rest)
    }

    #[test]
    fn a_line_limit_stops_at_the_break_after_the_last_line() {
        // One line: like a "\n" stop, leading whitespace aside.
        let (steps, _) = run_lines(&["\n", " \nhi", " there", "\nmore"], 1);
        assert_eq!(released(&steps), "\n \nhi there");
        assert!(matches!(steps.last(), Some(Step::Stop(_))));
        // Two: the first break is kept; blank lines between don't count.
        let (steps, _) = run_lines(&["one", "\n\n", "  \n", "two", "\nthree"], 2);
        assert_eq!(released(&steps), "one\n\n  \ntwo");
        assert!(matches!(steps.last(), Some(Step::Stop(_))));
        // Fewer lines than allowed: the reply runs to its end.
        let (steps, rest) = run_lines(&["one\ntwo"], 3);
        assert_eq!(released(&steps) + &rest, "one\ntwo");
        // The break may come mid-token.
        let (steps, _) = run_lines(&["a\nb\nc"], 2);
        assert_eq!(steps, vec![Step::Stop("a\nb".into())]);
    }

    #[test]
    fn held_tokens_wait_for_finish_and_truncate_drops_them() {
        let pieces = ["a", " b", " c", " d", " e"];
        let decode = |ids: &[u32]| Ok::<_, Infallible>(ids.iter().map(|&i| pieces[i as usize]).collect::<String>());
        let mut d = StreamDecoder::new(vec![]).hold_tokens(2);
        let mut out = String::new();
        for i in 0..4u32 {
            if let Step::Text(t) = d.push(i, decode).unwrap() {
                out += &t;
            }
        }
        // Four pushed, two held.
        assert_eq!(out, "a b");
        assert_eq!(d.emitted(), 3);
        // Roll back the held two and push another.
        d.truncate(2);
        if let Step::Text(t) = d.push(4, decode).unwrap() {
            out += &t;
        }
        out += &d.finish(decode).unwrap();
        assert_eq!(out, "a b e");
    }

}
