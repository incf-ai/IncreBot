//! Splitting replies to fit Discord's 2,000-character limit (design.md §6). One
//! splitter serves both streaming and non-streamed replies, so a reply is split
//! identically either way.

/// Characters per message we allow ourselves (Discord's hard limit is 2,000).
pub const MESSAGE_LIMIT: usize = 1900;

/// Where to cut `text` so the first part is at most `limit` characters: the last
/// paragraph break, else the last sentence end, else the last whitespace, else a
/// hard cut. Returns a byte index on a char boundary, or `None` if it already fits.
pub fn split_point(text: &str, limit: usize) -> Option<usize> {
    if text.chars().count() <= limit {
        return None;
    }
    let max = text
        .char_indices()
        .nth(limit)
        .map_or(text.len(), |(i, _)| i);
    let head = &text[..max];
    let min = max / 4; // never make tiny parts
    if let Some(i) = head.rfind("\n\n").filter(|&i| i > min) {
        return Some(i + 2);
    }
    let sentence = [". ", "! ", "? ", ".\n", "!\n", "?\n"]
        .iter()
        .filter_map(|p| head.rfind(p).map(|i| i + p.len()))
        .max()
        .filter(|&i| i > min);
    if let Some(i) = sentence {
        return Some(i);
    }
    if let Some(i) = head.rfind(char::is_whitespace).filter(|&i| i > min) {
        return Some(i + head[i..].chars().next().map_or(1, char::len_utf8));
    }
    Some(max)
}

/// Splits a whole reply into message-sized parts.
pub fn split_reply(text: &str, limit: usize) -> Vec<String> {
    let mut parts = Vec::new();
    let mut rest = text;
    while let Some(at) = split_point(rest, limit) {
        parts.push(rest[..at].trim_end().to_string());
        rest = rest[at..].trim_start();
    }
    if !rest.trim().is_empty() || parts.is_empty() {
        parts.push(rest.trim_end().to_string());
    }
    parts
}

/// A reply being streamed into a sequence of messages.
#[derive(Debug, Default)]
pub struct StreamingReply {
    /// Messages already finalised (content fixed).
    pub done: Vec<String>,
    /// The message currently being edited.
    pub current: String,
}

impl StreamingReply {
    /// Appends text. Returns the parts that just became final, in order; the caller
    /// finalises the current message with the first one and posts new messages
    /// for the rest.
    pub fn push(&mut self, text: &str, limit: usize) -> Vec<String> {
        self.current.push_str(text);
        let mut finished = Vec::new();
        while let Some(at) = split_point(&self.current, limit) {
            let part = self.current[..at].trim_end().to_string();
            self.current = self.current[at..].trim_start().to_string();
            finished.push(part.clone());
            self.done.push(part);
        }
        finished
    }

    /// Every part, as `split_reply` would produce for the full text.
    pub fn parts(&self) -> Vec<String> {
        let mut v = self.done.clone();
        if !self.current.trim().is_empty() || v.is_empty() {
            v.push(self.current.trim_end().to_string());
        }
        v
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prefers_paragraph_then_sentence_then_word() {
        assert_eq!(
            split_reply("aaaa bbbb.\n\ncccc", 12),
            vec!["aaaa bbbb.", "cccc"]
        );
        assert_eq!(
            split_reply("aaaa. bbbb cccc", 12),
            vec!["aaaa.", "bbbb cccc"]
        );
        assert_eq!(split_reply("aaaa bbbb cccc", 12), vec!["aaaa bbbb", "cccc"]);
        assert_eq!(
            split_reply("abcdefghijklmnop", 10),
            vec!["abcdefghij", "klmnop"]
        );
        assert_eq!(split_reply("short", 10), vec!["short"]);
    }

    #[test]
    fn respects_char_limit_with_multibyte_text() {
        let text = "日本語のテキスト。".repeat(300);
        for part in split_reply(&text, MESSAGE_LIMIT) {
            assert!(part.chars().count() <= MESSAGE_LIMIT);
        }
    }

    #[test]
    fn streaming_splits_like_whole_reply() {
        let text = "Lorem ipsum dolor sit amet. ".repeat(200) + "\n\nThe end.";
        for chunk in [1usize, 7, 50, 333] {
            let mut s = StreamingReply::default();
            let chars: Vec<char> = text.chars().collect();
            for piece in chars.chunks(chunk) {
                s.push(&piece.iter().collect::<String>(), MESSAGE_LIMIT);
            }
            assert_eq!(
                s.parts(),
                split_reply(&text, MESSAGE_LIMIT),
                "chunk size {chunk}"
            );
        }
    }
}
