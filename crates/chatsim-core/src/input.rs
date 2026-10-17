//! The single-line message editor with history recall.

/// Text being typed, a cursor (in chars), and previously sent lines.
#[derive(Debug, Default)]
pub struct InputLine {
    text: String,
    cursor: usize,
    history: Vec<String>,
    /// Index into `history` while browsing it with ↑/↓.
    browsing: Option<usize>,
    /// What was typed before browsing started, restored by ↓ past the newest.
    draft: String,
}

impl InputLine {
    pub fn text(&self) -> &str {
        &self.text
    }

    /// Cursor position in chars.
    pub fn cursor(&self) -> usize {
        self.cursor
    }

    /// The text left of the cursor.
    pub fn before_cursor(&self) -> &str {
        &self.text[..self.byte(self.cursor)]
    }

    pub fn is_empty(&self) -> bool {
        self.text.is_empty()
    }

    fn len(&self) -> usize {
        self.text.chars().count()
    }

    fn byte(&self, chars: usize) -> usize {
        self.text
            .char_indices()
            .nth(chars)
            .map_or(self.text.len(), |(i, _)| i)
    }

    pub fn insert(&mut self, c: char) {
        // A pasted newline would otherwise split one message into two.
        let c = if c == '\n' || c == '\r' { ' ' } else { c };
        let at = self.byte(self.cursor);
        self.text.insert(at, c);
        self.cursor += 1;
        self.browsing = None;
    }

    pub fn backspace(&mut self) {
        if self.cursor > 0 {
            self.cursor -= 1;
            let at = self.byte(self.cursor);
            self.text.remove(at);
        }
    }

    pub fn delete(&mut self) {
        if self.cursor < self.len() {
            let at = self.byte(self.cursor);
            self.text.remove(at);
        }
    }

    /// Deletes the word before the cursor and any spaces after it (Ctrl-W).
    pub fn delete_word(&mut self) {
        let chars: Vec<char> = self.text.chars().collect();
        let mut start = self.cursor;
        while start > 0 && chars[start - 1].is_whitespace() {
            start -= 1;
        }
        while start > 0 && !chars[start - 1].is_whitespace() {
            start -= 1;
        }
        let (a, b) = (self.byte(start), self.byte(self.cursor));
        self.text.replace_range(a..b, "");
        self.cursor = start;
    }

    /// Ctrl-U.
    pub fn clear(&mut self) {
        self.text.clear();
        self.cursor = 0;
    }

    pub fn left(&mut self) {
        self.cursor = self.cursor.saturating_sub(1);
    }

    pub fn right(&mut self) {
        self.cursor = (self.cursor + 1).min(self.len());
    }

    pub fn home(&mut self) {
        self.cursor = 0;
    }

    pub fn end(&mut self) {
        self.cursor = self.len();
    }

    /// ↑: the previous sent line (the draft is kept for ↓).
    pub fn history_prev(&mut self) {
        let i = match self.browsing {
            None if self.history.is_empty() => return,
            None => {
                self.draft = std::mem::take(&mut self.text);
                self.history.len() - 1
            }
            Some(i) => i.saturating_sub(1),
        };
        self.show_history(i);
    }

    /// ↓: the next sent line, then back to the draft.
    pub fn history_next(&mut self) {
        let Some(i) = self.browsing else { return };
        if i + 1 < self.history.len() {
            self.show_history(i + 1);
        } else {
            self.browsing = None;
            self.text = std::mem::take(&mut self.draft);
            self.end();
        }
    }

    fn show_history(&mut self, i: usize) {
        self.browsing = Some(i);
        self.text = self.history[i].clone();
        self.end();
    }

    /// Takes the line for sending and records it in the history (no blank lines,
    /// no immediate repeats).
    pub fn take(&mut self) -> String {
        let line = std::mem::take(&mut self.text);
        self.cursor = 0;
        self.browsing = None;
        self.draft.clear();
        if !line.trim().is_empty() && self.history.last() != Some(&line) {
            self.history.push(line.clone());
        }
        line
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn typed(s: &str) -> InputLine {
        let mut l = InputLine::default();
        s.chars().for_each(|c| l.insert(c));
        l
    }

    #[test]
    fn editing_handles_multibyte_text() {
        let mut l = typed("héllo 🌍");
        assert_eq!(l.cursor(), 7);
        l.left();
        l.left();
        l.insert('!');
        assert_eq!(l.text(), "héllo! 🌍");
        assert_eq!(l.before_cursor(), "héllo!");
        l.backspace();
        l.delete();
        assert_eq!(l.text(), "héllo🌍");
        l.home();
        l.delete();
        assert_eq!(l.text(), "éllo🌍");
        l.end();
        l.right();
        assert_eq!(l.cursor(), 5);
        l.backspace();
        assert_eq!(l.text(), "éllo");
    }

    #[test]
    fn boundaries_are_safe() {
        let mut l = InputLine::default();
        l.backspace();
        l.delete();
        l.left();
        l.right();
        l.delete_word();
        assert_eq!((l.text(), l.cursor()), ("", 0));
    }

    #[test]
    fn newlines_become_spaces() {
        assert_eq!(typed("a\nb\r").text(), "a b ");
    }

    #[test]
    fn delete_word_and_clear() {
        let mut l = typed("one two  three  ");
        l.delete_word();
        assert_eq!(l.text(), "one two  ");
        l.left();
        l.left();
        l.delete_word();
        assert_eq!(l.text(), "one   ");
        assert_eq!(l.cursor(), 4);
        l.clear();
        assert!(l.is_empty());
    }

    #[test]
    fn history_recall_keeps_the_draft() {
        let mut l = typed("first");
        assert_eq!(l.take(), "first");
        "second".chars().for_each(|c| l.insert(c));
        l.take();
        "second".chars().for_each(|c| l.insert(c));
        l.take(); // repeat not recorded
        "   ".chars().for_each(|c| l.insert(c));
        l.take(); // blank not recorded
        "draft".chars().for_each(|c| l.insert(c));
        l.history_prev();
        assert_eq!(l.text(), "second");
        assert_eq!(l.cursor(), 6);
        l.history_prev();
        l.history_prev();
        assert_eq!(l.text(), "first");
        l.history_next();
        assert_eq!(l.text(), "second");
        l.history_next();
        assert_eq!(l.text(), "draft");
        l.history_next();
        assert_eq!(l.text(), "draft");
    }

    #[test]
    fn typing_while_browsing_edits_a_copy() {
        let mut l = typed("hello");
        l.take();
        l.history_prev();
        l.insert('!');
        assert_eq!(l.take(), "hello!");
        l.history_prev();
        assert_eq!(l.text(), "hello!");
        l.history_prev();
        assert_eq!(l.text(), "hello");
    }
}
