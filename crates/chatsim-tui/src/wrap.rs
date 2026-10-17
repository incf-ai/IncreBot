//! Word wrapping by display width. The renderer wraps itself (rather than
//! letting `Paragraph` do it) so it knows the transcript's height for scrolling.

use unicode_width::{UnicodeWidthChar, UnicodeWidthStr};

/// Wraps `text` into lines at most `first` columns wide for the first line and
/// `rest` for the others. Breaks at spaces where it can and splits words longer
/// than a line; `\n` always breaks. Always returns at least one line.
pub fn wrap(text: &str, first: usize, rest: usize) -> Vec<String> {
    let (first, rest) = (first.max(1), rest.max(1));
    let mut lines: Vec<String> = Vec::new();
    let width = |lines: &Vec<String>| if lines.is_empty() { first } else { rest };
    for para in text.split('\n') {
        let mut line = String::new();
        let mut used = 0;
        for word in para.split(' ').filter(|w| !w.is_empty()) {
            let w = word.width();
            let gap = usize::from(used > 0);
            if used + gap + w <= width(&lines) {
                if gap == 1 {
                    line.push(' ');
                }
                line.push_str(word);
                used += gap + w;
                continue;
            }
            if used > 0 {
                lines.push(std::mem::take(&mut line));
                used = 0;
            }
            if w <= width(&lines) {
                line.push_str(word);
                used = w;
                continue;
            }
            // A word wider than a line: split it by characters.
            for c in word.chars() {
                let cw = c.width().unwrap_or(0);
                if used + cw > width(&lines) && used > 0 {
                    lines.push(std::mem::take(&mut line));
                    used = 0;
                }
                line.push(c);
                used += cw;
            }
        }
        lines.push(line);
    }
    lines
}

/// The part of `text` to show in a box `width` columns wide so that the
/// cursor (after `before_cursor`) stays visible, and the cursor's column in it.
pub fn scroll_input(text: &str, before_cursor: &str, width: usize) -> (String, usize) {
    let width = width.max(1);
    let cursor = before_cursor.width();
    // Drop leading characters until the cursor fits.
    let mut skipped = 0;
    let mut start = 0;
    for (i, c) in before_cursor.char_indices() {
        if cursor - skipped < width {
            break;
        }
        skipped += c.width().unwrap_or(0);
        start = i + c.len_utf8();
    }
    let mut shown = String::new();
    let mut used = 0;
    for c in text[start..].chars() {
        let cw = c.width().unwrap_or(0);
        if used + cw > width {
            break;
        }
        shown.push(c);
        used += cw;
    }
    (shown, cursor - skipped)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fits_on_one_line() {
        assert_eq!(wrap("hello world", 20, 20), ["hello world"]);
        assert_eq!(wrap("", 10, 10), [""]);
    }

    #[test]
    fn breaks_at_spaces_with_a_narrower_first_line() {
        assert_eq!(
            wrap("the quick brown fox jumps", 9, 12),
            ["the quick", "brown fox", "jumps"]
        );
        assert_eq!(wrap("aaa bbb", 3, 10), ["aaa", "bbb"]);
    }

    #[test]
    fn newlines_always_break_and_blank_lines_survive() {
        assert_eq!(wrap("a\n\nb", 10, 10), ["a", "", "b"]);
    }

    #[test]
    fn long_words_are_split() {
        assert_eq!(wrap("abcdefghij", 4, 4), ["abcd", "efgh", "ij"]);
        assert_eq!(wrap("x abcdefgh", 4, 4), ["x", "abcd", "efgh"]);
    }

    #[test]
    fn wide_characters_count_double() {
        assert_eq!(wrap("日本語テキスト", 6, 6), ["日本語", "テキス", "ト"]);
        for line in wrap("🌍🌍🌍 ok", 5, 5) {
            assert!(line.width() <= 5, "{line}");
        }
    }

    #[test]
    fn zero_width_is_treated_as_one() {
        assert_eq!(wrap("ab", 0, 0), ["a", "b"]);
    }

    #[test]
    fn input_scrolls_to_keep_the_cursor_visible() {
        assert_eq!(scroll_input("hello", "hel", 10), ("hello".into(), 3));
        assert_eq!(
            scroll_input("abcdefghij", "abcdefghij", 5),
            ("ghij".into(), 4)
        );
        assert_eq!(scroll_input("abcdefghij", "ab", 5), ("abcde".into(), 2));
        assert_eq!(
            scroll_input("abcdefghij", "abcdefg", 5),
            ("defgh".into(), 4)
        );
        let (shown, x) = scroll_input("日本語テキスト", "日本語テキ", 6);
        assert!(shown.width() <= 6 && x < 6, "{shown} {x}");
    }
}
