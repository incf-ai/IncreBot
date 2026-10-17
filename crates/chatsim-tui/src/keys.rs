//! Key bindings.

use chatsim_core::Action;
use ratatui::crossterm::event::{KeyCode, KeyEvent, KeyEventKind, KeyModifiers};

/// Lines per PgUp/PgDn.
pub const PAGE: usize = 10;

/// The action for a key press, if it has one. Releases and repeats of
/// non-character keys are ignored by the caller via [`KeyEventKind`].
pub fn action(key: KeyEvent) -> Option<Action> {
    if key.kind == KeyEventKind::Release {
        return None;
    }
    let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
    let shift = key.modifiers.contains(KeyModifiers::SHIFT);
    let alt = key.modifiers.contains(KeyModifiers::ALT);
    Some(match key.code {
        KeyCode::Char('c' | 'd') if ctrl => Action::Quit,
        KeyCode::Char('a') if ctrl => Action::Home,
        KeyCode::Char('e') if ctrl => Action::End,
        KeyCode::Char('u') if ctrl => Action::ClearLine,
        KeyCode::Char('w') if ctrl => Action::DeleteWord,
        KeyCode::Backspace if ctrl || alt => Action::DeleteWord,
        KeyCode::Char(_) if ctrl || alt => return None,
        KeyCode::Char(c) => Action::Insert(c),
        KeyCode::Enter => Action::Submit,
        KeyCode::Esc => Action::Cancel,
        KeyCode::Backspace => Action::Backspace,
        KeyCode::Delete => Action::Delete,
        KeyCode::Left => Action::Left,
        KeyCode::Right => Action::Right,
        KeyCode::Home if ctrl => Action::ScrollUp(usize::MAX),
        KeyCode::End if ctrl => Action::ScrollBottom,
        KeyCode::Home => Action::Home,
        KeyCode::End => Action::End,
        KeyCode::Up if shift => Action::ScrollUp(1),
        KeyCode::Down if shift => Action::ScrollDown(1),
        KeyCode::Up => Action::HistoryPrev,
        KeyCode::Down => Action::HistoryNext,
        KeyCode::PageUp => Action::ScrollUp(PAGE),
        KeyCode::PageDown => Action::ScrollDown(PAGE),
        KeyCode::Tab => Action::Insert(' '),
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key(code: KeyCode, mods: KeyModifiers) -> Option<Action> {
        action(KeyEvent::new(code, mods))
    }

    fn plain(code: KeyCode) -> Option<Action> {
        key(code, KeyModifiers::NONE)
    }

    #[test]
    fn typing_and_editing() {
        assert_eq!(plain(KeyCode::Char('x')), Some(Action::Insert('x')));
        assert_eq!(
            key(KeyCode::Char('X'), KeyModifiers::SHIFT),
            Some(Action::Insert('X'))
        );
        assert_eq!(plain(KeyCode::Enter), Some(Action::Submit));
        assert_eq!(plain(KeyCode::Backspace), Some(Action::Backspace));
        assert_eq!(plain(KeyCode::Delete), Some(Action::Delete));
        assert_eq!(plain(KeyCode::Home), Some(Action::Home));
        assert_eq!(
            key(KeyCode::Char('a'), KeyModifiers::CONTROL),
            Some(Action::Home)
        );
        assert_eq!(
            key(KeyCode::Char('e'), KeyModifiers::CONTROL),
            Some(Action::End)
        );
        assert_eq!(
            key(KeyCode::Char('u'), KeyModifiers::CONTROL),
            Some(Action::ClearLine)
        );
        assert_eq!(
            key(KeyCode::Char('w'), KeyModifiers::CONTROL),
            Some(Action::DeleteWord)
        );
        assert_eq!(
            key(KeyCode::Backspace, KeyModifiers::ALT),
            Some(Action::DeleteWord)
        );
        assert_eq!(plain(KeyCode::Tab), Some(Action::Insert(' ')));
    }

    #[test]
    fn control_keys() {
        assert_eq!(
            key(KeyCode::Char('c'), KeyModifiers::CONTROL),
            Some(Action::Quit)
        );
        assert_eq!(
            key(KeyCode::Char('d'), KeyModifiers::CONTROL),
            Some(Action::Quit)
        );
        assert_eq!(plain(KeyCode::Esc), Some(Action::Cancel));
        assert_eq!(key(KeyCode::Char('z'), KeyModifiers::CONTROL), None);
        assert_eq!(key(KeyCode::Char('x'), KeyModifiers::ALT), None);
        assert_eq!(plain(KeyCode::F(5)), None);
    }

    #[test]
    fn history_and_scrolling() {
        assert_eq!(plain(KeyCode::Up), Some(Action::HistoryPrev));
        assert_eq!(plain(KeyCode::Down), Some(Action::HistoryNext));
        assert_eq!(
            key(KeyCode::Up, KeyModifiers::SHIFT),
            Some(Action::ScrollUp(1))
        );
        assert_eq!(
            key(KeyCode::Down, KeyModifiers::SHIFT),
            Some(Action::ScrollDown(1))
        );
        assert_eq!(plain(KeyCode::PageUp), Some(Action::ScrollUp(PAGE)));
        assert_eq!(plain(KeyCode::PageDown), Some(Action::ScrollDown(PAGE)));
        assert_eq!(
            key(KeyCode::End, KeyModifiers::CONTROL),
            Some(Action::ScrollBottom)
        );
        assert_eq!(
            key(KeyCode::Home, KeyModifiers::CONTROL),
            Some(Action::ScrollUp(usize::MAX))
        );
    }

    #[test]
    fn releases_are_ignored() {
        let mut k = KeyEvent::new(KeyCode::Char('x'), KeyModifiers::NONE);
        k.kind = KeyEventKind::Release;
        assert_eq!(action(k), None);
    }
}
