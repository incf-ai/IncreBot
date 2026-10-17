//! Drawing the room: a header, the transcript, a status line and the input box.

use std::time::Instant;

use chatsim_core::{Activity, App, Backend, ChatMessage, Kind};
use ratatui::Frame;
use ratatui::layout::{Constraint, Layout, Position, Rect};
use ratatui::style::{Color, Modifier, Style, Stylize};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Paragraph};
use unicode_width::UnicodeWidthStr;

use crate::wrap::{scroll_input, wrap};

const SPINNER: [&str; 10] = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];
const NAME_COLORS: [Color; 6] = [
    Color::Cyan,
    Color::Green,
    Color::Yellow,
    Color::Blue,
    Color::LightRed,
    Color::LightCyan,
];
const BOT_COLOR: Color = Color::Magenta;
/// Indent of wrapped continuation lines.
const INDENT: usize = 2;

/// Draws everything and returns the largest useful scroll offset, which the
/// caller feeds back into [`App::clamp_scroll`].
pub fn draw<B: Backend>(f: &mut Frame, app: &App<B>) -> usize {
    let [header, body, status, input] = Layout::vertical([
        Constraint::Length(1),
        Constraint::Min(1),
        Constraint::Length(1),
        Constraint::Length(3),
    ])
    .areas(f.area());
    draw_header(f, header, app);
    let max_scroll = draw_transcript(f, body, app);
    draw_status(f, status, app, max_scroll);
    draw_input(f, input, app);
    max_scroll
}

fn draw_header<B: Backend>(f: &mut Frame, area: Rect, app: &App<B>) {
    let room = &app.room;
    let line = Line::from(vec![
        Span::raw(format!(" #{} ", room.channel_name())).bold(),
        Span::raw(format!(
            "· {} · mode: {} · {} ",
            room.bot_name(),
            room.mode(),
            app.backend.describe()
        )),
    ]);
    f.render_widget(
        Paragraph::new(line).style(Style::new().add_modifier(Modifier::REVERSED)),
        area,
    );
}

/// A stable colour per name; the bot has its own.
pub fn name_color(name: &str, bot: &str) -> Color {
    if name == bot {
        return BOT_COLOR;
    }
    let h = name
        .bytes()
        .fold(5381usize, |h, b| h.wrapping_mul(33) ^ b as usize);
    NAME_COLORS[h % NAME_COLORS.len()]
}

/// The transcript as display lines for a pane `width` columns wide.
pub fn transcript_lines<B: Backend>(app: &App<B>, width: usize) -> Vec<Line<'static>> {
    let room = &app.room;
    let mut out = Vec::new();
    for m in room.messages() {
        message_lines(m, room.bot_name(), room.in_history(m), width, &mut out);
    }
    out
}

fn message_lines(
    m: &ChatMessage,
    bot: &str,
    in_history: bool,
    width: usize,
    out: &mut Vec<Line<'static>>,
) {
    let indent = " ".repeat(INDENT);
    if m.kind == Kind::Notice {
        let style = Style::new()
            .fg(Color::DarkGray)
            .add_modifier(Modifier::ITALIC);
        for (i, l) in wrap(
            &m.text,
            width.saturating_sub(INDENT),
            width.saturating_sub(INDENT),
        )
        .into_iter()
        .enumerate()
        {
            let lead = if i == 0 { "· " } else { "  " };
            out.push(Line::styled(format!("{lead}{l}"), style));
        }
        return;
    }
    // Messages the bot no longer sees (or never saw: `/ask`) are dimmed.
    let dim = if in_history || m.streaming {
        Style::new()
    } else {
        Style::new().add_modifier(Modifier::DIM)
    };
    let mut prefix = vec![Span::styled(
        m.author.clone(),
        dim.fg(name_color(&m.author, bot))
            .add_modifier(Modifier::BOLD),
    )];
    if let Some(to) = &m.reply_to {
        prefix.push(Span::styled(format!(" → {to}"), dim.fg(Color::DarkGray)));
    }
    prefix.push(Span::styled(": ", dim));
    let prefix_width: usize = prefix.iter().map(|s| s.content.width()).sum();
    let text = match (m.streaming, m.text.is_empty()) {
        (true, true) => "…".to_string(),
        (true, false) => format!("{} ▌", m.text),
        _ => m.text.clone(),
    };
    let first = width.saturating_sub(prefix_width).max(1);
    for (i, l) in wrap(&text, first, width.saturating_sub(INDENT))
        .into_iter()
        .enumerate()
    {
        let body = Span::styled(l, dim);
        out.push(if i == 0 {
            let mut spans = prefix.clone();
            spans.push(body);
            Line::from(spans)
        } else {
            Line::from(vec![Span::raw(indent.clone()), body])
        });
    }
    if let Some(note) = m.note {
        out.push(Line::styled(
            format!("{indent}({note})"),
            Style::new()
                .fg(Color::DarkGray)
                .add_modifier(Modifier::ITALIC),
        ));
    }
}

fn draw_transcript<B: Backend>(f: &mut Frame, area: Rect, app: &App<B>) -> usize {
    let lines = transcript_lines(app, area.width as usize);
    let height = area.height as usize;
    let max_scroll = lines.len().saturating_sub(height);
    let scroll = app.scroll().min(max_scroll);
    let end = lines.len() - scroll;
    let start = end.saturating_sub(height);
    // Bottom-aligned, like a chat: pad above when the transcript is short.
    let pad = height.saturating_sub(end - start);
    let mut shown = vec![Line::default(); pad];
    shown.extend(lines[start..end].iter().cloned());
    f.render_widget(Paragraph::new(shown), area);
    max_scroll
}

fn spinner(since: Instant) -> &'static str {
    SPINNER[(since.elapsed().as_millis() / 100) as usize % SPINNER.len()]
}

/// The status line's text, without styling.
pub fn status_text<B: Backend>(app: &App<B>, max_scroll: usize) -> (String, Color) {
    let room = &app.room;
    let bot = room.bot_name();
    let (mut text, color) = match room.activity() {
        Activity::Idle => ("● ready".to_string(), Color::Green),
        Activity::Thinking { since } => (
            format!(
                "{} {bot} is thinking… {}s · Esc to stop",
                spinner(since),
                since.elapsed().as_secs()
            ),
            Color::Yellow,
        ),
        Activity::Writing { since } => (
            format!(
                "{} {bot} is typing… {}s · Esc to stop",
                spinner(since),
                since.elapsed().as_secs()
            ),
            Color::Yellow,
        ),
    };
    if room.queued() > 0 {
        text.push_str(&format!(" · {} queued", room.queued()));
    }
    let scroll = app.scroll().min(max_scroll);
    if scroll > 0 {
        text.push_str(&format!(" · ↑ scrolled {scroll} lines (PgDn / Ctrl-End)"));
    }
    (text, color)
}

fn draw_status<B: Backend>(f: &mut Frame, area: Rect, app: &App<B>, max_scroll: usize) {
    let (text, color) = status_text(app, max_scroll);
    f.render_widget(Paragraph::new(format!(" {text}")).fg(color), area);
}

fn draw_input<B: Backend>(f: &mut Frame, area: Rect, app: &App<B>) {
    let block = Block::bordered()
        .title(format!(" {} ", app.room.nick()))
        .border_style(Style::new().fg(name_color(app.room.nick(), app.room.bot_name())));
    let inner = block.inner(area);
    f.render_widget(block, area);
    if app.input.is_empty() {
        let hint = format!(
            "Message #{} as {} · /help for commands",
            app.room.channel_name(),
            app.room.nick()
        );
        f.render_widget(Paragraph::new(hint).fg(Color::DarkGray), inner);
        f.set_cursor_position(Position::new(inner.x, inner.y));
        return;
    }
    let (shown, x) = scroll_input(
        app.input.text(),
        app.input.before_cursor(),
        inner.width as usize,
    );
    f.render_widget(Paragraph::new(shown), inner);
    f.set_cursor_position(Position::new(inner.x + x as u16, inner.y));
}

#[cfg(test)]
mod tests {
    use super::*;
    use chatsim_core::{Action, Pacing, Room, RoomConfig, ScriptedBackend};
    use ratatui::Terminal;
    use ratatui::backend::TestBackend;

    fn app() -> App<ScriptedBackend> {
        let b = ScriptedBackend::new(Pacing::PerPoll(1), |r| {
            format!(" hi {} how are you", r.trigger_author)
        });
        App::new(Room::new(RoomConfig::default()).unwrap(), b).unwrap()
    }

    fn send(a: &mut App<ScriptedBackend>, s: &str) {
        s.chars().for_each(|c| a.handle(Action::Insert(c)));
        a.handle(Action::Submit);
    }

    fn render(a: &App<ScriptedBackend>, w: u16, h: u16) -> (Vec<String>, Position) {
        let mut t = Terminal::new(TestBackend::new(w, h)).unwrap();
        t.draw(|f| {
            draw(f, a);
        })
        .unwrap();
        let buf = t.backend().buffer().clone();
        let rows = (0..h)
            .map(|y| {
                (0..w)
                    .map(|x| buf[(x, y)].symbol().to_string())
                    .collect::<String>()
            })
            .collect();
        (rows, t.get_cursor_position().unwrap())
    }

    fn plain(lines: &[Line]) -> Vec<String> {
        lines
            .iter()
            .map(|l| l.spans.iter().map(|s| s.content.as_ref()).collect())
            .collect()
    }

    #[test]
    fn layout_has_header_transcript_status_and_input() {
        let (rows, cursor) = render(&app(), 60, 12);
        assert!(
            rows[0].contains("#general") && rows[0].contains("mode: every"),
            "{}",
            rows[0]
        );
        assert!(
            rows[1..8].iter().any(|r| r.contains("Welcome to #general")),
            "{rows:#?}"
        );
        assert!(rows[7].contains("/help"), "bottom-aligned: {rows:#?}");
        assert!(rows[8].contains("● ready"), "{}", rows[8]);
        assert!(rows[9].contains("Alice"), "input title: {}", rows[9]);
        assert!(rows[10].contains("Message #general as Alice"));
        assert_eq!(cursor, Position::new(1, 10));
    }

    #[test]
    fn messages_wrap_with_author_prefix_and_indent() {
        let mut a = app();
        send(&mut a, "/mode mention");
        send(&mut a, "one two three four five six");
        let lines = plain(&transcript_lines(&a, 20));
        let n = lines.len();
        assert_eq!(lines[n - 2..], ["Alice: one two three", "  four five six"]);
    }

    #[test]
    fn streaming_reply_shows_progress_then_final_text() {
        let mut a = app();
        send(&mut a, "hello");
        let lines = plain(&transcript_lines(&a, 60));
        assert_eq!(lines.last().unwrap(), "floob → Alice: …");
        a.pump();
        let lines = plain(&transcript_lines(&a, 60));
        assert_eq!(lines.last().unwrap(), "floob → Alice: hi ▌");
        assert!(status_text(&a, 0).0.contains("is typing"));
        while a.pump() {}
        let lines = plain(&transcript_lines(&a, 60));
        assert_eq!(lines.last().unwrap(), "floob → Alice: hi Alice how are you");
    }

    #[test]
    fn stopped_replies_show_a_note() {
        let mut a = app();
        send(&mut a, "hello");
        a.pump();
        a.handle(Action::Cancel);
        while a.pump() {}
        let lines = plain(&transcript_lines(&a, 60));
        assert_eq!(
            lines[lines.len() - 2..],
            ["floob → Alice: hi", "  (stopped)"]
        );
    }

    #[test]
    fn messages_outside_the_history_are_dimmed() {
        let mut a = app();
        send(&mut a, "/mode mention");
        send(&mut a, "remember this");
        let styled = transcript_lines(&a, 60);
        let last = styled.last().unwrap();
        assert!(!last.spans[1].style.add_modifier.contains(Modifier::DIM));
        send(&mut a, "/reset");
        let styled = transcript_lines(&a, 60);
        let old = styled
            .iter()
            .find(|l| l.spans.iter().any(|s| s.content == "remember this"))
            .unwrap();
        assert!(
            old.spans
                .iter()
                .all(|s| s.style.add_modifier.contains(Modifier::DIM))
        );
    }

    #[test]
    fn transcript_scrolls_and_reports_its_range() {
        let mut a = app();
        send(&mut a, "/mode mention");
        for i in 0..30 {
            send(&mut a, &format!("line {i}"));
        }
        let (rows, _) = render(&a, 40, 12);
        assert!(rows[7].contains("line 29"), "{rows:#?}");
        a.handle(Action::ScrollUp(5));
        let mut t = Terminal::new(TestBackend::new(40, 12)).unwrap();
        let mut max = 0;
        t.draw(|f| max = draw(f, &a)).unwrap();
        let total = transcript_lines(&a, 40).len();
        assert_eq!(max, total - 7, "{total} transcript lines in a 7-line pane");
        let (rows, _) = render(&a, 40, 12);
        assert!(rows[7].contains("line 24"), "{rows:#?}");
        assert!(rows[8].contains("scrolled 5 lines"), "{}", rows[8]);
    }

    #[test]
    fn input_box_shows_text_and_cursor() {
        let mut a = app();
        "héllo".chars().for_each(|c| a.handle(Action::Insert(c)));
        a.handle(Action::Left);
        let (rows, cursor) = render(&a, 30, 8);
        assert!(rows[6].contains("héllo"));
        assert_eq!(cursor, Position::new(1 + 4, 6));
        "x".repeat(60)
            .chars()
            .for_each(|c| a.handle(Action::Insert(c)));
        a.handle(Action::End);
        let (rows, cursor) = render(&a, 30, 8);
        assert!(rows[6].contains("xxo"), "{}", rows[6]);
        assert!(cursor.x < 29);
    }

    #[test]
    fn status_shows_thinking_and_queue() {
        let mut a = app();
        send(&mut a, "first");
        let (s, color) = status_text(&a, 0);
        assert!(s.contains("floob is thinking") && s.contains("Esc"), "{s}");
        assert_eq!(color, Color::Yellow);
        send(&mut a, "/ask question");
        assert!(status_text(&a, 0).0.contains("1 queued"));
    }

    #[test]
    fn names_get_stable_distinct_colours() {
        assert_eq!(name_color("IncreBot", "IncreBot"), BOT_COLOR);
        assert_eq!(
            name_color("Alice", "IncreBot"),
            name_color("Alice", "IncreBot")
        );
        assert!(NAME_COLORS.contains(&name_color("Bob", "IncreBot")));
    }

    #[test]
    fn tiny_terminals_do_not_panic() {
        let mut a = app();
        send(&mut a, "hello there");
        a.pump();
        for (w, h) in [(1, 1), (5, 3), (10, 6), (3, 20)] {
            render(&a, w, h);
        }
    }
}
