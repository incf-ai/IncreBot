//! Slash command answers in the channel's history (design.md §6). For a
//! command the persona's `remember_commands` lists, a public answer enters its
//! channel's history window as two turns, as if said there: the requester's
//! (from the message's first line, the header: an `/ask` quote or a novelty
//! card) and the bot's reply below it. Live replies (`reply::stream`) and the
//! REST seed (`dispatch::seed_window`) both build the requester's turn from
//! the header as posted, so a cold window matches a warm one.
//!
//! The requester's turn has the interaction's ID, which sorts just before the
//! answer, and is tied to the answer's message (`ChannelWindow::push_tied`):
//! deleting that message removes it, and so does anyone the header mentions
//! opting out.

use crate::chat::window::WindowMsg;

/// Longest question an `/ask` quote shows, in characters.
pub const ASK_QUOTE_CHARS: usize = 300;

/// Each novelty card's first character, and its command.
const CARDS: &[(&str, &str)] = &[
    ("🌭", "hotdog-rater"),
    ("🦎", "kobold-name"),
    ("👅", "lick-test"),
    ("🔥", "roast"),
    ("🦴", "fortune"),
    ("💰", "hoard-value"),
    ("🕰️", "kobold-age"),
    ("⚖️", "judge"),
    ("🪤", "trap-check"),
    ("🍸", "drink"),
    ("🎲", "gamble"),
    ("🐉", "dragon-alert"),
    ("🔮", "vibe-check"),
];

/// The header above an `/ask` answer: the question, on one line, quoted.
pub fn ask_header(asker_mention: &str, question: &str) -> String {
    let q = question.split_whitespace().collect::<Vec<_>>().join(" ");
    let q = if q.chars().count() > ASK_QUOTE_CHARS {
        q.chars().take(ASK_QUOTE_CHARS).collect::<String>() + "…"
    } else {
        q
    };
    format!("> {asker_mention} asks: {q}")
}

/// The command a header is for: `ask` for a quote, else by the card's emoji.
pub fn command_of(header: &str) -> Option<&'static str> {
    if header.starts_with("> ") && header.contains(" asks: ") {
        return Some("ask");
    }
    CARDS.iter().find(|(e, _)| header.starts_with(e)).map(|&(_, c)| c)
}

/// What the requester "said", from a header whose mentions are already
/// resolved to names: an `/ask` question as is, a card as
/// `[used /roast: floob roasts @Bob]` (no emoji, no bold).
pub fn requester_text(header: &str) -> Option<String> {
    let cmd = command_of(header)?;
    if cmd == "ask" {
        let (_, q) = header.split_once(" asks: ")?;
        return Some(q.trim().to_string()).filter(|q| !q.is_empty());
    }
    let (emoji, _) = CARDS.iter().find(|&&(_, c)| c == cmd)?;
    let card = header[emoji.len()..].replace('*', "");
    let card = card.trim().trim_end_matches(':').trim();
    Some(format!("[used /{cmd}: {card}]"))
}

/// The user IDs a header mentions (`<@id>`, `<@!id>`), in order, each once.
pub fn mentioned(header: &str) -> Vec<u64> {
    let mut out = Vec::new();
    for part in header.split("<@").skip(1) {
        let part = part.strip_prefix('!').unwrap_or(part);
        let Some((id, _)) = part.split_once('>') else { continue };
        if let Ok(id) = id.parse::<u64>()
            && !out.contains(&id)
        {
            out.push(id);
        }
    }
    out
}

/// A posted answer's text without the notice lines the bot adds
/// (`*(stopped)*`, `*(error: …)*`): what the history keeps of it.
pub fn without_notices(text: &str) -> String {
    text.lines()
        .filter(|l| {
            let l = l.trim();
            !(l.starts_with(super::NOTICE_PREFIX) && l.ends_with(")*"))
        })
        .collect::<Vec<_>>()
        .join("\n")
        .trim()
        .to_string()
}

/// A command's requester turn, for the history.
#[derive(Clone, Debug)]
pub struct Asked {
    pub slash: &'static str,
    /// Its ID is the interaction's.
    pub turn: WindowMsg,
    /// Who the header mentions besides the requester: their opting out
    /// removes it.
    pub about: Vec<u64>,
}

/// The requester turn for `header` (raw, mentions as `<@id>`) once
/// `resolve` has turned its mentions into names; `None` for a header that is
/// no command's.
pub fn asked(
    header: &str,
    interaction: u64,
    requester: (u64, &str),
    resolve: &dyn Fn(&str) -> String,
) -> Option<Asked> {
    let slash = command_of(header)?;
    let text = requester_text(&resolve(header))?;
    Some(Asked {
        slash,
        turn: WindowMsg {
            id: interaction,
            author_id: requester.0,
            author: requester.1.to_string(),
            text,
            extras: Default::default(),
            mentions: Vec::new(),
        },
        about: mentioned(header).into_iter().filter(|&u| u != requester.0).collect(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::discord::novelty;

    #[test]
    fn every_card_is_recognized() {
        let k = crate::discord::seeded::Key::new(1);
        let cards = [
            (novelty::name_card("<@5>", "Snek"), "kobold-name"),
            (novelty::lick_card(&k, "floob", 7, ("<@5>", 5), ("<@6>", 6)), "lick-test"),
            (novelty::self_lick_card(&k, "floob", 7, ("<@5>", 5)), "lick-test"),
            (novelty::roast_card("floob", "<@5>"), "roast"),
            (novelty::fortune_card("floob", "<@5>", novelty::omen()), "fortune"),
            (novelty::hoard_card("floob", "<@5>", &["a", "b", "c"], 3), "hoard-value"),
            (novelty::age_card("floob", "<@5>", 80), "kobold-age"),
            (novelty::bot_age_card("floob"), "kobold-age"),
            (novelty::judge_card("floob", "soup", 4), "judge"),
            (novelty::trap_card("floob", "<@5>", novelty::trap()), "trap-check"),
            (novelty::drink_card("floob", "<@5>", novelty::drink()), "drink"),
            (novelty::Gamble::roll(5).card("floob", "<@5>"), "gamble"),
            (novelty::dragon_card(novelty::dragon_sign()), "dragon-alert"),
            (novelty::vibe_card("floob"), "vibe-check"),
            (crate::discord::hotdog::measure(&k, 5).card("<@5>"), "hotdog-rater"),
            (ask_header("<@5>", "why?"), "ask"),
        ];
        for (card, cmd) in &cards {
            assert_eq!(command_of(card), Some(*cmd), "{card}");
            assert!(!card.contains('\n'), "a header is one line: {card}");
            assert!(crate::persona::REMEMBERABLE.contains(cmd));
        }
        assert_eq!(crate::persona::REMEMBERABLE.len(), CARDS.len() + 1);
        assert_eq!(command_of("hewwo"), None);
        assert_eq!(command_of("> just a quote"), None);
    }

    #[test]
    fn the_requester_says_the_question_or_uses_the_command() {
        let resolve = |s: &str| s.replace("<@5>", "@Alice").replace("<@6>", "@Bob");
        let a = asked(&ask_header("<@5>", "  why is\nthe sky   blue? "), 100, (5, "Alice"), &resolve).unwrap();
        assert_eq!((a.slash, a.turn.text.as_str(), a.turn.id, a.turn.author_id), ("ask", "why is the sky blue?", 100, 5));
        assert!(a.about.is_empty());
        let r = asked(&novelty::roast_card("floob", "<@6>"), 100, (5, "Alice"), &resolve).unwrap();
        assert_eq!(r.turn.text, "[used /roast: floob roasts @Bob]");
        assert_eq!(r.about, [6]);
        let h = asked(&novelty::hoard_card("floob", "<@5>", &["a", "b", "c"], 30), 1, (5, "Alice"), &resolve).unwrap();
        assert_eq!(h.turn.text, "[used /hoard-value: floob appraises @Alice's hoard: a, b and c. total value: 30 gold]");
        assert!(h.about.is_empty(), "the requester is not 'about'");
        assert!(asked("hewwo", 1, (5, "Alice"), &resolve).is_none());
    }

    #[test]
    fn long_questions_are_cut() {
        let h = ask_header("<@5>", &"a".repeat(1000));
        assert!(h.ends_with('…'));
        assert_eq!(h.chars().count(), "> <@5> asks: ".len() + ASK_QUOTE_CHARS + 1);
        // Exactly at the limit: kept whole, no ellipsis.
        let h = ask_header("<@5>", &"a".repeat(ASK_QUOTE_CHARS));
        assert!(h.ends_with('a') && !h.contains('…'));
    }

    #[test]
    fn cards_with_longer_emoji_strip_their_own() {
        // 🕰️ and ⚖️ are longer than 🔥 or 🌭: the text must lose exactly its
        // own card's emoji, not the first card's worth of bytes.
        let resolve = |s: &str| s.replace("<@5>", "@Alice");
        let a = asked(&novelty::age_card("floob", "<@5>", 80), 1, (9, "Bob"), &resolve).unwrap();
        assert_eq!(a.turn.text, "[used /kobold-age: @Alice is 80 kobold years old. (floob is 471.)]");
        let j = asked(&novelty::judge_card("floob", "soup", 3), 1, (9, "Bob"), &resolve).unwrap();
        assert_eq!(j.turn.text, "[used /judge: floob judges “soup”: 3/10 shiny rocks]");
    }

    #[test]
    fn mentions_and_notices() {
        assert_eq!(mentioned("<@5> and <@!6>, <@5>, <@&7> <#8> <@x>"), [5, 6]);
        assert_eq!(without_notices("hewwo\n*(stopped)*"), "hewwo");
        assert_eq!(without_notices("*(error: generation failed)*"), "");
        assert_eq!(without_notices("*(waves)* hi"), "*(waves)* hi");
    }
}
