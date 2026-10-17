//! Prompt rendering with the minijinja template (design.md §7).

use anyhow::{Result, anyhow, ensure};
use minijinja::{Environment, context};
use serde::Serialize;

use super::sanitize::{clean_name, lowercase_mentions, rename_aliases};

#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
pub struct Turn {
    pub author: String,
    pub content: String,
}

pub struct PromptBuilder {
    env: Environment<'static>,
    /// The name the bot has inside the prompt (its persona name).
    pub bot_name: String,
    /// The character card, at the top.
    pub persona: String,
    /// Example messages, after the persona; may be empty.
    pub examples: String,
    /// The short reminder after the history, or the persona when the persona
    /// file has none (design.md §7).
    pub reminder: String,
    /// Other names for the bot (configured name, Discord account name), rewritten
    /// to `bot_name` in message text.
    pub aliases: Vec<String>,
    /// Users' names in lowercase: the history's speaker labels, the `@name`s
    /// their mentions resolved to, and the names command prompts use
    /// (`Persona::lowercase_names`).
    pub lowercase_names: bool,
}

impl PromptBuilder {
    /// A builder for the persona file's three sections (each trimmed).
    pub fn new(template: &str, bot_name: &str, persona: &str, examples: &str, reminder: &str) -> Result<Self> {
        let mut env = Environment::new();
        env.add_template_owned("prompt", template.to_string())
            .map_err(|e| anyhow!("prompt template: {e}"))?;
        let (persona, examples, reminder) = (persona.trim(), examples.trim(), reminder.trim());
        Ok(Self {
            reminder: if reminder.is_empty() { persona } else { reminder }.to_string(),
            env,
            bot_name: bot_name.to_string(),
            persona: persona.to_string(),
            examples: examples.to_string(),
            aliases: Vec::new(),
            lowercase_names: false,
        })
    }

    /// Sets [`lowercase_names`](Self::lowercase_names).
    pub fn with_lowercase_names(mut self, on: bool) -> Self {
        self.lowercase_names = on;
        self
    }

    /// A user's (cleaned) name as the prompt shows it.
    pub fn user_name(&self, name: &str) -> String {
        if self.lowercase_names { name.to_lowercase() } else { name.to_string() }
    }

    /// `text` with its `@mentions` of `names` as the prompt shows them.
    pub fn user_mentions(&self, text: &str, names: &[&str]) -> String {
        if self.lowercase_names { lowercase_mentions(text, names) } else { text.to_string() }
    }

    /// Sets the names rewritten to `bot_name` in message text.
    pub fn with_aliases(mut self, aliases: impl IntoIterator<Item = String>) -> Self {
        let mut list: Vec<String> = Vec::new();
        for a in aliases.into_iter().map(|a| clean_name(&a)) {
            if !a.eq_ignore_ascii_case(&self.bot_name)
                && !list.iter().any(|b| b.eq_ignore_ascii_case(&a))
            {
                list.push(a);
            }
        }
        self.aliases = list;
        self
    }

    /// `text` with every alias of the bot replaced by its prompt name.
    pub fn rename(&self, text: &str) -> String {
        rename_aliases(text, &self.aliases, &self.bot_name)
    }

    pub fn render(&self, history: &[Turn]) -> Result<String> {
        self.render_reply(history, None, true, None, None)
    }

    /// Renders with `reply_to`, the author the bot is answering. It comes after
    /// the history in the template, so it never breaks the cached prefix; so does
    /// `repeat_persona` (false for a CPU reply, see `cpu_drop_persona_repeat`).
    /// `scene` (what the bot was up to, `chat::idle`) goes after the history
    /// and the reminder, where the reply heeds it, so it doesn't break the
    /// cached prefix either.
    pub fn render_reply(
        &self,
        history: &[Turn],
        reply_to: Option<&str>,
        repeat_persona: bool,
        note: Option<&str>,
        scene: Option<&str>,
    ) -> Result<String> {
        self.env
            .get_template("prompt")
            .and_then(|t| {
                let full = if self.examples.is_empty() {
                    self.persona.clone()
                } else {
                    format!("{}\n{}", self.persona, self.examples)
                };
                t.render(context! {
                    system_prompt => full,
                    persona => &self.persona,
                    reminder => &self.reminder,
                    examples => &self.examples,
                    bot_name => &self.bot_name,
                    history => history,
                    reply_to => reply_to,
                    repeat_persona => repeat_persona,
                    note => note,
                    scene => scene,
                })
            })
            .map_err(|e| anyhow!("rendering prompt: {e}"))
    }

    /// Tokens used by the template with no history (system prompt + fixed header).
    /// Rejects a header over `limit` (`GenerationConfig::header_limit`), which
    /// would leave too little room for history (design.md §7).
    pub fn check_header(&self, limit: usize, count: &dyn Fn(&str) -> usize) -> Result<usize> {
        let n = count(&self.render(&[])?);
        ensure!(
            n <= limit,
            "system prompt + template header is {n} tokens; it must be at most {limit} (max_header_tokens; 0 = half of max_prompt_tokens)"
        );
        Ok(n)
    }
}

/// The prompt notes whose phrases appear in the user messages they look at,
/// joined by newlines, `Safety` notes first and the rest in file order;
/// `None` if none match (design.md §7). `recent(n)` gives
/// the user messages since the bot's `n`-th last reply: a note with `turns:
/// n` looks there, so it stays on for `n` replies. `Safety` notes look at
/// everyone's messages there; the others only at the asker's (`asker(n)`,
/// the trigger's author), so a shared channel's backlog of other people's
/// chatter doesn't set them off.
/// `replies(phrases, span)` is how many replies the bot has made since a
/// push with one of `phrases` started, within its last `span` replies
/// (`ChannelWindow::replies_since`); a note with `after_replies: n` goes by
/// it instead (span `n + turns`): on for `turns` replies from the `n`-th.
/// All phrases match whole words (`Phrases::found_whole_in`).
pub fn notes_for(
    notes: &[crate::config::PromptNote],
    recent: &dyn Fn(usize) -> String,
    asker: &dyn Fn(usize) -> String,
    replies: &dyn Fn(&[String], usize) -> Option<usize>,
) -> Option<String> {
    use crate::config::NoteKind;
    use crate::engine::sampler::{normalized_words, Phrases};
    let mut words: std::collections::HashMap<(usize, bool), Vec<String>> = Default::default();
    let mut hits: Vec<&crate::config::PromptNote> = notes
        .iter()
        .filter(|n| {
            if let Some(after) = n.after_replies {
                let span = after + n.turns.max(1);
                return replies(&n.prompted_by, span).is_some_and(|c| c >= after && c < span);
            }
            let (turns, safety) = (n.turns.max(1), n.kind == NoteKind::Safety);
            let w = words.entry((turns, safety)).or_insert_with(|| {
                normalized_words(&if safety { recent(turns) } else { asker(turns) })
            });
            Phrases::new(&n.prompted_by).found_whole_in(w)
        })
        .collect();
    // Safety notes first; personality and humor after, in file order.
    hits.sort_by_key(|n| n.kind != crate::config::NoteKind::Safety);
    let texts: Vec<&str> = hits.iter().map(|n| n.text.trim()).collect();
    (!texts.is_empty()).then(|| texts.join("\n"))
}

/// `"\n### "` plus `"\n<name>:"` for the bot and every author in the window, so the
/// model doesn't speak for other users, and `"(<bot> is replying to"`, the
/// default template's reply marker, which the model sometimes copies after a
/// blank line (design.md §7).
pub fn stop_strings<'a>(bot_name: &str, authors: impl IntoIterator<Item = &'a str>) -> Vec<String> {
    let mut stops = vec![
        "\n### ".to_string(),
        format!("\n{bot_name}:"),
        format!("({bot_name} is replying to"),
    ];
    for a in authors {
        let s = format!("\n{a}:");
        if !stops.contains(&s) {
            stops.push(s);
        }
    }
    stops
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn safety_notes_come_first() {
        use crate::config::{NoteKind, PromptNote};
        let note = |text: &str, kind| PromptNote {
            text: text.into(),
            prompted_by: vec!["hi".into()],
            turns: 1,
            after_replies: None,
            kind,
        };
        let notes = [
            note("(gives in)", NoteKind::Personality),
            note("(a joke)", NoteKind::Humor),
            note("(grown up)", NoteKind::Safety),
            note("(stays himself)", NoteKind::Safety),
        ];
        let got = notes_for(&notes, &|_| "hi there".to_string(), &|_| "hi there".to_string(), &|_, _| None).unwrap();
        assert_eq!(got, "(grown up)\n(stays himself)\n(gives in)\n(a joke)");
    }

    #[test]
    fn a_third_section_replaces_the_persona_after_the_history() {
        let turns = [Turn {
            author: "A".into(),
            content: "hi".into(),
        }];
        let p = PromptBuilder::new(DEFAULT_TEMPLATE, "B", "long persona", "A: hi\nB: hello", "short").unwrap();
        let text = p.render_reply(&turns, None, true, None, None).unwrap();
        let after = &text[text.find("A: hi\n").unwrap()..];
        assert!(after.contains("short") && !after.contains("long persona"), "{text}");
        assert!(text.starts_with("long persona\nA: hi\nB: hello"), "{text}");
        assert!(!text[..text.find("A: hi\n").unwrap()].contains("short"));
        // No reminder: the persona is repeated.
        let p = PromptBuilder::new(DEFAULT_TEMPLATE, "B", "long persona", "A: hi\nB: hello", "").unwrap();
        let text = p.render_reply(&turns, None, true, None, None).unwrap();
        assert!(text[text.find("A: hi\n").unwrap()..].contains("long persona"));
    }

    #[test]
    fn only_safety_notes_read_other_people() {
        use crate::config::{NoteKind, PromptNote};
        let note = |text: &str, phrase: &str, kind| PromptNote {
            text: text.into(),
            prompted_by: vec![phrase.into()],
            turns: 1,
            after_replies: None,
            kind,
        };
        let notes = [
            note("(grown up)", "grade", NoteKind::Safety),
            note("(kobold tastes)", "coal", NoteKind::Personality),
            note("(skittish)", "fire", NoteKind::Humor),
        ];
        // The backlog has everyone's words; the asker said only the last line.
        let all = |_: usize| "what grade is ur kid in\nthis coal is hot\nthe fire is out\n@floob hi".to_string();
        let asker = |_: usize| "@floob hi".to_string();
        assert_eq!(notes_for(&notes, &all, &asker, &|_, _| None).as_deref(), Some("(grown up)"));
        // The asker's own words, whole: inside a longer word is no match.
        let asker = |_: usize| "@floob my coalition is on fire".to_string();
        assert_eq!(notes_for(&notes, &all, &asker, &|_, _| None).as_deref(), Some("(grown up)\n(skittish)"));
        let all = |_: usize| "time to upgrade\n@floob my coalition is on fire".to_string();
        assert_eq!(notes_for(&notes, &all, &asker, &|_, _| None).as_deref(), Some("(skittish)"), "upgrade isn't grade");
    }

    #[test]
    fn after_replies_notes_wait_for_the_count() {
        use crate::config::PromptNote;
        let notes = [PromptNote {
            text: "(B gives in now.)".into(),
            prompted_by: vec!["for your".into()],
            turns: 2,
            after_replies: Some(2),
            kind: Default::default(),
        }];
        let note = |c: Option<usize>| notes_for(&notes, &|_| String::new(), &|_| String::new(), &|_, _| c);
        assert_eq!(note(None), None);
        assert_eq!(note(Some(0)), None);
        assert_eq!(note(Some(1)), None);
        assert_eq!(note(Some(2)).as_deref(), Some("(B gives in now.)"));
        assert_eq!(note(Some(3)).as_deref(), Some("(B gives in now.)"));
        assert_eq!(note(Some(4)), None, "on for `turns` replies only");
    }

    #[test]
    fn notes_match_the_message_and_sit_after_the_history() {
        use crate::config::PromptNote;
        let notes = [
            PromptNote {
                text: "(B is a grown-up.)".into(),
                prompted_by: vec!["mommy".into(), "tuck you in".into()],
                turns: 3,
                after_replies: None,
                kind: Default::default(),
            },
            PromptNote {
                text: "(B skips kids' games.)".into(),
                prompted_by: vec!["seek".into()],
                turns: 1,
                after_replies: None,
                kind: Default::default(),
            },
        ];
        let now = |t: &'static str| move |_: usize| t.to_string();
        assert_eq!(notes_for(&notes, &now("hi B"), &now("hi B"), &|_, _| None), None);
        assert_eq!(notes_for(&notes, &now("want me 2 tuck u in?"), &now("want me 2 tuck u in?"), &|_, _| None).as_deref(), Some("(B is a grown-up.)"));
        assert_eq!(
            notes_for(&notes, &now("im ur mommy\nhide n seeek?"), &now("im ur mommy\nhide n seeek?"), &|_, _| None).as_deref(),
            Some("(B is a grown-up.)\n(B skips kids' games.)")
        );
        // Two replies later: "mommy" is still in the 3-reply note's window,
        // "seek" is out of the 1-reply note's.
        let later = |n: usize| if n >= 3 { "im ur mommy\nhide n seek?\nok\nfloob" } else { "floob" }.to_string();
        assert_eq!(notes_for(&notes, &later, &later, &|_, _| None).as_deref(), Some("(B is a grown-up.)"));
        let p = PromptBuilder::new(DEFAULT_TEMPLATE, "B", "persona", "", "").unwrap();
        let turns = [Turn {
            author: "A".into(),
            content: "im ur mommy".into(),
        }];
        let with = p.render_reply(&turns, None, true, Some("(B is a grown-up.)"), None).unwrap();
        let without = p.render_reply(&turns, None, true, None, None).unwrap();
        assert!(!without.contains("grown-up"));
        let at = with.find("(B is a grown-up.)").unwrap();
        assert!(with.find("A: im ur mommy").unwrap() < at && at < with.rfind("### Instruction:").unwrap());
        // Everything before the note is unchanged, so the cached prefix holds.
        assert!(with.starts_with(&without[..without.rfind("### Instruction:").unwrap()]));
    }
    use crate::config::DEFAULT_TEMPLATE;

    #[test]
    fn renders_default_template() {
        let p = PromptBuilder::new(DEFAULT_TEMPLATE, "Increbot", "Be nice.\n", "", "").unwrap();
        let out = p
            .render(&[
                Turn {
                    author: "Alice".into(),
                    content: "hi".into(),
                },
                Turn {
                    author: "Increbot".into(),
                    content: "hello".into(),
                },
            ])
            .unwrap();
        assert_eq!(
            out,
            "Be nice.\n\n### Instruction:\nContinue the conversation below as Increbot.\n\nAlice: hi\nIncrebot: hello\n\n\
             Be nice.\n\n### Instruction:\nContinue the conversation above as Increbot.\n\n### Response:\nIncrebot:"
        );
        // Prefix stability: one more turn only appends before the repeated
        // system prompt and instruction.
        let shorter = p
            .render(&[Turn {
                author: "Alice".into(),
                content: "hi".into(),
            }])
            .unwrap();
        let cut = shorter.rfind("\nBe nice.").unwrap();
        assert!(out.starts_with(&shorter[..cut]));
    }

    #[test]
    fn persona_repeat_can_be_left_out() {
        let p = PromptBuilder::new(DEFAULT_TEMPLATE, "floob", "Be nice.", "", "").unwrap();
        let turns = [Turn {
            author: "Carol".into(),
            content: "hi".into(),
        }];
        let with = p.render_reply(&turns, Some("Carol"), true, None, None).unwrap();
        let without = p.render_reply(&turns, Some("Carol"), false, None, None).unwrap();
        assert!(
            without.ends_with(
                "Carol: hi\n\n### Instruction:\nContinue the conversation above as floob.\n\n\
                 ### Response:\n(floob is replying to Carol)\nfloob:"
            ),
            "{without}"
        );
        assert_eq!(without.matches("Be nice.").count(), 1);
        // The cached prefix (through the history) is the same either way.
        let cut = without.rfind("\n### Instruction:").unwrap();
        assert!(with.starts_with(&without[..cut]));
    }

    #[test]
    fn examples_appear_only_at_the_top() {
        let p = PromptBuilder::new(DEFAULT_TEMPLATE, "B", "Be nice.", "A: hi\nB: hello\n", "").unwrap();
        let out = p
            .render(&[Turn {
                author: "Carol".into(),
                content: "hey".into(),
            }])
            .unwrap();
        assert_eq!(
            out,
            "Be nice.\nA: hi\nB: hello\n\n### Instruction:\nContinue the conversation below as B.\n\n\
             Carol: hey\n\nBe nice.\n\n### Instruction:\nContinue the conversation above as B.\n\n### Response:\nB:"
        );
    }

    #[test]
    fn reply_to_names_the_author_after_the_history() {
        let p = PromptBuilder::new(DEFAULT_TEMPLATE, "floob", "Be nice.", "", "").unwrap();
        let turns = [Turn {
            author: "Carol".into(),
            content: "hi".into(),
        }];
        let with = p.render_reply(&turns, Some("Carol"), true, None, None).unwrap();
        assert!(
            with.ends_with(
                "Carol: hi\n\nBe nice.\n\n### Instruction:\nContinue the conversation above as floob.\n\n\
                 ### Response:\n(floob is replying to Carol)\nfloob:"
            ),
            "{with}"
        );
        let without = p.render(&turns).unwrap();
        assert!(without.ends_with("### Response:\nfloob:"), "{without}");
        let cut = without.find("\n### Response:").unwrap();
        assert_eq!(
            with[..cut],
            without[..cut],
            "the cached prefix is unchanged"
        );
    }

    #[test]
    fn header_size_check() {
        let words = |s: &str| s.split_whitespace().count();
        let p = PromptBuilder::new(DEFAULT_TEMPLATE, "B", "short", "", "").unwrap();
        assert!(p.check_header(100, &words).is_ok());
        let long = "word ".repeat(80);
        let p = PromptBuilder::new(DEFAULT_TEMPLATE, "B", &long, "", "").unwrap();
        assert!(
            p.check_header(100, &words)
                .unwrap_err()
                .to_string()
                .contains("max_header_tokens")
        );
    }

    #[test]
    fn aliases_are_renamed_to_the_persona_name() {
        let p = PromptBuilder::new(DEFAULT_TEMPLATE, "floob", "sys", "", "")
            .unwrap()
            .with_aliases(["IncreBot".to_string(), "IncreBot#1234".into(), "floob".into()]);
        assert_eq!(p.aliases, ["IncreBot", "IncreBot#1234"]);
        assert!(p.render(&[]).unwrap().ends_with("floob:"));
        assert_eq!(
            p.rename("hey @IncreBot#1234 and IncreBot"),
            "hey @floob and floob"
        );
    }

    #[test]
    fn stops_cover_bot_and_authors_once() {
        let s = stop_strings("Bot", ["Alice", "Bob", "Alice"]);
        assert_eq!(s, vec!["\n### ", "\nBot:", "(Bot is replying to", "\nAlice:", "\nBob:"]);
    }
}
