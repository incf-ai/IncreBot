//! Parsing a submitted input line into a [`Command`].

use std::fmt;
use std::str::FromStr;

/// When the bot replies on its own.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReplyMode {
    /// Every message, like a `discord.dedicated_channels` channel.
    Every,
    /// Only messages that mention `@<bot name>`, like a `discord.shared_channels` channel.
    Mention,
}

impl FromStr for ReplyMode {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, String> {
        match s.trim().to_ascii_lowercase().as_str() {
            "every" | "all" => Ok(Self::Every),
            "mention" | "mentions" => Ok(Self::Mention),
            other => Err(format!("unknown mode {other:?} (every | mention)")),
        }
    }
}

impl fmt::Display for ReplyMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Every => "every",
            Self::Mention => "mention",
        })
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Command {
    /// A chat message from the current user.
    Say(String),
    /// A chat message from another user, without switching nick.
    As {
        nick: String,
        text: String,
    },
    Nick(String),
    /// A one-shot question that bypasses the channel history (`/ask`).
    Ask(String),
    Reset,
    Cancel,
    Mode(ReplyMode),
    Stats,
    /// Show the last prompt the bot was given.
    Prompt,
    Clear,
    Help,
    Quit,
}

pub const HELP: &str = "\
/nick NAME        speak as someone else from now on
/as NAME TEXT     send one message as NAME
/ask QUESTION     one-shot question without channel history (like /ask)
/mode every|mention   reply to every message, or only to @mentions
/cancel           stop the reply in progress and drop queued ones (also Esc)
/reset            forget the channel history and the bot's cache (like /reset)
/stats            engine and history numbers (like /stats)
/prompt           the last prompt the bot was given, exactly (like /context)
/clear            clear the screen (the bot's history is kept)
/quit             leave (also Ctrl-C)
Start a message with // to send a literal leading / (so ///note sends a '//note'
side note, which floob ignores, as on Discord).
Keys: Enter send · ↑/↓ input history · PgUp/PgDn scroll · Esc stop reply";

/// Parses one submitted line. `Ok(None)` for a blank line; `Err` holds a message
/// for the user.
pub fn parse(line: &str) -> Result<Option<Command>, String> {
    let line = line.trim();
    if line.is_empty() {
        return Ok(None);
    }
    if let Some(rest) = line.strip_prefix("//") {
        return Ok(Some(Command::Say(format!("/{rest}"))));
    }
    let Some(rest) = line.strip_prefix('/') else {
        return Ok(Some(Command::Say(line.to_string())));
    };
    let (name, arg) = match rest.split_once(char::is_whitespace) {
        Some((n, a)) => (n, a.trim()),
        None => (rest, ""),
    };
    let need = |usage: &str| {
        if arg.is_empty() {
            Err(format!("usage: {usage}"))
        } else {
            Ok(arg.to_string())
        }
    };
    let no_arg = |cmd: Command| {
        if arg.is_empty() {
            Ok(cmd)
        } else {
            Err(format!("/{name} takes no arguments"))
        }
    };
    let cmd = match name.to_ascii_lowercase().as_str() {
        "nick" | "name" => Command::Nick(need("/nick NAME")?),
        "as" => {
            let usage = "usage: /as NAME TEXT";
            let (nick, text) = arg.split_once(char::is_whitespace).ok_or(usage)?;
            let text = text.trim();
            if text.is_empty() {
                return Err(usage.into());
            }
            Command::As {
                nick: nick.to_string(),
                text: text.to_string(),
            }
        }
        "ask" => Command::Ask(need("/ask QUESTION")?),
        "mode" => Command::Mode(need("/mode every|mention")?.parse()?),
        "reset" => no_arg(Command::Reset)?,
        "cancel" | "stop" => no_arg(Command::Cancel)?,
        "stats" => no_arg(Command::Stats)?,
        "prompt" | "context" => no_arg(Command::Prompt)?,
        "clear" => no_arg(Command::Clear)?,
        "help" | "?" => Command::Help,
        "quit" | "exit" | "q" => Command::Quit,
        _ => return Err(format!("unknown command /{name}; try /help")),
    };
    Ok(Some(cmd))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ok(s: &str) -> Command {
        parse(s).unwrap().unwrap()
    }

    #[test]
    fn plain_text_and_blank_lines() {
        assert_eq!(parse("   ").unwrap(), None);
        assert_eq!(ok("  hello there  "), Command::Say("hello there".into()));
        assert_eq!(ok("//shrug"), Command::Say("/shrug".into()));
        assert_eq!(ok("a / b"), Command::Say("a / b".into()));
    }

    #[test]
    fn commands_with_arguments() {
        assert_eq!(ok("/nick  Bob Smith "), Command::Nick("Bob Smith".into()));
        assert_eq!(ok("/ASK why?"), Command::Ask("why?".into()));
        assert_eq!(
            ok("/as Carol  hi all"),
            Command::As {
                nick: "Carol".into(),
                text: "hi all".into()
            }
        );
        assert_eq!(ok("/mode Mention"), Command::Mode(ReplyMode::Mention));
        assert_eq!(ok("/mode every"), Command::Mode(ReplyMode::Every));
    }

    #[test]
    fn commands_without_arguments() {
        assert_eq!(ok("/reset"), Command::Reset);
        assert_eq!(ok("/stop"), Command::Cancel);
        assert_eq!(ok("/stats"), Command::Stats);
        assert_eq!(ok("/prompt"), Command::Prompt);
        assert_eq!(ok("/context"), Command::Prompt);
        assert_eq!(ok("/clear"), Command::Clear);
        assert_eq!(ok("/help"), Command::Help);
        assert_eq!(ok("/q"), Command::Quit);
    }

    #[test]
    fn errors_explain_usage() {
        assert_eq!(parse("/nick").unwrap_err(), "usage: /nick NAME");
        assert_eq!(parse("/as Bob").unwrap_err(), "usage: /as NAME TEXT");
        assert!(parse("/mode loud").unwrap_err().contains("unknown mode"));
        assert!(parse("/reset now").unwrap_err().contains("no arguments"));
        assert!(parse("/frobnicate").unwrap_err().contains("/help"));
    }

    #[test]
    fn mode_round_trips() {
        for m in [ReplyMode::Every, ReplyMode::Mention] {
            assert_eq!(m.to_string().parse::<ReplyMode>().unwrap(), m);
        }
    }
}
