//! Whether to reply to a message, and who counts as an admin (design.md §6).

use crate::config::{Channel, DiscordConfig, listed};

/// What the gate needs to know about a message.
#[derive(Clone, Debug, Default)]
pub struct MessageFacts {
    pub author_id: u64,
    pub author_is_bot: bool,
    /// `None` for a DM.
    pub guild_id: Option<u64>,
    pub channel_id: u64,
    /// The parent channel when the message is in a thread.
    pub parent_channel_id: Option<u64>,
    pub mentions_bot: bool,
    pub replies_to_bot: bool,
}

/// How the bot behaves in a guild channel (design.md §6).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ChannelKind {
    /// `dedicated_channels`: replies to every message.
    Dedicated,
    /// `shared_channels`: replies only to mentions and replies to its messages.
    Shared,
}

/// A channel's kind; threads inherit their parent's. `None`: the bot ignores it.
pub fn channel_kind(channel: u64, parent: Option<u64>, cfg: &DiscordConfig) -> Option<ChannelKind> {
    let has =
        |list: &[Channel]| listed(list, channel) || parent.is_some_and(|p| listed(list, p));
    if has(&cfg.dedicated_channels) {
        Some(ChannelKind::Dedicated)
    } else if has(&cfg.shared_channels) {
        Some(ChannelKind::Shared)
    } else {
        None
    }
}

/// Whether the bot reads a message at all (keeps it as history): DMs from
/// `dm_users`, and dedicated or shared channels (and their threads) in
/// allowlisted guilds. Everything else is ignored.
pub fn in_scope(m: &MessageFacts, cfg: &DiscordConfig) -> bool {
    match m.guild_id {
        None => listed(&cfg.dm_users, m.author_id),
        Some(g) => {
            listed(&cfg.guilds, g)
                && channel_kind(m.channel_id, m.parent_channel_id, cfg).is_some()
        }
    }
}

/// Replies: every in-scope DM, every message in a dedicated channel, and
/// mentions of the bot or replies to it in a shared channel. Never to bots.
pub fn should_reply(m: &MessageFacts, cfg: &DiscordConfig) -> bool {
    if m.author_is_bot || !in_scope(m, cfg) {
        return false;
    }
    match channel_kind(m.channel_id, m.parent_channel_id, cfg) {
        _ if m.guild_id.is_none() => true,
        Some(ChannelKind::Dedicated) => true,
        Some(ChannelKind::Shared) => m.mentions_bot || m.replies_to_bot,
        None => false,
    }
}

/// Messages starting with `//` (after leading whitespace) are invisible to the
/// bot: no reply, even when pinged, and never part of its history. People use it
/// for side remarks the bot shouldn't see.
pub fn is_ignored_text(text: &str) -> bool {
    text.trim_start().starts_with("//")
}

/// Whether a message addresses the bot: it mentions the bot user, or one of the
/// bot's own roles. Discord creates a managed role with the bot's name when the
/// bot is invited with permissions, and the @ menu offers it next to the bot, so
/// users often pick it by mistake.
pub fn mentions_bot(users: &[u64], roles: &[u64], bot_id: u64, bot_roles: &[u64]) -> bool {
    users.contains(&bot_id) || roles.iter().any(|r| bot_roles.contains(r))
}

/// Admins: exactly the user IDs in `admins`. Server permissions such as Manage
/// Server don't count, so other servers' staff can't see prompts (`/super-admin context`),
/// reset the bot or stop other people's replies.
pub fn is_admin(user_id: u64, cfg: &DiscordConfig) -> bool {
    listed(&cfg.admins, user_id)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> DiscordConfig {
        DiscordConfig {
            guilds: vec![10.into()],
            dedicated_channels: vec![100.into()],
            shared_channels: vec![300.into()],
            dm_users: vec![7.into()],
            admins: vec![8.into()],
            ..Default::default()
        }
    }

    fn msg() -> MessageFacts {
        MessageFacts {
            author_id: 1,
            guild_id: Some(10),
            channel_id: 200,
            ..Default::default()
        }
    }

    #[test]
    fn the_bots_own_role_counts_as_a_mention() {
        let (bot, bot_roles) = (42, [7]);
        assert!(mentions_bot(&[42], &[], bot, &bot_roles));
        assert!(mentions_bot(&[], &[7], bot, &bot_roles));
        assert!(mentions_bot(&[5], &[3, 7], bot, &bot_roles));
        // Other users and other roles don't address the bot.
        assert!(!mentions_bot(&[5], &[3], bot, &bot_roles));
        assert!(!mentions_bot(&[], &[], bot, &bot_roles));
        // Roles unknown yet (cache not filled): only user mentions count.
        assert!(!mentions_bot(&[], &[7], bot, &[]));
    }

    fn at(channel: u64, parent: Option<u64>) -> MessageFacts {
        MessageFacts {
            channel_id: channel,
            parent_channel_id: parent,
            ..msg()
        }
    }

    fn mentioned(m: MessageFacts) -> MessageFacts {
        MessageFacts {
            mentions_bot: true,
            ..m
        }
    }

    fn replied(m: MessageFacts) -> MessageFacts {
        MessageFacts {
            replies_to_bot: true,
            ..m
        }
    }

    #[test]
    fn channel_kinds_and_threads() {
        let c = cfg();
        assert_eq!(channel_kind(100, None, &c), Some(ChannelKind::Dedicated));
        assert_eq!(channel_kind(300, None, &c), Some(ChannelKind::Shared));
        assert_eq!(channel_kind(200, None, &c), None);
        // Threads inherit their parent's kind.
        assert_eq!(
            channel_kind(555, Some(100), &c),
            Some(ChannelKind::Dedicated)
        );
        assert_eq!(channel_kind(556, Some(300), &c), Some(ChannelKind::Shared));
        assert_eq!(channel_kind(557, Some(200), &c), None);
    }

    #[test]
    fn dedicated_channels_answer_everything() {
        let c = cfg();
        assert!(should_reply(&at(100, None), &c));
        assert!(should_reply(&at(555, Some(100)), &c));
        assert!(in_scope(&at(100, None), &c));
    }

    #[test]
    fn shared_channels_answer_only_mentions_and_replies() {
        let c = cfg();
        assert!(in_scope(&at(300, None), &c), "kept as history");
        assert!(!should_reply(&at(300, None), &c));
        assert!(should_reply(&mentioned(at(300, None)), &c));
        assert!(should_reply(&replied(at(300, None)), &c));
        assert!(!should_reply(&at(556, Some(300)), &c));
        assert!(should_reply(&mentioned(at(556, Some(300))), &c));
    }

    #[test]
    fn unlisted_channels_and_other_guilds_are_ignored() {
        let c = cfg();
        for m in [at(200, None), at(557, Some(200))] {
            assert!(!in_scope(&m, &c), "not even kept as history");
            assert!(!should_reply(&mentioned(m.clone()), &c));
            assert!(!should_reply(&replied(m), &c));
        }
        let elsewhere = MessageFacts {
            guild_id: Some(11),
            ..mentioned(at(100, None))
        };
        assert!(!in_scope(&elsewhere, &c) && !should_reply(&elsewhere, &c));
    }

    #[test]
    fn double_slash_messages_are_ignored() {
        assert!(is_ignored_text("// side note, floob ignore this"));
        assert!(is_ignored_text("   //no space"));
        assert!(is_ignored_text("//"));
        assert!(!is_ignored_text("/ask something"));
        assert!(!is_ignored_text("see https://example.com"));
        assert!(!is_ignored_text("hi // trailing comment"));
        assert!(!is_ignored_text(""));
    }

    #[test]
    fn bots_never_get_replies() {
        let c = cfg();
        let bot = MessageFacts {
            author_is_bot: true,
            ..mentioned(at(100, None))
        };
        assert!(!should_reply(&bot, &c));
    }

    #[test]
    fn dms_only_from_allowlisted_users() {
        let c = cfg();
        let dm = MessageFacts {
            guild_id: None,
            ..msg()
        };
        assert!(!should_reply(&dm, &c));
        assert!(should_reply(&MessageFacts { author_id: 7, ..dm }, &c));
    }

    #[test]
    fn admins() {
        let c = cfg();
        assert!(is_admin(8, &c));
        assert!(!is_admin(1, &c));
        assert!(
            !is_admin(8, &DiscordConfig::default()),
            "no admins by default"
        );
    }
}
