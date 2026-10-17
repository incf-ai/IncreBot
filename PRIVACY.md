# FoobBot Privacy Policy

**Effective date:** October 15, 2026

This policy explains what information the FoobBot Discord bot (the "Bot", which
talks as a character called "floob") handles, why, and for how long. The Bot is
operated by **incf_** on Discord ("we", "us") as a hobby project, for fun and geek
cred. It runs in several public Discord servers, but only ones we have chosen to
allow: nobody else can add it to a server.

The short version: the Bot reads recent chat messages so it can reply in context.
It keeps them **only in memory**, doesn't save them to disk (apart from occasional
troubleshooting logs, see section 3), doesn't send them to any service other than
Discord, and forgets them when they age out, when we use `/super-admin reset`, or when
the Bot restarts.

## 1. What the Bot processes

In the servers where it is allowed, the Bot receives messages from Discord in the
channels it can see, but it keeps only those from the channels it has been set up to
talk in (and their threads); it ignores the rest. For each message it keeps, it may
hold:

- the message text, with mentions shown as display names;
- the author's Discord user ID and display name or server nickname;
- the message ID and the channel or thread it was sent in;
- short descriptions of non-text content: attachment **file names** (the files
  themselves are not downloaded), sticker names, link-preview titles and poll
  questions;
- whether the message mentions the Bot or replies to it.

The Bot also processes:

- **slash commands** you use with it (for example, the question you type into
  `/ask`);
- **reactions** on its replies (a ❌ reaction stops a reply);
- **direct messages**, only from users we have specifically allowed. Everyone
  else's DMs are ignored.

The first time the Bot replies in a channel after it starts or after `/super-admin reset`, it
fetches a few recent messages from that channel through Discord, so it knows what
the conversation is about.

The Bot doesn't collect anything from outside Discord.

## 2. How it is used

Only to generate the Bot's replies. Recent messages are combined with the Bot's
character description into a text prompt, which an AI language model reads to
write the next reply. The model runs **on our own computer**: your messages are
not sent to any AI provider or other third party.

We do not use your messages to train or fine-tune any AI model. We do not sell or
share them, and we do not use them for advertising or profiling.

## 3. Where it is kept and for how long

| What | Where | How long |
|---|---|---|
| Recent messages in each channel | Memory (RAM) only | Until newer messages push them out, we run `/super-admin reset`, or the Bot restarts |
| The AI model's working memory of recent conversations (numbers computed from the prompt) | Memory only | A few recent conversations; older ones are dropped automatically. Cleared by `/super-admin reset` or a restart |
| The last prompt sent to the model in each channel | Memory only | Until the next reply in that channel, `/super-admin reset`, or a restart |
| For the last 20 messages in each channel that the Bot replied to: the prompt the model read for that reply (which includes the recent messages it was given) and the settings it was made with | Memory only | Until newer messages push it out, the reply is retried (replaced by the new prompt), anyone uses `/opt-out`, we run `/super-admin reset`, or the Bot restarts |
| For the last 20 messages in each channel: who sent it, its first 80 characters, and what the Bot decided and did about it | Memory only | Until newer messages push it out, you use `/opt-out`, we run `/super-admin reset`, or the Bot restarts |
| The opt-out list: your user ID and when you opted out, if you used `/opt-out` | A file on our computer | Until you use `/opt-in`, which replaces it with your user ID and the time you opted back in (kept so the Bot keeps ignoring what you wrote while opted out, even after a restart) |
| Technical logs | Log files on our computer | As long as they're useful for running and troubleshooting the Bot |

**Logs** normally contain only technical information: times, channel IDs, how
much text (in tokens) was processed, speed, and error messages. They are not
meant to contain message text or user IDs, though an error message may
occasionally include a fragment of one.

To fix problems, we may occasionally turn on detailed logging that also records
message text and prompts. Those logs are kept only as long as needed to
troubleshoot.

Replies the Bot posts are ordinary Discord messages. They stay in the channel
like anyone else's messages, and Discord's own policies apply to them.

## 4. Who can see it

- **Everyone in the channel** can see the Bot's replies, which may quote or refer
  to things that were said.
- **We, the operators,** can use `/super-admin context`. It privately shows us the last
  prompt the Bot read in a channel, which includes the recent messages it was
  given. We use it to check and improve the Bot's behavior. Server owners and
  moderators can't.
- **We, the operators,** can also use the "Admin actions" → Trace button. It privately
  sends us, for one of the last 20 messages in a channel, its first 80 characters and
  what the Bot decided and did about it (whether it replied and why, how long it waited,
  how the reply was made). We use it to find out why the Bot behaved as it did.
- **We, the operators,** can also use the "Admin actions" → Context button. It
  privately shows us, for one of the last 20 messages in a channel that the Bot
  replied to, the prompt the model read for that reply, which includes the recent
  messages it was given, and the settings it was made with. We use it to check and
  improve the Bot's behavior.
- **Discord** handles all messages under
  [Discord's Privacy Policy](https://discord.com/privacy).

We don't give anyone else access, and we don't share the information unless the
law requires it.

## 5. Your choices and rights

- **Don't want the Bot to read you?** Use `/opt-out`. From then on, the Bot drops
  your messages as soon as Discord delivers them: it doesn't keep them, use them
  in replies, or reply to them. It also forgets the messages of yours it was
  holding. To remember your choice, it saves your user ID and the time you opted
  out. `/opt-in` makes your messages visible again from that moment on: what you
  wrote while opted out stays invisible, even if the Bot restarts, which is why
  the record is replaced by your user ID and the time you opted back in rather
  than deleted. `/ask` still works while you are opted out, because you send
  that question to the Bot directly.
- **Skip a single message:** start it with `//`. The Bot ignores it completely.
- You can also talk in channels the Bot can't see. The Bot only reads channels it
  has been set up to talk in.
- **Erasing a conversation:** we can run `/super-admin reset` in a channel to make the Bot
  forget it immediately; ask us if you want a conversation forgotten. Everything
  is also forgotten when the Bot restarts.
- **Requests and questions:** contact **incf_** on Discord. Because we keep
  messages only briefly and only in memory, there is usually nothing left to hand
  over or delete by the time a request arrives.

**If you are in the EU, UK or a similar jurisdiction:** we process this
information on the basis of our legitimate interest in running a chat bot for the
communities where it is used (GDPR Art. 6(1)(f)). You have the right to access,
correct or delete your personal data, to object to or restrict its processing,
and to complain to your local data protection authority. Contact us to exercise
these rights. The Bot is operated from the United States, so your messages are
processed there.

## 6. Age

The Bot is for people **18 and older** (see the [Terms of Service](TERMS.md)). We
do not knowingly process information from anyone younger.

## 7. Security

Messages are held only in memory on a computer we control, and the Bot's access
token is kept private. No system is perfectly secure, but keeping almost nothing, for a
short time, limits what could ever be exposed.

## 8. Changes

If this policy changes, we will update this page and its effective date.

## 9. Contact

**incf_** on Discord.
