//! `/classify-text` and `/classify-image`: up to five yes/no questions about
//! a text or a picture, each answered on its own, answered by d1 (`crate::classify`) rather than the chat
//! model. Billed like `/ask`. The reply names the asker (without a ping) and
//! reposts what was judged: the text in full, the picture spoilered.

use std::sync::Arc;
use std::sync::atomic::Ordering;

use poise::serenity_prelude as serenity;
use tracing::{info, warn};

use super::{Context, Error, NOTICE_PREFIX, notice};
use crate::classify::image::ImageLimits;
use crate::classify::{Classifier, Decision, Json, Question};

/// A model used one call at a time: each uses every compute thread (or the
/// GPU) on its own.
pub struct Gated<M> {
    pub model: Arc<M>,
    pub turn: tokio::sync::Semaphore,
}

impl<M> Gated<M> {
    pub fn new(model: M) -> Self {
        Self { model: Arc::new(model), turn: tokio::sync::Semaphore::new(1) }
    }
}

/// The d1 classifier.
pub type Classify = Gated<Classifier>;

/// Ask d1 up to five yes/no questions about a text.
#[poise::command(
    slash_command,
    rename = "classify-text",
    install_context = "Guild",
    interaction_context = "Guild|BotDm"
)]
pub async fn classify_text(
    ctx: Context<'_>,
    #[description = "The text to judge"]
    #[max_length = 6000]
    text: String,
    #[description = "A yes/no question about it, e.g. \"Is this a complaint?\""]
    #[max_length = 300]
    question: String,
    #[description = "Another yes/no question, answered on its own"]
    #[max_length = 300]
    question2: Option<String>,
    #[description = "A third yes/no question, answered on its own"]
    #[max_length = 300]
    question3: Option<String>,
    #[description = "A fourth yes/no question, answered on its own"]
    #[max_length = 300]
    question4: Option<String>,
    #[description = "A fifth yes/no question, answered on its own"]
    #[max_length = 300]
    question5: Option<String>,
) -> Result<(), Error> {
    let data = ctx.data().clone();
    let Some(classify) = data.classify.clone() else {
        return notice(ctx, format!("{NOTICE_PREFIX}classification is turned off here)*")).await;
    };
    if data.shutting_down.load(Ordering::Relaxed) {
        return notice(ctx, format!("{NOTICE_PREFIX}restarting, ask again shortly)*")).await;
    }
    let text = text.trim().to_owned();
    let questions = questions([question, or(question2), or(question3), or(question4), or(question5)]);
    if text.is_empty() || questions.is_empty() {
        return notice(ctx, format!("{NOTICE_PREFIX}needs both a text and a question)*")).await;
    }
    let limit = data.cfg.classify.classify_max_text_tokens;
    let tokens = classify.model.tokenizer().encode(&text).map_or(usize::MAX, |t| t.len());
    if tokens > limit {
        return notice(
            ctx,
            format!("{NOTICE_PREFIX}that text is {tokens} tokens; the limit is {limit})*"),
        )
        .await;
    }
    let hold = match data.limit_try_hold(ctx.author().id.get(), 1.0) {
        Ok(hold) => hold,
        Err(why) => return notice(ctx, why).await,
    };
    ctx.defer().await?;

    let qs = asked(&questions);
    let state = Json::String(text.clone());
    let run = move |c: &Classifier| c.classify(Some(&state), &qs);
    let Some(yes) = decide(ctx, &classify, hold, "classify-text", tokens, run).await? else {
        return Ok(());
    };
    // The text first, quoted line by line (`>>>` would take in the answers
    // too), or attached when it doesn't fit in one message.
    let (who, answers) = (asker(ctx), answers(&questions, &yes));
    let inline = format!("{who}\n{}\n{answers}", block_quote(&text));
    let out = if inline.chars().count() <= MESSAGE_CHARS {
        poise::CreateReply::default().content(inline)
    } else {
        poise::CreateReply::default()
            .content(format!("{who} *(the text is attached)*\n{answers}"))
            .attachment(serenity::CreateAttachment::bytes(text.into_bytes(), "text.txt"))
    };
    reply(ctx, out).await
}

/// Ask d1 up to five yes/no questions about a picture.
#[poise::command(
    slash_command,
    rename = "classify-image",
    install_context = "Guild",
    interaction_context = "Guild|BotDm"
)]
pub async fn classify_image(
    ctx: Context<'_>,
    #[description = "The picture to judge (PNG, JPEG or WebP)"] picture: serenity::Attachment,
    #[description = "A yes/no question about it, e.g. \"Is there a cat in this picture?\""]
    #[max_length = 300]
    question: String,
    #[description = "Another yes/no question, answered on its own"]
    #[max_length = 300]
    question2: Option<String>,
    #[description = "A third yes/no question, answered on its own"]
    #[max_length = 300]
    question3: Option<String>,
    #[description = "A fourth yes/no question, answered on its own"]
    #[max_length = 300]
    question4: Option<String>,
    #[description = "A fifth yes/no question, answered on its own"]
    #[max_length = 300]
    question5: Option<String>,
) -> Result<(), Error> {
    let data = ctx.data().clone();
    let Some(classify) = data.classify.clone().filter(|c| c.model.has_vision()) else {
        return notice(ctx, format!("{NOTICE_PREFIX}picture classification is turned off here)*")).await;
    };
    if data.shutting_down.load(Ordering::Relaxed) {
        return notice(ctx, format!("{NOTICE_PREFIX}restarting, ask again shortly)*")).await;
    }
    let questions = questions([question, or(question2), or(question3), or(question4), or(question5)]);
    if questions.is_empty() {
        return notice(ctx, format!("{NOTICE_PREFIX}needs a question)*")).await;
    }
    let k = &data.cfg.classify;
    let limits = ImageLimits {
        max_bytes: k.max_image_mib << 20,
        max_pixels: k.max_image_pixels,
        max_side: k.image_max_side,
    };
    // Discord's size, before fetching anything; `load` checks the bytes again.
    let size = u64::from(picture.size);
    if size > limits.max_bytes {
        let why = crate::classify::image::Rejected::TooLarge { bytes: size, max: limits.max_bytes };
        return notice(ctx, format!("{NOTICE_PREFIX}{why})*")).await;
    }
    let hold = match data.limit_try_hold(ctx.author().id.get(), 1.0) {
        Ok(hold) => hold,
        Err(why) => return notice(ctx, why).await,
    };

    // Fetched and decoded before deferring when that's quick, so a bad file's
    // notice stays ephemeral (after a public defer it can't be).
    let fetch = fetch_picture(&picture, limits);
    tokio::pin!(fetch);
    let (fetched, deferred) = match tokio::time::timeout(FETCH_BEFORE_DEFER, &mut fetch).await {
        Ok(fetched) => (fetched, false),
        Err(_) => {
            ctx.defer().await?;
            (fetch.await, true)
        }
    };
    let (rgb, jpeg) = match fetched {
        Ok(fetched) => fetched,
        Err(why) => {
            data.limit_release(hold);
            return notice(ctx, format!("{NOTICE_PREFIX}{why})*")).await;
        }
    };
    if !deferred {
        ctx.defer().await?;
    }
    let pixels = rgb.width() as usize * rgb.height() as usize;

    let qs = asked(&questions);
    let run = move |c: &Classifier| c.classify_image(&[rgb], None, &qs);
    let Some(yes) = decide(ctx, &classify, hold, "classify-image", pixels, run).await? else {
        return Ok(());
    };
    let name = format!("SPOILER_{}.jpg", file_stem(&picture.filename));
    reply_with_picture(ctx, jpeg, &name, &answers(&questions, &yes)).await
}

/// The asker, the picture (spoilered), then `answers`, as one reply.
async fn reply_with_picture(ctx: Context<'_>, jpeg: Vec<u8>, name: &str, answers: &str) -> Result<(), Error> {
    let components = vec![text_display(&asker(ctx)), gallery(&[name]), text_display(answers)];
    let file = serenity::CreateAttachment::bytes(jpeg, name);
    let plain = poise::CreateReply::default()
        .content(format!("{}\n{answers}", asker(ctx)))
        .attachment(file.clone());
    reply_v2(ctx, components, vec![file], plain).await
}

/// Components V2 (which serenity 0.12 has no builders for): a text display.
pub(super) fn text_display(content: &str) -> serde_json::Value {
    serde_json::json!({ "type": 10, "content": content })
}

/// Components V2: the attachments `names`, spoilered, as one media gallery
/// (side by side, in order).
pub(super) fn gallery(names: &[&str]) -> serde_json::Value {
    let items: Vec<_> = names
        .iter()
        .map(|name| serde_json::json!({ "media": { "url": format!("attachment://{name}") }, "spoiler": true }))
        .collect();
    serde_json::json!({ "type": 12, "items": items })
}

/// Components V2: the attachment `name` as a file.
pub(super) fn file(name: &str) -> serde_json::Value {
    serde_json::json!({ "type": 13, "file": { "url": format!("attachment://{name}") } })
}

/// The deferred response, edited into a Components V2 message of `components`
/// with `files` attached (a plain message shows attachments under its text, and
/// an embed's picture can't be spoilered). If Discord refuses it, `plain` is
/// sent instead: still one reply, with the files under the text.
pub(super) async fn reply_v2(
    ctx: Context<'_>,
    components: Vec<serde_json::Value>,
    files: Vec<serenity::CreateAttachment>,
    plain: poise::CreateReply,
) -> Result<(), Error> {
    let poise::Context::Application(app) = ctx else {
        unreachable!("only slash commands reply this way")
    };
    const IS_COMPONENTS_V2: u64 = 1 << 15;
    let attachments: Vec<_> = files
        .iter()
        .enumerate()
        .map(|(id, f)| serde_json::json!({ "id": id, "filename": f.filename }))
        .collect();
    let body = serde_json::json!({
        "flags": IS_COMPONENTS_V2,
        "allowed_mentions": { "parse": [] },
        "attachments": attachments,
        "components": components,
    });
    // Each file is the part `files[id]` for its id above. Serenity 0.12 names
    // every part `files[0]` here (the ids it would set are private to it), so
    // two or more files were refused; this is its request, numbered.
    let mut form = reqwest::multipart::Form::new().text("payload_json", body.to_string());
    for (id, f) in files.into_iter().enumerate() {
        let mime = if f.filename.ends_with(".txt") { "text/plain; charset=utf-8" } else { "image/jpeg" };
        let part = reqwest::multipart::Part::bytes(f.data).file_name(f.filename).mime_str(mime)?;
        form = form.part(format!("files[{id}]"), part);
    }
    // The interaction token is the credential; no bot token goes with it.
    let url = format!(
        "https://discord.com/api/v10/webhooks/{}/{}/messages/@original",
        app.interaction.application_id, app.interaction.token
    );
    let sent = async {
        let response = HTTP.patch(url).multipart(form).send().await?;
        let status = response.status();
        if status.is_success() {
            return Ok::<(), Error>(());
        }
        Err(format!("{status}: {}", response.text().await.unwrap_or_default()).into())
    };
    match sent.await {
        Ok(()) => Ok(()),
        Err(e) => {
            warn!(error = %e, command = %ctx.command().name, "Components V2 reply refused; sending it plainly");
            reply(ctx, plain).await
        }
    }
}

/// For `reply_v2`'s requests: one connection pool, a timeout like serenity's.
static HTTP: std::sync::LazyLock<reqwest::Client> = std::sync::LazyLock::new(|| {
    reqwest::Client::builder().timeout(std::time::Duration::from_secs(30)).build().unwrap_or_default()
});

/// How long a command may fetch and decode a picture before it must defer
/// (Discord wants a response within 3 s).
pub(super) const FETCH_BEFORE_DEFER: std::time::Duration = std::time::Duration::from_secs(2);

/// The attachment downloaded into memory (it never touches the disk), decoded
/// within `limits` and re-encoded as the JPEG to repost; or why not, worded
/// for the sender.
pub(super) async fn fetch_picture(
    picture: &serenity::Attachment,
    limits: ImageLimits,
) -> Result<(::image::RgbImage, Vec<u8>), String> {
    let bytes = picture.download().await.map_err(|e| {
        warn!(error = %e, "picture download failed");
        "couldn't fetch the picture".to_owned()
    })?;
    let decode = move || {
        let rgb = crate::classify::image::load(&bytes, &limits)?;
        let jpeg = crate::classify::image::to_jpeg(&rgb);
        Ok::<_, crate::classify::image::Rejected>((rgb, jpeg))
    };
    match tokio::task::spawn_blocking(decode).await {
        Ok(decoded) => decoded.map_err(|why| why.to_string()),
        Err(e) => {
            warn!(error = %e, "picture decode task failed");
            Err("the picture could not be read".to_owned())
        }
    }
}

/// Runs `run` on the classifier in its turn: P(yes) of each question, or
/// `None` once the failure has been said (`hold` released).
async fn decide(
    ctx: Context<'_>,
    classify: &Classify,
    hold: Option<super::limits::Hold>,
    what: &'static str,
    size: usize,
    run: impl FnOnce(&Classifier) -> anyhow::Result<Vec<Decision>> + Send + 'static,
) -> Result<Option<Vec<f64>>, Error> {
    let decisions = in_turn(ctx, classify, hold, what, size, run).await?;
    Ok(decisions.map(|d| d.iter().map(|d| d.probabilities[0]).collect()))
}

/// Runs `run` on the model in its turn, off the async threads: its result, or
/// `None` once the failure has been said (`hold` released). `size` (tokens,
/// pixels) goes to the log.
pub(super) async fn in_turn<M: Send + Sync + 'static, T: Send + 'static>(
    ctx: Context<'_>,
    gated: &Gated<M>,
    hold: Option<super::limits::Hold>,
    what: &'static str,
    size: usize,
    run: impl FnOnce(&M) -> anyhow::Result<T> + Send + 'static,
) -> Result<Option<T>, Error> {
    let data = ctx.data();
    let result = {
        let _turn = gated.turn.acquire().await?;
        let model = gated.model.clone();
        tokio::task::spawn_blocking(move || {
            let start = std::time::Instant::now();
            let out = run(&model);
            (out, start.elapsed())
        })
        .await
    };
    match result {
        Ok((Ok(out), took)) => {
            info!(size, took_ms = took.as_millis() as u64, "{what}");
            Ok(Some(out))
        }
        Ok((Err(e), _)) => {
            warn!(error = %e, "{what} failed");
            data.limit_release(hold);
            ctx.say(format!("{NOTICE_PREFIX}{what} failed)*")).await?;
            Ok(None)
        }
        Err(e) => {
            data.limit_release(hold);
            Err(e.into())
        }
    }
}

/// Sends `out` with every ping turned off: the asker's mention only names them,
/// and nothing in a reposted text pings anyone.
pub(super) async fn reply(ctx: Context<'_>, out: poise::CreateReply) -> Result<(), Error> {
    ctx.send(out.allowed_mentions(serenity::CreateAllowedMentions::new())).await?;
    Ok(())
}

/// Discord's limit on a message's text.
pub(super) const MESSAGE_CHARS: usize = 2000;

/// Questions per call. At 300 characters each, all of them with their
/// verdicts and the asker still fit in one message ([`MESSAGE_CHARS`]).
const MAX_QUESTIONS: usize = 5;

/// An optional question, absent as empty.
fn or(question: Option<String>) -> String {
    question.unwrap_or_default()
}

/// The questions given, trimmed, without the empty ones.
fn questions(given: [String; MAX_QUESTIONS]) -> Vec<String> {
    given.into_iter().map(|q| one_line(&q)).filter(|q| !q.is_empty()).collect()
}

/// `questions` for d1, each answered on its own (they share the prompt up to
/// the question, not each other's answers).
fn asked(questions: &[String]) -> Vec<(String, Question)> {
    questions
        .iter()
        .enumerate()
        .map(|(i, q)| (format!("q{i}"), Question::Noul { instructions: q.clone(), criteria: None }))
        .collect()
}

/// "@asker asked about:", naming them without a ping (see [`reply`]).
fn asker(ctx: Context<'_>) -> String {
    format!("<@{}> asked about:", ctx.author().id)
}

/// Each question with its verdict.
fn answers(questions: &[String], yes: &[f64]) -> String {
    let lines: Vec<String> =
        questions.iter().zip(yes).map(|(q, &p)| format!("**{q}**\n{}", verdict(p))).collect();
    lines.join("\n")
}

/// `text` as a block quote that ends with it: every line behind `> `.
pub(super) fn block_quote(text: &str) -> String {
    text.lines().map(|l| format!("> {l}")).collect::<Vec<_>>().join("\n")
}

/// The sender's file name without its extension, kept to safe characters
/// (it becomes the reposted attachment's name).
pub(super) fn file_stem(name: &str) -> String {
    let stem = name.rsplit_once('.').map_or(name, |(stem, _)| stem);
    let safe: String = stem
        .chars()
        .filter(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_'))
        .take(60)
        .collect();
    if safe.is_empty() { "picture".to_owned() } else { safe }
}

/// "Yes (87%)" and the like, by P(yes).
fn verdict(yes: f64) -> String {
    let pct = (yes * 100.0).round();
    let word = match yes {
        p if p >= 0.8 => "Yes",
        p if p >= 0.6 => "Probably yes",
        p if p > 0.4 => "Unsure",
        p if p > 0.2 => "Probably no",
        _ => "No",
    };
    format!("{word} — P(yes) = {pct}%")
}

pub(super) fn one_line(s: &str) -> String {
    s.split_whitespace().collect::<Vec<_>>().join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn verdict_bands_and_file_names() {
        assert_eq!(verdict(0.873), "Yes — P(yes) = 87%");
        assert_eq!(verdict(0.5), "Unsure — P(yes) = 50%");
        assert_eq!(verdict(0.01), "No — P(yes) = 1%");
        assert_eq!(file_stem("my cat (1).PNG"), "mycat1");
        assert_eq!(file_stem("../../etc/passwd"), "picture");
        assert_eq!(file_stem(".png"), "picture");
        assert_eq!(file_stem("ünï.webp"), "n");
        assert_eq!(file_stem(&"a".repeat(100)).len(), 60);
        let qs = questions(["  Is it\nred? ".into(), String::new(), " Is it big?".into(), " ".into(), "Is it?".into()]);
        assert_eq!(qs, ["Is it red?", "Is it big?", "Is it?"]);
        assert_eq!(asked(&qs).len(), 3);
        // The longest reply without a text: the mention, five full questions, five verdicts.
        let verdicts = "\n**".len() + 300 + "**\n".len() + verdict(0.65).len();
        assert!("<@18446744073709551615> asked about:".len() + MAX_QUESTIONS * verdicts <= MESSAGE_CHARS);
        assert_eq!(block_quote("one\n\ntwo\r\n"), "> one\n> \n> two");
        assert_eq!(answers(&qs[..2], &[0.9, 0.1]), "**Is it red?**\nYes — P(yes) = 90%\n**Is it big?**\nNo — P(yes) = 10%");
    }
}
