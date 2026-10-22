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

/// The d1 classifier, one call at a time: each uses every compute thread (or
/// the GPU) on its own.
pub struct Classify {
    pub classifier: Arc<Classifier>,
    pub turn: tokio::sync::Semaphore,
}

impl Classify {
    pub fn new(classifier: Classifier) -> Self {
        Self {
            classifier: Arc::new(classifier),
            turn: tokio::sync::Semaphore::new(1),
        }
    }
}

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
    let tokens = classify.classifier.tokenizer().encode(&text).map_or(usize::MAX, |t| t.len());
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
    let Some(classify) = data.classify.clone().filter(|c| c.classifier.has_vision()) else {
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
        max_bytes: k.max_image_bytes,
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

/// The deferred response, edited into a Components V2 message (which serenity
/// 0.12 has no builders for): the asker, the picture, then `answers`. A plain
/// message shows attachments under its text, and an embed's picture can't be
/// spoilered.
async fn reply_with_picture(ctx: Context<'_>, jpeg: Vec<u8>, name: &str, answers: &str) -> Result<(), Error> {
    let poise::Context::Application(app) = ctx else {
        unreachable!("/classify-image is a slash command")
    };
    const IS_COMPONENTS_V2: u64 = 1 << 15;
    const TEXT_DISPLAY: u8 = 10;
    const MEDIA_GALLERY: u8 = 12;
    let body = serde_json::json!({
        "flags": IS_COMPONENTS_V2,
        "allowed_mentions": { "parse": [] },
        "attachments": [{ "id": 0, "filename": name }],
        "components": [
            { "type": TEXT_DISPLAY, "content": asker(ctx) },
            { "type": MEDIA_GALLERY, "items": [
                { "media": { "url": format!("attachment://{name}") }, "spoiler": true },
            ] },
            { "type": TEXT_DISPLAY, "content": answers },
        ],
    });
    let file = serenity::CreateAttachment::bytes(jpeg.clone(), name);
    match ctx.http().edit_original_interaction_response(&app.interaction.token, &body, vec![file]).await {
        Ok(_) => Ok(()),
        Err(e) => {
            // Still one reply, with the picture under the text.
            warn!(error = %e, "classify-image Components V2 reply refused; sending it plainly");
            let plain = poise::CreateReply::default()
                .content(format!("{}\n{answers}", asker(ctx)))
                .attachment(serenity::CreateAttachment::bytes(jpeg, name));
            reply(ctx, plain).await
        }
    }
}

/// How long `/classify-image` may fetch and decode before it must defer
/// (Discord wants a response within 3 s).
const FETCH_BEFORE_DEFER: std::time::Duration = std::time::Duration::from_secs(2);

/// The attachment downloaded into memory (it never touches the disk), decoded
/// within `limits` and re-encoded as the JPEG to repost; or why not, worded
/// for the sender.
async fn fetch_picture(
    picture: &serenity::Attachment,
    limits: ImageLimits,
) -> Result<(::image::RgbImage, Vec<u8>), String> {
    let bytes = picture.download().await.map_err(|e| {
        warn!(error = %e, "classify-image download failed");
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
            warn!(error = %e, "classify-image decode task failed");
            Err("the picture could not be read".to_owned())
        }
    }
}

/// Runs `run` on the classifier in its turn, off the async threads: P(yes) of
/// each question, or `None` once the failure has been said (`hold`
/// released). `size` (tokens, pixels) goes to the log.
async fn decide(
    ctx: Context<'_>,
    classify: &Classify,
    hold: Option<super::limits::Hold>,
    what: &'static str,
    size: usize,
    run: impl FnOnce(&Classifier) -> anyhow::Result<Vec<Decision>> + Send + 'static,
) -> Result<Option<Vec<f64>>, Error> {
    let data = ctx.data();
    let result = {
        let _turn = classify.turn.acquire().await?;
        let classifier = classify.classifier.clone();
        tokio::task::spawn_blocking(move || {
            let start = std::time::Instant::now();
            let out = run(&classifier);
            (out, start.elapsed())
        })
        .await
    };
    match result {
        Ok((Ok(decisions), took)) => {
            info!(size, took_ms = took.as_millis() as u64, "{what}");
            Ok(Some(decisions.iter().map(|d| d.probabilities[0]).collect()))
        }
        Ok((Err(e), _)) => {
            warn!(error = %e, "{what} failed");
            data.limit_release(hold);
            ctx.say(format!("{NOTICE_PREFIX}classification failed)*")).await?;
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
async fn reply(ctx: Context<'_>, out: poise::CreateReply) -> Result<(), Error> {
    ctx.send(out.allowed_mentions(serenity::CreateAllowedMentions::new())).await?;
    Ok(())
}

/// Discord's limit on a message's text.
const MESSAGE_CHARS: usize = 2000;

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
fn block_quote(text: &str) -> String {
    text.lines().map(|l| format!("> {l}")).collect::<Vec<_>>().join("\n")
}

/// The sender's file name without its extension, kept to safe characters
/// (it becomes the reposted attachment's name).
fn file_stem(name: &str) -> String {
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

fn one_line(s: &str) -> String {
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
