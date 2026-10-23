//! `/describe-image`: Qwen3-VL-8B describes a picture, or answers a question
//! about it (`crate::describe`). Billed like `/ask`. The reply reposts the
//! picture (spoilered) at the top, then names the asker (without a ping), quotes
//! the question and gives the answer, which is not spoilered. A long answer
//! goes in a previewed `description.txt`, under the text and the picture.

use std::sync::atomic::Ordering;

use poise::serenity_prelude as serenity;

use super::classify::{
    FETCH_BEFORE_DEFER, Gated, MESSAGE_CHARS, block_quote, fetch_picture, file_stem, gallery, in_turn, reply,
    reply_v2, text_display,
};
use super::{Context, Error, NOTICE_PREFIX, notice};
use crate::classify::image::{ImageLimits, Rejected};
use crate::classify::vision::Rgb;
use crate::describe::{Description, Describer};

/// Qwen3-VL.
pub type Describe = Gated<Describer>;

/// Pictures are scaled to at most this on the longer side when fetched; the
/// model's own resize (to `describe_max_image_tokens`) takes them smaller still.
const IMAGE_MAX_SIDE: u32 = 2048;

/// Longer answers (in tokens, or in lines with text) go in an attached file.
const INLINE_TOKENS: usize = 100;
const INLINE_LINES: usize = 3;

/// Describe a picture, or ask about it.
#[poise::command(
    slash_command,
    rename = "describe-image",
    install_context = "Guild",
    interaction_context = "Guild|BotDm"
)]
pub async fn describe_image(
    ctx: Context<'_>,
    #[description = "The picture (PNG, JPEG or WebP)"] image: serenity::Attachment,
    #[description = "A question about it, or what to describe (default: describe it)"]
    #[max_length = 4000]
    prompt: Option<String>,
) -> Result<(), Error> {
    let data = ctx.data().clone();
    let Some(describe) = data.describe.clone() else {
        return notice(ctx, format!("{NOTICE_PREFIX}describing pictures is turned off here)*")).await;
    };
    if data.shutting_down.load(Ordering::Relaxed) {
        return notice(ctx, format!("{NOTICE_PREFIX}restarting, ask again shortly)*")).await;
    }
    let prompt = prompt.map(|p| p.trim().to_owned()).unwrap_or_default();
    let limit = data.cfg.describe.describe_max_prompt_chars;
    let chars = prompt.chars().count();
    if chars > limit {
        return notice(ctx, format!("{NOTICE_PREFIX}the question is {chars} characters; the limit is {limit})*")).await;
    }
    if let Err(why) = describe.model.prompt_tokens(&prompt) {
        return notice(ctx, format!("{NOTICE_PREFIX}{why})*")).await;
    }
    let k = &data.cfg.classify;
    let limits = ImageLimits { max_bytes: k.max_image_mib << 20, max_pixels: k.max_image_pixels, max_side: IMAGE_MAX_SIDE };
    // Discord's size, before fetching anything; `load` checks the bytes again.
    let size = u64::from(image.size);
    if size > limits.max_bytes {
        let why = Rejected::TooLarge { bytes: size, max: limits.max_bytes };
        return notice(ctx, format!("{NOTICE_PREFIX}{why})*")).await;
    }
    let hold = match data.limit_try_hold(ctx.author().id.get(), 1.0) {
        Ok(hold) => hold,
        Err(why) => return notice(ctx, why).await,
    };

    // Fetched and decoded before deferring when that's quick, so a bad file's
    // notice stays ephemeral (after a public defer it can't be).
    let fetch = fetch_picture(&image, limits);
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

    let question = prompt.clone();
    let shared = data.clone();
    let run = move |m: &Describer| {
        let stop = || shared.shutting_down.load(Ordering::Relaxed);
        m.describe(&Rgb::from_image(&rgb)?, &question, &stop)
    };
    let Some(d) = in_turn(ctx, &describe, hold, "describe-image", pixels, run).await? else {
        return Ok(());
    };
    tracing::info!(
        image_tokens = d.image_tokens,
        generated = d.generated,
        vision_s = d.vision_secs,
        prefill_s = d.prefill_secs,
        decode_s = d.decode_secs,
        "described"
    );

    // The picture first; then who asked what, and the answer.
    let header = header(&format!("<@{}>", ctx.author().id), &prompt);
    let answer = answer(&d);
    let name = format!("SPOILER_{}.jpg", file_stem(&image.filename));
    let picture = serenity::CreateAttachment::bytes(jpeg, name.as_str());
    let text = serenity::CreateAttachment::bytes(d.text.clone().into_bytes(), "description.txt");

    if !(short(&d) && fits(&header, &answer, MESSAGE_CHARS)) {
        // A plain message, so Discord previews the text file (a Components V2
        // file is only a download); the picture shows under the text.
        let note = attached(&d);
        let out = poise::CreateReply::default().content(format!("{header}\n{note}")).attachment(picture).attachment(text);
        return reply(ctx, out).await;
    }
    // A plain message shows its attachments under its text.
    let plain = poise::CreateReply::default().content(format!("{header}\n{answer}")).attachment(picture.clone());
    let components = vec![gallery(&[&name]), text_display(&header), text_display(&answer)];
    reply_v2(ctx, components, vec![picture], plain).await
}

/// "@asker asked:" and the question quoted, or that they asked for a description.
fn header(asker: &str, prompt: &str) -> String {
    if prompt.is_empty() {
        format!("{asker} asked for a description:")
    } else {
        format!("{asker} asked:\n{}", block_quote(prompt))
    }
}

/// The answer, and a note when it ran out of tokens.
fn answer(d: &Description) -> String {
    let text = if d.text.is_empty() { "*(no answer)*" } else { d.text.as_str() };
    if d.truncated { format!("{text}\n-# (cut off at the length limit)") } else { text.to_owned() }
}

/// Whether the answer is short enough to show in the message: at most
/// [`INLINE_TOKENS`] tokens and [`INLINE_LINES`] lines with text.
fn short(d: &Description) -> bool {
    d.generated <= INLINE_TOKENS && d.text.lines().filter(|l| !l.trim().is_empty()).count() <= INLINE_LINES
}

/// What the message says in place of an attached answer.
fn attached(d: &Description) -> String {
    let cut = if d.truncated { ", cut off at the length limit" } else { "" };
    let s = if d.generated == 1 { "" } else { "s" };
    format!("*(the answer is attached: {} token{s}{cut})*", d.generated)
}

/// Whether the header and the answer fit in one message of `limit` characters.
fn fits(header: &str, answer: &str, limit: usize) -> bool {
    header.chars().count() + 1 + answer.chars().count() <= limit
}

#[cfg(test)]
mod tests {
    use super::*;

    fn described(text: &str, truncated: bool) -> Description {
        Description {
            text: text.into(),
            tokens: Vec::new(),
            image_tokens: 0,
            grid: (0, 0),
            prompt_tokens: 0,
            generated: text.split_whitespace().count(),
            truncated,
            vision_secs: 0.0,
            prefill_secs: 0.0,
            decode_secs: 0.0,
        }
    }

    #[test]
    fn headers_and_answers() {
        assert_eq!(header("<@1>", ""), "<@1> asked for a description:");
        assert_eq!(header("<@1>", "What is it?\nReally?"), "<@1> asked:\n> What is it?\n> Really?");
        assert_eq!(answer(&described("Two cats.", false)), "Two cats.");
        assert_eq!(answer(&described("Two", true)), "Two\n-# (cut off at the length limit)");
        assert_eq!(answer(&described("", false)), "*(no answer)*");
        assert!(fits("a", "b", 3) && !fits("a", "bc", 3));
    }

    #[test]
    fn long_answers_are_attached() {
        assert!(short(&described("One.\n\nTwo.\nThree.", false)));
        assert!(!short(&described("One.\nTwo.\nThree.\nFour.", false)));
        assert!(short(&described(&"word ".repeat(100), false)));
        assert!(!short(&described(&"word ".repeat(101), false)));
        assert_eq!(attached(&described("a b", false)), "*(the answer is attached: 2 tokens)*");
        assert_eq!(attached(&described("a", true)), "*(the answer is attached: 1 token, cut off at the length limit)*");
    }
}
