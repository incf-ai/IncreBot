//! `/embed-compare`: how alike up to two pictures and two texts are, by the
//! cosine similarity of their EmbeddingGemma 2 embeddings (`crate::embed`).
//! Billed like `/ask`. The reply reposts the pictures (spoilered) at the top,
//! names the asker (without a ping), quotes or attaches the texts and shows the
//! pairs as `embed::matrix` lays them out.

use std::sync::atomic::Ordering;

use poise::serenity_prelude as serenity;

use super::classify::{
    FETCH_BEFORE_DEFER, Gated, MESSAGE_CHARS, block_quote, fetch_picture, file, file_stem, gallery, in_turn,
    reply, reply_v2, text_display,
};
use super::{Context, Error, NOTICE_PREFIX, notice};
use crate::classify::image::{ImageLimits, Rejected};
use crate::classify::vision::Rgb;
use crate::embed::{Embedder, MAX_SOFT_TOKENS, matrix};

/// EmbeddingGemma 2.
pub type Embed = Gated<Embedder>;

/// Compare up to two pictures and two texts.
#[poise::command(
    slash_command,
    rename = "embed-compare",
    install_context = "Guild",
    interaction_context = "Guild|BotDm"
)]
pub async fn embed_compare(
    ctx: Context<'_>,
    #[description = "A picture (PNG, JPEG or WebP)"] image: Option<serenity::Attachment>,
    #[description = "Another picture"] image2: Option<serenity::Attachment>,
    #[description = "A text"]
    #[max_length = 6000]
    text: Option<String>,
    #[description = "Another text"]
    #[max_length = 6000]
    text2: Option<String>,
) -> Result<(), Error> {
    let data = ctx.data().clone();
    let Some(embed) = data.embed.clone() else {
        return notice(ctx, format!("{NOTICE_PREFIX}comparing is turned off here)*")).await;
    };
    if data.shutting_down.load(Ordering::Relaxed) {
        return notice(ctx, format!("{NOTICE_PREFIX}restarting, ask again shortly)*")).await;
    }
    let pictures: Vec<serenity::Attachment> = [image, image2].into_iter().flatten().collect();
    let texts: Vec<String> =
        [text, text2].into_iter().flatten().map(|t| t.trim().to_owned()).filter(|t| !t.is_empty()).collect();
    if pictures.len() + texts.len() < 2 {
        return notice(ctx, format!("{NOTICE_PREFIX}needs two things to compare: pictures, texts or both)*")).await;
    }
    let limit = data.cfg.embed.embed_max_text_tokens;
    let mut tokens = 0;
    for t in &texts {
        let n = embed.model.tokens(t).map_or(usize::MAX, |t| t.len());
        if n > limit {
            return notice(ctx, format!("{NOTICE_PREFIX}a text is {n} tokens; the limit is {limit})*")).await;
        }
        tokens += n;
    }
    let k = &data.cfg.classify;
    let limits = ImageLimits {
        max_bytes: k.max_image_mib << 20,
        max_pixels: k.max_image_pixels,
        max_side: data.cfg.embed.embed_image_max_side,
    };
    // Discord's sizes, before fetching anything; `load` checks the bytes again.
    for p in &pictures {
        let size = u64::from(p.size);
        if size > limits.max_bytes {
            let why = Rejected::TooLarge { bytes: size, max: limits.max_bytes };
            return notice(ctx, format!("{NOTICE_PREFIX}{why})*")).await;
        }
    }
    let hold = match data.limit_try_hold(ctx.author().id.get(), 1.0) {
        Ok(hold) => hold,
        Err(why) => return notice(ctx, why).await,
    };

    // Fetched and decoded before deferring when that's quick, so a bad file's
    // notice stays ephemeral (after a public defer it can't be).
    let fetch = async {
        let mut fetched = Vec::new();
        for p in &pictures {
            fetched.push(fetch_picture(p, limits).await?);
        }
        Ok::<_, String>(fetched)
    };
    tokio::pin!(fetch);
    let (fetched, deferred) = match tokio::time::timeout(FETCH_BEFORE_DEFER, &mut fetch).await {
        Ok(fetched) => (fetched, false),
        Err(_) => {
            ctx.defer().await?;
            (fetch.await, true)
        }
    };
    let fetched = match fetched {
        Ok(fetched) => fetched,
        Err(why) => {
            data.limit_release(hold);
            return notice(ctx, format!("{NOTICE_PREFIX}{why})*")).await;
        }
    };
    if !deferred {
        ctx.defer().await?;
    }
    let (rgbs, jpegs): (Vec<_>, Vec<_>) = fetched.into_iter().unzip();

    let to_embed = texts.clone();
    let run = move |m: &Embedder| {
        let images = rgbs.iter().map(|p| m.embed_image(&Rgb::from_image(p)?)).collect::<anyhow::Result<Vec<_>>>()?;
        let texts = to_embed.iter().map(|t| m.embed_text(t)).collect::<anyhow::Result<Vec<_>>>()?;
        Ok((images, texts))
    };
    // About how many tokens the text tower runs over.
    let size = tokens + pictures.len() * MAX_SOFT_TOKENS;
    let Some((images, embedded)) = in_turn(ctx, &embed, hold, "embed-compare", size, run).await? else {
        return Ok(());
    };
    let Some(table) = matrix::table(&images, &embedded) else {
        unreachable!("two or more things were compared")
    };
    let result = format!("```\n{table}\n```\n{}", footnote(!images.is_empty() && !embedded.is_empty()));

    // The pictures first, side by side; then who compared them, the texts
    // (quoted, or attached when they don't fit in one message) and the table.
    let who = format!("<@{}> compared", ctx.author().id);
    let header = match pictures.len() {
        0 => format!("{who}:"),
        1 => format!("{who} **I1** (above):"),
        _ => format!("{who} **I1** (left) and **I2** (right):"),
    };
    let quoted = quote(&texts);
    let inline = header.chars().count() + quoted.chars().count() + result.chars().count() + 2 <= MESSAGE_CHARS;
    let shown = if inline { quoted } else { "*(the texts are attached)*".to_owned() };
    let mut files: Vec<serenity::CreateAttachment> = jpegs
        .into_iter()
        .zip(&pictures)
        .enumerate()
        .map(|(i, (jpeg, p))| {
            serenity::CreateAttachment::bytes(jpeg, format!("SPOILER_I{}_{}.jpg", i + 1, file_stem(&p.filename)))
        })
        .collect();
    let images: Vec<String> = files.iter().map(|f| f.filename.clone()).collect();
    if !inline {
        files.extend(texts.iter().enumerate().map(|(i, t)| {
            serenity::CreateAttachment::bytes(t.clone().into_bytes(), format!("T{}.txt", i + 1))
        }));
    }
    // Discord puts a plain message's attachments under its text.
    let plain_header = if images.is_empty() { format!("{who}:") } else { format!("{who} (pictures below):") };
    let mut plain = poise::CreateReply::default().content(
        [plain_header.as_str(), &shown, &result].into_iter().filter(|s| !s.is_empty()).collect::<Vec<_>>().join("\n"),
    );
    for f in &files {
        plain = plain.attachment(f.clone());
    }
    if images.is_empty() {
        return reply(ctx, plain).await;
    }

    let names: Vec<&str> = images.iter().map(String::as_str).collect();
    let mut components = vec![gallery(&names), text_display(&header)];
    if inline {
        if !shown.is_empty() {
            components.push(text_display(&shown));
        }
    } else {
        components.extend(files[images.len()..].iter().map(|f| file(&f.filename)));
    }
    components.push(text_display(&result));
    reply_v2(ctx, components, files, plain).await
}

/// Each text as `**T1**` and a block quote.
fn quote(texts: &[String]) -> String {
    let parts: Vec<String> =
        texts.iter().enumerate().map(|(i, t)| format!("**T{}**\n{}", i + 1, block_quote(t))).collect();
    parts.join("\n")
}

/// How to read the scores, which are compressed and never near 0 for real inputs.
fn footnote(mixed: bool) -> String {
    let mut s = "-# Cosine similarity: higher is more alike. Read the order, not the size: \
                 even unrelated things score well above 0"
        .to_owned();
    if mixed {
        s.push_str(", and picture↔text pairs score lower than text↔text ones");
    }
    s.push('.');
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quotes_and_footnotes() {
        assert_eq!(quote(&[]), "");
        assert_eq!(quote(&["a\nb".into(), "c".into()]), "**T1**\n> a\n> b\n**T2**\n> c");
        assert!(footnote(false).ends_with("well above 0."));
        assert!(footnote(true).ends_with("than text↔text ones."));
        // All four take a small part of a message, leaving the rest for the texts.
        let one = [vec![1.0], vec![1.0]];
        let table = matrix::table(&one, &one).unwrap();
        assert!(table.chars().count() + footnote(true).chars().count() < MESSAGE_CHARS / 4);
    }
}
