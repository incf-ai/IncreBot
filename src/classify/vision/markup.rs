//! The prompt text for pictures: the runner's `<image>` markup and the processor's expansion of
//! each `<image>` into image tokens with row/column and thumbnail tags
//! (`Lfm2VlProcessor._build_image_tokens`).

use super::{Config, Layout};
use crate::classify::Error;

/// The placeholder the chat template writes per picture.
pub const IMAGE: &str = "<image>";
const START: &str = "<|image_start|>";
const END: &str = "<|image_end|>";
const THUMBNAIL: &str = "<|img_thumbnail|>";

/// `runner._image_markup(n)`: what the chat template writes for `n` pictures at the head of the
/// user turn (one `<image>` each on LFM2-VL).
#[must_use]
pub fn image_markup(images: usize) -> String {
    IMAGE.repeat(images)
}

/// Image tokens for a tile, and for the whole picture or thumbnail (`_get_image_num_tokens`).
#[must_use]
pub fn token_counts(layout: &Layout, config: &Config) -> (usize, usize) {
    let d = config.downsample;
    let tile = (config.tile_size / config.patch).div_ceil(d).pow(2);
    let (h, w) = layout.size;
    let image = (h / config.patch).div_ceil(d) * (w / config.patch).div_ceil(d);
    (tile, image)
}

/// The text one picture's `<image>` becomes (`replace_image_token`, image special tokens on).
#[must_use]
pub fn expansion(layout: &Layout, config: &Config) -> String {
    let (per_tile, image) = token_counts(layout, config);
    let mut out = String::from(START);
    if layout.rows > 1 || layout.cols > 1 {
        for row in 1..=layout.rows {
            for col in 1..=layout.cols {
                out.push_str(&format!("<|img_row_{row}_col_{col}|>"));
                out.push_str(&IMAGE.repeat(per_tile));
            }
        }
        if config.use_thumbnail {
            out.push_str(THUMBNAIL);
            out.push_str(&IMAGE.repeat(image));
        }
    } else {
        out.push_str(&IMAGE.repeat(image));
    }
    out.push_str(END);
    out
}

/// `get_text_with_replacements`: `text` with its `i`-th `<image>` replaced by picture `i`'s
/// expansion.
///
/// # Errors
///
/// If `text` doesn't hold exactly one `<image>` per layout.
pub fn expand(text: &str, layouts: &[Layout], config: &Config) -> Result<String, Error> {
    let parts: Vec<&str> = text.split(IMAGE).collect();
    if parts.len() != layouts.len() + 1 {
        return Err(Error::new(format!(
            "{} {IMAGE} in the text for {} pictures",
            parts.len().saturating_sub(1),
            layouts.len()
        )));
    }
    let mut out = String::with_capacity(text.len());
    let mut layouts = layouts.iter();
    for (i, part) in parts.iter().enumerate() {
        if i > 0
            && let Some(layout) = layouts.next()
        {
            out.push_str(&expansion(layout, config));
        }
        out.push_str(part);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_and_tiled() {
        let c = Config::default();
        let one = Layout {
            rows: 1,
            cols: 1,
            size: (448, 576),
        };
        assert_eq!(token_counts(&one, &c), (256, 14 * 18));
        let e = expansion(&one, &c);
        assert!(e.starts_with("<|image_start|><image>") && e.ends_with("<image><|image_end|>"));
        assert_eq!(e.matches(IMAGE).count(), 252);
        let tiled = Layout {
            rows: 2,
            cols: 3,
            size: (384, 512),
        };
        let e = expansion(&tiled, &c);
        assert_eq!(e.matches(IMAGE).count(), 6 * 256 + 12 * 16);
        assert!(e.contains("<|img_row_2_col_3|>") && e.contains(THUMBNAIL));
        let text = expand("a<image>b<image>c", &[one, tiled], &c).unwrap();
        assert!(text.starts_with("a<|image_start|>") && text.ends_with("<|image_end|>c"));
        assert!(expand("<image>", &[], &c).is_err());
        assert_eq!(image_markup(2), "<image><image>");
    }
}
