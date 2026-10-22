//! d1's picture path after decoding, ported from d1-burn: the runner's pixel
//! cap and transformers' `Lfm2VlImageProcessor` (smart resize, tiles and
//! thumbnail, normalisation, 16-pixel patches), the prompt markup each picture
//! expands into, and the SigLIP2 tower with its projector (`siglip2`).
//!
//! The preprocessing matches the Python reference bit for bit (see `resample`
//! for the two resizers).

pub mod markup;
pub mod resample;
pub mod siglip2;

use image::RgbImage;

use super::prompt::IM_START;
use super::{Error, Json, Question, Renderer, Tokenizer};
use resample::{Flavour, resize};
pub use siglip2::Vision;

/// `runner.VISION_MAX_PIXELS`: every picture is bounded at this many pixels before the processor.
pub const MAX_PIXELS: usize = 1024 * 1024;
/// The `<image>` token's id (`image_token_id` in config.json).
pub const IMAGE_TOKEN_ID: u32 = 124_907;
/// Values per patch: 16 · 16 pixels · 3 channels.
pub const PATCH_VALUES: usize = 16 * 16 * 3;

/// An 8-bit RGB picture, rows top to bottom, `[r, g, b]` per pixel.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Rgb {
    width: usize,
    height: usize,
    data: Vec<u8>,
}

impl Rgb {
    /// A picture of `width × height` pixels from `data` (`3 · width · height` bytes).
    pub fn new(width: usize, height: usize, data: Vec<u8>) -> Result<Self, Error> {
        let fits = |n: usize| n > 0 && u32::try_from(n).is_ok();
        if !fits(width) || !fits(height) {
            return Err(Error::new(format!("a {width}×{height} picture")));
        }
        if width.checked_mul(height).and_then(|n| n.checked_mul(3)) != Some(data.len()) {
            return Err(Error::new(format!(
                "{} bytes for a {width}×{height} RGB picture",
                data.len()
            )));
        }
        Ok(Self { width, height, data })
    }

    /// From a decoded picture (`crate::classify::image::load`).
    pub fn from_image(picture: &RgbImage) -> Result<Self, Error> {
        let (w, h) = picture.dimensions();
        Self::new(w as usize, h as usize, picture.as_raw().clone())
    }

    pub const fn width(&self) -> usize {
        self.width
    }

    pub const fn height(&self) -> usize {
        self.height
    }

    /// The pixels, rows top to bottom, three bytes each.
    pub fn data(&self) -> &[u8] {
        &self.data
    }

    /// The `width × height` block at column `x`, row `y` (inside the picture).
    fn crop(&self, x: usize, y: usize, width: usize, height: usize) -> Self {
        let stride = self.width * 3;
        let data = self
            .data
            .chunks_exact(stride)
            .skip(y)
            .take(height)
            .flat_map(|row| row.get(x * 3..(x + width) * 3).unwrap_or_default())
            .copied()
            .collect();
        Self { width, height, data }
    }
}

/// `n` as f64; every size here is far below 2^32.
fn float(n: usize) -> f64 {
    u32::try_from(n).map_or(f64::INFINITY, f64::from)
}

/// Python's `int(v)` / `math.floor` on a non-negative float, saturating.
fn to_usize(v: f64) -> usize {
    v as usize
}

/// `runner.cap_pixels`: the picture downscaled (Pillow bicubic) to at most `max_pixels`, each side
/// `max(1, int(side · sqrt(max_pixels / (w · h))))`; unchanged if it already fits.
pub fn cap_pixels(image: Rgb, max_pixels: usize) -> Rgb {
    let (w, h) = (image.width, image.height);
    if w * h <= max_pixels {
        return image;
    }
    let scale = (float(max_pixels) / float(w * h)).sqrt();
    let width = to_usize(float(w) * scale).max(1);
    let height = to_usize(float(h) * scale).max(1);
    resize(&image, width, height, Flavour::Pillow)
}

/// The image processor's settings: d1's `processor_config.json` by default.
#[derive(Debug, Clone, PartialEq)]
pub struct Config {
    /// `tile_size`: each tile is this many pixels square.
    pub tile_size: usize,
    /// `min_tiles`, `max_tiles`: the tile grid's bounds (`do_image_splitting` is on).
    pub min_tiles: usize,
    pub max_tiles: usize,
    /// `min_image_tokens`, `max_image_tokens`: the token range of a single image or thumbnail.
    pub min_image_tokens: usize,
    pub max_image_tokens: usize,
    /// `encoder_patch_size`.
    pub patch: usize,
    /// `downsample_factor`.
    pub downsample: usize,
    /// `use_thumbnail`.
    pub use_thumbnail: bool,
    /// `max_pixels_tolerance`.
    pub max_pixels_tolerance: f64,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            tile_size: 512,
            min_tiles: 2,
            max_tiles: 10,
            min_image_tokens: 64,
            max_image_tokens: 256,
            patch: 16,
            downsample: 2,
            use_thumbnail: true,
            max_pixels_tolerance: 2.0,
        }
    }
}

/// The candidate grids `(columns, rows)` for 2 to 10 tiles, in the order the processor's
/// `_target_ratios(2, 10)` gives them (sorted by tile count from a Python set): the order decides
/// ties in `find_closest_aspect_ratio`.
const RATIOS_2_10: [(usize, usize); 26] = [
    (1, 2),
    (2, 1),
    (3, 1),
    (1, 3),
    (2, 2),
    (4, 1),
    (1, 4),
    (5, 1),
    (1, 5),
    (1, 6),
    (6, 1),
    (3, 2),
    (2, 3),
    (7, 1),
    (1, 7),
    (4, 2),
    (2, 4),
    (1, 8),
    (8, 1),
    (1, 9),
    (3, 3),
    (9, 1),
    (2, 5),
    (5, 2),
    (10, 1),
    (1, 10),
];

/// How the processor laid a picture out: `image_rows`, `image_cols` and `image_sizes`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Layout {
    /// Tile rows and columns (1 × 1 for a picture that isn't split).
    pub rows: usize,
    pub cols: usize,
    /// The smart-resized size `(height, width)`: the whole picture's, or the thumbnail's.
    pub size: (usize, usize),
}

/// One picture through the processor: its tiles (then the thumbnail) as patches.
#[derive(Debug, Clone, PartialEq)]
pub struct Processed {
    pub layout: Layout,
    /// Per tile: the normalised patches, `[patches, PATCH_VALUES]` flattened, and the patch grid
    /// `(rows, columns)`.
    pub tiles: Vec<(Vec<f32>, (usize, usize))>,
}

/// Python's `round(n / factor) * factor` (half to even).
fn round_by_factor(n: usize, factor: usize) -> usize {
    to_usize((float(n) / float(factor)).round_ties_even()) * factor
}

impl Config {
    fn factor(&self) -> usize {
        self.patch * self.downsample
    }

    /// `smart_resize`: `(width, height)`, multiples of `patch · downsample`, within the token range.
    pub fn smart_resize(&self, height: usize, width: usize) -> (usize, usize) {
        let f = self.factor();
        let min_pixels = self.min_image_tokens * self.patch * self.patch * self.downsample.pow(2);
        let max_pixels = self.max_image_tokens * self.patch * self.patch * self.downsample.pow(2);
        let mut h_bar = f.max(round_by_factor(height, f));
        let mut w_bar = f.max(round_by_factor(width, f));
        if h_bar * w_bar > max_pixels {
            let beta = (float(height * width) / float(max_pixels)).sqrt();
            h_bar = f.max(to_usize((float(height) / beta / float(f)).floor()) * f);
            w_bar = f.max(to_usize((float(width) / beta / float(f)).floor()) * f);
        } else if h_bar * w_bar < min_pixels {
            let beta = (float(min_pixels) / float(height * width)).sqrt();
            h_bar = to_usize((float(height) * beta / float(f)).ceil()) * f;
            w_bar = to_usize((float(width) * beta / float(f)).ceil()) * f;
        }
        (w_bar, h_bar)
    }

    /// `_is_image_too_large`: whether the picture is split into tiles.
    pub fn is_too_large(&self, height: usize, width: usize) -> bool {
        let f = self.factor();
        let h_bar = self.patch.max(round_by_factor(height, f));
        let w_bar = self.patch.max(round_by_factor(width, f));
        let limit = self.max_image_tokens * self.patch * self.patch * self.downsample.pow(2);
        float(h_bar * w_bar) > float(limit) * self.max_pixels_tolerance
    }

    /// `find_closest_aspect_ratio` over `_target_ratios`: the grid `(columns, rows)`.
    pub fn grid(&self, height: usize, width: usize) -> (usize, usize) {
        let ratios: Vec<(usize, usize)> = if (self.min_tiles, self.max_tiles) == (2, 10) {
            RATIOS_2_10.to_vec()
        } else {
            // Any other bounds: the same set, sorted by tile count; ties keep (w, h) order.
            let n = self.max_tiles;
            let mut r: Vec<_> = (1..=n)
                .flat_map(|w| (1..=n).map(move |h| (w, h)))
                .filter(|&(w, h)| (self.min_tiles..=self.max_tiles).contains(&(w * h)))
                .collect();
            r.sort_by_key(|&(w, h)| w * h);
            r
        };
        let aspect = float(width) / float(height);
        let area = float(width * height);
        let mut best_diff = f64::INFINITY;
        let mut best = (1, 1);
        for (w, h) in ratios {
            let diff = (aspect - float(w) / float(h)).abs();
            if diff < best_diff {
                best_diff = diff;
                best = (w, h);
            } else if diff == best_diff {
                let target = float(self.tile_size * self.tile_size * w * h);
                if area > 0.5 * target {
                    best = (w, h);
                }
            }
        }
        best
    }

    /// Runs the image processor on one (already capped) picture.
    pub fn process(&self, image: &Rgb) -> Processed {
        let (h, w) = (image.height, image.width);
        let (new_w, new_h) = self.smart_resize(h, w);
        let splits = !(self.min_tiles == 1 && self.max_tiles == 1);
        let torch = |img: &Rgb, w: usize, h: usize| resize(img, w, h, Flavour::Torch);
        let (rows, cols, pictures) = if splits && self.is_too_large(h, w) {
            let (gw, gh) = self.grid(h, w);
            let t = self.tile_size;
            let big = torch(image, t * gw, t * gh);
            let mut tiles: Vec<Rgb> = (0..gh)
                .flat_map(|r| (0..gw).map(move |c| (r, c)))
                .map(|(r, c)| big.crop(c * t, r * t, t, t))
                .collect();
            if self.use_thumbnail && gw * gh != 1 {
                tiles.push(torch(image, new_w, new_h));
            }
            (gh, gw, tiles)
        } else {
            (1, 1, vec![torch(image, new_w, new_h)])
        };
        Processed {
            layout: Layout { rows, cols, size: (new_h, new_w) },
            tiles: pictures.iter().map(|p| self.patches(p)).collect(),
        }
    }

    /// `rescale_and_normalize` (`(v - 127.5) / 127.5` in f32) and `convert_image_to_patches`: patches
    /// rows first, each `[y, x, channel]`.
    fn patches(&self, tile: &Rgb) -> (Vec<f32>, (usize, usize)) {
        let p = self.patch;
        let (ph, pw) = (tile.height / p, tile.width / p);
        let stride = tile.width * 3;
        let mut out = Vec::with_capacity(ph * pw * p * p * 3);
        for band in tile.data.chunks_exact(stride * p).take(ph) {
            for px in 0..pw {
                for row in band.chunks_exact(stride) {
                    let pixels = row.get(px * p * 3..(px + 1) * p * 3).unwrap_or_default();
                    out.extend(pixels.iter().map(|&v| (f32::from(v) - 127.5) / 127.5));
                }
            }
        }
        (out, (ph, pw))
    }
}

/// Each question's token ids for a prompt about `layouts`' pictures, as d1-burn's runner renders
/// them: one `<image>` per picture right after the user header (`prefix_text`), each expanded into
/// its image tokens; then rows as [`Renderer::rows`] (one question encoded whole, several sharing
/// the prefix encoded apart).
///
/// # Errors
///
/// As [`Renderer::rows`].
pub fn rows(
    renderer: &Renderer,
    tokenizer: &Tokenizer,
    layouts: &[Layout],
    state: Option<&Json>,
    questions: &[Question],
) -> Result<Vec<Vec<u32>>, Error> {
    let config = Config::default();
    let mut prefix = renderer.prefix(state)?;
    let head = format!("{IM_START}user\n");
    let at = prefix
        .find(&head)
        .ok_or_else(|| Error::new("no user turn in the prompt"))?
        + head.len();
    prefix.insert_str(at, &markup::image_markup(layouts.len()));
    let encode = |text: &str| tokenizer.encode(&markup::expand(text, layouts, &config)?);
    if let [q] = questions {
        return Ok(vec![encode(&(prefix + &renderer.suffix(tokenizer, q)?))?]);
    }
    let prefix = encode(&prefix)?;
    questions
        .iter()
        .map(|q| {
            let mut row = prefix.clone();
            row.extend(tokenizer.encode(&renderer.suffix(tokenizer, q)?)?);
            Ok(row)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn smart_resize_like_python() {
        let c = Config::default();
        // Values from the processor (prep fixtures' image_sizes).
        assert_eq!(c.smart_resize(480, 640), (576, 416));
        assert_eq!(c.smart_resize(1, 1), (256, 256));
        assert_eq!(c.smart_resize(30, 10000), (9344, 32));
        assert!(!c.is_too_large(480, 640));
        assert!(c.is_too_large(886, 1182));
        assert_eq!(c.grid(886, 1182), (3, 2));
        assert_eq!(c.smart_resize(886, 1182), (576, 416));
    }

    /// Nothing at most 640 on a side is ever split into tiles.
    #[test]
    fn pictures_within_640_are_one_tile() {
        let c = Config::default();
        for (h, w) in [(640, 640), (480, 640), (640, 1), (1, 640), (360, 640)] {
            assert!(!c.is_too_large(h, w), "{h}×{w}");
        }
    }

    #[test]
    fn rejects_bad_pictures() {
        assert!(Rgb::new(0, 3, vec![]).is_err());
        assert!(Rgb::new(2, 2, vec![0; 11]).is_err());
    }

    #[test]
    fn patches_are_rows_then_columns() {
        let c = Config::default();
        // 32 × 16, red = the column: two patches side by side, each [y, x, channel].
        let data = (0..16).flat_map(|_| (0..32_u8).flat_map(|x| [x, 7, 9])).collect();
        let (v, shape) = c.patches(&Rgb::new(32, 16, data).unwrap());
        assert_eq!(shape, (1, 2));
        assert_eq!(v.len(), 2 * PATCH_VALUES);
        let norm = |x: u8| (f32::from(x) - 127.5) / 127.5;
        assert_eq!(v[3], norm(1));
        assert_eq!(v[4], norm(7));
        assert_eq!(v[3 * 16], norm(0));
        assert_eq!(v[PATCH_VALUES], norm(16));
    }
}
