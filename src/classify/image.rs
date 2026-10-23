//! Untrusted pictures in, small RGB out, never touching disk: PNG, JPEG or
//! WebP only, judged by their bytes; the size, then the header's dimensions,
//! checked before any pixel is decoded; the decoders' allocations capped; a
//! decoder panic caught as an error. EXIF rotation is applied, then the picture
//! is scaled down so its longer side is at most `max_side`.

use std::io::Cursor;
use std::panic::{AssertUnwindSafe, catch_unwind};

use image::{DynamicImage, ImageDecoder, ImageFormat, ImageReader, Limits, RgbImage};
use image::imageops::FilterType;

/// What a picture may be.
#[derive(Clone, Copy, Debug)]
pub struct ImageLimits {
    pub max_bytes: u64,
    pub max_pixels: u64,
    pub max_side: u32,
}

/// Why a picture was turned away, worded for the person who sent it.
#[derive(Debug, PartialEq, Eq)]
pub enum Rejected {
    TooLarge { bytes: u64, max: u64 },
    TooManyPixels { width: u32, height: u32, max: u64 },
    Format,
    Corrupt,
}

impl std::fmt::Display for Rejected {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::TooLarge { bytes, max } => write!(f, "the file is {} MiB; the limit is {} MiB", mib(*bytes), mib(*max)),
            Self::TooManyPixels { width, height, max } => write!(
                f,
                "the picture is {width}×{height}; the limit is {:.0} megapixels",
                *max as f64 / 1e6
            ),
            Self::Format => f.write_str("only PNG, JPEG and WebP pictures are taken"),
            Self::Corrupt => f.write_str("the picture could not be read"),
        }
    }
}

impl std::error::Error for Rejected {}

/// `bytes` in MiB: whole when it is, else to a tenth (rounded up, so a file
/// just over the limit never reads as equal to it).
fn mib(bytes: u64) -> String {
    if bytes.is_multiple_of(1 << 20) {
        return (bytes >> 20).to_string();
    }
    format!("{:.1}", (bytes as f64 / f64::from(1 << 20) * 10.0).ceil() / 10.0)
}

/// Decodes `bytes` within `limits` and scales the result (see the module doc).
pub fn load(bytes: &[u8], limits: &ImageLimits) -> Result<RgbImage, Rejected> {
    let len = bytes.len() as u64;
    if len > limits.max_bytes {
        return Err(Rejected::TooLarge { bytes: len, max: limits.max_bytes });
    }
    // The decoders are fuzzed but not proven; a panic in one is a bad file.
    let decoded = catch_unwind(AssertUnwindSafe(|| decode(bytes, limits)))
        .map_err(|_| Rejected::Corrupt)??;
    Ok(fit(decoded, limits.max_side))
}

fn decode(bytes: &[u8], limits: &ImageLimits) -> Result<RgbImage, Rejected> {
    let mut reader = ImageReader::new(Cursor::new(bytes))
        .with_guessed_format()
        .map_err(|_| Rejected::Corrupt)?;
    // Never trusted from a file name or a content type: from the bytes alone.
    if !matches!(
        reader.format(),
        Some(ImageFormat::Png | ImageFormat::Jpeg | ImageFormat::WebP)
    ) {
        return Err(Rejected::Format);
    }
    let mut caps = Limits::default();
    // Room for the decoded picture at 16 bits × 4 channels, plus the decoders'
    // own working memory.
    caps.max_alloc = Some(limits.max_pixels.saturating_mul(8).saturating_add(64 << 20));
    reader.limits(caps);
    let mut decoder = reader.into_decoder().map_err(|_| Rejected::Corrupt)?;
    let (width, height) = decoder.dimensions();
    if width == 0 || height == 0 {
        return Err(Rejected::Corrupt);
    }
    if u64::from(width) * u64::from(height) > limits.max_pixels {
        return Err(Rejected::TooManyPixels { width, height, max: limits.max_pixels });
    }
    let orientation = decoder.orientation().ok();
    let mut picture = DynamicImage::from_decoder(decoder).map_err(|_| Rejected::Corrupt)?;
    if let Some(o) = orientation {
        picture.apply_orientation(o);
    }
    // Alpha is dropped, not composited, as d1's processor (PIL `convert("RGB")`) does.
    Ok(picture.into_rgb8())
}

/// `picture` scaled down (never up) so its longer side is at most `max_side`.
pub fn fit(picture: RgbImage, max_side: u32) -> RgbImage {
    let (w, h) = picture.dimensions();
    let long = w.max(h);
    if long <= max_side {
        return picture;
    }
    let scale = f64::from(max_side) / f64::from(long);
    let side = |n: u32| ((f64::from(n) * scale).round() as u32).clamp(1, max_side);
    image::imageops::resize(&picture, side(w), side(h), FilterType::CatmullRom)
}

/// `picture` as a JPEG, in memory: what gets reposted, so none of the sender's
/// bytes (metadata, anything crafted) go back out.
pub fn to_jpeg(picture: &RgbImage) -> Vec<u8> {
    let mut out = Vec::new();
    image::codecs::jpeg::JpegEncoder::new_with_quality(&mut out, 90)
        .encode_image(picture)
        .expect("encoding to memory cannot fail");
    out
}

/// The picture as `Rgb` bytes for the vision tower: width, height, `[r, g, b]` rows.
pub fn into_parts(picture: RgbImage) -> (u32, u32, Vec<u8>) {
    let (w, h) = picture.dimensions();
    (w, h, picture.into_raw())
}

#[cfg(test)]
mod tests {
    use super::*;

    const LIMITS: ImageLimits = ImageLimits { max_bytes: 1 << 20, max_pixels: 4_000_000, max_side: 640 };

    fn encode(picture: &DynamicImage, format: ImageFormat) -> Vec<u8> {
        let mut out = Cursor::new(Vec::new());
        picture.write_to(&mut out, format).unwrap();
        out.into_inner()
    }

    #[test]
    fn scales_the_longer_side_to_the_cap_and_keeps_small_ones() {
        let wide = DynamicImage::ImageRgb8(RgbImage::new(1600, 900));
        let got = load(&encode(&wide, ImageFormat::Png), &LIMITS).unwrap();
        assert_eq!(got.dimensions(), (640, 360));
        let tall = DynamicImage::ImageRgba8(image::RgbaImage::new(300, 1000));
        let got = load(&encode(&tall, ImageFormat::Png), &LIMITS).unwrap();
        assert_eq!(got.dimensions(), (192, 640));
        let small = DynamicImage::ImageRgb8(RgbImage::new(200, 100));
        let got = load(&encode(&small, ImageFormat::Jpeg), &LIMITS).unwrap();
        assert_eq!(got.dimensions(), (200, 100));
        let back = load(&to_jpeg(&got), &LIMITS).unwrap();
        assert_eq!(back.dimensions(), (200, 100));
    }

    #[test]
    fn turns_away_large_files_other_formats_and_garbage() {
        let big = vec![0u8; (LIMITS.max_bytes + 1) as usize];
        let err = load(&big, &LIMITS).unwrap_err();
        assert_eq!(err.to_string(), "the file is 1.1 MiB; the limit is 1 MiB");
        let err = Rejected::TooLarge { bytes: 12_345_678, max: 10 << 20 };
        assert_eq!(err.to_string(), "the file is 11.8 MiB; the limit is 10 MiB");
        assert_eq!(load(b"GIF89a\x01\x00\x01\x00", &LIMITS), Err(Rejected::Format));
        assert_eq!(load(b"<svg xmlns='http://www.w3.org/2000/svg'/>", &LIMITS), Err(Rejected::Format));
        let mut png = encode(&DynamicImage::ImageRgb8(RgbImage::new(50, 50)), ImageFormat::Png);
        png.truncate(png.len() / 2);
        assert_eq!(load(&png, &LIMITS), Err(Rejected::Corrupt));
    }

    /// A decompression bomb: a tiny PNG whose header claims 60000 × 60000
    /// (3.6 gigapixels) is turned away from the header, before any pixel.
    #[test]
    fn turns_away_bombs_from_the_header() {
        let mut png = encode(&DynamicImage::ImageRgb8(RgbImage::new(1, 1)), ImageFormat::Png);
        // IHDR: the signature (8), length (4), "IHDR" (4), then width and height.
        png[16..20].copy_from_slice(&60_000u32.to_be_bytes());
        png[20..24].copy_from_slice(&60_000u32.to_be_bytes());
        // The IHDR CRC over type + data, so the header itself parses.
        let crc = crc32(&png[12..29]);
        png[29..33].copy_from_slice(&crc.to_be_bytes());
        assert!(png.len() < 100);
        assert_eq!(
            load(&png, &LIMITS),
            Err(Rejected::TooManyPixels { width: 60_000, height: 60_000, max: LIMITS.max_pixels })
        );
    }

    fn crc32(bytes: &[u8]) -> u32 {
        let mut crc = !0u32;
        for &b in bytes {
            crc ^= u32::from(b);
            for _ in 0..8 {
                crc = if crc & 1 != 0 { (crc >> 1) ^ 0xEDB8_8320 } else { crc >> 1 };
            }
        }
        !crc
    }
}
