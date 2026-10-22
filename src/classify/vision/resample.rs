//! The two bicubic resamplers d1's picture path runs, bit for bit (ported
//! from d1-burn).
//!
//! Both are separable, two-pass (horizontal, then vertical) and integer: each output sample is
//! `Σ pixel · weight` over fixed-point weights, rounded by adding half a unit, shifted back and
//! clamped to `0..=255`, with an 8-bit image between the passes. They differ in the details:
//!
//! - [`Flavour::Pillow`] is Pillow's `ImagingResample` (`libImaging/Resample.c`), what
//!   `Image.resize(.., BICUBIC)` runs in `runner.cap_pixels`: the cubic as
//!   `(((x - 5)x + 8)x - 4)·a` and 22 fractional bits (`PRECISION_BITS`).
//! - [`Flavour::Torch`] is PyTorch's antialiased bicubic on uint8 (`_upsample_bicubic2d_aa`, the
//!   AVX2 kernel in `UpSampleKernelAVXAntialias.h`, a port of Pillow-SIMD), what torchvision's
//!   `resize(.., BICUBIC, antialias=True)` runs on a CPU uint8 tensor in the image processor: the
//!   cubic as `((a·x - 5a)x + 8a)x - 4a`, and as many fractional bits as keep the largest weight of
//!   the axis below 2^15 (the weights are int16).
//!
//! Both use Keys' `a = -0.5` and the same support: two input pixels on each side of the centre,
//! scaled by the downscale factor.

use super::Rgb;

/// Which implementation to reproduce.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Flavour {
    /// Pillow's `Image.resize(.., Resampling.BICUBIC)`.
    Pillow,
    /// PyTorch's `interpolate(.., mode="bicubic", antialias=True)` on uint8, on an AVX2 CPU.
    Torch,
}

/// Keys' cubic coefficient.
const A: f64 = -0.5;
/// Pillow's fixed-point fraction bits for 8-bit images.
const PILLOW_BITS: u32 = 22;
/// PyTorch's int16 weights stay below this.
const TORCH_LIMIT: i64 = 1 << 15;

/// One output sample: the first input sample it reads and its weights, in fixed point.
#[derive(Debug, Clone)]
struct Taps {
    start: usize,
    weights: Vec<i64>,
}

/// The weights of one axis.
#[derive(Debug, Clone)]
struct Kernel {
    taps: Vec<Taps>,
    /// Fraction bits of the weights.
    bits: u32,
}

/// `n` as f64; every size here is far below 2^32.
fn float(n: usize) -> f64 {
    u32::try_from(n).map_or(f64::INFINITY, f64::from)
}

/// C's `(int)v` / `static_cast<int64_t>(v)`: truncation toward zero.
fn trunc(v: f64) -> i64 {
    v as i64
}

fn pillow_cubic(x: f64) -> f64 {
    let x = x.abs();
    if x < 1.0 {
        ((A + 2.0) * x - (A + 3.0)) * x * x + 1.0
    } else if x < 2.0 {
        (((x - 5.0) * x + 8.0) * x - 4.0) * A
    } else {
        0.0
    }
}

fn torch_cubic(x: f64) -> f64 {
    let x = x.abs();
    if x < 1.0 {
        ((A + 2.0) * x - (A + 3.0)) * x * x + 1.0
    } else if x < 2.0 {
        ((A * x - 5.0 * A) * x + 8.0 * A) * x - 4.0 * A
    } else {
        0.0
    }
}

/// `(start, weights)` per output sample in f64, normalised to sum 1, and the largest weight.
fn float_weights(input: usize, output: usize, flavour: Flavour) -> (Vec<(usize, Vec<f64>)>, f64) {
    let scale = float(input) / float(output);
    let filterscale = scale.max(1.0);
    let support = 2.0 * filterscale;
    let max_taps = usize::try_from(trunc(support.ceil())).unwrap_or(0) * 2 + 1;
    let inv = 1.0 / filterscale;
    let filter = match flavour {
        Flavour::Pillow => pillow_cubic,
        Flavour::Torch => torch_cubic,
    };
    let mut max = 0.0_f64;
    let rows = (0..output)
        .map(|i| {
            let center = (float(i) + 0.5) * scale;
            let start = usize::try_from(trunc(center - support + 0.5)).unwrap_or(0);
            let end = usize::try_from(trunc(center + support + 0.5))
                .unwrap_or(0)
                .min(input);
            let mut count = end.saturating_sub(start);
            if flavour == Flavour::Torch {
                // PyTorch clips a count one too large through rounding; Pillow doesn't.
                count = count.min(max_taps);
            }
            let mut w: Vec<f64> = (start..start + count)
                .map(|j| filter((float(j) - center + 0.5) * inv))
                .collect();
            let total: f64 = w.iter().sum();
            if total != 0.0 {
                for v in &mut w {
                    *v /= total;
                    max = max.max(*v);
                }
            }
            (start, w)
        })
        .collect();
    (rows, max)
}

/// The fixed-point kernel of one axis, `input` samples to `output`.
fn kernel(input: usize, output: usize, flavour: Flavour) -> Kernel {
    let (rows, max) = float_weights(input, output, flavour);
    let bits = match flavour {
        Flavour::Pillow => PILLOW_BITS,
        Flavour::Torch => {
            let mut bits = 0;
            while bits < 22 && trunc(0.5 + max * f64::from(1_u32 << (bits + 1))) < TORCH_LIMIT {
                bits += 1;
            }
            bits
        }
    };
    let unit = f64::from(1_u32 << bits);
    let taps = rows
        .into_iter()
        .map(|(start, w)| Taps {
            start,
            weights: w
                .into_iter()
                .map(|v| {
                    let v = v * unit;
                    trunc(if v < 0.0 { v - 0.5 } else { v + 0.5 })
                })
                .collect(),
        })
        .collect();
    Kernel { taps, bits }
}

impl Kernel {
    fn rounding(&self) -> i64 {
        if self.bits == 0 {
            0
        } else {
            1 << (self.bits - 1)
        }
    }

    fn finish(&self, acc: i64) -> u8 {
        u8::try_from((acc >> self.bits).clamp(0, 255)).unwrap_or(u8::MAX)
    }
}

/// Resamples every row of `src` to `kernel.taps.len()` pixels.
fn horizontal(src: &Rgb, kernel: &Kernel) -> Rgb {
    let width = kernel.taps.len();
    let mut data = Vec::with_capacity(width * src.height * 3);
    for row in src.data.chunks_exact(src.width * 3) {
        for t in &kernel.taps {
            let mut acc = [kernel.rounding(); 3];
            let (pixels, _) = row.get(t.start * 3..).unwrap_or_default().as_chunks::<3>();
            for (p, &w) in pixels.iter().zip(&t.weights) {
                for (a, &v) in acc.iter_mut().zip(p) {
                    *a += i64::from(v) * w;
                }
            }
            data.extend(acc.map(|a| kernel.finish(a)));
        }
    }
    Rgb {
        width,
        height: src.height,
        data,
    }
}

/// Resamples every column of `src` to `kernel.taps.len()` pixels.
fn vertical(src: &Rgb, kernel: &Kernel) -> Rgb {
    let stride = src.width * 3;
    let mut data = Vec::with_capacity(stride * kernel.taps.len());
    let mut acc = vec![0_i64; stride];
    for t in &kernel.taps {
        acc.fill(kernel.rounding());
        let rows = src.data.get(t.start * stride..).unwrap_or_default();
        for (row, &w) in rows.chunks_exact(stride).zip(&t.weights) {
            for (a, &v) in acc.iter_mut().zip(row) {
                *a += i64::from(v) * w;
            }
        }
        data.extend(acc.iter().map(|&a| kernel.finish(a)));
    }
    Rgb {
        width: src.width,
        height: kernel.taps.len(),
        data,
    }
}

/// `src` resized to `width × height` (both at least 1) the way `flavour` does it.
#[must_use]
pub fn resize(src: &Rgb, width: usize, height: usize, flavour: Flavour) -> Rgb {
    let mut out = None;
    if width != src.width {
        out = Some(horizontal(src, &kernel(src.width, width, flavour)));
    }
    if height != src.height {
        let from = out.as_ref().unwrap_or(src);
        out = Some(vertical(from, &kernel(src.height, height, flavour)));
    }
    out.unwrap_or_else(|| src.clone())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn weights_sum_to_one_unit() {
        for flavour in [Flavour::Pillow, Flavour::Torch] {
            for (i, o) in [
                (7, 256),
                (640, 512),
                (4096, 1182),
                (3, 1),
                (1, 5),
                (10000, 32),
            ] {
                let k = kernel(i, o, flavour);
                assert_eq!(k.taps.len(), o);
                for t in &k.taps {
                    let s: i64 = t.weights.iter().sum();
                    // Each weight is rounded on its own: at most half a unit off.
                    let slack = i64::try_from(t.weights.len()).unwrap();
                    assert!(
                        (s - (1 << k.bits)).abs() <= slack,
                        "{flavour:?} {i}->{o}: {s}"
                    );
                    assert!(t.start + t.weights.len() <= i);
                }
                if flavour == Flavour::Torch {
                    assert!(
                        k.taps
                            .iter()
                            .flat_map(|t| &t.weights)
                            .all(|w| w.abs() < TORCH_LIMIT)
                    );
                }
            }
        }
    }

    #[test]
    fn flat_stays_flat() {
        let src = Rgb {
            width: 13,
            height: 9,
            data: [10_u8, 128, 250].repeat(13 * 9),
        };
        for flavour in [Flavour::Pillow, Flavour::Torch] {
            for (w, h) in [(5, 4), (40, 31), (13, 2), (1, 1)] {
                let out = resize(&src, w, h, flavour);
                assert_eq!((out.width, out.height), (w, h));
                assert_eq!(out.data, [10_u8, 128, 250].repeat(w * h));
            }
            assert_eq!(resize(&src, 13, 9, flavour).data, src.data);
        }
    }
}
