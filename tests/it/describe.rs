//! Qwen3-VL against transformers in float32: the fixtures of
//! `tmp/qwen3vl/fixtures.py` (kept out of the repo), in `tmp/qwen3vl/fixtures` or `DESCRIBE_FIXTURES`.
//! Without them each test says so and passes.
//!
//! The tower runs from the F16 mmproj (f16 weights, so a loose tolerance) on
//! the CPU (the reference) and on the GPU; the language model is Q4_K_M on the first GPU in
//! `INCREBOT_GPUS`, fed the reference's own picture rows, so its check is the
//! top token and how far the greedy answers agree.
//! Ignored by default: `cargo test --release --test it describe -- --include-ignored --nocapture`

use std::fs;
use std::path::PathBuf;
use std::sync::OnceLock;

use increbot::classify::vision::Rgb;
use increbot::describe::vision::{self, Features, Vision};
use increbot::describe::{self as qwen, Describer, Params};
use increbot::embed::safetensors::SafeTensors;
use increbot::gpu::Device;

fn fixtures() -> PathBuf {
    std::env::var_os("DESCRIBE_FIXTURES").map_or_else(
        || PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tmp/qwen3vl/fixtures"),
        PathBuf::from,
    )
}

/// The fixture names with their question (`None`: the tower alone), or `None` (said).
fn cases() -> Option<Vec<(String, Option<String>)>> {
    let index = fixtures().join("index.json");
    let Ok(text) = fs::read_to_string(&index) else {
        eprintln!("SKIPPED: needs {}", index.display());
        return None;
    };
    let json: serde_json::Value = serde_json::from_str(&text).unwrap();
    let cases = json["cases"].as_object().unwrap();
    Some(cases.iter().map(|(k, v)| (k.clone(), v.as_str().map(str::to_owned))).collect())
}

fn open(name: &str) -> SafeTensors {
    SafeTensors::open(&fixtures().join(format!("{name}.safetensors"))).unwrap()
}

fn floats(st: &SafeTensors, name: &str) -> Vec<f32> {
    let shape = st.info(name).unwrap().shape.clone();
    st.f32(name, &shape).unwrap()
}

fn ints(st: &SafeTensors, name: &str) -> Vec<u32> {
    st.i64(name).unwrap().into_iter().map(|i| u32::try_from(i).unwrap()).collect()
}

fn picture(st: &SafeTensors) -> Rgb {
    let shape = &st.info("pixels").unwrap().shape;
    Rgb::new(shape[1], shape[0], st.bytes("pixels").unwrap().to_vec()).unwrap()
}

/// The reference's merged grid and picture rows.
fn reference(st: &SafeTensors) -> Features {
    let grid = ints(st, "grid_thw");
    let main = floats(st, "main");
    let levels = st.info("deepstack").unwrap().shape[0];
    let all = floats(st, "deepstack");
    let deepstack = all.chunks_exact(all.len() / levels).map(<[f32]>::to_vec).collect();
    Features { main, deepstack, grid: (grid[1] as usize / vision::MERGE, grid[2] as usize / vision::MERGE) }
}

/// The largest difference relative to `b`'s largest magnitude.
fn rel_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "lengths");
    let scale = b.iter().fold(0f32, |m, v| m.max(v.abs()));
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max) / scale
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f64 = a.iter().zip(b).map(|(x, y)| f64::from(*x) * f64::from(*y)).sum();
    let na: f64 = a.iter().map(|x| f64::from(*x).powi(2)).sum();
    let nb: f64 = b.iter().map(|x| f64::from(*x).powi(2)).sum();
    (dot / (na.sqrt() * nb.sqrt())) as f32
}

fn hub(repo: &str, file: &str) -> PathBuf {
    increbot::hub::resolve(repo, file).unwrap()
}

#[test]
fn preprocessing_matches_the_processor() {
    let Some(cases) = cases() else { return };
    for (name, _) in cases {
        let st = open(&name);
        let p = vision::preprocess(&picture(&st), qwen::MAX_IMAGE_TOKENS).unwrap();
        let grid = ints(&st, "grid_thw");
        assert_eq!((1, p.grid.0, p.grid.1), (grid[0] as usize, grid[1] as usize, grid[2] as usize), "{name} grid");
        // The processor's patches are [channel, frame, y, x] with both frames equal.
        let theirs = floats(&st, "pixel_values");
        let per = 3 * 2 * vision::PATCH * vision::PATCH;
        let mut worst = 0f32;
        for (mine, theirs) in p.values.chunks_exact(vision::PATCH * vision::PATCH * 3).zip(theirs.chunks_exact(per)) {
            for c in 0..3 {
                for yx in 0..vision::PATCH * vision::PATCH {
                    worst = worst.max((mine[yx * 3 + c] - theirs[c * 2 * 256 + yx]).abs());
                }
            }
        }
        eprintln!("{name}: grid {:?}, worst pixel difference {worst}", p.grid);
        // Bicubic resampling rounds to whole 8-bit values on each side.
        assert!(worst <= 2.0 / 127.5 + 1e-4, "{name}: {worst}");
    }
}

#[test]
#[ignore = "loads the Qwen3-VL mmproj"]
fn tower_matches_transformers() {
    let Some(cases) = cases() else { return };
    let tower = Vision::load(&hub(qwen::REPO, qwen::MMPROJ_FILE)).unwrap();
    for (name, _) in cases {
        let st = open(&name);
        let want = reference(&st);
        let p = vision::preprocess(&picture(&st), qwen::MAX_IMAGE_TOKENS).unwrap();
        let start = std::time::Instant::now();
        let got = tower.features(&p).unwrap();
        check_tower(&name, &got, &want, start.elapsed().as_secs_f32());
    }
}

/// The checks of [`tower_matches_transformers`] on `got`.
fn check_tower(name: &str, got: &Features, want: &Features, secs: f32) {
    assert_eq!(got.grid, want.grid, "{name} grid");
    let main = (rel_diff(&got.main, &want.main), cosine(&got.main, &want.main));
    eprintln!("{name}: {} rows in {secs:.2}s, main rel {:.4} cos {:.6}", got.grid.0 * got.grid.1, main.0, main.1);
    assert!(main.1 > 0.999, "{name} main: {main:?}");
    for (i, (g, w)) in got.deepstack.iter().zip(&want.deepstack).enumerate() {
        let c = cosine(g, w);
        eprintln!("  deepstack {i}: rel {:.4} cos {c:.6}", rel_diff(g, w));
        assert!(c > 0.999, "{name} deepstack {i}: {c}");
    }
}

#[test]
#[ignore = "loads Qwen3-VL-8B onto a GPU"]
fn gpu_tower_matches_transformers() {
    if cfg!(debug_assertions) {
        eprintln!("SKIPPED: run with --release");
        return;
    }
    let Some(cases) = cases() else { return };
    let _cards = crate::CARDS.lock();
    let Some(m) = describer() else { return };
    for (name, _) in cases {
        let st = open(&name);
        let start = std::time::Instant::now();
        let got = m.features(&picture(&st), &|| false).unwrap();
        check_tower(&name, &got, &reference(&st), start.elapsed().as_secs_f32());
    }
}

fn describer() -> Option<&'static Describer> {
    static MODEL: OnceLock<Option<Describer>> = OnceLock::new();
    MODEL
        .get_or_init(|| {
            let Some(pci) = std::env::var("INCREBOT_GPUS").ok().and_then(|v| v.split([',', '=']).next().map(str::to_string)).filter(|v| !v.is_empty()) else {
                eprintln!("SKIPPED: INCREBOT_GPUS is not set");
                return None;
            };
            let dev = Device::open(pci.parse().unwrap()).unwrap();
            let params = Params { max_new_tokens: 64, temperature: 0.0, ..Params::default() };
            Some(
                Describer::open(
                    &dev,
                    &hub(qwen::REPO, qwen::MODEL_FILE),
                    &hub(qwen::REPO, qwen::MMPROJ_FILE),
                    &hub(qwen::TOKENIZER_REPO, "tokenizer.json"),
                    qwen::MAX_IMAGE_TOKENS,
                    params,
                )
                .unwrap(),
            )
        })
        .as_ref()
}

#[test]
#[ignore = "loads Qwen3-VL-8B onto a GPU"]
fn answers_match_transformers() {
    if cfg!(debug_assertions) {
        eprintln!("SKIPPED: run with --release");
        return;
    }
    let Some(cases) = cases() else { return };
    let _cards = crate::CARDS.lock();
    let Some(m) = describer() else { return };
    for (name, question) in cases {
        let Some(question) = question else { continue };
        let st = open(&name);
        let f = reference(&st);
        let q = m.prompt_tokens(&question).unwrap();
        let turn = m.turn(f.grid, &q).unwrap();
        assert_eq!(turn.ids, ints(&st, "input_ids"), "{name} tokens");
        let want_pos = ints(&st, "position_ids");
        let n = turn.pos.len();
        for (i, p) in turn.pos.iter().enumerate() {
            assert_eq!(*p, [want_pos[i], want_pos[n + i], want_pos[2 * n + i]], "{name} position {i}");
        }

        let logits = m.prefill(&f, &q).unwrap();
        let want = floats(&st, "logits");
        let argmax = |v: &[f32]| (0..v.len()).max_by(|&i, &j| v[i].total_cmp(&v[j])).unwrap();
        eprintln!("{name}: top {} (want {}), logits cos {:.5}", argmax(&logits), argmax(&want), cosine(&logits, &want));
        assert_eq!(argmax(&logits), argmax(&want), "{name} top token");
        assert!(cosine(&logits, &want) > 0.98, "{name} logits");

        let d = m.answer(&f, &q, &|| false).unwrap();
        let greedy = ints(&st, "greedy");
        let same = d.tokens.iter().zip(&greedy).take_while(|(a, b)| a == b).count();
        eprintln!("{name}: {same}/{} greedy tokens agree, {:.1} tok/s\n  {}", greedy.len(), d.generated as f32 / d.decode_secs, d.text);
        // Q4_K_M drifts from float32 at near ties ("vibrant pink" / "vibrant
        // stripes"); the opening words should agree.
        assert!(same >= 3, "{name}: {same} tokens agree");
    }
}

#[test]
#[ignore = "loads Qwen3-VL-8B onto a GPU"]
fn describes_a_picture_end_to_end() {
    if cfg!(debug_assertions) {
        eprintln!("SKIPPED: run with --release");
        return;
    }
    let Some(cases) = cases() else { return };
    let _cards = crate::CARDS.lock();
    let Some(m) = describer() else { return };
    let (name, _) = cases.into_iter().find(|(_, q)| q.is_some()).unwrap();
    let d = m.describe(&picture(&open(&name)), "", &|| false).unwrap();
    eprintln!(
        "{name}: {} rows, vision {:.1}s, prefill {:.1}s, {} tokens in {:.1}s\n  {}",
        d.image_tokens, d.vision_secs, d.prefill_secs, d.generated, d.decode_secs, d.text
    );
    assert!(d.generated > 5 && !d.text.is_empty());
}
