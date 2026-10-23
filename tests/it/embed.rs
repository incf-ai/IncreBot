//! EmbeddingGemma 2 against sentence-transformers in float32: the fixtures of
//! `tmp/eg2/fixtures.py` (kept out of the repo), in `tmp/eg2/fixtures` or `EMBED_FIXTURES`.
//! Without them each test says so and passes.
//!
//! The model tests load `google/embeddinggemma-2` from the HF cache, so they are
//! ignored by default: `cargo test --release --test it embed -- --include-ignored --nocapture`

use std::fs;
use std::path::PathBuf;
use std::sync::OnceLock;

use increbot::classify::vision::Rgb;
use increbot::embed::safetensors::SafeTensors;
use increbot::embed::{Embedder, Patches, cosine, preprocess};

fn fixtures() -> PathBuf {
    std::env::var_os("EMBED_FIXTURES").map_or_else(
        || PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tmp/eg2/fixtures"),
        PathBuf::from,
    )
}

/// The fixture names of one kind (`texts` or `images`) with what went in, or `None` (said).
fn cases(kind: &str) -> Option<Vec<(String, String)>> {
    let index = fixtures().join("index.json");
    let Ok(text) = fs::read_to_string(&index) else {
        eprintln!("SKIPPED: needs {}", index.display());
        return None;
    };
    let json: serde_json::Value = serde_json::from_str(&text).unwrap();
    let cases = json[kind].as_object().unwrap();
    Some(cases.iter().map(|(k, v)| (k.clone(), v.as_str().unwrap().to_owned())).collect())
}

fn open(kind: &str, name: &str) -> SafeTensors {
    SafeTensors::open(&fixtures().join(format!("{kind}_{name}.safetensors"))).unwrap()
}

fn floats(st: &SafeTensors, name: &str) -> Vec<f32> {
    let shape = st.info(name).unwrap().shape.clone();
    st.f32(name, &shape).unwrap()
}

fn ids(st: &SafeTensors) -> Vec<u32> {
    st.i64("input_ids").unwrap().into_iter().map(|i| u32::try_from(i).unwrap()).collect()
}

fn picture(st: &SafeTensors) -> Rgb {
    let shape = &st.info("pixels").unwrap().shape;
    Rgb::new(shape[1], shape[0], st.bytes("pixels").unwrap().to_vec()).unwrap()
}

/// The fixture's patches as the processor made them.
fn patches(st: &SafeTensors) -> Patches {
    let positions = st.i64("positions").unwrap();
    Patches {
        values: floats(st, "pixel_values"),
        positions: positions
            .chunks_exact(2)
            .map(|p| (usize::try_from(p[0]).unwrap(), usize::try_from(p[1]).unwrap()))
            .collect(),
    }
}

fn max_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "lengths");
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max)
}

fn model() -> &'static Embedder {
    static MODEL: OnceLock<Embedder> = OnceLock::new();
    MODEL.get_or_init(|| Embedder::load().unwrap())
}

#[test]
fn preprocessing_matches_the_processor() {
    let Some(cases) = cases("images") else { return };
    for (name, _) in cases {
        let st = open("image", &name);
        let (want, got) = (patches(&st), preprocess(&picture(&st)));
        assert_eq!(got.positions, want.positions, "{name}: positions");
        let d = max_diff(&got.values, &want.values);
        eprintln!("{name}: {} patches, pixel values within {d:e}", got.len());
        // Bit-exact resize and the same 1/255; a 1-level rounding difference shows as 0.0039.
        assert!(d < 1e-6, "{name}: pixel values differ by {d}");
    }
}

#[test]
#[ignore = "loads google/embeddinggemma-2"]
fn texts_match_sentence_transformers() {
    let Some(cases) = cases("texts") else { return };
    let m = model();
    for (name, text) in cases {
        let st = open("text", &name);
        assert_eq!(m.tokens(&text).unwrap(), ids(&st), "{name}: token ids");
        let (got, want) = (m.embed_text(&text).unwrap(), floats(&st, "embedding"));
        let c = cosine(&got, &want);
        eprintln!("{name}: {} tokens, cosine {c:.6}, max diff {:e}", ids(&st).len(), max_diff(&got, &want));
        assert!(c > 0.9995, "{name}: cosine {c}");
    }
}

#[test]
#[ignore = "loads google/embeddinggemma-2"]
fn images_match_sentence_transformers() {
    let Some(cases) = cases("images") else { return };
    let m = model();
    for (name, _) in cases {
        let st = open("image", &name);
        let p = patches(&st);
        let soft = m.soft_tokens(&p);
        let want = floats(&st, "soft_tokens");
        let rows = |v: &[f32]| v.chunks_exact(512).map(<[f32]>::to_vec).collect::<Vec<_>>();
        let worst = rows(&soft).iter().zip(rows(&want)).map(|(a, b)| cosine(a, &b)).fold(1.0, f32::min);
        assert_eq!(Embedder::image_tokens(soft.len() / 512), ids(&st), "{name}: token ids");
        let (got, want) = (m.embed_image(&picture(&st)).unwrap(), floats(&st, "embedding"));
        let c = cosine(&got, &want);
        eprintln!(
            "{name}: {} soft tokens (worst row cosine {worst:.6}, max diff {:e}), cosine {c:.6}",
            soft.len() / 512,
            max_diff(&soft, &floats(&st, "soft_tokens")),
        );
        assert!(worst > 0.999, "{name}: soft token cosine {worst}");
        assert!(c > 0.9995, "{name}: cosine {c}");
    }
}
