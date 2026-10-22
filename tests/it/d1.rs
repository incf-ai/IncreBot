//! d1 classification against the Python reference (ported from d1-burn's
//! `tests/text.rs`): the fixtures of `discovery/bench/d1_text_fixtures.py` in
//! the d1 bench project, `D1_FIXTURES` to point elsewhere. Without the fixtures
//! or the tokenizer each test says so and passes.
//!
//! The end-to-end test loads d1-3B Q8_0 from the HF cache, so it is ignored by
//! default: `cargo test --release --test it d1 -- --ignored --nocapture`

use std::fs;
use std::path::PathBuf;

use increbot::classify::prompt::{self, OptionStyle, StateStyle, System};
use increbot::classify::{Classifier, Json, Question, Renderer, Tokenizer};
use increbot::model::LinearImpl;

const TOKENIZER_REVISION: &str = "051bcc464b01b9f92942b364d9586b0ef5912432";

fn home() -> String {
    std::env::var("HOME").unwrap_or_default()
}

fn fixtures() -> PathBuf {
    std::env::var_os("D1_FIXTURES").map_or_else(
        || {
            PathBuf::from(format!(
                "{}/Programming/bench_LiquidAI_d1-omni-600M/fixtures/d1-3b/text-e2e",
                home()
            ))
        },
        PathBuf::from,
    )
}

fn tokenizer_json() -> PathBuf {
    PathBuf::from(format!(
        "{}/.cache/huggingface/hub/models--LiquidAI--d1-3B/snapshots/{TOKENIZER_REVISION}/tokenizer.json",
        home()
    ))
}

/// The tokenizer and a fixture file's rows, or `None` (said) when either is missing.
fn setup(file: &str) -> Option<(Tokenizer, Vec<Json>)> {
    let (tok, path) = (tokenizer_json(), fixtures().join(file));
    if !tok.exists() || !path.exists() {
        eprintln!("SKIPPED: needs {} and {}", tok.display(), path.display());
        return None;
    }
    let tokenizer = Tokenizer::from_file(&tok, "<|startoftext|>").unwrap();
    let rows = fs::read_to_string(&path)
        .unwrap()
        .lines()
        .map(|l| Json::parse(l).unwrap())
        .collect();
    Some((tokenizer, rows))
}

fn str_of<'a>(row: &'a Json, key: &str) -> &'a str {
    match row.get(key) {
        Some(Json::String(s)) => s,
        other => panic!("{key}: {other:?}"),
    }
}

fn ints(v: &Json) -> Vec<u32> {
    list(v)
        .iter()
        .map(|i| match i {
            Json::Int(i) => u32::try_from(*i).unwrap(),
            other => panic!("id {other:?}"),
        })
        .collect()
}

fn floats(v: &Json) -> Vec<f64> {
    list(v)
        .iter()
        .map(|i| match i {
            Json::Float(f) => *f,
            Json::Int(i) => *i as f64,
            other => panic!("probability {other:?}"),
        })
        .collect()
}

fn list(v: &Json) -> &[Json] {
    match v {
        Json::Array(items) => items,
        other => panic!("list {other:?}"),
    }
}

/// A fixture's state: the JSON text of the Python state, `null` being no state.
fn state(row: &Json) -> Option<Json> {
    match Json::parse(str_of(row, "state")).unwrap() {
        Json::Null => None,
        s => Some(s),
    }
}

#[test]
fn d1_tokenizer_matches_python() {
    let Some((tokenizer, rows)) = setup("tokens.jsonl") else {
        return;
    };
    let bad: Vec<&str> = rows
        .iter()
        .filter(|row| tokenizer.encode(str_of(row, "text")).unwrap() != ints(row.get("ids").unwrap()))
        .map(|row| str_of(row, "text"))
        .collect();
    eprintln!("tokenizer: {} texts, {} mismatches", rows.len(), bad.len());
    assert!(bad.is_empty(), "mismatches: {bad:?}");
}

#[test]
fn d1_prompts_match_python() {
    let Some((tokenizer, rows)) = setup("renders.jsonl") else {
        return;
    };
    for row in &rows {
        let name = format!(
            "{} on {} ({})",
            str_of(row, "question_name"),
            str_of(row, "state_name"),
            str_of(row, "style")
        );
        let renderer = Renderer {
            style: match str_of(row, "style") {
                "json_only" => StateStyle::JsonOnly,
                "json" => StateStyle::Json,
                "sections" => StateStyle::Sections,
                "sections_only" => StateStyle::SectionsOnly,
                other => panic!("style {other}"),
            },
            system: match str_of(row, "system") {
                "none" => System::None,
                _ => System::Isolated,
            },
            option_style: match str_of(row, "option_style") {
                "desc" => OptionStyle::Desc,
                _ => OptionStyle::NameDesc,
            },
            ..Renderer::new(&tokenizer)
        };
        let q = Question::from_json(row.get("question").unwrap()).unwrap();
        let state = state(row);
        let text = renderer.render(&tokenizer, state.as_ref(), &q).unwrap();
        assert_eq!(text, str_of(row, "prompt"), "{name}");
        assert_eq!(tokenizer.encode(&text).unwrap(), ints(row.get("ids").unwrap()), "{name}");
        if let Some(Json::Array(aliases)) = row.get("aliases") {
            let want: Vec<(String, u32)> = aliases
                .iter()
                .map(|a| match list(a) {
                    [Json::String(code), Json::Int(id)] => (code.clone(), u32::try_from(*id).unwrap()),
                    other => panic!("alias {other:?}"),
                })
                .collect();
            let labels: Vec<&str> = match &q {
                Question::Choice { criteria, .. } => criteria.iter().map(|(l, _)| l.as_str()).collect(),
                _ => panic!("{name}: aliases for a non-choice"),
            };
            assert_eq!(prompt::aliases(&tokenizer, &labels).unwrap(), want, "{name}");
        }
        let got = prompt::readout_ids(&tokenizer, &q);
        match row.get("readout_ids").unwrap() {
            Json::String(_) => assert!(got.is_err(), "{name}: Python raised"),
            groups => {
                let want: Vec<Vec<u32>> = list(groups).iter().map(ints).collect();
                assert_eq!(got.unwrap(), want, "{name}");
            }
        }
    }
    eprintln!("prompts: {} renders match", rows.len());
}

/// d1-3B `file` from the HF cache, or `None` (said) when it isn't there.
fn cached_model(file: &str) -> Option<PathBuf> {
    let path = hf_hub::Cache::from_env()
        .model("LiquidAI/d1-3B-GGUF".into())
        .get(file);
    if path.is_none() {
        eprintln!("SKIPPED: {file} not in the HF cache");
    }
    path
}

/// Runs every fixture case through `classifier`: every question's
/// probabilities, in fixture order.
fn probabilities(classifier: &Classifier, cases: &[Json]) -> Vec<(String, Vec<f64>)> {
    let mut out = Vec::new();
    for case in cases {
        let name = str_of(case, "case");
        let questions: Vec<(String, Question)> = match case.get("questions").unwrap() {
            Json::Object(items) => items
                .iter()
                .map(|(n, q)| (n.clone(), Question::from_json(q).unwrap()))
                .collect(),
            other => panic!("questions {other:?}"),
        };
        let state = state(case);
        let qs: Vec<Question> = questions.iter().map(|(_, q)| q.clone()).collect();
        let want_rows: Vec<Vec<u32>> = list(case.get("rows").unwrap()).iter().map(ints).collect();
        assert_eq!(classifier.rows(state.as_ref(), &qs).unwrap(), want_rows, "{name}: rows");

        let got = classifier.classify(state.as_ref(), &questions).unwrap();
        out.extend(got.into_iter().map(|d| (format!("{name}.{}", d.name), d.probabilities)));
    }
    out
}

/// The torch probabilities of every fixture question, in fixture order.
fn torch_probabilities(cases: &[Json]) -> Vec<Vec<f64>> {
    cases
        .iter()
        .flat_map(|case| list(case.get("probabilities").unwrap()).iter().map(floats))
        .collect()
}

/// The worst |Δp| of `got` against `want`, saying each question's.
fn worst_difference(got: &[(String, Vec<f64>)], want: &[Vec<f64>]) -> f64 {
    assert_eq!(got.len(), want.len());
    let mut worst = 0f64;
    for ((name, got), want) in got.iter().zip(want) {
        let diff = got.iter().zip(want).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
        eprintln!("{name}: max |Δp| {diff:.4}");
        worst = worst.max(diff);
    }
    eprintln!("worst |Δp| over every question: {worst:.4}");
    worst
}

#[test]
#[ignore = "loads d1-3B Q8_0 (3.6 GB) and runs every fixture case"]
fn d1_3b_probabilities_match_torch() {
    let Some((_, cases)) = setup("probs.jsonl") else {
        return;
    };
    let Some(model) = cached_model("d1-3B-Q8_0.gguf") else {
        return;
    };
    let classifier = Classifier::load(&model, &tokenizer_json(), LinearImpl::Packed, 32_768).unwrap();
    // Q8_0 weights against the fp32 torch reference.
    let worst = worst_difference(&probabilities(&classifier, &cases), &torch_probabilities(&cases));
    assert!(worst < 0.03, "worst |Δp| {worst}");
}

/// The GPU path (Q4_K_M, the only kind it takes) on the first GPU in
/// `INCREBOT_GPUS`: `INCREBOT_GPUS=0000:03:00.0 cargo test --release --test it
/// d1_3b_gpu -- --ignored --nocapture`
#[test]
#[ignore = "loads d1-3B Q4_K_M on a GPU and runs every fixture case"]
fn d1_3b_gpu_probabilities_match_torch() {
    let Some(pci) = std::env::var("INCREBOT_GPUS")
        .ok()
        .and_then(|s| s.split([',', '=']).next().map(str::to_owned))
        .filter(|s| !s.is_empty())
    else {
        eprintln!("SKIPPED: INCREBOT_GPUS is not set");
        return;
    };
    let Some((_, cases)) = setup("probs.jsonl") else {
        return;
    };
    let Some(model) = cached_model("d1-3B-Q4_K_M.gguf") else {
        return;
    };
    let cpu = Classifier::load(&model, &tokenizer_json(), LinearImpl::Packed, 8192).unwrap();
    let start = std::time::Instant::now();
    let cpu_probs = probabilities(&cpu, &cases);
    let cpu_s = start.elapsed().as_secs_f64();
    drop(cpu);
    let gpu = Classifier::load(&model, &tokenizer_json(), LinearImpl::Packed, 8192)
        .unwrap()
        .with_gpu(pci.parse().unwrap())
        .unwrap();
    let start = std::time::Instant::now();
    let gpu_probs = probabilities(&gpu, &cases);
    eprintln!(
        "{} cases: CPU {cpu_s:.1} s, GPU {:.1} s",
        cases.len(),
        start.elapsed().as_secs_f64()
    );
    eprintln!("GPU against torch:");
    let torch = worst_difference(&gpu_probs, &torch_probabilities(&cases));
    eprintln!("GPU against the CPU on the same weights:");
    let cpu: Vec<Vec<f64>> = cpu_probs.into_iter().map(|(_, p)| p).collect();
    let worst = worst_difference(&gpu_probs, &cpu);
    // The GPU quantizes activations to 8 bits for its integer matmuls and
    // keeps K/V in f16; the CPU does neither.
    assert!(worst < 0.05, "GPU against CPU worst |Δp| {worst}");
    assert!(torch < 0.1, "GPU against torch worst |Δp| {torch}");
}

// Pictures: the image fixtures of the d1 bench project (`image-prep/` for the
// processor and the end-to-end prompts, `image_*.safetensors` for the tower).

use increbot::classify::vision::{self, Config, Processed, Rgb, Vision};

fn image_fixtures() -> PathBuf {
    fixtures().parent().unwrap().to_path_buf()
}

/// A safetensors file's tensors by name: dtype, shape and raw bytes.
struct SafeTensors {
    header: serde_json::Value,
    data: Vec<u8>,
}

impl SafeTensors {
    fn open(path: &std::path::Path) -> Self {
        let bytes = fs::read(path).unwrap();
        let n = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
        let header = serde_json::from_slice(&bytes[8..8 + n]).unwrap();
        Self { header, data: bytes[8 + n..].to_vec() }
    }

    fn raw(&self, name: &str) -> (&str, Vec<usize>, &[u8]) {
        let t = &self.header[name];
        let shape = t["shape"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as usize).collect();
        let at = t["data_offsets"].as_array().unwrap();
        let (a, b) = (at[0].as_u64().unwrap() as usize, at[1].as_u64().unwrap() as usize);
        (t["dtype"].as_str().unwrap(), shape, &self.data[a..b])
    }

    fn f32s(&self, name: &str) -> (Vec<usize>, Vec<f32>) {
        let (dtype, shape, bytes) = self.raw(name);
        assert_eq!(dtype, "F32", "{name}");
        (shape, bytes.as_chunks::<4>().0.iter().map(|b| f32::from_le_bytes(*b)).collect())
    }

    fn i64s(&self, name: &str) -> Vec<i64> {
        let (dtype, _, bytes) = self.raw(name);
        assert_eq!(dtype, "I64", "{name}");
        bytes.as_chunks::<8>().0.iter().map(|b| i64::from_le_bytes(*b)).collect()
    }

    fn picture(&self, name: &str) -> Rgb {
        let (dtype, shape, bytes) = self.raw(name);
        assert_eq!((dtype, shape.len(), shape[2]), ("U8", 3, 3), "{name}");
        Rgb::new(shape[1], shape[0], bytes.to_vec()).unwrap()
    }
}

/// The `image-prep` directory, or `None` (said) when it isn't there.
fn image_prep() -> Option<PathBuf> {
    let dir = image_fixtures().join("image-prep");
    if !dir.join("prompts.json").exists() {
        eprintln!("SKIPPED: needs {}", dir.display());
        return None;
    }
    Some(dir)
}

fn usizes(v: &[i64]) -> Vec<usize> {
    v.iter().map(|&i| usize::try_from(i).unwrap()).collect()
}

/// Every prep case: the decoded picture capped as the runner caps it, then the
/// processor's layout and patch values, against Python's.
#[test]
fn d1_image_processor_matches_python() {
    let Some(dir) = image_prep() else {
        return;
    };
    let config = Config::default();
    let mut cases = 0;
    for entry in fs::read_dir(&dir).unwrap() {
        let path = entry.unwrap().path();
        let name = path.file_name().unwrap().to_string_lossy().into_owned();
        if !name.starts_with("prep_") {
            continue;
        }
        let f = SafeTensors::open(&path);
        let capped = f.picture("capped");
        let ours = vision::cap_pixels(f.picture("decoded"), vision::MAX_PIXELS);
        assert!(ours == capped, "{name}: capped picture");
        let p = config.process(&capped);
        let layout = usizes(&f.i64s("layout"));
        assert_eq!(
            [p.layout.rows, p.layout.cols, p.layout.size.0, p.layout.size.1],
            layout[..],
            "{name}: layout"
        );
        let shapes = usizes(&f.i64s("spatial_shapes"));
        let (shape, values) = f.f32s("pixel_values");
        assert_eq!(p.tiles.len(), shape[0], "{name}: tiles");
        let padded = shape[1] * vision::PATCH_VALUES;
        for (t, (got, grid)) in p.tiles.iter().enumerate() {
            assert_eq!([grid.0, grid.1], shapes[2 * t..2 * t + 2], "{name}: tile {t} grid");
            let want = &values[t * padded..t * padded + got.len()];
            let worst = got.iter().zip(want).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
            assert!(worst == 0.0, "{name}: tile {t} values off by {worst}");
        }
        cases += 1;
    }
    eprintln!("image processor: {cases} cases match");
}

/// A prompts.json row's pictures through the processor (from the prep cases'
/// capped pictures).
fn prompt_pictures(dir: &std::path::Path, row: &serde_json::Value) -> Vec<(Rgb, Processed)> {
    row["images"]
        .as_array()
        .unwrap()
        .iter()
        .map(|n| {
            let f = SafeTensors::open(&dir.join(format!("prep_{}.safetensors", n.as_str().unwrap())));
            let rgb = f.picture("capped");
            let p = Config::default().process(&rgb);
            (rgb, p)
        })
        .collect()
}

fn prompt_rows(dir: &std::path::Path) -> Vec<serde_json::Value> {
    serde_json::from_str(&fs::read_to_string(dir.join("prompts.json")).unwrap()).unwrap()
}

/// A prompts.json row's state and question (stored as JSON text to keep key order).
fn prompt_case(row: &serde_json::Value) -> (Option<Json>, Question) {
    let state = Some(row["state_json"].as_str().unwrap())
        .filter(|s| *s != "null")
        .map(|s| Json::parse(s).unwrap());
    let q = Question::from_json(&Json::parse(row["question_json"].as_str().unwrap()).unwrap()).unwrap();
    (state, q)
}

#[test]
fn d1_image_prompts_match_python() {
    let Some(dir) = image_prep() else {
        return;
    };
    let tok = tokenizer_json();
    if !tok.exists() {
        eprintln!("SKIPPED: needs {}", tok.display());
        return;
    }
    let tokenizer = Tokenizer::from_file(&tok, "<|startoftext|>").unwrap();
    let renderer = Renderer::new(&tokenizer);
    let rows = prompt_rows(&dir);
    for row in &rows {
        let name = row["name"].as_str().unwrap();
        let (state, q) = prompt_case(row);
        let layouts: Vec<_> = prompt_pictures(&dir, row).iter().map(|(_, p)| p.layout).collect();
        let got = vision::rows(&renderer, &tokenizer, &layouts, state.as_ref(), &[q]).unwrap();
        let want: Vec<u32> = serde_json::from_value(row["ids"].clone()).unwrap();
        assert_eq!(got, [want], "{name}: ids");
    }
    eprintln!("image prompts: {} match", rows.len());
}

/// Max |Δ| and |Δ| relative to the reference's RMS.
fn compare(name: &str, got: &[f32], want: &[f32]) -> (f32, f32) {
    assert_eq!(got.len(), want.len(), "{name}: length");
    let worst = got.iter().zip(want).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
    let rms = (want.iter().map(|v| v * v).sum::<f32>() / want.len() as f32).sqrt();
    let err = (got.iter().zip(want).map(|(a, b)| (a - b) * (a - b)).sum::<f32>() / want.len() as f32).sqrt();
    eprintln!("{name}: max |Δ| {worst:.4}, RMS error / RMS {:.5}", err / rms);
    (worst, err / rms)
}

/// The tower and projector (Q8_0 mmproj) against the fp32 torch reference.
#[test]
#[ignore = "loads the d1-3B mmproj and runs the vision fixtures"]
fn d1_3b_vision_matches_torch() {
    let Some(mmproj) = cached_model("mmproj-d1-3B-Q8_0.gguf") else {
        return;
    };
    let tower = Vision::load(&mmproj).unwrap();
    let d = 1152;
    for case in ["image_coco_cats", "image_wide", "image_large_tiled"] {
        let path = image_fixtures().join(format!("{case}.safetensors"));
        if !path.exists() {
            eprintln!("SKIPPED: {}", path.display());
            continue;
        }
        let f = SafeTensors::open(&path);
        let (shape, values) = f.f32s("pixel_values");
        let shapes = usizes(&f.i64s("spatial_shapes"));
        let (_, embed) = f.f32s("embed.out.fp32");
        let (_, encoded) = f.f32s("vision.out.fp32");
        let (_, projected) = f.f32s("proj.out.fp32");
        let (_, proj_only) = f.f32s("proj_only.out.fp32");
        let start = std::time::Instant::now();
        let (mut proj, mut proj_ref) = (Vec::new(), Vec::new());
        let (mut enc, mut enc_want) = (Vec::new(), Vec::new());
        for t in 0..shape[0] {
            let grid = (shapes[2 * t], shapes[2 * t + 1]);
            let n = grid.0 * grid.1;
            let tile = &values[t * shape[1] * 768..][..n * 768];
            if t == 0 {
                compare(&format!("{case} embed"), &tower.embed(tile, grid), &embed[..n * d]);
            }
            let x = tower.encode(tile, grid);
            let want = &encoded[t * shape[1] * d..][..n * d];
            proj_ref.extend(tower.project(want, grid));
            proj.extend(tower.project(&x, grid));
            enc.extend(x);
            enc_want.extend_from_slice(want);
        }
        let took = start.elapsed().as_secs_f64();
        let (_, enc_err) = compare(&format!("{case} tower"), &enc, &enc_want);
        let (_, only_err) = compare(&format!("{case} projector alone"), &proj_ref, &proj_only);
        let (_, err) = compare(&format!("{case} tower + projector"), &proj, &projected);
        // Torch's own bf16 run, for scale: our weights are Q8_0, not fp32.
        compare(
            &format!("{case} torch bf16 projector alone"),
            &f.f32s("proj_only.out.bf16").1,
            &proj_only,
        );
        let (_, torch_err) =
            compare(&format!("{case} torch bf16 tower + projector"), &f.f32s("proj.out.bf16").1, &projected);
        eprintln!("{case}: {} tiles, {} tokens in {took:.2} s", shape[0], proj.len() / 2048);
        assert!(enc_err < 0.02, "{case}: tower");
        // Q8_0 weights (~1-2% rel-RMS here) are coarser than torch's bf16 (~0.3%).
        assert!(only_err < 0.03, "{case}: projector");
        assert!(err < 2.0 * torch_err.max(0.01), "{case}: tower + projector");
    }
}

/// End to end: every prompts.json case, pictures through the tower into d1-3B
/// Q8_0, against the fp32 torch probabilities.
#[test]
#[ignore = "loads d1-3B Q8_0 and its mmproj and runs every picture prompt"]
fn d1_3b_image_probabilities_match_torch() {
    let Some(dir) = image_prep() else {
        return;
    };
    let (Some(model), Some(mmproj)) = (cached_model("d1-3B-Q8_0.gguf"), cached_model("mmproj-d1-3B-Q8_0.gguf"))
    else {
        return;
    };
    let classifier = Classifier::load(&model, &tokenizer_json(), LinearImpl::Packed, 32_768)
        .unwrap()
        .with_vision(&mmproj)
        .unwrap();
    let mut worst = 0f64;
    for row in &prompt_rows(&dir) {
        let name = row["name"].as_str().unwrap();
        let (state, q) = prompt_case(row);
        let pictures: Vec<_> = prompt_pictures(&dir, row)
            .into_iter()
            .map(|(rgb, _)| {
                image::RgbImage::from_raw(rgb.width() as u32, rgb.height() as u32, rgb.data().to_vec()).unwrap()
            })
            .collect();
        let start = std::time::Instant::now();
        let got = classifier
            .classify_image(&pictures, state.as_ref(), &[("answer".into(), q)])
            .unwrap();
        let took = start.elapsed().as_secs_f64();
        let want: Vec<f64> = serde_json::from_value(row["probs"].clone()).unwrap();
        let diff = got[0].probabilities.iter().zip(&want).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
        eprintln!(
            "{name}: {:?} vs torch {want:?}: max |Δp| {diff:.4}, {took:.1} s (torch {:.1} s)",
            got[0].probabilities,
            row["seconds"].as_f64().unwrap_or(0.0)
        );
        worst = worst.max(diff);
    }
    eprintln!("worst |Δp| over every picture prompt: {worst:.4}");
    assert!(worst < 0.05, "worst |Δp| {worst}");
}
