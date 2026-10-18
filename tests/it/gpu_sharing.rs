//! Sharing the GPUs (gpu-plan.md, milestone 3): an "intruder" allocates 200 MiB
//! on the preferred card in the middle of a reply. The reply must continue on
//! the other card, our VRAM on the first card must be released quickly, and the
//! first card must be used again once it has been free for the hold-off. With
//! both cards taken, replies fall back to the CPU.
//!
//! The intruder is a second Vulkan device in this process: its allocations
//! aren't in the manager's tracker, so the monitor sees them as another
//! program's. Needs both GPUs in `INCREBOT_GPUS` and TinyLlama Q4_K_M; skipped
//! otherwise and in debug builds. `GPU_SHARING_13B=1` runs it on the 13B
//! instead, for the release time of ~8 GB of weights.

use std::sync::Arc;
use std::time::{Duration, Instant};

use increbot::engine::{Backends, Forward};
use increbot::gguf::Gguf;
use increbot::gpu::manager::{GpuSettings, GpuSpec, Gpus};
use increbot::gpu::monitor::{read_u64, vram_used_path};
use increbot::gpu::policy::State;
use increbot::gpu::{Device, PciAddr};
use increbot::model::{KvCache, LinearImpl, Llama};

const HOLDOFF: Duration = Duration::from_secs(2);

fn setup() -> Option<(Arc<Llama>, Vec<PciAddr>)> {
    // The manager's log lines (state changes, release breakdown).
    let _ = tracing_subscriber::fmt()
        .with_env_filter("increbot::gpu=info")
        .with_test_writer()
        .try_init();
    if cfg!(debug_assertions) {
        eprintln!("SKIPPED: run with --release");
        return None;
    }
    let gpus: Vec<PciAddr> = std::env::var("INCREBOT_GPUS")
        .unwrap_or_default()
        .split(',')
        .filter(|s| !s.trim().is_empty())
        .map(|s| s.split('=').next().unwrap().trim().parse().unwrap())
        .collect();
    if gpus.len() < 2 {
        eprintln!("SKIPPED: needs two GPUs in INCREBOT_GPUS");
        return None;
    }
    let (repo, file) = if std::env::var_os("GPU_SHARING_13B").is_some() {
        (
            "TheBloke/LLaMA2-13B-Tiefighter-GGUF",
            "llama2-13b-tiefighter.Q4_K_M.gguf",
        )
    } else {
        (
            "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
            "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
        )
    };
    let Some(path) = hf_hub::Cache::from_env().model(repo.into()).get(file) else {
        eprintln!("SKIPPED: {file} not in the HF cache");
        return None;
    };
    let mut cpu = Llama::load(
        &Gguf::open(&path).unwrap(),
        &Default::default(),
        LinearImpl::Packed,
    )
    .unwrap();
    cpu.set_kv_f16(true);
    Some((Arc::new(cpu), gpus))
}

fn wait_for(what: &str, limit: Duration, mut ok: impl FnMut() -> bool) {
    let start = Instant::now();
    while !ok() {
        assert!(start.elapsed() < limit, "timed out waiting for {what}");
        std::thread::sleep(Duration::from_millis(10));
    }
    eprintln!("{what}: after {:.2}s", start.elapsed().as_secs_f64());
}

fn state(gpus: &Gpus, i: usize) -> State {
    gpus.cards[i].status().state
}

fn argmax(v: &[f32]) -> usize {
    (0..v.len()).max_by(|&i, &j| v[i].total_cmp(&v[j])).unwrap()
}

#[test]
fn intruder_moves_the_reply_and_the_card_comes_back() {
    let _cards = crate::CARDS.lock().unwrap_or_else(|e| e.into_inner());
    let Some((cpu, pcis)) = setup() else {
        return;
    };
    let settings = GpuSettings {
        gpus: pcis.iter().map(|&pci| GpuSpec { pci, label: pci.to_string(), name: String::new() }).collect(),
        other_mib: 100,
        poll_hz: 20,
        holdoff: HOLDOFF,
        standby: true,
        n_ctx: 2048,
    };
    let gpus = Gpus::start(settings, cpu.clone()).expect("GPUs available");
    wait_for("both GPUs ready", Duration::from_secs(60), || {
        state(&gpus, 0) == State::Ready && state(&gpus, 1) == State::Ready
    });
    let mut backends = Backends::new(cpu.clone(), Some(gpus.clone()), 2048);
    let never = || false;

    // A reply: prefill, then decode steps; an intruder arrives on GPU 1 midway.
    let prompt: Vec<u32> = (0..100).map(|i| 300 + i * 7).collect();
    let mut cache = cpu.new_cache();
    let mut reference: KvCache = cpu.new_cache();
    backends.start_job();
    (&mut backends)
        .forward(&prompt, &mut cache, &never)
        .unwrap()
        .unwrap();
    assert_eq!(backends.route(), format!("gpu {}", pcis[0]));
    cpu.forward(&prompt, &mut reference).unwrap();

    let intruder_dev = Device::open(pcis[0]).unwrap();
    let baseline = gpus.cards[0].status().reading.baseline;
    let path = vram_used_path(pcis[0]);
    let mut intruder = None;
    let mut poller = None;
    let mut last = Vec::new();
    for step in 0..400u32 {
        if step == 20 {
            intruder = Some(intruder_dev.vram(200 << 20).unwrap());
            // From the intruder's allocation until the card holds only the
            // baseline and the intruder (± 64 MiB): our release time.
            let (path, arrived) = (path.clone(), Instant::now());
            poller = Some(std::thread::spawn(move || {
                loop {
                    if read_u64(&path).unwrap() < baseline + (200 << 20) + (64 << 20) {
                        return arrived.elapsed();
                    }
                    std::thread::sleep(Duration::from_micros(500));
                }
            }));
        }
        let tok = 1000 + step % 500;
        last = (&mut backends)
            .forward(&[tok], &mut cache, &never)
            .unwrap()
            .unwrap();
        cpu.forward(&[tok], &mut reference).unwrap();
        if step > 20 && backends.route().contains(&pcis[1].to_string()) && step > 60 {
            break;
        }
    }
    assert_eq!(
        backends.route(),
        format!("gpu {} → gpu {}", pcis[0], pcis[1]),
        "the reply moved to the second GPU"
    );
    assert_eq!(cache.tokens, reference.tokens);
    let want = cpu.forward(&[7], &mut reference).unwrap();
    let got = (&mut backends)
        .forward(&[7], &mut cache, &never)
        .unwrap()
        .unwrap();
    assert_eq!(
        argmax(&got),
        argmax(&want),
        "continued reply agrees with the CPU"
    );
    drop(last);

    // Our VRAM on GPU 1 is gone (the monitor's release time, and sysfs).
    wait_for("GPU 1 in backoff", Duration::from_secs(2), || {
        state(&gpus, 0) == State::Backoff
    });
    let st = gpus.cards[0].status();
    let p = st.last_preemption.expect("a preemption was recorded");
    let detect_to_free = p.release;
    let total = poller.unwrap().join().unwrap();
    eprintln!(
        "release: {:.1} ms from detection to freed; {:.1} ms from the intruder's allocation \
         until our VRAM was gone from sysfs",
        detect_to_free.as_secs_f64() * 1e3,
        total.as_secs_f64() * 1e3
    );
    assert!(
        detect_to_free < Duration::from_millis(150),
        "freed in {detect_to_free:?}"
    );
    // Detection (≤ 50 ms at 20 Hz) + stop + free + the kernel's ~12 ms/GiB.
    assert!(
        total < Duration::from_millis(300),
        "released after {total:?}"
    );

    // The intruder leaves: GPU 1 is back after the hold-off and serves new jobs.
    drop(intruder.take());
    let gone = Instant::now();
    wait_for("GPU 1 ready again", Duration::from_secs(30), || {
        state(&gpus, 0) == State::Ready
    });
    assert!(gone.elapsed() >= HOLDOFF, "not before the hold-off");
    backends.start_job();
    (&mut backends)
        .forward(&[8], &mut cache, &never)
        .unwrap()
        .unwrap();
    assert_eq!(backends.route(), format!("gpu {}", pcis[0]));

    // Both cards taken: the CPU.
    let other_dev = Device::open(pcis[1]).unwrap();
    let a = intruder_dev.vram(200 << 20).unwrap();
    let b = other_dev.vram(200 << 20).unwrap();
    wait_for("both GPUs in backoff", Duration::from_secs(2), || {
        state(&gpus, 0) == State::Backoff && state(&gpus, 1) == State::Backoff
    });
    backends.start_job();
    (&mut backends)
        .forward(&[9], &mut cache, &never)
        .unwrap()
        .unwrap();
    assert_eq!(backends.route(), "cpu");
    drop((a, b));
}

/// A whole job through the inference thread: generated on the preferred GPU,
/// and the next job in the same channel reuses the cached prefix.
#[test]
fn worker_replies_on_the_gpu() {
    use increbot::engine::sampler::SamplerParams;
    use increbot::engine::worker::{CacheKey, Event, Job, spawn};
    use increbot::tokenizer::Tokenizer;

    let _cards = crate::CARDS.lock().unwrap_or_else(|e| e.into_inner());
    let Some((cpu, pcis)) = setup() else {
        return;
    };
    let settings = GpuSettings {
        gpus: pcis.iter().map(|&pci| GpuSpec { pci, label: pci.to_string(), name: String::new() }).collect(),
        other_mib: 100,
        poll_hz: 20,
        holdoff: HOLDOFF,
        standby: false,
        n_ctx: 2048,
    };
    let gpus = Gpus::start(settings, cpu.clone()).expect("GPUs available");
    wait_for("GPU ready", Duration::from_secs(60), || {
        state(&gpus, 0) == State::Ready
    });
    // Standby off: the second card stays empty while the first is fine.
    assert_eq!(state(&gpus, 1), State::Free);
    let path = hf_hub::Cache::from_env()
        .model("hf-internal-testing/llama-tokenizer".into())
        .get("tokenizer.json")
        .expect("tokenizer in the HF cache");
    let gguf_path = hf_hub::Cache::from_env()
        .model("TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF".into())
        .get("tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf")
        .unwrap();
    let tok = Arc::new(Tokenizer::load(&path, &Gguf::open(&gguf_path).unwrap()).unwrap());
    let h = spawn(Backends::new(cpu, Some(gpus), 2048), tok.clone(), 2).unwrap();
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let job = |prompt: &str| {
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
        let job = Job {
            key: CacheKey::Channel(1),
            prompt: tok.encode_prompt(prompt).unwrap(),
            sampler: SamplerParams::greedy(),
            seed: 1,
            history: vec![],
            max_new_tokens: 5,
            max_new_tokens_gpu: 30,
                        wrap_up_after: 0,
                wrap_up_after_gpu: 0,
stops: vec![],
        max_lines: None,
        labels: false,
            timeout: Duration::from_secs(60),
            cancel: Arc::default(),
            events: tx,
            best_of: std::num::NonZeroUsize::MIN,
            score: None,
            bans: None,
        };
        rt.block_on(async {
            h.jobs.send(job).await.unwrap();
            let mut text = String::new();
            loop {
                match rx.recv().await.unwrap() {
                    Event::Text(t) => text += &t,
                    Event::Finished(out) => break (text, out),
                    Event::Failed(e) => panic!("job failed: {e}"),
                    Event::Started | Event::Paused(_) => {}
                }
            }
        })
    };
    let (text, out) = job("The capital of France is");
    eprintln!("reply: {text:?}");
    // The GPU limit (30), not the CPU one (5).
    assert!(out.tokens.len() > 5, "{} tokens", out.tokens.len());
    assert!(text.to_lowercase().contains("paris"), "{text:?}");
    assert_eq!(
        h.stats.lock().unwrap().last_backend,
        format!("gpu {}", pcis[0])
    );
    let (_, out) = job("The capital of France is Paris. The capital of Italy is");
    assert!(out.reused_tokens > 0, "prefix reused");
    assert_eq!(
        h.stats.lock().unwrap().last_backend,
        format!("gpu {}", pcis[0])
    );
}
