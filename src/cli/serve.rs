//! `increbot serve`: run the Discord bot.

use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context as _, Result};
use tracing::info;

use crate::persona::{Active, Persona};
use crate::config::Config;
use crate::discord;
use crate::engine::Backends;
use crate::gguf::Gguf;
use crate::gpu::manager::Gpus;
use crate::model::Llama;
use crate::tokenizer::Tokenizer;
use crate::{hub, runtime};

/// Startup order (design.md §6): config → compute pool → model → self-tests →
/// Discord. The bot shows offline until it can answer.
pub fn serve(config_path: &Path) -> Result<()> {
    info!(version = crate::VERSION, "increbot starting");
    let cfg = Config::load(config_path, &|k| std::env::var(k).ok())?;
    // Before the model loads: a broken persona file fails fast.
    let persona = Persona::load(Path::new(&cfg.discord.persona_file), &|k| std::env::var(k).ok())?;
    persona.check_against(&cfg)?;
    // Before the model loads: an unwritable opt-out file fails fast.
    let optouts = discord::OptOuts::load(Path::new(&cfg.discord.optout_file))?;
    info!(opted_out = optouts.len(), "opt-out list loaded");
    let token = std::env::var("DISCORD_TOKEN").unwrap_or_default();
    anyhow::ensure!(
        !token.trim().is_empty(),
        "DISCORD_TOKEN is not set (see .env.example)"
    );

    runtime::init_pool(cfg.runtime.threads, cfg.runtime.pin_threads)?;
    let gguf = Gguf::open(&hub::resolve(&cfg.model.repo, &cfg.model.file)?)?;
    let tok = Arc::new(Tokenizer::load(
        &hub::resolve(&cfg.model.tokenizer_repo, "tokenizer.json")?,
        &gguf,
    )?);
    let mut model = Llama::load(&gguf, &Default::default(), cfg.model.linear_impl)?;
    model.set_kv_f16(cfg.runtime.kv_cache_f16);
    drop(gguf);
    anyhow::ensure!(
        cfg.model.n_ctx <= model.config.n_ctx,
        "n_ctx {} exceeds the model's context {}",
        cfg.model.n_ctx,
        model.config.n_ctx
    );

    // Self-tests: the persona's prompt fits, and one forward pass produces sane logits.
    let d = &cfg.discord;
    let persona = Active::new(persona, &d.prompt_template, &d.bot_name, &[], &tok)?;
    let count = |s: &str| tok.encode_prompt(s).map_or(usize::MAX, |v| v.len());
    let header = persona.prompt.check_header(cfg.generation.header_limit(), &count)?;
    let logits = model.forward(&[tok.bos], &mut model.new_cache())?;
    anyhow::ensure!(
        logits.iter().all(|l| l.is_finite()),
        "self-test forward produced non-finite logits"
    );
    info!(
        header_tokens = header,
        persona = %persona.persona.line(),
        "self-tests passed"
    );

    // The GPUs (gpu-plan.md): each card's manager uploads the weights in the
    // background; until one is ready, jobs run on the CPU.
    let model = Arc::new(model);
    let gpus = if cfg.runtime.gpus.is_empty() {
        None
    } else {
        Gpus::start(cfg.runtime.gpu_settings(cfg.model.n_ctx)?, model.clone())
    };
    let classifier = if cfg.classify.classify_enabled { Some(load_classifier(&cfg)?) } else { None };
    let backends = Backends::new(model, gpus, cfg.model.n_ctx);
    let engine = crate::engine::worker::spawn(backends, tok.clone(), cfg.runtime.kv_cache_slots)?;
    let rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()?;
    rt.block_on(async move {
        use poise::serenity_prelude as serenity;
        let (framework, shared) = discord::framework(cfg, tok, persona, engine, optouts, classifier);
        let mut client = serenity::ClientBuilder::new(token.trim(), discord::intents())
            .framework(framework)
            .await
            .context("building the Discord client")?;
        let shards = client.shard_manager.clone();
        tokio::spawn(signals(shared, shards));
        client.start().await.context("Discord client")
    })
}

/// d1 for the /classify-* commands (`classify` config section), checked with
/// one question before the bot goes online.
fn load_classifier(cfg: &Config) -> Result<crate::classify::Classifier> {
    use crate::classify::{Classifier, Json, Question};
    let k = &cfg.classify;
    let model = hub::resolve(&k.classify_repo, &k.classify_file)?;
    let tokenizer = hub::resolve(&k.classify_tokenizer_repo, "tokenizer.json")?;
    // The text, the question and the prompt around them; also the GPU's cap.
    // A picture is at most 10 tiles and a thumbnail, ~2.8k image tokens.
    let max_ctx = (k.classify_max_text_tokens + 1024).clamp(4096, 8192);
    let mut classifier = Classifier::load(&model, &tokenizer, crate::model::LinearImpl::Packed, max_ctx)
        .context("loading the classifier")?;
    if !k.classify_gpu.is_empty() {
        classifier = classifier.with_gpu(k.classify_gpu.parse()?).context("classify_gpu")?;
    }
    // Pictures run on the CPU even with a GPU set.
    let mmproj = hub::resolve(&k.classify_repo, &k.classify_mmproj)?;
    classifier = classifier.with_vision(&mmproj).context("classify_mmproj")?;
    let q = Question::Noul { instructions: "Is this a greeting?".into(), criteria: None };
    let d = classifier.classify(Some(&Json::String("Hello there!".into())), &[("check".into(), q)])?;
    let yes = d[0].probabilities[0];
    anyhow::ensure!(yes.is_finite() && yes > 0.5, "classifier self-test: P(yes) = {yes} for a greeting");
    info!(file = %k.classify_file, gpu = %k.classify_gpu, p_yes = yes, "classifier ready");
    Ok(classifier)
}

/// SIGHUP reloads the persona; SIGTERM/SIGINT shut down within ~5 s (design.md
/// §6): stop accepting, cancel the running job (its reply keeps the partial text
/// with "(stopped: restarting)"), tell queued requesters, close the gateway.
async fn signals(shared: discord::Data, shards: Arc<poise::serenity_prelude::ShardManager>) {
    use std::sync::atomic::Ordering;
    use tokio::signal::unix::{SignalKind, signal};
    let (Ok(mut term), Ok(mut int), Ok(mut hup)) = (
        signal(SignalKind::terminate()),
        signal(SignalKind::interrupt()),
        signal(SignalKind::hangup()),
    ) else {
        tracing::error!("could not install signal handlers");
        return;
    };
    loop {
        tokio::select! {
            _ = hup.recv() => match shared.reload_persona() {
                Ok(h) => info!(header_tokens = h, "SIGHUP: persona reloaded"),
                Err(e) => tracing::warn!(error = %e, "SIGHUP: reload failed, keeping the old persona"),
            },
            _ = term.recv() => break,
            _ = int.recv() => break,
        }
    }
    info!("shutting down (press Ctrl-C again to force)");
    // A second signal skips the graceful steps.
    tokio::spawn(async move {
        tokio::select! {
            _ = term.recv() => {}
            _ = int.recv() => {}
        }
        tracing::warn!("second signal: exiting now");
        std::process::exit(130);
    });
    shared.shutting_down.store(true, Ordering::Relaxed);
    let (tx, rx) = tokio::sync::oneshot::channel();
    if shared
        .dispatch
        .send(discord::DispatchMsg::Shutdown(tx))
        .is_ok()
    {
        let _ = tokio::time::timeout(Duration::from_secs(4), rx).await;
    }
    // Let reply tasks post their final edits.
    tokio::time::sleep(Duration::from_millis(800)).await;
    // Closing the gateway cleanly makes the bot show offline at once, but
    // serenity's shutdown_all can wait forever (seen in practice); exiting also
    // closes the connection, and Discord marks the bot offline shortly after.
    if tokio::time::timeout(Duration::from_secs(3), shards.shutdown_all())
        .await
        .is_err()
    {
        tracing::warn!("Discord connection didn't close within 3 s; exiting anyway");
    }
    info!("stopped");
    std::process::exit(0);
}
