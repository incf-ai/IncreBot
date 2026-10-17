//! Tokenizer + streaming decode against the real Llama 2 tokenizer (design.md §5, §10).
//! Skipped when the TinyLlama GGUF (for its vocab) is not in the HF cache.

use increbot::engine::decode::{Step, StreamDecoder};
use increbot::gguf::Gguf;
use increbot::hub;
use increbot::tokenizer::Tokenizer;

fn tokenizer() -> Option<Tokenizer> {
    let gguf = hf_hub::Cache::from_env()
        .model("TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF".into())
        .get("tinyllama-1.1b-chat-v1.0.Q8_0.gguf");
    let Some(gguf) = gguf else {
        eprintln!("SKIPPED: TinyLlama GGUF not in the HF cache");
        return None;
    };
    let gguf = Gguf::open(&gguf).unwrap();
    Some(
        Tokenizer::load(
            &hub::resolve("hf-internal-testing/llama-tokenizer", "tokenizer.json").unwrap(),
            &gguf,
        )
        .unwrap(),
    )
}

#[test]
fn exactly_one_bos() {
    let Some(tok) = tokenizer() else { return };
    let ids = tok.encode_prompt("Hello there").unwrap();
    assert_eq!(ids[0], tok.bos);
    assert_eq!(ids.iter().filter(|&&t| t == tok.bos).count(), 1);
}

#[test]
fn streaming_round_trips_multibyte_text_without_replacement_chars() {
    let Some(tok) = tokenizer() else { return };
    let text = "Cats 🐈 like naïve façades — 日本語 too 😀!";
    let ids = tok.encode_prompt(text).unwrap();
    // Emoji and CJK go through byte-fallback tokens, splitting characters across tokens.
    assert!(
        ids.len() > text.split_whitespace().count() + 6,
        "expected byte-fallback tokens: {ids:?}"
    );

    let mut dec = StreamDecoder::new(vec![]);
    let decode = |ids: &[u32]| tok.decode(ids);
    let mut out = String::new();
    for &id in &ids[1..] {
        match dec.push(id, decode).unwrap() {
            Step::Text(t) => {
                assert!(
                    !t.contains('\u{FFFD}'),
                    "released a replacement char: {t:?}"
                );
                out += &t;
            }
            Step::Stop(_) => unreachable!(),
        }
    }
    out += &dec.finish(decode).unwrap();
    assert_eq!(out, text);
}

#[test]
fn dry_breakers_cover_newlines_colons_quotes_and_asterisks() {
    let Some(tok) = tokenizer() else { return };
    let breakers = tok.ids_containing("\n:\"*");
    let ids = |s: &str| tok.encode_prompt(s).unwrap().split_off(1);
    // "floob: hi *wiggles*\n": the colon, asterisks and newline are breakers;
    // the words are not.
    for (text, is_breaker) in [
        ("\n", true),
        (":", true),
        ("*", true),
        (" *", true),
        ("\"", true),
        (" floob", false),
        (" hi", false),
    ] {
        let t = *ids(&format!("a{text}")).last().unwrap();
        assert_eq!(breakers.contains(&t), is_breaker, "{text:?} = token {t}");
    }
    assert!(!breakers.contains(&tok.bos) && !breakers.contains(&tok.eos));
    assert!(breakers.len() < 1000, "{} breakers", breakers.len());
}

#[test]
fn first_char_ids_cover_every_token_opening_with_it() {
    let Some(tok) = tokenizer() else { return };
    let banned = tok.ids_starting_with("!");
    let ids = |s: &str| tok.encode_prompt(s).unwrap().split_off(1);
    for (text, is_banned) in [("!", true), (" !", true), ("!!", true), (" !!!", true), (" hi", false), (" ?", false)] {
        let t = *ids(&format!("a{text}")).last().unwrap();
        assert_eq!(banned.contains(&t), is_banned, "{text:?} = token {t}");
    }
    assert!(!banned.contains(&tok.bos) && !banned.contains(&tok.eos));
    // Special tokens aren't their text ("</s>").
    let angle = tok.ids_starting_with("<");
    assert!(!angle.contains(&tok.bos) && !angle.contains(&tok.eos) && !angle.contains(&0));
    assert!(angle.contains(ids("a <").last().unwrap()));
}

#[test]
fn newline_ids_are_whitespace_with_a_line_break() {
    let Some(tok) = tokenizer() else { return };
    let nl = tok.newline_ids();
    let ids = |s: &str| tok.encode_prompt(s).unwrap().split_off(1);
    assert!(nl.contains(ids("a\n").last().unwrap()));
    assert!(!nl.contains(ids("a b").last().unwrap()) && !nl.contains(&tok.eos));
}
