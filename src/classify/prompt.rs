//! The prompt and the readout, ported from the pinned remote code's `prompt.py` (with the
//! defaults `runner.py` uses): a System One decision is one forward pass that stops at the answer
//! slot, and the answer is a softmax over its options' tokens.
//!
//! A state is any JSON value, a string state being the string itself; `None` is no state.
//! [`Json`] keeps object keys in their order, and [`Json::dumps`] writes what Python's
//! `json.dumps(state, ensure_ascii=False, indent=2)` writes. Integers outside `i64`/`u64` arrive
//! from the JSON parser as floats, where Python keeps them exact. Calibration is not ported (the
//! runner's default is none), nor are pictures (`images` markup).

use std::collections::HashSet;

use serde::de::{self, Deserialize, Deserializer, MapAccess, SeqAccess, Visitor};
use serde_json::{Map, Value, json};

use super::Error;
use super::tokenizer::Tokenizer;

/// Opens a chat turn.
pub const IM_START: &str = "<|im_start|>";
/// Closes a chat turn.
pub const IM_END: &str = "<|im_end|>";
const YES_FORMS: [&str; 3] = ["yes", "Yes", "YES"];
const NO_FORMS: [&str; 3] = ["no", "No", "NO"];
const ISOLATED: &str = "You are a System One decision model. Answer only the current isolated \
                        question from the shared state. Other questions do not exist.";

// --------------------------------------------------------------------------------------------- //
// JSON with Python's semantics
// --------------------------------------------------------------------------------------------- //

/// A JSON value as Python's `json.loads` gives it: objects keep their key order, and a repeated key
/// keeps its first place with its last value.
#[derive(Debug, Clone, PartialEq)]
pub enum Json {
    /// `null` (`None`).
    Null,
    /// `true` or `false`.
    Bool(bool),
    /// An integer.
    Int(i128),
    /// A number with a fraction or an exponent.
    Float(f64),
    /// A string.
    String(String),
    /// An array.
    Array(Vec<Json>),
    /// An object, in key order.
    Object(Vec<(String, Json)>),
}

impl<'de> Deserialize<'de> for Json {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        deserializer.deserialize_any(JsonVisitor)
    }
}

struct JsonVisitor;

impl<'de> Visitor<'de> for JsonVisitor {
    type Value = Json;

    fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("a JSON value")
    }

    fn visit_unit<E: de::Error>(self) -> Result<Json, E> {
        Ok(Json::Null)
    }

    fn visit_bool<E: de::Error>(self, v: bool) -> Result<Json, E> {
        Ok(Json::Bool(v))
    }

    fn visit_i64<E: de::Error>(self, v: i64) -> Result<Json, E> {
        Ok(Json::Int(i128::from(v)))
    }

    fn visit_u64<E: de::Error>(self, v: u64) -> Result<Json, E> {
        Ok(Json::Int(i128::from(v)))
    }

    fn visit_f64<E: de::Error>(self, v: f64) -> Result<Json, E> {
        Ok(Json::Float(v))
    }

    fn visit_str<E: de::Error>(self, v: &str) -> Result<Json, E> {
        Ok(Json::String(v.to_owned()))
    }

    fn visit_string<E: de::Error>(self, v: String) -> Result<Json, E> {
        Ok(Json::String(v))
    }

    fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<Json, A::Error> {
        let mut out = Vec::new();
        while let Some(v) = seq.next_element()? {
            out.push(v);
        }
        Ok(Json::Array(out))
    }

    fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<Json, A::Error> {
        let mut out: Vec<(String, Json)> = Vec::new();
        while let Some((k, v)) = map.next_entry::<String, Json>()? {
            match out.iter_mut().find(|(key, _)| *key == k) {
                Some(slot) => slot.1 = v,
                None => out.push((k, v)),
            }
        }
        Ok(Json::Object(out))
    }
}

impl Json {
    /// Parses JSON text.
    ///
    /// # Errors
    ///
    /// If `text` is not JSON.
    pub fn parse(text: &str) -> Result<Self, Error> {
        serde_json::from_str(text).map_err(|e| Error::new(format!("JSON: {e}")))
    }

    /// The value of `key` in an object.
    #[must_use]
    pub fn get(&self, key: &str) -> Option<&Self> {
        match self {
            Self::Object(items) => items.iter().find(|(k, _)| k == key).map(|(_, v)| v),
            _ => None,
        }
    }

    /// `json.dumps(self, ensure_ascii=False, indent=2)`.
    #[must_use]
    pub fn dumps(&self) -> String {
        let mut out = String::new();
        self.dump(&mut out, 0);
        out
    }

    fn dump(&self, out: &mut String, level: usize) {
        let indent = |out: &mut String, level: usize| {
            out.push('\n');
            out.extend(std::iter::repeat_n(' ', 2 * level));
        };
        match self {
            Self::Array(items) if items.is_empty() => out.push_str("[]"),
            Self::Object(items) if items.is_empty() => out.push_str("{}"),
            Self::Array(items) => {
                out.push('[');
                for (i, v) in items.iter().enumerate() {
                    if i > 0 {
                        out.push(',');
                    }
                    indent(out, level + 1);
                    v.dump(out, level + 1);
                }
                indent(out, level);
                out.push(']');
            }
            Self::Object(items) => {
                out.push('{');
                for (i, (k, v)) in items.iter().enumerate() {
                    if i > 0 {
                        out.push(',');
                    }
                    indent(out, level + 1);
                    json_string(out, k);
                    out.push_str(": ");
                    v.dump(out, level + 1);
                }
                indent(out, level);
                out.push('}');
            }
            Self::Null => out.push_str("null"),
            Self::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
            Self::Int(i) => {
                out.push_str(&i.to_string());
            }
            Self::Float(x) if x.is_nan() => out.push_str("NaN"),
            Self::Float(x) if x.is_infinite() => {
                out.push_str(if *x > 0.0 { "Infinity" } else { "-Infinity" });
            }
            Self::Float(x) => out.push_str(&py_float(*x)),
            Self::String(s) => json_string(out, s),
        }
    }

    /// Python's `str()` of a scalar: `None`, `True`, `1.0`, or the string itself.
    fn py_str(&self) -> Result<String, Error> {
        Ok(match self {
            Self::Null => "None".to_owned(),
            Self::Bool(true) => "True".to_owned(),
            Self::Bool(false) => "False".to_owned(),
            Self::Int(i) => i.to_string(),
            Self::Float(x) if x.is_nan() => "nan".to_owned(),
            Self::Float(x) if x.is_infinite() => if *x > 0.0 { "inf" } else { "-inf" }.to_owned(),
            Self::Float(x) => py_float(*x),
            Self::String(s) => s.clone(),
            Self::Array(_) | Self::Object(_) => {
                return Err(Error::new(
                    "Python's str() of a list or an object is not ported",
                ));
            }
        })
    }

    /// `_is_scalar`: `None`, or a string, number or bool whose `str()` has no newline.
    fn is_scalar(&self) -> bool {
        match self {
            Self::Null | Self::Bool(_) | Self::Int(_) | Self::Float(_) => true,
            Self::String(s) => !s.contains('\n'),
            Self::Array(_) | Self::Object(_) => false,
        }
    }

    /// `str(v)` with `None` as the empty string, for the `sections` style.
    fn section_str(&self) -> Result<String, Error> {
        match self {
            Self::Null => Ok(String::new()),
            v => v.py_str(),
        }
    }
}

/// A string as Python's JSON encoder writes it with `ensure_ascii=False`.
fn json_string(out: &mut String, s: &str) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{8}' => out.push_str("\\b"),
            '\u{c}' => out.push_str("\\f"),
            c if u32::from(c) < 0x20 => {
                out.push_str(&format!("\\u{:04x}", u32::from(c)));
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

/// Python's `repr()` of a finite float: the shortest digits that round-trip, positional when the
/// decimal exponent is in `-4..16`, else `d.ddde±XX`.
fn py_float(x: f64) -> String {
    // Rust's `{:e}` gives the same shortest round-trip digits: `-1.2345e-5`.
    let sci = format!("{x:e}");
    let (mantissa, exp) = sci.split_once('e').unwrap_or((sci.as_str(), "0"));
    let exp: i32 = exp.parse().unwrap_or(0);
    let negative = mantissa.starts_with('-');
    let digits: String = mantissa.chars().filter(char::is_ascii_digit).collect();
    let n = i32::try_from(digits.len()).unwrap_or(i32::MAX);
    let decpt = exp + 1;
    let mut out = String::from(if negative { "-" } else { "" });
    if (-3..=16).contains(&decpt) {
        if decpt <= 0 {
            out.push_str("0.");
            out.extend(std::iter::repeat_n(
                '0',
                usize::try_from(-decpt).unwrap_or(0),
            ));
            out.push_str(&digits);
        } else if decpt >= n {
            out.push_str(&digits);
            out.extend(std::iter::repeat_n(
                '0',
                usize::try_from(decpt - n).unwrap_or(0),
            ));
            out.push_str(".0");
        } else {
            let (int, frac) = digits.split_at(usize::try_from(decpt).unwrap_or(0));
            out.push_str(&format!("{int}.{frac}"));
        }
    } else {
        let mut chars = digits.chars();
        out.extend(chars.next());
        let rest: String = chars.collect();
        if !rest.is_empty() {
            out.push_str(&format!(".{rest}"));
        }
        let sign = if exp < 0 { '-' } else { '+' };
        out.push_str(&format!("e{sign}{:02}", exp.unsigned_abs()));
    }
    out
}

// --------------------------------------------------------------------------------------------- //
// questions
// --------------------------------------------------------------------------------------------- //

/// A question in the Decision Index's schema.
#[derive(Debug, Clone, PartialEq)]
pub enum Question {
    /// Yes or no; the optional criteria say what each side means (`true`, `false`).
    Noul {
        /// The question.
        instructions: String,
        /// `(true, false)` when the criteria are given and not empty; a missing side is `None`.
        criteria: Option<(Option<Json>, Option<Json>)>,
    },
    /// One of named options, each with an optional description.
    Choice {
        /// The question.
        instructions: String,
        /// `(label, description)` in order.
        criteria: Vec<(String, Option<String>)>,
    },
    /// A level `0..n` with a name for each.
    Score {
        /// The question.
        instructions: String,
        /// Each level's name.
        criteria: Vec<String>,
    },
}

fn string(v: &Json, what: &str) -> Result<String, Error> {
    match v {
        Json::String(s) => Ok(s.clone()),
        _ => Err(Error::new(format!("{what}: not a string"))),
    }
}

impl Question {
    /// A question in the Decision Index's JSON, `{"type": "noul" | "choice" | "score",
    /// "instructions", "criteria"}`; the type defaults to `choice` (`as_question`).
    ///
    /// # Errors
    ///
    /// If a field is missing or of the wrong type.
    pub fn from_json(q: &Json) -> Result<Self, Error> {
        let instructions = string(
            q.get("instructions")
                .ok_or_else(|| Error::new("question: no instructions"))?,
            "instructions",
        )?;
        let kind = match q.get("type") {
            None => "choice".to_owned(),
            Some(t) => string(t, "type")?,
        };
        let criteria = q.get("criteria");
        match kind.as_str() {
            "noul" => {
                let criteria = match criteria {
                    None | Some(Json::Null) => None,
                    Some(Json::Object(items)) if items.is_empty() => None,
                    Some(c @ Json::Object(_)) => {
                        Some((c.get("true").cloned(), c.get("false").cloned()))
                    }
                    Some(_) => return Err(Error::new("noul criteria: not an object")),
                };
                Ok(Self::Noul {
                    instructions,
                    criteria,
                })
            }
            "score" => match criteria {
                Some(Json::Array(levels)) => Ok(Self::Score {
                    instructions,
                    criteria: levels
                        .iter()
                        .map(|l| string(l, "score level"))
                        .collect::<Result<_, _>>()?,
                }),
                _ => Err(Error::new("score criteria: not a list")),
            },
            _ => match criteria {
                Some(Json::Object(options)) => Ok(Self::Choice {
                    instructions,
                    criteria: options
                        .iter()
                        .map(|(k, v)| match v {
                            Json::Null => Ok((k.clone(), None)),
                            v => Ok((k.clone(), Some(string(v, "option description")?))),
                        })
                        .collect::<Result<_, Error>>()?,
                }),
                _ => Err(Error::new("choice criteria: not an object")),
            },
        }
    }
}

// --------------------------------------------------------------------------------------------- //
// verbalizer and readout
// --------------------------------------------------------------------------------------------- //

/// Python's `str.isspace`: Unicode whitespace plus the ASCII separators `\x1c`–`\x1f`.
fn py_space(c: char) -> bool {
    c.is_whitespace() || ('\u{1c}'..='\u{1f}').contains(&c)
}

/// Native letters when the labels already are single letters, else `A`..`Z`, else `00`..
/// (`option_codes`).
#[must_use]
pub fn option_codes<S: AsRef<str>>(labels: &[S]) -> Vec<String> {
    let labs: Vec<&str> = labels
        .iter()
        .map(|l| l.as_ref().trim_matches(py_space))
        .collect();
    let letter = |k: &&str| {
        let mut chars = k.chars();
        matches!((chars.next(), chars.next()), (Some(c), None) if c.is_alphabetic())
    };
    if !labs.is_empty() && labs.iter().all(letter) {
        return labs.into_iter().map(str::to_owned).collect();
    }
    if labs.len() <= 26 {
        return ('A'..='Z').take(labs.len()).map(String::from).collect();
    }
    (0..labs.len()).map(|i| format!("{i:02}")).collect()
}

/// Codes to try, in order, when an option's own code is not one free token (`_FALLBACK_POOL`).
fn fallback_pool() -> impl Iterator<Item = String> {
    let upper = ('A'..='Z').map(String::from);
    let pairs = (0..100).map(|i| format!("{i:02}"));
    let lower = ('a'..='z').map(String::from);
    let hashes = (0..200).map(|i| format!("#{i}"));
    let two = ('A'..='Z').flat_map(|a| ('A'..='Z').map(move |b| format!("{a}{b}")));
    upper.chain(pairs).chain(lower).chain(hashes).chain(two)
}

/// Every label's distinct single-token code and its id (`aliases`).
///
/// # Errors
///
/// If the tokenizer fails, or no single-token code is left.
pub fn aliases<S: AsRef<str>>(
    tokenizer: &Tokenizer,
    labels: &[S],
) -> Result<Vec<(String, u32)>, Error> {
    let mut used = HashSet::new();
    let mut out = Vec::new();
    let mut take = |raw: String, used: &mut HashSet<u32>| -> Result<bool, Error> {
        match tokenizer.single(&raw)? {
            Some(id) if used.insert(id) => {
                out.push((raw, id));
                Ok(true)
            }
            _ => Ok(false),
        }
    };
    for code in option_codes(labels) {
        if take(code, &mut used)? {
            continue;
        }
        let mut found = false;
        for raw in fallback_pool() {
            if take(raw, &mut used)? {
                found = true;
                break;
            }
        }
        if !found {
            return Err(Error::new(format!(
                "no single-token alias left for {} options",
                labels.len()
            )));
        }
    }
    Ok(out)
}

/// The single-token ids among `texts`, without repeats (`_ids`).
fn ids<S: AsRef<str>>(tokenizer: &Tokenizer, texts: &[S]) -> Result<Vec<u32>, Error> {
    let mut out = Vec::new();
    for t in texts {
        if let Some(id) = tokenizer.single(t.as_ref())?
            && !out.contains(&id)
        {
            out.push(id);
        }
    }
    Ok(out)
}

/// The token ids to score, one group per option, max-pooled: the answer is a softmax over these
/// and nothing else (`readout_ids`). A noul's groups are yes then no.
///
/// # Errors
///
/// If the tokenizer fails, or an option has no single-token form.
pub fn readout_ids(tokenizer: &Tokenizer, q: &Question) -> Result<Vec<Vec<u32>>, Error> {
    match q {
        Question::Noul { .. } => {
            let (yes, no) = (ids(tokenizer, &YES_FORMS)?, ids(tokenizer, &NO_FORMS)?);
            if yes.is_empty() || no.is_empty() {
                return Err(Error::new("tokenizer has no single-token yes/no"));
            }
            Ok(vec![yes, no])
        }
        Question::Score { criteria, .. } => {
            let groups = (0..criteria.len())
                .map(|i| ids(tokenizer, &[i.to_string()]))
                .collect::<Result<Vec<_>, _>>()?;
            if groups.iter().any(Vec::is_empty) {
                return Err(Error::new(format!(
                    "score with {} levels needs single-token digits; the primitive is defined for 2 to 10",
                    criteria.len()
                )));
            }
            Ok(groups)
        }
        Question::Choice { criteria, .. } => {
            let labels: Vec<&str> = criteria.iter().map(|(l, _)| l.as_str()).collect();
            let mut groups = Vec::new();
            for (code, id) in aliases(tokenizer, &labels)? {
                let mut group = vec![id];
                group.extend(
                    ids(tokenizer, &[format!(" {code}")])?
                        .into_iter()
                        .filter(|&i| i != id),
                );
                groups.push(group);
            }
            if groups.is_empty() {
                return Err(Error::new("choice with no options"));
            }
            Ok(groups)
        }
    }
}

/// The option probabilities from the log-probabilities at the answer slot (`logz`, indexed by
/// token id): a softmax over each option's best form (`readout`, with no calibration).
///
/// # Errors
///
/// As [`readout_ids`], or if an option's id is outside `logz`.
pub fn readout(tokenizer: &Tokenizer, q: &Question, logz: &[f32]) -> Result<Vec<f64>, Error> {
    let scores = readout_ids(tokenizer, q)?
        .iter()
        .map(|group| {
            group
                .iter()
                .map(|&id| {
                    usize::try_from(id)
                        .ok()
                        .and_then(|i| logz.get(i))
                        .map(|&v| f64::from(v))
                        .ok_or_else(|| Error::new(format!("token {id} outside the vocabulary")))
                })
                .try_fold(f64::NEG_INFINITY, |m, v| v.map(|v| m.max(v)))
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(softmax(&scores))
}

fn softmax(scores: &[f64]) -> Vec<f64> {
    let m = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = scores.iter().map(|s| (s - m).exp()).collect();
    let sum: f64 = exps.iter().sum();
    exps.iter().map(|e| e / sum).collect()
}

/// A question's answer as `api.answer` gives it: a noul's P(yes); a choice's pick, confidence
/// and probabilities; a score's expected level, confidence, probabilities and legend.
#[must_use]
pub fn answer(q: &Question, probs: &[f64]) -> Value {
    let best = probs
        .iter()
        .enumerate()
        .fold(None, |b: Option<(usize, f64)>, (i, &p)| match b {
            Some((_, bp)) if bp >= p => b,
            _ => Some((i, p)),
        });
    let (best, confidence) = best.unwrap_or((0, f64::NAN));
    match q {
        Question::Noul { .. } => json!({"type": "noul", "noul": probs.first()}),
        Question::Choice { criteria, .. } => {
            let names: Map<String, Value> = criteria
                .iter()
                .zip(probs)
                .map(|((n, _), &p)| (n.clone(), json!(p)))
                .collect();
            json!({
                "type": "choice",
                "choice": criteria.get(best).map(|(n, _)| n),
                "confidence": confidence,
                "probabilities": names,
            })
        }
        Question::Score { criteria, .. } => {
            let score: f64 = probs
                .iter()
                .enumerate()
                .map(|(i, p)| f64::from(u32::try_from(i).unwrap_or(u32::MAX)) * p)
                .sum();
            let probabilities: Map<String, Value> = probs
                .iter()
                .enumerate()
                .map(|(i, &p)| (i.to_string(), json!(p)))
                .collect();
            let legend: Map<String, Value> = criteria
                .iter()
                .enumerate()
                .map(|(i, t)| (i.to_string(), json!(t)))
                .collect();
            json!({
                "type": "score",
                "score": score,
                "confidence": confidence,
                "probabilities": probabilities,
                "legend": legend,
            })
        }
    }
}

// --------------------------------------------------------------------------------------------- //
// state and question rendering
// --------------------------------------------------------------------------------------------- //

/// How a state is rendered (`DEFAULT_STATE_STYLE` is `JsonOnly`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum StateStyle {
    /// Every state as the JSON object it is.
    #[default]
    JsonOnly,
    /// JSON, with three shortcuts: `Message:`, `Passage:` / `Asked:`, a lone question's text.
    Json,
    /// Nested states as labelled blocks, with the shortcuts.
    Sections,
    /// Nested states as labelled blocks.
    SectionsOnly,
}

/// The system turn (`SYSTEMS`): none by default.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum System {
    /// No system turn.
    #[default]
    None,
    /// Answer only the current question.
    Isolated,
}

/// How an option line is written.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum OptionStyle {
    /// `A description` (the label, `_` as spaces, when there is none).
    #[default]
    Desc,
    /// `A label: description` when the label says something the description does not.
    NameDesc,
}

/// Renders prompts as `runner.SystemOne` does.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Renderer {
    /// The BOS string at the head of every prompt.
    pub bos: String,
    /// What sits between the assistant header and the answer slot: nothing on LFM2-VL.
    pub lead: String,
    /// How the state is written.
    pub style: StateStyle,
    /// The system turn.
    pub system: System,
    /// How option lines are written.
    pub option_style: OptionStyle,
}

fn sections(obj: &Json, path: &str, out: &mut Vec<String>) -> Result<(), Error> {
    let head = if path.is_empty() {
        String::new()
    } else {
        format!("[{path}]\n")
    };
    match obj {
        Json::Object(items) => {
            let scalars = items
                .iter()
                .filter(|(_, v)| v.is_scalar())
                .map(|(k, v)| Ok(format!("{k}: {}", v.section_str()?)))
                .collect::<Result<Vec<_>, Error>>()?;
            if !scalars.is_empty() {
                out.push(format!("{head}{}", scalars.join("\n")));
            }
            for (k, v) in items.iter().filter(|(_, v)| !v.is_scalar()) {
                let sub = if path.is_empty() {
                    k.clone()
                } else {
                    format!("{path}.{k}")
                };
                sections(v, &sub, out)?;
            }
        }
        Json::Array(items) if !items.is_empty() && items.iter().all(Json::is_scalar) => {
            let lines = items
                .iter()
                .map(|v| Ok(format!("- {}", v.section_str()?)))
                .collect::<Result<Vec<_>, Error>>()?;
            out.push(format!("{head}{}", lines.join("\n")));
        }
        Json::Array(items) => {
            let n = items.len();
            for (i, v) in items.iter().enumerate() {
                let i = i + 1;
                let sub = if path.is_empty() {
                    format!("{i}/{n}")
                } else {
                    format!("{path} {i}/{n}")
                };
                sections(v, &sub, out)?;
            }
        }
        v => out.push(format!("{head}{}", v.section_str()?)),
    }
    Ok(())
}

fn render_state(state: &Json) -> Result<String, Error> {
    if let Json::String(s) = state {
        return Ok(s.clone());
    }
    let mut out = Vec::new();
    sections(state, "", &mut out)?;
    Ok(out.join("\n\n"))
}

/// `opt` and digits: a placeholder label (`^opt\d+$`; Python's `$` also matches before a final
/// newline). Only ASCII digits count, where Python's `\d` takes any decimal digit.
fn placeholder(label: &str) -> bool {
    let label = label.strip_suffix('\n').unwrap_or(label);
    label
        .strip_prefix("opt")
        .is_some_and(|d| !d.is_empty() && d.chars().all(|c| c.is_ascii_digit()))
}

impl Renderer {
    /// The runner's defaults with `tokenizer`'s BOS: no lead, `json_only` states, no system turn,
    /// `desc` option lines.
    #[must_use]
    pub fn new(tokenizer: &Tokenizer) -> Self {
        Self {
            bos: tokenizer.bos().to_owned(),
            lead: String::new(),
            style: StateStyle::default(),
            system: System::default(),
            option_style: OptionStyle::default(),
        }
    }

    /// The block that precedes `QUESTION:` (`state_block`).
    ///
    /// # Errors
    ///
    /// If a shortcut or a section needs Python's `str()` of a list or an object.
    pub fn state_block(&self, state: &Json) -> Result<String, Error> {
        let shortcuts = matches!(self.style, StateStyle::Json | StateStyle::Sections);
        if let (true, Json::Object(items)) = (shortcuts, state) {
            let keys: HashSet<&str> = items.iter().map(|(k, _)| k.as_str()).collect();
            let field = |k: &str| state.get(k).map_or(Ok(String::new()), Json::py_str);
            if keys == HashSet::from(["text"]) {
                return Ok(format!("Message: {}\n\n", field("text")?));
            }
            if keys == HashSet::from(["passage", "question"]) {
                return Ok(format!(
                    "Passage: {}\n\nAsked: {}\n\n",
                    field("passage")?,
                    field("question")?
                ));
            }
            if keys == HashSet::from(["question"]) {
                return Ok(format!("{}\n\n", field("question")?));
            }
        }
        match (self.style, state) {
            (StateStyle::Json | StateStyle::JsonOnly, Json::String(s)) => Ok(format!("{s}\n\n")),
            (StateStyle::Json | StateStyle::JsonOnly, state) => {
                Ok(format!("{}\n\n", state.dumps()))
            }
            (StateStyle::Sections | StateStyle::SectionsOnly, state) => {
                Ok(format!("{}\n\n", render_state(state)?))
            }
        }
    }

    /// Everything before the question, shared by every question on one state (`prefix_text`).
    /// With no state the question follows the user header directly.
    ///
    /// # Errors
    ///
    /// As [`Renderer::state_block`].
    pub fn prefix(&self, state: Option<&Json>) -> Result<String, Error> {
        let turn = match self.system {
            System::None => String::new(),
            System::Isolated => format!("{IM_START}system\n{ISOLATED}{IM_END}\n"),
        };
        let body = match state {
            None => String::new(),
            Some(s) => format!("{}\nQUESTION:\n", self.state_block(s)?),
        };
        Ok(format!("{}{turn}{IM_START}user\n{body}", self.bos))
    }

    fn option_line(&self, code: &str, label: &str, desc: Option<&str>) -> String {
        let text = match desc {
            Some(d) if !d.is_empty() => d.to_owned(),
            _ => label.replace('_', " "),
        };
        if self.option_style == OptionStyle::NameDesc && !placeholder(label) && label != text {
            format!("{code} {label}: {text}")
        } else {
            format!("{code} {text}")
        }
    }

    /// The question's text (`question_block`).
    ///
    /// # Errors
    ///
    /// If the tokenizer fails, no option code is left, or a noul's criterion is a list or an
    /// object.
    pub fn question_block(&self, tokenizer: &Tokenizer, q: &Question) -> Result<String, Error> {
        match q {
            Question::Choice {
                instructions,
                criteria,
            } => {
                let labels: Vec<&str> = criteria.iter().map(|(l, _)| l.as_str()).collect();
                let lines: Vec<String> = aliases(tokenizer, &labels)?
                    .iter()
                    .zip(criteria)
                    .map(|((code, _), (label, desc))| {
                        self.option_line(code, label, desc.as_deref())
                    })
                    .collect();
                Ok(format!(
                    "{instructions}\n\nOptions:\n{}\n\nReply with the option code only.",
                    lines.join("\n")
                ))
            }
            Question::Noul {
                instructions,
                criteria,
            } => {
                let extra = match criteria {
                    None => String::new(),
                    Some((yes, no)) => {
                        let side = |v: &Option<Json>| v.as_ref().unwrap_or(&Json::Null).py_str();
                        format!("\nYes: {}\nNo: {}", side(yes)?, side(no)?)
                    }
                };
                Ok(format!(
                    "{instructions}{extra}\n\nReply with yes or no only."
                ))
            }
            Question::Score {
                instructions,
                criteria,
            } => {
                let legend: Vec<String> = criteria
                    .iter()
                    .enumerate()
                    .map(|(i, name)| format!("{i} {name}"))
                    .collect();
                let top = i128::try_from(criteria.len()).unwrap_or(i128::MAX) - 1;
                Ok(format!(
                    "{instructions}\n\n{}\n\nReply with a single digit 0-{top} only.",
                    legend.join("\n")
                ))
            }
        }
    }

    /// The question and the assistant header, up to the answer slot (`suffix_text`).
    ///
    /// # Errors
    ///
    /// As [`Renderer::question_block`].
    pub fn suffix(&self, tokenizer: &Tokenizer, q: &Question) -> Result<String, Error> {
        Ok(format!(
            "{}{IM_END}\n{IM_START}assistant\n{}",
            self.question_block(tokenizer, q)?,
            self.lead
        ))
    }

    /// The whole prompt for one question (`render`).
    ///
    /// # Errors
    ///
    /// As [`Renderer::prefix`] and [`Renderer::suffix`].
    pub fn render(
        &self,
        tokenizer: &Tokenizer,
        state: Option<&Json>,
        q: &Question,
    ) -> Result<String, Error> {
        Ok(self.prefix(state)? + &self.suffix(tokenizer, q)?)
    }

    /// Each question's token ids as the runner reads a text request: one question is its whole
    /// prompt encoded at once; several share the prefix, encoded apart, each followed by its own
    /// encoded suffix.
    ///
    /// # Errors
    ///
    /// As [`Renderer::render`], or if the tokenizer fails.
    pub fn rows(
        &self,
        tokenizer: &Tokenizer,
        state: Option<&Json>,
        questions: &[Question],
    ) -> Result<Vec<Vec<u32>>, Error> {
        if let [q] = questions {
            return Ok(vec![tokenizer.encode(&self.render(tokenizer, state, q)?)?]);
        }
        let prefix = tokenizer.encode(&self.prefix(state)?)?;
        questions
            .iter()
            .map(|q| {
                let mut row = prefix.clone();
                row.extend(tokenizer.encode(&self.suffix(tokenizer, q)?)?);
                Ok(row)
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn python_float_repr() {
        let cases = [
            (1.0, "1.0"),
            (0.1, "0.1"),
            (-12.5, "-12.5"),
            (1e16, "1e+16"),
            (1e15, "1000000000000000.0"),
            (1e22, "1e+22"),
            (0.0001, "0.0001"),
            (0.00001, "1e-05"),
            (1e-7, "1e-07"),
            (2.5e-5, "2.5e-05"),
            (123_456_789.123, "123456789.123"),
            (3.0e-310, "3e-310"),
            (0.0, "0.0"),
            (-0.0, "-0.0"),
            (1.234_567_890_123_456_7e300, "1.2345678901234567e+300"),
        ];
        for (x, want) in cases {
            assert_eq!(py_float(x), want, "{x:e}");
        }
    }

    #[test]
    fn dumps_like_python() {
        let v = Json::parse(r#"{"b": [], "a": {"x": [1, 2.0, null, true]}, "c": {}, "s": "é\"\\\n\u0007\u007f", "b": 3}"#)
            .unwrap();
        assert_eq!(
            v.dumps(),
            "{\n  \"b\": 3,\n  \"a\": {\n    \"x\": [\n      1,\n      2.0,\n      null,\n      true\n    ]\n  },\n  \"c\": {},\n  \"s\": \"é\\\"\\\\\\n\\u0007\u{7f}\"\n}"
        );
    }

    #[test]
    fn codes() {
        assert_eq!(option_codes(&["b", " Q\u{1f}"]), ["b", "Q"]);
        assert_eq!(option_codes(&["one", "two"]), ["A", "B"]);
        let many: Vec<String> = (0..27).map(|i| format!("x{i}")).collect();
        assert_eq!(option_codes(&many).last().map(String::as_str), Some("26"));
        assert!(placeholder("opt12") && placeholder("opt1\n") && !placeholder("opt"));
        assert_eq!(softmax(&[0.0, 0.0]), [0.5, 0.5]);
    }
}
