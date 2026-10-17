//! `/opt-out` and `/opt-in`: users who don't want the bot to read their messages.
//!
//! Everyone is opted in by default. The file lists the users who opted out,
//! each with when (UTC, RFC 3339). `/opt-in` turns the entry into a
//! *visible-from* time instead of deleting it: a cold start (`dispatch::
//! seed_messages`) must keep skipping what they said while opted out, which
//! the live windows never saw, and without the time a restart would read it
//! back in (PRIVACY.md says why the entry stays). The file is rewritten
//! whole on each change (write a temporary file, then rename), so a crash
//! never leaves it half-written.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context as _, Result};
use serde::{Deserialize, Serialize};

#[derive(Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct File {
    opted_out: Vec<Entry>,
    /// Users who opted back in, with when: their earlier messages stay
    /// invisible. Left out when empty, so files from before this field load.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    opted_in: Vec<Entry>,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Entry {
    user_id: u64,
    since: String,
}

/// One user's standing.
#[derive(Clone, PartialEq, Debug)]
enum State {
    /// Opted out since (RFC 3339): invisible.
    Out(String),
    /// Opted back in: visible from this snowflake on; older messages stay
    /// invisible to a cold start.
    VisibleFrom(u64),
}

pub struct OptOuts {
    path: PathBuf,
    /// User id → their standing (absent: opted in all along). Held while the
    /// file is written, so writes happen one at a time and in order.
    users: Mutex<BTreeMap<u64, State>>,
}

impl OptOuts {
    /// Reads `path` (none yet = nobody opted out) and writes it back, so a
    /// directory the bot can't write fails at startup, not on the first `/opt-out`.
    /// An unreadable file is an error: carrying on would opt everyone back in.
    pub fn load(path: &Path) -> Result<Self> {
        let file: File = match std::fs::read_to_string(path) {
            Ok(text) => serde_json::from_str(&text)
                .with_context(|| format!("invalid opt-out file {}", path.display()))?,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => File::default(),
            Err(e) => return Err(e).with_context(|| format!("reading {}", path.display())),
        };
        let users = file
            .opted_out
            .into_iter()
            .map(|e| (e.user_id, State::Out(e.since)))
            .chain(file.opted_in.into_iter().map(|e| {
                (e.user_id, State::VisibleFrom(crate::chat::window::snowflake_at(parse_rfc3339(&e.since))))
            }))
            .collect();
        let this = Self {
            path: path.to_owned(),
            users: Mutex::new(users),
        };
        this.save(&this.lock())?;
        Ok(this)
    }

    pub fn is_out(&self, user: u64) -> bool {
        matches!(self.lock().get(&user), Some(State::Out(_)))
    }

    /// Whether `user`'s message `id` is hidden: they are opted out, or it is
    /// from before they opted back in. What a cold start checks
    /// (`dispatch::seed_messages`); live messages only need
    /// [`is_out`](Self::is_out), since they are always from now.
    pub fn hidden(&self, user: u64, id: u64) -> bool {
        match self.lock().get(&user) {
            Some(State::Out(_)) => true,
            Some(State::VisibleFrom(from)) => id <= *from,
            None => false,
        }
    }

    /// How many users are opted out.
    pub fn len(&self) -> usize {
        self.lock().values().filter(|s| matches!(s, State::Out(_))).count()
    }

    /// Records `user` as opted out now. `Ok(false)`: they already were.
    pub fn opt_out(&self, user: u64) -> Result<bool> {
        self.change(user, State::Out(rfc3339(SystemTime::now())))
    }

    /// Removes `user`'s opt-out, keeping from when they are visible again.
    /// `Ok(false)`: they weren't opted out.
    pub fn opt_in(&self, user: u64) -> Result<bool> {
        self.change(user, State::VisibleFrom(crate::chat::window::snowflake_at(SystemTime::now())))
    }

    /// Applies the change and saves it; if saving fails, nothing changes.
    fn change(&self, user: u64, to: State) -> Result<bool> {
        let mut users = self.lock();
        let out = matches!(users.get(&user), Some(State::Out(_)));
        if out == matches!(to, State::Out(_)) {
            return Ok(false);
        }
        let mut next = users.clone();
        next.insert(user, to);
        self.save(&next)?;
        *users = next;
        Ok(true)
    }

    fn save(&self, users: &BTreeMap<u64, State>) -> Result<()> {
        let entry = |(&user_id, state): (&u64, &State)| match state {
            State::Out(since) => Entry { user_id, since: since.clone() },
            State::VisibleFrom(id) => Entry {
                user_id,
                since: rfc3339(UNIX_EPOCH + std::time::Duration::from_millis(crate::chat::window::snowflake_ms(*id))),
            },
        };
        let file = File {
            opted_out: users.iter().filter(|(_, s)| matches!(s, State::Out(_))).map(entry).collect(),
            opted_in: users.iter().filter(|(_, s)| matches!(s, State::VisibleFrom(_))).map(entry).collect(),
        };
        let dir = self.path.parent().filter(|d| !d.as_os_str().is_empty());
        if let Some(dir) = dir {
            std::fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;
        }
        let tmp = self.path.with_extension("json.tmp");
        let text = serde_json::to_string_pretty(&file)? + "\n";
        std::fs::write(&tmp, text).with_context(|| format!("writing {}", tmp.display()))?;
        std::fs::rename(&tmp, &self.path)
            .with_context(|| format!("replacing {}", self.path.display()))
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, BTreeMap<u64, State>> {
        self.users.lock().unwrap_or_else(|e| e.into_inner())
    }
}

/// `YYYY-MM-DDTHH:MM:SSZ` → the time it names; a malformed one reads as the
/// epoch, which hides nothing extra (it is before every message).
fn parse_rfc3339(text: &str) -> SystemTime {
    let n = |r: std::ops::Range<usize>| -> Option<i64> { text.get(r)?.parse().ok() };
    let (Some(y), Some(mo), Some(d), Some(h), Some(mi), Some(se)) =
        (n(0..4), n(5..7), n(8..10), n(11..13), n(14..16), n(17..19))
    else {
        return UNIX_EPOCH;
    };
    // Howard Hinnant's days-from-civil, the inverse of `rfc3339` below.
    let (y, m) = (y - i64::from(mo <= 2), mo);
    let era = y.div_euclid(400);
    let yoe = y - era * 400;
    let mp = if m > 2 { m - 3 } else { m + 9 };
    let doy = (153 * mp + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    let days = era * 146_097 + doe - 719_468;
    let secs = days * 86_400 + h * 3_600 + mi * 60 + se;
    UNIX_EPOCH + std::time::Duration::from_secs(secs.max(0) as u64)
}

/// `YYYY-MM-DDTHH:MM:SSZ` (Howard Hinnant's civil-from-days, as in build.rs).
fn rfc3339(t: SystemTime) -> String {
    let secs = t.duration_since(UNIX_EPOCH).map_or(0, |d| d.as_secs());
    let (days, rem) = ((secs / 86_400) as i64, secs % 86_400);
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = yoe + era * 400 + i64::from(month <= 2);
    format!(
        "{year:04}-{month:02}-{day:02}T{:02}:{:02}:{:02}Z",
        rem / 3_600,
        rem % 3_600 / 60,
        rem % 60
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    fn temp_path(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("increbot-optout-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        dir.join("sub").join("optouts.json")
    }

    #[test]
    fn opt_out_and_in_persist() {
        let path = temp_path("persist");
        let o = OptOuts::load(&path).unwrap();
        assert!(path.exists(), "load creates the directory and file");
        assert!(!o.is_out(7));
        assert!(o.opt_out(7).unwrap());
        assert!(!o.opt_out(7).unwrap(), "already out");
        assert!(o.opt_out(9).unwrap());

        let o = OptOuts::load(&path).unwrap();
        assert!(o.is_out(7) && o.is_out(9));
        assert!(o.opt_in(7).unwrap());
        assert!(!o.opt_in(7).unwrap(), "already in");

        let o = OptOuts::load(&path).unwrap();
        assert!(!o.is_out(7) && o.is_out(9));
        assert_eq!(o.len(), 1, "len counts only opted-out users");
        // 7 opted back in: what they said while out stays hidden, and the
        // visible-from time survives a reload.
        // The visible-from time survives the file's RFC 3339 (whole seconds),
        // so the margins are seconds, not milliseconds.
        let now = crate::chat::window::snowflake_at(SystemTime::now());
        let (before, after) = (now - (10_000 << 22), now + (10_000 << 22));
        assert!(o.hidden(7, before) && o.hidden(9, after));
        assert!(!o.hidden(7, after) && !o.hidden(5, 1));
        let o = OptOuts::load(&path).unwrap();
        assert!(o.hidden(7, before));
        assert!(!o.hidden(7, after), "visible from the opt-in on");
        let _ = std::fs::remove_dir_all(path.parent().unwrap().parent().unwrap());
    }

    #[test]
    fn old_files_without_opted_in_load() {
        let path = temp_path("oldformat");
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(&path, r#"{ "opted_out": [{ "user_id": 7, "since": "2026-10-12T13:30:45Z" }] }"#).unwrap();
        let o = OptOuts::load(&path).unwrap();
        assert!(o.is_out(7) && o.len() == 1);
        let _ = std::fs::remove_dir_all(path.parent().unwrap().parent().unwrap());
    }

    #[test]
    fn rfc3339_round_trips() {
        for t in ["1970-01-01T00:00:00Z", "2026-10-12T13:30:45Z", "2000-02-29T00:00:00Z"] {
            assert_eq!(rfc3339(parse_rfc3339(t)), t);
        }
        assert_eq!(parse_rfc3339("garbage"), UNIX_EPOCH);
    }

    #[test]
    fn corrupt_file_is_an_error() {
        let path = temp_path("corrupt");
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(&path, "{not json").unwrap();
        assert!(OptOuts::load(&path).is_err());
        let _ = std::fs::remove_dir_all(path.parent().unwrap().parent().unwrap());
    }

    #[test]
    fn timestamps() {
        assert_eq!(rfc3339(UNIX_EPOCH), "1970-01-01T00:00:00Z");
        let t = UNIX_EPOCH + Duration::from_secs(1_791_811_845);
        assert_eq!(rfc3339(t), "2026-10-12T13:30:45Z");
        let leap = UNIX_EPOCH + Duration::from_secs(951_782_400);
        assert_eq!(rfc3339(leap), "2000-02-29T00:00:00Z");
    }
}
