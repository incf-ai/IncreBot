//! Joke commands besides `/hotdog-rater` (design.md §6). `/kobold-name`,
//! `/lick-test`, `/hoard-value`, `/kobold-age` and `/judge` are fixed forever,
//! from the user IDs (or the judged text) and the novelty key alone, with
//! `seeded`'s draws (**changing a table or tag here changes everyone's
//! results**; the pinned tests guard that). The rest draw from `rand` each time. Every command
//! ends in an LLM reply to a set-up prompt.

use super::seeded::{Key, Stream, text_input};

/// Who a command is about, as the requester names them to the bot.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Who {
    /// The requester.
    Me,
    /// The bot itself.
    Bot,
    Other(String),
}

impl Who {
    /// "my", "ur", "Bob's".
    pub fn possessive(&self) -> String {
        match self {
            Who::Me => "my".into(),
            Who::Bot => "ur".into(),
            Who::Other(n) => format!("{n}'s"),
        }
    }

    /// "my own", "ur own", "Bob's": [`Who::possessive`] with "own" where the
    /// word alone is ambiguous. From the requester's mouth "ur hotdog" reads,
    /// to the model, as the requester's (README round 20); "ur own hotdog"
    /// says it's the bot's.
    pub fn possessive_own(&self) -> String {
        match self {
            Who::Me => "my own".into(),
            Who::Bot => "ur own".into(),
            Who::Other(n) => format!("{n}'s"),
        }
    }

    /// Whose it is, plainly, for a command's note: the set-up line says "my"
    /// or "ur" from the requester's side, which the model often flips.
    pub fn ownership(&self, thing: &str, requester: &str, bot: &str) -> String {
        match self {
            Who::Me => format!("(it's {requester}'s own {thing}; {requester} asked {bot} to measure it.)"),
            Who::Bot => format!("(it's {bot}'s own {thing}: {requester} asked {bot} to measure his own.)"),
            Who::Other(n) => format!("(the {thing} belongs to {n}, not to {requester}; {requester} asked {bot} to measure it.)"),
        }
    }

    /// "me", "urself", "Bob".
    pub fn object(&self) -> String {
        match self {
            Who::Me => "me".into(),
            Who::Bot => "urself".into(),
            Who::Other(n) => n.clone(),
        }
    }

    /// "i'm", "u are", "Bob is".
    fn is(&self) -> String {
        match self {
            Who::Me => "i'm".into(),
            Who::Bot => "u are".into(),
            Who::Other(n) => format!("{n} is"),
        }
    }

    /// "i taste", "u taste", "Bob tastes".
    fn tastes(&self) -> String {
        match self {
            Who::Me => "i taste".into(),
            Who::Bot => "u taste".into(),
            Who::Other(n) => format!("{n} tastes"),
        }
    }
}

fn pick(s: &mut Stream, list: &[&'static str]) -> &'static str {
    list[(s.next() % list.len() as u64) as usize]
}

// ---- /kobold-name ----

const FIRST_START: &[&str] = &[
    "Grim", "Snik", "Mog", "Tik", "Zub", "Krag", "Nib", "Rusk", "Glim", "Skree", "Dob",
    "Vex", "Murk", "Fizz", "Gob", "Skab", "Drek", "Yip", "Kurn", "Splat",
];
const FIRST_END: &[&str] = &[
    "ble", "nik", "gle", "rat", "zz", "wick", "kin", "grub", "tooth", "snout", "dle",
    "rik",
];
const LAST_START: &[&str] = &[
    "Moss", "Rust", "Ember", "Mud", "Pebble", "Soot", "Gravel", "Fungus", "Copper",
    "Cinder", "Bone", "Shale", "Grime", "Brass",
];
const LAST_END: &[&str] = &[
    "byte", "claw", "tail", "scale", "digger", "snout", "toe", "fang", "grub", "hoard",
    "licker", "sniffer",
];
const EPITHETS: &[&str] = &[
    "the Damp",
    "the Shiny",
    "Who Bites Ankles",
    "the Trapmaker",
    "of the Lower Tunnels",
    "the Slightly Singed",
    "Dragon-Botherer",
    "the Pebble Counter",
    "the Unwashed",
    "Rock-Licker",
    "the Persistent",
    "Three-Toes",
    "the Suspicious",
    "Hoarder of Spoons",
    "the Twice-Exploded",
    "Who Owes Everyone Gold",
];

/// The fixed kobold name of the user with this Discord ID.
pub fn kobold_name(key: &Key, user_id: u64) -> String {
    let mut s = Stream::new(key, "kobold-name", user_id);
    format!(
        "{}{} {}{} {}",
        pick(&mut s, FIRST_START),
        pick(&mut s, FIRST_END),
        pick(&mut s, LAST_START),
        pick(&mut s, LAST_END),
        pick(&mut s, EPITHETS),
    )
}

pub fn name_card(whose: &str, name: &str) -> String {
    format!("🦎 {whose}'s kobold name: **{name}**")
}

pub fn name_prompt(bot: &str, who: &Who, name: &str) -> String {
    format!(
        "{bot}, {} kobold name is {name}! what do u think of it?",
        who.possessive()
    )
}

// ---- /lick-test ----

const TASTES: &[&str] = &[
    "copper coins",
    "warm rocks",
    "cave moss",
    "old cheese",
    "dragon smoke",
    "pocket lint",
    "fresh beetles",
    "rust",
    "mushroom stew",
    "wet dog",
    "spicy peppers",
    "candle wax",
    "river clay",
    "salted fish",
    "stolen honey",
    "burnt toast",
    "cheap ale",
    "regret",
];

/// What the user with this ID tastes like, fixed forever.
pub fn taste(key: &Key, user_id: u64) -> &'static str {
    pick(&mut Stream::new(key, "lick-test-taste", user_id), TASTES)
}

/// How compatible two users are, 0–100, the same in either order.
pub fn compatibility(key: &Key, a: u64, b: u64) -> u8 {
    let (lo, hi) = (a.min(b), a.max(b));
    let x = text_input(&format!("{lo}/{hi}"));
    (Stream::new(key, "lick-test", x).next() % 101) as u8
}

/// `a` and `b` are mentions with their user IDs; the bot (`bot_id`) is
/// "himself" when licked and its name when tasted.
pub fn lick_card(key: &Key, bot: &str, bot_id: u64, (a, a_id): (&str, u64), (b, b_id): (&str, u64)) -> String {
    let licked = |m: &str, id: u64| if id == bot_id { "himself".to_string() } else { m.to_string() };
    let tasted = |m: &str, id: u64| if id == bot_id { bot.to_string() } else { m.to_string() };
    format!(
        "👅 {bot} licks {} and {}. {} tastes like **{}**, {} tastes like **{}**. \
         flavor match: **{}%**",
        licked(a, a_id),
        licked(b, b_id),
        tasted(a, a_id),
        taste(key, a_id),
        tasted(b, b_id),
        taste(key, b_id),
        compatibility(key, a_id, b_id),
    )
}

pub fn lick_prompt(key: &Key, bot: &str, (a, a_id): (&Who, u64), (b, b_id): (&Who, u64)) -> String {
    format!(
        "{bot}, lick {} and {} and tell everyone if it's a match! {} like {}, {} like {}, \
         and that's a {}% flavor match.",
        a.object(),
        b.object(),
        a.tastes(),
        taste(key, a_id),
        b.tastes(),
        taste(key, b_id),
        compatibility(key, a_id, b_id),
    )
}

/// A self-pair's score: the person's match with themselves, fixed like any
/// other pair's.
pub fn self_love(key: &Key, user_id: u64) -> u8 {
    compatibility(key, user_id, user_id)
}

/// The card when both people are the same (`whom` is a mention).
pub fn self_lick_card(key: &Key, bot: &str, bot_id: u64, (whom, id): (&str, u64)) -> String {
    let (flavor, love) = (taste(key, id), self_love(key, id));
    if id == bot_id {
        format!("👅 {bot} licks himself. twice. he tastes like **{flavor}**. inner flavor match: **{love}%**")
    } else {
        format!("👅 {bot} licks {whom} twice. {whom} tastes like **{flavor}**. inner flavor match: **{love}%**")
    }
}

pub fn self_lick_prompt(key: &Key, bot: &str, (who, id): (&Who, u64)) -> String {
    let (flavor, love) = (taste(key, id), self_love(key, id));
    match who {
        Who::Me => format!(
            "{bot}, lick me twice and tell everyone how well i match my own flavor! i taste like \
             {flavor}, that's a {love}% inner flavor match."
        ),
        Who::Bot => format!(
            "{bot}, lick urself twice and tell everyone how well u match ur own flavor! u taste \
             like {flavor}, that's a {love}% inner flavor match."
        ),
        Who::Other(n) => format!(
            "{bot}, lick {n} twice and tell everyone how well {n} matches their own flavor! {n} \
             tastes like {flavor}, that's a {love}% inner flavor match."
        ),
    }
}

// ---- /roast ----

pub fn roast_card(bot: &str, whom: &str) -> String {
    format!("🔥 {bot} roasts {whom}:")
}

pub fn roast_prompt(bot: &str, who: &Who) -> String {
    format!("{bot}, roast {}! be mean, kobold mean.", who.object())
}

// ---- /fortune ----

const OMENS: &[&str] = &[
    "in a spiral",
    "in a neat little pile",
    "pointing at the nearest exit",
    "in the shape of a dragon",
    "all over ur feet",
    "in a perfect circle",
    "and one rolls away into a crack",
    "face down, which is bad",
    "spelling something rude",
    "around a shiny pebble nobody put there",
    "in two even rows",
    "and one of them is missing",
    "and one is still twitching",
    "on top of each other, somehow",
];

/// A random omen (a new one every time, unlike the fixed commands).
pub fn omen() -> &'static str {
    OMENS[rand::random_range(0..OMENS.len())]
}

pub fn fortune_card(bot: &str, whom: &str, omen: &str) -> String {
    format!("🦴 {bot} throws the rat bones for {whom}. they land {omen}.")
}

pub fn fortune_prompt(bot: &str, omen: &str) -> String {
    format!("*hands {bot} a handful of rat bones* {bot}, read my fortune! *the bones land {omen}*")
}

// ---- /hoard-value ----

const HOARD_ITEMS: &[&str] = &[
    "3 bottle caps",
    "a suspicious ring",
    "a bent spoon",
    "half a sandwich",
    "a gold tooth",
    "17 buttons",
    "a dragon scale (probably fake)",
    "a single sock",
    "a jar of teeth",
    "a crown made of foil",
    "a map to nowhere",
    "4 gems (glass)",
    "a very good rock",
    "someone's keys",
    "a cursed coin",
    "a broken compass",
    "a candle stub",
    "an IOU from a goblin",
    "a lockpick (bent)",
    "a mysterious egg",
];

/// The fixed hoard of the user with this ID: three different items and its
/// value in gold (1–100,000, mostly small).
pub fn hoard(key: &Key, user_id: u64) -> ([&'static str; 3], u32) {
    let mut s = Stream::new(key, "hoard-value", user_id);
    let mut items: Vec<&'static str> = Vec::with_capacity(3);
    while items.len() < 3 {
        let item = pick(&mut s, HOARD_ITEMS);
        if !items.contains(&item) {
            items.push(item);
        }
    }
    let u = s.unit();
    let gold = 1 + (u * u * u * u * 99_999.0) as u32;
    ([items[0], items[1], items[2]], gold)
}

fn list3(items: &[&str; 3]) -> String {
    format!("{}, {} and {}", items[0], items[1], items[2])
}

pub fn hoard_card(bot: &str, whose: &str, items: &[&str; 3], gold: u32) -> String {
    format!(
        "💰 {bot} appraises {whose}'s hoard: {}. total value: **{gold} gold**",
        list3(items)
    )
}

pub fn hoard_prompt(bot: &str, who: &Who, items: &[&str; 3], gold: u32) -> String {
    format!(
        "{bot}, appraise {} hoard! it has {}, worth {gold} gold. is it any good?",
        who.possessive(),
        list3(items)
    )
}

// ---- /kobold-age ----

/// The bot's own age (the persona's; prompts/cN.rsn).
pub const BOT_AGE: u32 = 471;

/// The fixed kobold age of the user with this ID: 40–399, mostly under 150,
/// so always well under the bot's.
pub fn kobold_age(key: &Key, user_id: u64) -> u32 {
    let u = Stream::new(key, "kobold-age", user_id).unit();
    40 + (u * u * 360.0) as u32
}

pub fn age_card(bot: &str, whom: &str, age: u32) -> String {
    format!("🕰️ {whom} is **{age} kobold years old**. ({bot} is {BOT_AGE}.)")
}

pub fn bot_age_card(bot: &str) -> String {
    format!("🕰️ {bot} is **{BOT_AGE} kobold years old**, much older than other kobolds.")
}

pub fn age_prompt(bot: &str, who: &Who, age: u32) -> String {
    match who {
        Who::Bot => format!(
            "{bot}, u are {BOT_AGE} kobold years old! how does it feel being so much older \
             than other kobolds?"
        ),
        _ => format!(
            "{bot}, {} {age} kobold years old! that's {} years younger than u, old timer.",
            who.is(),
            BOT_AGE - age
        ),
    }
}

// ---- /judge ----

/// Longest thing shown or judged, in characters.
pub const JUDGE_MAX_CHARS: usize = 200;

/// The thing as shown and judged: whitespace collapsed, capped.
pub fn judge_text(thing: &str) -> String {
    let t = thing.split_whitespace().collect::<Vec<_>>().join(" ");
    if t.chars().count() > JUDGE_MAX_CHARS {
        t.chars().take(JUDGE_MAX_CHARS).collect::<String>() + "…"
    } else {
        t
    }
}

/// The fixed score, 0–10, of a thing (case and spacing don't matter).
pub fn judge_score(key: &Key, thing: &str) -> u8 {
    let x = text_input(&judge_text(thing).to_lowercase());
    (Stream::new(key, "judge", x).next() % 11) as u8
}

pub fn judge_card(bot: &str, thing: &str, score: u8) -> String {
    format!("⚖️ {bot} judges “{thing}”: **{score}/10 shiny rocks**")
}

pub fn judge_prompt(bot: &str, thing: &str, score: u8) -> String {
    format!(
        "{bot}, judge this out of 10 shiny rocks: {thing} *{bot} decides it's a {score}* why?"
    )
}

// ---- /trap-check (new each time) ----

const TRAPS: &[&str] = &[
    "the path is clear. suspiciously clear.",
    "the path is clear.",
    "the path is clear, except for a very judgmental beetle.",
    "a tripwire, already sprung by someone else.",
    "the bucket trap. there is now a bucket on ur head.",
    "a pit trap, about ankle deep.",
    "a net trap, and the bot is the one in it.",
    "a rolling boulder, but a small one.",
    "glue on the floor. ur boots live here now.",
    "a trap the bot made last week and forgot about.",
    "a sign that says NOT A TRAP. it is a trap.",
    "a mimic pretending to be a treasure chest.",
    "a swinging log, already swinging.",
];

pub fn trap() -> &'static str {
    TRAPS[rand::random_range(0..TRAPS.len())]
}

pub fn trap_card(bot: &str, whom: &str, trap: &str) -> String {
    format!("🪤 {bot} checks {whom}'s path for traps: {}", trap.replace("the bot", bot))
}

pub fn trap_prompt(bot: &str, trap: &str) -> String {
    format!("{bot}, check my path for traps! *{}*", trap.replace("the bot", bot))
}

// ---- /drink (new each time) ----

const SPIRITS: &[&str] = &[
    "mushroom brandy",
    "fermented moss",
    "dragon-pepper rum",
    "cave-drip gin",
    "beetle mead",
    "swamp whiskey",
    "glowcap schnapps",
];
const MIXERS: &[&str] = &[
    "warm puddle water",
    "glowworm juice",
    "a splash of lamp oil (probably fine)",
    "goat milk",
    "sparkling spring water",
    "bat-fruit nectar",
    "cold tunnel tea",
];
const GARNISHES: &[&str] = &[
    "a pebble",
    "a single beetle leg",
    "a sprig of glowmoss",
    "a leaf umbrella",
    "a shiny button",
    "a pinch of soot",
    "a salted rim (it's ash)",
];

/// Spirit, mixer and garnish, new every time.
pub fn drink() -> [&'static str; 3] {
    let p = |l: &[&'static str]| l[rand::random_range(0..l.len())];
    [p(SPIRITS), p(MIXERS), p(GARNISHES)]
}

pub fn drink_card(bot: &str, whom: &str, [spirit, mixer, garnish]: [&str; 3]) -> String {
    format!("🍸 {bot} mixes {whom} a drink: {spirit} and {mixer}, garnished with {garnish}")
}

pub fn drink_prompt(bot: &str, [spirit, mixer, garnish]: [&str; 3]) -> String {
    format!(
        "*{bot} mixes me a drink: {spirit} and {mixer}, garnished with {garnish}* {bot}, \
         what's it called? how does it taste?"
    )
}

// ---- /gamble (new each time; nothing is kept) ----

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Gamble {
    pub bet: u32,
    /// The requester's two bones, then the bot's.
    pub rolls: [u8; 4],
}

impl Gamble {
    pub fn roll(bet: u32) -> Gamble {
        Gamble {
            bet,
            rolls: std::array::from_fn(|_| rand::random_range(1..=6)),
        }
    }

    fn totals(&self) -> (u8, u8) {
        let r = self.rolls;
        (r[0] + r[1], r[2] + r[3])
    }

    pub fn card(&self, bot: &str, whom: &str) -> String {
        let r = self.rolls;
        let (mine, theirs) = self.totals();
        let result = match mine.cmp(&theirs) {
            std::cmp::Ordering::Greater => format!("{whom} wins {} shiny rocks!", self.bet),
            std::cmp::Ordering::Less => format!("{bot} wins. the rocks go in the hoard."),
            std::cmp::Ordering::Equal => "a tie. everyone keeps their rocks.".to_string(),
        };
        format!(
            "🎲 {whom} bets **{} shiny rocks**. {whom} rolls {}+{} = **{mine}**, {bot} rolls \
             {}+{} = **{theirs}**. **{result}**",
            self.bet, r[0], r[1], r[2], r[3]
        )
    }

    pub fn prompt(&self, bot: &str) -> String {
        let (mine, theirs) = self.totals();
        let result = match mine.cmp(&theirs) {
            std::cmp::Ordering::Greater => "i win! pay up!",
            std::cmp::Ordering::Less => "u win...",
            std::cmp::Ordering::Equal => "it's a tie!",
        };
        format!(
            "*bets {} shiny rocks with {bot} and throws the bones* i rolled {mine}, u rolled \
             {theirs}. {result}",
            self.bet
        )
    }
}

// ---- /dragon-alert (new each time) ----

const DRAGON_SIGNS: &[&str] = &[
    "a huge shadow passes over the tunnel entrance",
    "the ceiling has fresh scorch marks",
    "something very large is snoring nearby",
    "there's a pile of melted gold coins in the corner",
    "the air smells like smoke and sulfur",
    "a scale the size of a shield is lying on the floor",
    "all the warm rocks are suddenly much warmer",
    "a distant ROAR shakes dust from the ceiling",
    "a goat runs past, screaming",
];

pub fn dragon_sign() -> &'static str {
    DRAGON_SIGNS[rand::random_range(0..DRAGON_SIGNS.len())]
}

pub fn dragon_card(sign: &str) -> String {
    format!("🐉 **DRAGON ALERT**: {sign}.")
}

pub fn dragon_prompt(bot: &str, sign: &str) -> String {
    format!("*{sign}* {bot}!! is that a dragon?!")
}

// ---- /vibe-check ----

/// Messages a vibe check reads, newest last.
pub const VIBE_MESSAGES: usize = 15;
/// Longest single message quoted, in characters.
const VIBE_MAX_CHARS: usize = 200;

pub fn vibe_card(bot: &str) -> String {
    format!("🔮 {bot} sniffs the air to check the vibe in here…")
}

/// `recent` is (author, text), oldest first. Quoted inline rather than as chat
/// lines, so the model doesn't continue the transcript as someone else.
pub fn vibe_prompt(bot: &str, recent: &[(String, String)]) -> String {
    let quotes: Vec<String> = recent
        .iter()
        .filter(|(_, t)| !t.trim().is_empty())
        .map(|(a, t)| {
            let t = t.split_whitespace().collect::<Vec<_>>().join(" ");
            let t = if t.chars().count() > VIBE_MAX_CHARS {
                t.chars().take(VIBE_MAX_CHARS).collect::<String>() + "…"
            } else {
                t
            };
            format!("{a} said \"{t}\"")
        })
        .collect();
    if quotes.is_empty() {
        return format!("{bot}, vibe check! it's been totally quiet in here. what's the vibe?");
    }
    format!(
        "{bot}, vibe check! here's what everyone's been saying: {}. what's the vibe in here?",
        quotes.join("; ")
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ownership_lines_say_plainly_whose_it_is() {
        let o = |w: Who| w.ownership("hotdog", "thornback", "floob");
        assert_eq!(o(Who::Me), "(it's thornback's own hotdog; thornback asked floob to measure it.)");
        assert_eq!(o(Who::Bot), "(it's floob's own hotdog: thornback asked floob to measure his own.)");
        assert_eq!(
            o(Who::Other("Grimble".into())),
            "(the hotdog belongs to Grimble, not to thornback; thornback asked floob to measure it.)"
        );
    }
    use crate::discord::seeded::test_key;

    const IDS: [u64; 3] = [80351110224678912, 1234567890123456789, 999];

    /// Pinned: if this fails, a change renamed existing users. Update on purpose only.
    #[test]
    fn pinned_names() {
        assert_eq!(IDS.map(|id| kobold_name(&test_key(), id)), PINNED_NAMES);
    }

    const PINNED_NAMES: [&str; 3] = [
        "Kurnrat Copperscale the Twice-Exploded",
        "Kragkin Sootsnout Rock-Licker",
        "Glimsnout Embertail the Slightly Singed",
    ];

    /// Pinned like the names: tastes, and the matches between the IDs.
    #[test]
    fn pinned_lick_tests() {
        assert_eq!(IDS.map(|id| taste(&test_key(), id)), PINNED_TASTES);
        let m = [
            compatibility(&test_key(), IDS[0], IDS[1]),
            compatibility(&test_key(), IDS[1], IDS[2]),
            compatibility(&test_key(), IDS[0], IDS[2]),
        ];
        assert_eq!(m, PINNED_MATCHES);
    }

    const PINNED_TASTES: [&str; 3] = ["rust", "regret", "regret"];
    const PINNED_MATCHES: [u8; 3] = [99, 33, 77];

    #[test]
    fn self_pairs_get_their_own_cards_and_prompts() {
        let k = test_key();
        assert_eq!(self_love(&k, IDS[0]), compatibility(&k, IDS[0], IDS[0]));
        let (t, l) = (taste(&k, 5), self_love(&k, 5));
        assert_eq!(
            self_lick_card(&k, "floob", 7, ("<@5>", 5)),
            format!("👅 floob licks <@5> twice. <@5> tastes like **{t}**. inner flavor match: **{l}%**")
        );
        assert!(self_lick_card(&k, "floob", 7, ("<@7>", 7)).starts_with("👅 floob licks himself. twice. he tastes like"));
        assert!(self_lick_prompt(&k, "floob", (&Who::Me, 5)).starts_with("floob, lick me twice and tell everyone how well i match my own flavor!"));
        assert!(self_lick_prompt(&k, "floob", (&Who::Bot, 7)).contains("lick urself twice"));
        let bob = Who::Other("Bob".into());
        assert!(self_lick_prompt(&k, "floob", (&bob, 5)).contains(&format!("how well Bob matches their own flavor! Bob tastes like {t}, that's a {l}% inner flavor match.")));
    }

    #[test]
    fn compatibility_is_symmetric_and_in_range() {
        let mut seen = std::collections::HashSet::new();
        let key = test_key();
        for i in 0..2_000u64 {
            let (a, b) = (1_100_000_000_000_000_000 + i, 1_100_000_000_000_000_000 + i * 7 + 1);
            let c = compatibility(&key, a, b);
            assert_eq!(c, compatibility(&key, b, a));
            assert!(c <= 100);
            seen.insert(c);
        }
        assert!(seen.len() > 90, "only {} distinct values", seen.len());
    }

    #[test]
    fn names_vary_and_use_every_table_entry() {
        let names: std::collections::HashSet<_> = (0..5_000u64)
            .map(|i| kobold_name(&test_key(), 1_100_000_000_000_000_000 + i))
            .collect();
        assert!(names.len() > 4_900);
        for epithet in EPITHETS {
            assert!(names.iter().any(|n| n.ends_with(epithet)), "{epithet} unused");
        }
    }

    #[test]
    fn prompts_name_the_right_people() {
        let bob = Who::Other("Bob".into());
        assert_eq!(
            name_prompt("floob", &Who::Me, "Grimble Mossbyte the Damp"),
            "floob, my kobold name is Grimble Mossbyte the Damp! what do u think of it?"
        );
        assert!(name_prompt("floob", &Who::Bot, "X").starts_with("floob, ur kobold name"));
        assert_eq!(roast_prompt("floob", &bob), "floob, roast Bob! be mean, kobold mean.");
        assert!(roast_prompt("floob", &Who::Bot).contains("roast urself!"));
        let k = test_key();
        let p = lick_prompt(&k, "floob", (&Who::Me, 1), (&bob, 2));
        assert!(p.starts_with("floob, lick me and Bob and tell everyone"), "{p}");
        assert!(!p.contains("cup"));
        let card = lick_card(&k, "floob", 7, ("<@7>", 7), ("<@2>", 2));
        assert!(card.starts_with(&format!("👅 floob licks himself and <@2>. floob tastes like **{}**, <@2> tastes", taste(&k, 7))), "{card}");
        assert!(p.contains(&format!("i taste like {}, Bob tastes like {}", taste(&k, 1), taste(&k, 2))));
        assert!(fortune_prompt("floob", "in a spiral").ends_with("*the bones land in a spiral*"));
    }

    #[test]
    fn pinned_hoards_and_ages() {
        assert_eq!(IDS.map(|id| hoard(&test_key(), id)), PINNED_HOARDS);
        assert_eq!(IDS.map(|id| kobold_age(&test_key(), id)), PINNED_AGES);
        assert_eq!(["a hotdog", "  A   HOTDOG ", "pineapple pizza"].map(|t| judge_score(&test_key(), t)), PINNED_JUDGE);
    }

    const PINNED_HOARDS: [([&str; 3], u32); 3] = [
        (["a jar of teeth", "a bent spoon", "a suspicious ring"], 3915),
        (["an IOU from a goblin", "3 bottle caps", "a broken compass"], 73402),
        (["a dragon scale (probably fake)", "a bent spoon", "a jar of teeth"], 20532),
    ];
    const PINNED_AGES: [u32; 3] = [267, 219, 49];
    const PINNED_JUDGE: [u8; 3] = [5, 5, 10];

    #[test]
    fn hoards_ages_and_scores_stay_in_range() {
        let mut scores = std::collections::HashSet::new();
        let key = test_key();
        for i in 0..20_000u64 {
            let id = 1_100_000_000_000_000_000 + i;
            let (items, gold) = hoard(&key, id);
            assert!(items[0] != items[1] && items[1] != items[2] && items[0] != items[2]);
            assert!((1..=100_000).contains(&gold));
            assert!((40..400).contains(&kobold_age(&key, id)));
            scores.insert(judge_score(&key, &i.to_string()));
        }
        assert_eq!(scores.len(), 11);
    }

    #[test]
    fn age_prompts() {
        assert_eq!(
            age_prompt("floob", &Who::Me, 100),
            "floob, i'm 100 kobold years old! that's 371 years younger than u, old timer."
        );
        assert!(age_prompt("floob", &Who::Bot, 0).contains("u are 471 kobold years old"));
        assert!(bot_age_card("floob").contains("much older than other kobolds"));
    }

    #[test]
    fn judge_normalizes_and_caps() {
        assert_eq!(judge_text("  a\n  hotdog "), "a hotdog");
        assert_eq!(judge_text(&"x".repeat(500)).chars().count(), JUDGE_MAX_CHARS + 1);
    }

    #[test]
    fn gamble_outcomes() {
        let g = |rolls| Gamble { bet: 10, rolls };
        assert!(g([6, 6, 1, 1]).card("floob", "Bob").contains("Bob wins 10 shiny rocks!"));
        assert!(g([1, 1, 6, 6]).card("floob", "Bob").contains("floob wins."));
        assert!(g([3, 4, 5, 2]).card("floob", "Bob").contains("a tie"));
        assert!(g([6, 6, 1, 1]).prompt("floob").ends_with("i rolled 12, u rolled 2. i win! pay up!"));
        for _ in 0..200 {
            assert!(Gamble::roll(1).rolls.iter().all(|r| (1..=6).contains(r)));
        }
    }

    #[test]
    fn random_tables_render() {
        assert!(!trap_card("floob", "Bob", trap()).contains("the bot"));
        assert!(trap_prompt("floob", TRAPS[6]).contains("floob is the one in it"));
        let d = drink();
        assert!(drink_prompt("floob", d).contains(d[0]));
        assert!(dragon_prompt("floob", dragon_sign()).ends_with("floob!! is that a dragon?!"));
    }

    #[test]
    fn vibe_prompt_quotes_and_caps() {
        let msgs = vec![
            ("Bob".to_string(), "hi\nall".to_string()),
            ("Ann".to_string(), "  ".to_string()),
            ("Cy".to_string(), "y".repeat(300)),
        ];
        let p = vibe_prompt("floob", &msgs);
        assert!(p.contains("Bob said \"hi all\"; Cy said \"yyy"), "{p}");
        assert!(!p.contains("Ann"));
        assert!(!p.contains(&"y".repeat(201)));
        assert!(vibe_prompt("floob", &[]).contains("totally quiet"));
    }
}
