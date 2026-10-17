//! `/hotdog-rater`: a joke "measurement" derived from a Discord user ID (and
//! the novelty key) alone, so each user's hotdog is the same forever
//! (design.md §6; the draws are `seeded`'s). **Changing a constant here
//! resizes everyone's hotdog**; the `pinned_results` test guards against that.

use super::novelty::Who;
use super::seeded::{Key, Stream};

/// The set-up prompt when the persona has no `hotdog-rater` command setup.
pub const PROMPT: &str = "*hands {bot} a measuring stick* {bot}, measure {whose} hotdog! it's \
    {length} cm long and {girth} cm around, a {tier}. what do u think of it?";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tier {
    CocktailWeenie,
    BallparkFrank,
    RespectableBratwurst,
    Footlong,
    PrizeKielbasa,
    LegendaryDragonSausage,
}

impl Tier {
    fn of(length_cm: f64) -> Tier {
        match length_cm {
            l if l < 7.0 => Tier::CocktailWeenie,
            l if l < 11.0 => Tier::BallparkFrank,
            l if l < 15.0 => Tier::RespectableBratwurst,
            l if l < 19.0 => Tier::Footlong,
            l if l < 26.0 => Tier::PrizeKielbasa,
            _ => Tier::LegendaryDragonSausage,
        }
    }

    pub const ALL: [Tier; 6] = [
        Tier::CocktailWeenie,
        Tier::BallparkFrank,
        Tier::RespectableBratwurst,
        Tier::Footlong,
        Tier::PrizeKielbasa,
        Tier::LegendaryDragonSausage,
    ];

    pub fn name(self) -> &'static str {
        match self {
            Tier::CocktailWeenie => "Cocktail Weenie",
            Tier::BallparkFrank => "Ballpark Frank",
            Tier::RespectableBratwurst => "Respectable Bratwurst",
            Tier::Footlong => "Footlong",
            Tier::PrizeKielbasa => "Prize Kielbasa",
            Tier::LegendaryDragonSausage => "Legendary Dragon Sausage",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Hotdog {
    /// Both rounded to 0.1 cm.
    pub length_cm: f64,
    pub girth_cm: f64,
    pub tier: Tier,
}

impl Hotdog {
    /// A hotdog of the given size, with its tier (for the benchmark's
    /// `/hotdog-rater` scenarios; the command itself uses [`measure`]).
    pub fn of_size(length_cm: f64, girth_cm: f64) -> Hotdog {
        Hotdog {
            length_cm: round1(length_cm),
            girth_cm: round1(girth_cm),
            tier: Tier::of(length_cm),
        }
    }
}

fn round1(x: f64) -> f64 {
    (x * 10.0).round() / 10.0
}

fn inches(cm: f64) -> f64 {
    round1(cm / 2.54)
}

/// The hotdog of the user with this Discord ID. About 25% are fill (below);
/// the rest are normal around 13.5 cm long, 11.7 cm around, with girth
/// correlated to length (r = 0.6). The fill group tops every printable size
/// (2.5–40.0 cm long, 5.0–22.0 around) up to at least 10% of the chance of the
/// most likely one, so no size is out of reach and the tails are flat.
pub fn measure(key: &Key, user_id: u64) -> Hotdog {
    let mut s = Stream::new(key, "hotdog", user_id);
    let rarity = s.unit();
    let extra = s.unit();
    let z1 = s.normal();
    let z2 = s.normal();
    let girth_z = 0.6 * z1 + 0.8 * z2;
    // The clamps are the printable range, which only the far tails reach, so
    // no size collects the draws beyond it (no spikes).
    let (length, girth) = if rarity < FILL {
        // One draw for both, with the tables' short and long ends lined up:
        // a short fill hotdog is a thin one and a long one thick.
        let (short, thin) = (below(&LENGTH_FILL, 135), below(&GIRTH_FILL, 117));
        let girth_u = if extra < short {
            extra / short * thin
        } else {
            thin + (extra - short) / (1.0 - short) * (1.0 - thin)
        };
        (pick(&LENGTH_FILL, extra), pick(&GIRTH_FILL, girth_u))
    } else {
        (
            (13.5 + 2.8 * z1).clamp(2.5, 40.0),
            (11.7 + 1.6 * girth_z).clamp(5.0, 22.0),
        )
    };
    let length_cm = round1(length);
    Hotdog {
        length_cm,
        girth_cm: round1(girth),
        tier: Tier::of(length_cm),
    }
}

/// The share of `table`'s weight below `tenths`.
fn below(table: &[(u16, u32)], tenths: u16) -> f64 {
    let total: u64 = table.iter().map(|&(_, w)| w as u64).sum();
    let under: u64 = table.iter().filter(|&&(t, _)| t < tenths).map(|&(_, w)| w as u64).sum();
    under as f64 / total as f64
}

/// The size in `table` (tenths of a cm, weight) at quantile `u` in (0, 1).
fn pick(table: &[(u16, u32)], u: f64) -> f64 {
    let total: u64 = table.iter().map(|&(_, w)| w as u64).sum();
    let target = (u * total as f64) as u64;
    let mut seen = 0;
    for &(tenths, w) in table {
        seen += w as u64;
        if target < seen {
            return tenths as f64 / 10.0;
        }
    }
    table[table.len() - 1].0 as f64 / 10.0
}

// The fill group's share and sizes: what the normal group leaves short of the
// floor at each size, in parts per billion of the group. Regenerate with
// `python3 tools/hotdog-fill/gen.py` after changing any other constant in
// `measure`; `distribution_is_sane` checks the floor.
// Generated by tools/hotdog-fill/gen.py (FLOOR = 0.1): floor 0.1001 of peak for length, 0.1000 for girth.
const FILL: f64 = 0.2539;
const LENGTH_FILL: [(u16, u32); 257] = [
    (25, 4720308), (26, 4753066), (27, 4751164), (28, 4748923), (29, 4746294), (30, 4743217),
    (31, 4739626), (32, 4735449), (33, 4730603), (34, 4724996), (35, 4718527), (36, 4711081),
    (37, 4702535), (38, 4692749), (39, 4681574), (40, 4668841), (41, 4654370), (42, 4637962),
    (43, 4619401), (44, 4598452), (45, 4574862), (46, 4548358), (47, 4518644), (48, 4485403),
    (49, 4448298), (50, 4406967), (51, 4361022), (52, 4310056), (53, 4253635), (54, 4191300),
    (55, 4122568), (56, 4046934), (57, 3963866), (58, 3872810), (59, 3773189), (60, 3664405),
    (61, 3545840), (62, 3416854), (63, 3276793), (64, 3124985), (65, 2960748), (66, 2783386),
    (67, 2592197), (68, 2386473), (69, 2165505), (70, 1928587), (71, 1675017), (72, 1404104),
    (73, 1115171), (74, 807561), (75, 480639), (76, 133799), (196, 181133), (197, 488743),
    (198, 777676), (199, 1048589), (200, 1302159), (201, 1539077), (202, 1760045), (203, 1965768),
    (204, 2156958), (205, 2334320), (206, 2498557), (207, 2650364), (208, 2790426), (209, 2919411),
    (210, 3037977), (211, 3146761), (212, 3246382), (213, 3337438), (214, 3420506), (215, 3496140),
    (216, 3564872), (217, 3627207), (218, 3683628), (219, 3734594), (220, 3780538), (221, 3821870),
    (222, 3858975), (223, 3892215), (224, 3921929), (225, 3948434), (226, 3972024), (227, 3992972),
    (228, 4011534), (229, 4027942), (230, 4042413), (231, 4055145), (232, 4066321), (233, 4076106),
    (234, 4084653), (235, 4092099), (236, 4098568), (237, 4104175), (238, 4109021), (239, 4113198),
    (240, 4116789), (241, 4119866), (242, 4122495), (243, 4124735), (244, 4126638), (245, 4128249),
    (246, 4129609), (247, 4130753), (248, 4131712), (249, 4132513), (250, 4133180), (251, 4133732),
    (252, 4134189), (253, 4134565), (254, 4134873), (255, 4135125), (256, 4135329), (257, 4135494),
    (258, 4135627), (259, 4135733), (260, 4135818), (261, 4135886), (262, 4135939), (263, 4135980),
    (264, 4136013), (265, 4136038), (266, 4136057), (267, 4136072), (268, 4136083), (269, 4136091),
    (270, 4136098), (271, 4136102), (272, 4136106), (273, 4136108), (274, 4136110), (275, 4136111),
    (276, 4136112), (277, 4136112), (278, 4136113), (279, 4136113), (280, 4136113), (281, 4136113),
    (282, 4136114), (283, 4136114), (284, 4136114), (285, 4136114), (286, 4136114), (287, 4136114),
    (288, 4136114), (289, 4136114), (290, 4136114), (291, 4136114), (292, 4136114), (293, 4136114),
    (294, 4136114), (295, 4136114), (296, 4136114), (297, 4136114), (298, 4136114), (299, 4136114),
    (300, 4136114), (301, 4136114), (302, 4136114), (303, 4136114), (304, 4136114), (305, 4136114),
    (306, 4136114), (307, 4136114), (308, 4136114), (309, 4136114), (310, 4136114), (311, 4136114),
    (312, 4136114), (313, 4136114), (314, 4136114), (315, 4136114), (316, 4136114), (317, 4136114),
    (318, 4136114), (319, 4136114), (320, 4136114), (321, 4136114), (322, 4136114), (323, 4136114),
    (324, 4136114), (325, 4136114), (326, 4136114), (327, 4136114), (328, 4136114), (329, 4136114),
    (330, 4136114), (331, 4136114), (332, 4136114), (333, 4136114), (334, 4136114), (335, 4136114),
    (336, 4136114), (337, 4136114), (338, 4136114), (339, 4136114), (340, 4136114), (341, 4136114),
    (342, 4136114), (343, 4136114), (344, 4136114), (345, 4136114), (346, 4136114), (347, 4136114),
    (348, 4136114), (349, 4136114), (350, 4136114), (351, 4136114), (352, 4136114), (353, 4136114),
    (354, 4136114), (355, 4136114), (356, 4136114), (357, 4136114), (358, 4136114), (359, 4136114),
    (360, 4136114), (361, 4136114), (362, 4136114), (363, 4136114), (364, 4136114), (365, 4136114),
    (366, 4136114), (367, 4136114), (368, 4136114), (369, 4136114), (370, 4136114), (371, 4136114),
    (372, 4136114), (373, 4136114), (374, 4136114), (375, 4136114), (376, 4136114), (377, 4136114),
    (378, 4136114), (379, 4136114), (380, 4136114), (381, 4136114), (382, 4136114), (383, 4136114),
    (384, 4136114), (385, 4136114), (386, 4136114), (387, 4136114), (388, 4136114), (389, 4136114),
    (390, 4136114), (391, 4136114), (392, 4136114), (393, 4136114), (394, 4136114), (395, 4136114),
    (396, 4136114), (397, 4136114), (398, 4136114), (399, 4136114), (400, 4136114),
];
const GIRTH_FILL: [(u16, u32); 106] = [
    (50, 7252412), (51, 7267198), (52, 7264122), (53, 7260087), (54, 7254830), (55, 7248022),
    (56, 7239260), (57, 7228050), (58, 7213795), (59, 7195773), (60, 7173121), (61, 7144812),
    (62, 7109635), (63, 7066165), (64, 7012747), (65, 6947462), (66, 6868105), (67, 6772163),
    (68, 6656786), (69, 6518769), (70, 6354534), (71, 6160119), (72, 5931166), (73, 5662925),
    (74, 5350261), (75, 4987672), (76, 4569317), (77, 4089059), (78, 3540516), (79, 2917134),
    (80, 2212262), (81, 1419255), (82, 531581), (148, 792641), (149, 2117411), (150, 3325194),
    (151, 4420704), (152, 5409338), (153, 6297011), (154, 7090018), (155, 7794890), (156, 8418273),
    (157, 8966815), (158, 9447073), (159, 9865428), (160, 10228017), (161, 10540681), (162, 10808922),
    (163, 11037875), (164, 11232291), (165, 11396525), (166, 11534542), (167, 11649920), (168, 11745862),
    (169, 11825218), (170, 11890503), (171, 11943922), (172, 11987391), (173, 12022569), (174, 12050877),
    (175, 12073529), (176, 12091551), (177, 12105806), (178, 12117016), (179, 12125779), (180, 12132586),
    (181, 12137844), (182, 12141878), (183, 12144954), (184, 12147285), (185, 12149040), (186, 12150352),
    (187, 12151326), (188, 12152044), (189, 12152571), (190, 12152953), (191, 12153229), (192, 12153427),
    (193, 12153568), (194, 12153667), (195, 12153736), (196, 12153785), (197, 12153818), (198, 12153840),
    (199, 12153856), (200, 12153866), (201, 12153873), (202, 12153877), (203, 12153880), (204, 12153882),
    (205, 12153883), (206, 12153884), (207, 12153884), (208, 12153885), (209, 12153885), (210, 12153885),
    (211, 12153885), (212, 12153885), (213, 12153885), (214, 12153885), (215, 12153885), (216, 12153885),
    (217, 12153885), (218, 12153885), (219, 12153885), (220, 12153885),
];

impl Hotdog {
    /// The measurement card: `whose` is a mention or name.
    pub fn card(&self, whose: &str) -> String {
        format!(
            "🌭 {whose}'s hotdog: **{:.1} cm** long ({:.1} in), **{:.1} cm** around ({:.1} in) — *{}*",
            self.length_cm,
            inches(self.length_cm),
            self.girth_cm,
            inches(self.girth_cm),
            self.tier.name(),
        )
    }

    /// What the requester "says" to the bot, for its LLM commentary, from a
    /// persona's `commands` prompt or [`PROMPT`] (placeholders: `persona::CommandName::placeholders`).
    pub fn prompt(&self, template: &str, bot: &str, who: &Who) -> String {
        crate::persona::render_command(
            template,
            &[
                ("bot", bot),
                ("whose", &who.possessive()),
                ("whose_own", &who.possessive_own()),
                ("length", &format!("{:.1}", self.length_cm)),
                ("girth", &format!("{:.1}", self.girth_cm)),
                ("tier", &self.tier.name().to_lowercase()),
            ],
        )
    }
}

/// `/super-admin test`: the hotdog and inner flavor match (`novelty::self_love`) of
/// `user_id` under every key in `keys`, as a monospace table with a summary;
/// the row for `live` (the configured key) is marked.
pub fn key_table(label: &str, user_id: u64, keys: std::ops::RangeInclusive<u64>, live: u64) -> String {
    let rows: Vec<(u64, Hotdog, u8)> = keys
        .map(|k| {
            let key = Key::new(k);
            (k, measure(&key, user_id), super::novelty::self_love(&key, user_id))
        })
        .collect();
    let mut out = format!(
        "{label} ({user_id})\n\
         key │ length cm (in) │ girth cm (in) │ inner match │ tier\n\
         ────┼────────────────┼───────────────┼─────────────┼─────────────────────────\n"
    );
    for (k, d, love) in &rows {
        let row = format!(
            "{k:>3} │ {:>6.1} ({:>4.1})  │ {:>5.1} ({:>4.1})  │ {love:>10}% │ {:<24}",
            d.length_cm,
            inches(d.length_cm),
            d.girth_cm,
            inches(d.girth_cm),
            d.tier.name(),
        );
        out += if *k == live { row + " ← live" } else { row.trim_end().to_string() }.as_str();
        out += "\n";
    }
    let n = rows.len().max(1) as f64;
    let mean = |f: &dyn Fn(&(u64, Hotdog, u8)) -> f64| rows.iter().map(f).sum::<f64>() / n;
    let tiers: Vec<String> = Tier::ALL
        .iter()
        .map(|t| format!("{} {}", t.name(), rows.iter().filter(|(_, d, _)| d.tier == *t).count()))
        .collect();
    out += &format!(
        "summary: mean length {:.1} cm, mean girth {:.1} cm, mean inner flavor match {:.0}% · {}\n",
        mean(&|r| r.1.length_cm),
        mean(&|r| r.1.girth_cm),
        mean(&|r| r.2 as f64),
        tiers.join(", "),
    );
    out
}

/// What `/super-admin test` searches for: a key under which the caller and
/// the bot get these tiers and these inner flavor matches.
pub struct Wish {
    pub me_tier: Tier,
    pub bot_tier: Tier,
    pub me_love: std::ops::RangeInclusive<u8>,
    pub bot_love: std::ops::RangeInclusive<u8>,
}

pub const WISH: Wish = Wish {
    me_tier: Tier::CocktailWeenie,
    bot_tier: Tier::BallparkFrank,
    me_love: 0..=24,
    bot_love: 76..=100,
};

/// One key's results for the caller and the bot: hotdog and inner flavor match.
pub struct KeyResult {
    pub key: u64,
    pub me: (Hotdog, u8),
    pub bot: (Hotdog, u8),
}

/// The first `max` keys in `keys` that give `me` and `bot` what `wish` asks
/// for, and the last key looked at (the end of `keys` if fewer were found).
pub fn scan(me: u64, bot: u64, keys: std::ops::RangeInclusive<u64>, wish: &Wish, max: usize) -> (Vec<KeyResult>, u64) {
    let mut found = Vec::new();
    let mut last = *keys.start();
    for k in keys {
        last = k;
        let key = Key::new(k);
        // Self-love first: one block each, against thirteen for a hotdog.
        let (me_love, bot_love) = (super::novelty::self_love(&key, me), super::novelty::self_love(&key, bot));
        if !wish.me_love.contains(&me_love) || !wish.bot_love.contains(&bot_love) {
            continue;
        }
        let (me_dog, bot_dog) = (measure(&key, me), measure(&key, bot));
        if me_dog.tier == wish.me_tier && bot_dog.tier == wish.bot_tier {
            found.push(KeyResult { key: k, me: (me_dog, me_love), bot: (bot_dog, bot_love) });
            if found.len() == max {
                break;
            }
        }
    }
    (found, last)
}

/// A key as `config.rsn` writes it: `0x0000_0000_0000_000E`.
pub fn hex_key(k: u64) -> String {
    format!("0x{:04X}_{:04X}_{:04X}_{:04X}", k >> 48, (k >> 32) & 0xFFFF, (k >> 16) & 0xFFFF, k & 0xFFFF)
}

/// `wish` in words, for the table and the reply.
pub fn wish_text(me: &str, bot: &str, wish: &Wish) -> String {
    format!(
        "{me} gets a {} with inner flavor match {}–{}% and {bot} a {} with inner flavor match {}–{}%",
        wish.me_tier.name(),
        wish.me_love.start(),
        wish.me_love.end(),
        wish.bot_tier.name(),
        wish.bot_love.start(),
        wish.bot_love.end(),
    )
}

/// The keys `scan` found (up to `last`), as a monospace table; the row for
/// `live` is marked.
pub fn matches_table(me: &str, bot: &str, wish: &Wish, found: &[KeyResult], last: u64, live: u64) -> String {
    let mut out = format!(
        "matches: keys where {}\nsearched keys 0–{last}, {} found\n",
        wish_text(me, bot, wish),
        found.len()
    );
    if found.is_empty() {
        return out;
    }
    let name = |s: &str| format!("{:<14}", s.chars().take(14).collect::<String>());
    out += &format!(
        "\n    key │ config value          │ {} │ match │ {} │ match\n\
         ────────┼───────────────────────┼────────────────┼───────┼────────────────┼───────\n",
        name(me),
        name(bot),
    );
    for r in found {
        out += &format!(
            "{:>7} │ {} │ {:>4.1} × {:>4.1} cm │ {:>4}% │ {:>4.1} × {:>4.1} cm │ {:>4}%{}\n",
            r.key,
            hex_key(r.key),
            r.me.0.length_cm,
            r.me.0.girth_cm,
            r.me.1,
            r.bot.0.length_cm,
            r.bot.0.girth_cm,
            r.bot.1,
            if r.key == live { "  ← live" } else { "" },
        );
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::discord::seeded::test_key;

    #[test]
    fn same_id_same_hotdog() {
        for id in [0, 1, 80351110224678912, u64::MAX] {
            assert_eq!(measure(&test_key(), id), measure(&test_key(), id));
        }
        assert_ne!(measure(&test_key(), 1), measure(&test_key(), 2));
    }

    /// Results for real-looking IDs, pinned: if this fails, a change resized
    /// existing users' hotdogs. Update it only on purpose.
    #[test]
    fn pinned_results() {
        let got: Vec<_> = [80351110224678912u64, 1234567890123456789, 999]
            .map(|id| {
                let h = measure(&test_key(), id);
                (h.length_cm, h.girth_cm, h.tier)
            })
            .into();
        assert_eq!(got, PINNED);
    }

    const PINNED: [(f64, f64, Tier); 3] = [
        (15.5, 11.5, Tier::Footlong),
        (7.0, 7.9, Tier::BallparkFrank),
        (15.0, 11.9, Tier::Footlong),
    ];

    #[test]
    fn distribution_is_sane() {
        let n = 200_000u64;
        let (mut tiny, mut legend, mut sum) = (0, 0, 0.0);
        let mut tiers = std::collections::HashSet::new();
        // Users at each printable size, by tenths of a cm.
        let (mut lengths, mut girths) = ([0u32; 401], [0u32; 221]);
        let key = test_key();
        // Snowflake-like IDs: sequential, as close IDs are the hard case.
        for id in (0..n).map(|i| 1_100_000_000_000_000_000 + i) {
            let h = measure(&key, id);
            assert!((2.5..=40.0).contains(&h.length_cm), "{h:?}");
            assert!((5.0..=22.0).contains(&h.girth_cm), "{h:?}");
            tiny += (h.length_cm < 6.5) as u32;
            legend += (h.length_cm >= 26.0) as u32;
            sum += h.length_cm;
            tiers.insert(h.tier.name());
            lengths[(h.length_cm * 10.0).round() as usize] += 1;
            girths[(h.girth_cm * 10.0).round() as usize] += 1;
        }
        let rate = |k: u32| k as f64 / n as f64;
        // Under 6.5 cm and from 26 cm: expected 4.8% and 14.8% (tools/hotdog-fill/gen.py).
        assert!((0.043..0.054).contains(&rate(tiny)), "tiny {}", rate(tiny));
        assert!((0.14..0.158).contains(&rate(legend)), "legendary {}", rate(legend));
        let mean = sum / n as f64;
        assert!((16.0..17.0).contains(&mean), "mean {mean}");
        assert_eq!(tiers.len(), 6);
        // The floor: every size at least 10% as likely as the likeliest
        // (7% here, for sampling noise at ~210 users a size).
        for (name, lo, counts) in [("length", 25, &lengths[25..]), ("girth", 50, &girths[50..])] {
            let (min, max) = (counts.iter().min().unwrap(), counts.iter().max().unwrap());
            assert!(*min as f64 >= 0.07 * *max as f64, "{name}: least {min}, most {max}");
            // No spikes: no size much likelier than both its neighbours (a
            // clamp inside the range made the old 6.5 cm one 9×).
            for (i, w) in counts.windows(3).enumerate() {
                assert!(w[1] as f64 <= 1.5 * w[0].max(w[2]) as f64, "{name}: spike at {:.1} cm ({w:?})", (lo + i + 1) as f64 / 10.0);
            }
        }
    }

    /// Where each `│` sits in a line, counted in characters.
    fn bars(r: &str) -> Vec<usize> {
        r.char_indices().filter(|(_, c)| *c == '│').map(|(i, _)| r[..i].chars().count()).collect()
    }

    #[test]
    fn key_table_rows_line_up_and_mark_the_live_key() {
        let t = key_table("incf", 80351110224678912, 0..=100, 0);
        let lines: Vec<&str> = t.lines().collect();
        // Label, header, rule, 101 rows, summary.
        assert_eq!(lines.len(), 105);
        assert_eq!(lines[0], "incf (80351110224678912)");
        let rows = &lines[3..104];
        assert!(rows[0].ends_with("← live") && rows[0].starts_with("  0 │"));
        assert!(rows[1..].iter().all(|r| !r.contains("live")));
        assert!(rows.iter().all(|r| bars(r) == bars(lines[1])), "{t}");
        let key = Key::new(0);
        let (d, love) = (measure(&key, 80351110224678912), super::super::novelty::self_love(&key, 80351110224678912));
        assert!(rows[0].contains(&format!("{:.1}", d.length_cm)) && rows[0].contains(&format!("{love:>8}%")));
        assert!(lines[104].starts_with("summary: mean length ") && lines[104].contains("mean inner flavor match"));
        assert!(!key_table("x", 1, 0..=100, 500).contains("live"));
    }

    #[test]
    fn scan_returns_only_matches_and_stops_at_max() {
        let (me, bot) = (80351110224678912, 1234567890123456789);
        let any = Wish { me_tier: Tier::Footlong, bot_tier: Tier::RespectableBratwurst, me_love: 0..=100, bot_love: 0..=100 };
        let (found, last) = scan(me, bot, 0..=999, &any, 3);
        assert_eq!(found.len(), 3);
        assert_eq!(last, found[2].key);
        for r in &found {
            let key = Key::new(r.key);
            assert_eq!(r.me.0, measure(&key, me));
            assert_eq!((r.me.0.tier, r.bot.0.tier), (Tier::Footlong, Tier::RespectableBratwurst));
        }
        let t = matches_table("incf", "FloobBot", &any, &found, last, found[1].key);
        let lines: Vec<&str> = t.lines().collect();
        assert!(lines[1].ends_with("3 found"));
        assert!(lines[5..].iter().all(|r| bars(r) == bars(lines[3])), "{t}");
        assert!(lines[6].ends_with("← live") && lines[6].contains(&hex_key(found[1].key)));
        // None: the whole range is searched.
        let none = Wish { me_tier: Tier::CocktailWeenie, bot_tier: Tier::CocktailWeenie, me_love: 0..=0, bot_love: 100..=100 };
        let (empty, last) = scan(me, bot, 0..=50, &none, 5);
        assert!(empty.is_empty() && last == 50);
        assert_eq!(matches_table("a", "b", &none, &empty, last, 0).lines().count(), 2);
        assert_eq!(hex_key(14), "0x0000_0000_0000_000E");
    }

    #[test]
    fn card_and_prompt() {
        let h = Hotdog {
            length_cm: 14.2,
            girth_cm: 12.0,
            tier: Tier::RespectableBratwurst,
        };
        assert_eq!(
            h.card("<@1>"),
            "🌭 <@1>'s hotdog: **14.2 cm** long (5.6 in), **12.0 cm** around (4.7 in) — *Respectable Bratwurst*"
        );
        assert!(h.prompt(PROMPT, "floob", &Who::Me).contains("measure my hotdog!"));
        assert!(h.prompt(PROMPT, "floob", &Who::Bot).contains("measure ur hotdog!"));
        let bob = Who::Other("Bob".into());
        assert_eq!(
            h.prompt(PROMPT, "floob", &bob),
            "*hands floob a measuring stick* floob, measure Bob's hotdog! it's 14.2 cm long and \
             12.0 cm around, a respectable bratwurst. what do u think of it?"
        );
        assert_eq!(h.prompt("{bot}: {length}x{girth}", "floob", &bob), "floob: 14.2x12.0");
        // "own" where "my"/"ur" alone is ambiguous (README round 20)
        assert_eq!(h.prompt("measure {whose_own} hotdog", "floob", &Who::Bot), "measure ur own hotdog");
        assert_eq!(h.prompt("measure {whose_own} hotdog", "floob", &Who::Me), "measure my own hotdog");
        assert_eq!(h.prompt("measure {whose_own} hotdog", "floob", &bob), "measure Bob's hotdog");
    }
}
