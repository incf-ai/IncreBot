//! The similarity table `/embed-compare` and `increbot embed` print: pictures
//! `I1`, `I2`, then texts `T1`, `T2`, each pair once (the strict lower triangle,
//! no diagonal), with `│` and `─┼─` between the picture and text blocks.
//!
//! ```text
//!       I1    I2 │    T1
//! I2  0.41       │
//! ───────────────┼──────
//! T1  0.28  0.31 │
//! T2  0.20  0.25 │  0.55
//! ```
//!
//! Two items make one line, `I1 ↔ T1  0.28`.

use super::cosine;

const CELL: usize = 6;
const LABEL: usize = 2;

/// The table for these embeddings, or `None` for fewer than two.
pub fn table(images: &[Vec<f32>], texts: &[Vec<f32>]) -> Option<String> {
    let items: Vec<(String, &[f32])> = images
        .iter()
        .enumerate()
        .map(|(i, e)| (format!("I{}", i + 1), e.as_slice()))
        .chain(texts.iter().enumerate().map(|(i, e)| (format!("T{}", i + 1), e.as_slice())))
        .collect();
    let n = items.len();
    if n < 2 {
        return None;
    }
    let value = |i: usize, j: usize| two_places(cosine(items[i].1, items[j].1));
    if n == 2 {
        return Some(format!("{} ↔ {}  {}", items[0].0, items[1].0, value(1, 0)));
    }

    // Columns are items 0..n-1, rows 1..n; picture columns, then text columns.
    let image_cols = images.len().min(n - 1);
    let split = image_cols > 0 && image_cols < n - 1;
    let line = |label: &str, cell: &dyn Fn(usize) -> String| {
        let mut s = format!("{label:<LABEL$}");
        for j in 0..n - 1 {
            if split && j == image_cols {
                s.push_str(" │");
            }
            s.push_str(&format!("{:>CELL$}", cell(j)));
        }
        s.trim_end().to_owned()
    };

    let mut out = vec![line("", &|j| items[j].0.clone())];
    for (i, (label, _)) in items.iter().enumerate().skip(1) {
        if i == images.len() && i > 1 {
            // Between the picture rows and the text rows.
            let left = LABEL + CELL * image_cols;
            let rule = if split {
                format!("{}┼{}", "─".repeat(left + 1), "─".repeat(CELL * (n - 1 - image_cols)))
            } else {
                "─".repeat(left)
            };
            out.push(rule);
        }
        out.push(line(label, &|j| if j < i { value(i, j) } else { String::new() }));
    }
    Some(out.join("\n"))
}

/// `v` to two places, without `-0.00`.
fn two_places(v: f32) -> String {
    let s = format!("{v:.2}");
    if s == "-0.00" { "0.00".to_owned() } else { s }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Unit vectors at these angles (degrees) in the plane: cosines are cos(a − b).
    fn at(degrees: &[f32]) -> Vec<Vec<f32>> {
        degrees.iter().map(|d| vec![d.to_radians().cos(), d.to_radians().sin()]).collect()
    }

    #[test]
    fn fewer_than_two_is_none() {
        assert_eq!(table(&[], &[]), None);
        assert_eq!(table(&at(&[0.0]), &[]), None);
        assert_eq!(table(&[], &at(&[0.0])), None);
    }

    #[test]
    fn two_make_one_line() {
        assert_eq!(table(&at(&[0.0]), &at(&[60.0])).unwrap(), "I1 ↔ T1  0.50");
        assert_eq!(table(&at(&[0.0, 90.0]), &[]).unwrap(), "I1 ↔ I2  0.00");
        assert_eq!(table(&[], &at(&[0.0, 180.0])).unwrap(), "T1 ↔ T2  -1.00");
    }

    #[test]
    fn all_four() {
        let want = ["      I1    I2 │    T1", "I2  0.50       │", "───────────────┼──────", "T1  0.00  0.87 │", "T2 -0.50  0.50 │  0.87"].join("\n");
        assert_eq!(table(&at(&[0.0, 60.0]), &at(&[90.0, 120.0])).unwrap(), want);
    }

    #[test]
    fn one_picture_two_texts() {
        let want = ["      I1 │    T1", "T1  0.50 │", "T2  0.00 │  0.87"].join("\n");
        assert_eq!(table(&at(&[0.0]), &at(&[60.0, 90.0])).unwrap(), want);
    }

    #[test]
    fn two_pictures_one_text() {
        let want = ["      I1    I2", "I2  0.50", "──────────────", "T1  0.00  0.87"].join("\n");
        assert_eq!(table(&at(&[0.0, 60.0]), &at(&[90.0])).unwrap(), want);
    }

    #[test]
    fn one_kind_only() {
        assert_eq!(table(&[], &at(&[0.0, 60.0, 90.0])).unwrap(), "      T1    T2\nT2  0.50\nT3  0.00  0.87");
    }
}
