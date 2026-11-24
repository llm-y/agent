//! Memori: bagian "## Memory" di AGENTS.md, satu fakta per baris. Tanpa database: file biasa
//! yang bisa dibaca, diedit, dan masuk git. Memori proyek ada di akar proyek; memori untuk
//! semua proyek di ~/.config/llmy/AGENTS.md, jadi ikut terbawa ke proyek lain.

use crate::rules;
use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};

const HEAD: &str = "## Memory";
const MAX_ITEMS: usize = 40;
const MAX_CHARS: usize = 240;

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Scope {
    Project,
    All,
}

/// File tujuan. Yang dipilih adalah file yang memang dibaca `rules`: membuat AGENTS.md di
/// samping CLAUDE.md akan membuat CLAUDE.md tidak terbaca lagi.
pub fn target(cwd: &Path, scope: Scope) -> Result<PathBuf, String> {
    target_in(cwd, scope, rules::global())
}

fn target_in(cwd: &Path, scope: Scope, global: Option<PathBuf>) -> Result<PathBuf, String> {
    if scope == Scope::All {
        let p = global.ok_or("folder konfigurasi tidak ditemukan")?;
        return Ok(fs::canonicalize(&p).unwrap_or(p));
    }
    let top = rules::top(cwd);
    for n in ["AGENTS.md", "CLAUDE.md"] {
        let p = top.join(n);
        if p.symlink_metadata().is_ok() {
            return match fs::canonicalize(&p) {
                Ok(r) if r.starts_with(top) => Ok(r),
                _ => Err(format!("{n} menunjuk ke luar proyek; memori tidak disimpan")),
            };
        }
    }
    Ok(top.join("AGENTS.md"))
}

/// (awal, akhir) bagian memori dalam byte, termasuk judulnya.
fn section(s: &str) -> Option<(usize, usize)> {
    let (mut pos, mut start, mut fence) = (0, None, false);
    for l in s.split_inclusive('\n') {
        let t = l.trim_end();
        if t.starts_with("```") {
            fence = !fence;
        }
        match start {
            None if !fence && t == HEAD => start = Some(pos),
            Some(a) if !fence && (t.starts_with("# ") || t.starts_with("## ")) => return Some((a, pos)),
            _ => {}
        }
        pos += l.len();
    }
    start.map(|a| (a, s.len()))
}

pub fn items(s: &str) -> Vec<String> {
    let Some((a, b)) = section(s) else { return vec![] };
    s[a..b].lines().filter_map(|l| l.strip_prefix("- ")).map(|l| l.trim().to_string()).collect()
}

/// Memori ditaruh di depan agar tidak ikut terpotong bila file terlalu panjang.
pub fn front(s: &str) -> String {
    match section(s) {
        Some((a, b)) if a > 0 && !items(s).is_empty() => {
            format!("{}\n\n{}{}", s[a..b].trim_end(), &s[..a], &s[b..]).trim_end().to_string()
        }
        _ => s.to_string(),
    }
}

fn key(s: &str) -> String {
    s.trim_end_matches('.').split_whitespace().collect::<Vec<_>>().join(" ").to_lowercase()
}

/// Setengah kata penting sama: kemungkinan besar fakta yang sama dengan nilai baru.
fn similar(a: &str, b: &str) -> bool {
    let words = |s: &str| -> HashSet<String> {
        s.to_lowercase().split(|c: char| !c.is_alphanumeric()).filter(|w| w.chars().count() >= 3).map(String::from).collect()
    };
    let (a, b) = (words(a), words(b));
    let n = a.len().min(b.len());
    n > 0 && a.intersection(&b).count() * 2 >= n.max(2)
}

/// Memori masuk git dan dikirim ke model: jangan sampai memuat kunci atau sandi.
fn secret(s: &str) -> bool {
    let l = s.to_lowercase();
    if s.contains("-----BEGIN") {
        return true;
    }
    let long = |p: &str, n: usize| {
        s.match_indices(p).any(|(i, _)| {
            !s[..i].ends_with(|c: char| c.is_alphanumeric())
                && s[i + p.len()..].bytes().take_while(|b| b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_')).count() >= n
        })
    };
    if [("sk-", 16), ("ghp_", 16), ("github_pat_", 16), ("AKIA", 12), ("xox", 12), ("AIza", 20)].iter().any(|(p, n)| long(p, *n)) {
        return true;
    }
    ["password", "passwd", "kata sandi", "api key", "api_key", "apikey", "token", "secret"].iter().any(|w| {
        l.match_indices(w).any(|(i, _)| {
            let r = l[i + w.len()..].trim_start();
            r.strip_prefix([':', '=']).is_some_and(|v| v.trim().len() >= 6)
        })
    })
}

/// Isi file sesudah fakta ditambah dan/atau fakta lama diganti.
pub fn apply(src: &str, fact: &str, replaces: &str) -> Result<String, String> {
    if secret(fact) {
        return Err("fakta ini tampak memuat kunci atau sandi; tidak disimpan".into());
    }
    let fact = fact.split_whitespace().collect::<Vec<_>>().join(" ");
    let fact = fact.trim_start_matches(['-', '*', '#', ' ']).to_string();
    let both = key(replaces) == "none";
    let replaces = if both { "" } else { replaces };
    let old = key(replaces.trim_start_matches(['-', '*', ' ']));
    if fact.is_empty() && old.is_empty() {
        return Err("`fact` kosong".into());
    }
    if fact.chars().count() > MAX_CHARS {
        return Err(format!("fakta terlalu panjang; maksimal {MAX_CHARS} karakter, satu fakta per panggilan"));
    }
    let mut v = items(src);
    if !old.is_empty() {
        let exact = v.iter().position(|i| key(i) == old);
        let part: Vec<usize> = (0..v.len()).filter(|i| key(&v[*i]).contains(&old)).collect();
        let i = match (exact, part.as_slice()) {
            (Some(i), _) => i,
            (None, [i]) => *i,
            (None, []) => return Err(format!("`replaces` tidak cocok dengan memori mana pun. Isinya:\n{}", list(&v))),
            _ => return Err(format!("`replaces` cocok dengan beberapa memori; salin satu baris utuh:\n{}", list(&v))),
        };
        v.remove(i);
    }
    if !fact.is_empty() {
        if old.is_empty()
            && !both
            && let Some(i) = v.iter().find(|i| key(i) != key(&fact) && similar(i, &fact))
        {
            return Err(format!("memori mirip sudah ada: \"{i}\". Bila fakta baru menggantikannya, panggil lagi dengan replaces berisi baris itu; bila keduanya tetap benar, pakai replaces \"none\""));
        }
        if v.iter().any(|i| key(i) == key(&fact)) {
            if old.is_empty() {
                return Err("sudah ada di memori".into());
            }
        } else {
            v.push(fact);
        }
    }
    if v.len() > MAX_ITEMS {
        return Err(format!("memori penuh ({MAX_ITEMS}); gabungkan atau ganti yang usang lewat `replaces`:\n{}", list(&v[..MAX_ITEMS])));
    }
    let nl = if src.contains("\r\n") { "\r\n" } else { "\n" };
    let body = format!("{HEAD}{nl}{nl}{}", v.iter().map(|i| format!("- {i}{nl}")).collect::<String>());
    Ok(match section(src) {
        Some((a, b)) => {
            // Catatan lain di bagian ini milik pengguna: pertahankan.
            let keep: Vec<&str> = src[a..b].lines().skip(1).filter(|l| !l.starts_with("- ")).map(str::trim_end).collect();
            let keep = keep.join(nl);
            let keep = keep.trim();
            let mid = if keep.is_empty() { body } else { body.replacen(HEAD, &format!("{HEAD}{nl}{nl}{keep}"), 1) };
            let after = &src[b..];
            format!("{}{mid}{}{after}", &src[..a], if after.is_empty() { "" } else { nl })
        }
        None if src.trim().is_empty() => body,
        None => format!("{}{nl}{nl}{body}", src.trim_end()),
    })
}

/// Permintaan mengingat yang jelas dari pengguna: (cakupan, kalimatnya sendiri).
pub fn wish(msg: &str) -> Option<(&'static str, String)> {
    let l = msg.trim().to_lowercase();
    if l.ends_with('?') || l.contains('\n') || l.chars().count() > MAX_CHARS {
        return None;
    }
    let skip = ["tolong", "please", "ok", "oke"];
    let first = l.split(|c: char| !c.is_alphanumeric()).find(|w| !w.is_empty() && !skip.contains(w));
    let trigger = matches!(first, Some("ingat" | "ingatlah" | "remember" | "memorize"));
    if !trigger && !["mulai sekarang", "untuk seterusnya", "from now on"].iter().any(|w| l.starts_with(w)) {
        return None;
    }
    // "Ingat untuk semua proyek: jawab singkat" -> "jawab singkat"
    let fact = match msg.split_once(':') {
        Some((p, f)) if trigger && p.split_whitespace().count() <= 5 && f.split_whitespace().count() >= 2 => f,
        _ => msg,
    };
    let all = ["semua proyek", "setiap proyek", "all project", "every project"].iter().any(|w| l.contains(w));
    Some((if all { "all" } else { "project" }, fact.trim().to_string()))
}

fn list(v: &[String]) -> String {
    v.iter().map(|i| format!("- {i}")).collect::<Vec<_>>().join("\n")
}

/// Untuk perintah /memory.
pub fn show(cwd: &Path) -> String {
    let mut out = vec![];
    for (name, scope) in [("Proyek ini", Scope::Project), ("Semua proyek", Scope::All)] {
        let Ok(p) = target(cwd, scope) else { continue };
        let v = items(&fs::read_to_string(&p).unwrap_or_default());
        let body = if v.is_empty() { "(kosong)".into() } else { list(&v) };
        out.push(format!("{name}: {}\n{body}", p.display()));
    }
    out.join("\n\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tambah_ganti_hapus() {
        let a = apply("", "Pakai pnpm", "").unwrap();
        assert_eq!(a, "## Memory\n\n- Pakai pnpm\n");
        assert!(apply(&a, " pakai  PNPM. ", "").is_err());
        let b = apply(&a, "- Jawab singkat\n# palsu", "").unwrap();
        assert_eq!(items(&b), ["Pakai pnpm", "Jawab singkat # palsu"]);
        assert!(apply(&b, "Pakai bun, bukan pnpm", "").unwrap_err().contains("Pakai pnpm"));
        assert_eq!(items(&apply(&b, "Pakai bun, bukan pnpm", "none").unwrap()).len(), 3);
        let c = apply(&b, "Pakai bun", "pnpm").unwrap();
        assert_eq!(items(&c), ["Jawab singkat # palsu", "Pakai bun"]);
        assert_eq!(items(&apply(&c, "", "Pakai bun").unwrap()), ["Jawab singkat # palsu"]);
        assert!(apply(&c, "x", "tidak ada").unwrap_err().contains("Pakai bun"));
        assert!(apply(&c, "", "").is_err());
        assert!(apply("## Memory\n- ab\n- abc\n", "x", "b").unwrap_err().contains("beberapa"));
        assert_eq!(items(&apply("## Memory\n- ab\n- abc\n", "x", "ab").unwrap()), ["abc", "x"]);
    }

    #[test]
    fn isi_lain_tetap() {
        let s = "# Proyek\r\n\r\n```\r\n## Memory\r\n```\r\n\r\n## Memory\r\ncatatan saya\r\n- lama\r\n\r\n## Lain\r\nisi\r\n";
        let r = apply(s, "baru", "").unwrap();
        assert_eq!(r, "# Proyek\r\n\r\n```\r\n## Memory\r\n```\r\n\r\n## Memory\r\n\r\ncatatan saya\r\n\r\n- lama\r\n- baru\r\n\r\n## Lain\r\nisi\r\n");
        assert_eq!(apply("# A\nisi", "baru", "").unwrap(), "# A\nisi\n\n## Memory\n\n- baru\n");
        assert!(front(s).starts_with("## Memory\r\ncatatan") && front(s).ends_with("## Lain\r\nisi"));
        assert_eq!(front("# A\n"), "# A\n");
    }

    #[test]
    fn batas_dan_rahasia() {
        let mut s = String::new();
        for i in 0..MAX_ITEMS {
            s = apply(&s, &format!("fakta {i}"), "none").unwrap();
        }
        assert!(apply(&s, "satu lagi", "none").unwrap_err().contains("penuh"));
        assert!(apply(&s, "pengganti", "fakta 0").is_ok());
        assert!(apply("", &"x".repeat(MAX_CHARS + 1), "").is_err());
        for x in ["kunci sk-abcdefghijklmnopqrst", "password: hunter22", "API_KEY = abcdef123", "-----BEGIN RSA"] {
            assert!(apply("", x, "").is_err(), "{x}");
        }
        for x in ["token disimpan di env OPENAI_API_KEY", "pakai risk-based-testing-approach", "password diambil dari pass"] {
            assert!(apply("", x, "").is_ok(), "{x}");
        }
    }

    #[test]
    fn permintaan_mengingat() {
        let w = |x| wish(x).map(|(s, f)| format!("{s}|{f}")).unwrap_or_default();
        assert_eq!(w("Ingat: pakai bun"), "project|pakai bun");
        assert_eq!(w("tolong ingat untuk semua proyek: jawab singkat"), "all|jawab singkat");
        assert_eq!(w("Ingat rilis: jumat"), "project|Ingat rilis: jumat");
        assert_eq!(w("Mulai sekarang jawab singkat"), "project|Mulai sekarang jawab singkat");
        for x in ["apa kamu ingat?", "Ingat tidak fungsi itu?", "perbaiki pemakaian memory", "ingat\nfoo"] {
            assert_eq!(w(x), "", "{x}");
        }
    }

    #[test]
    fn file_tujuan() {
        let d = Path::new(env!("CARGO_MANIFEST_DIR")).join("target/test/memory");
        let _ = fs::remove_dir_all(&d);
        fs::create_dir_all(d.join("p/.git")).unwrap();
        fs::create_dir_all(d.join("p/sub")).unwrap();
        let p = fs::canonicalize(d.join("p")).unwrap();
        let sub = p.join("sub");
        assert_eq!(target_in(&sub, Scope::Project, None).unwrap(), p.join("AGENTS.md"));
        fs::write(p.join("CLAUDE.md"), "x").unwrap();
        assert_eq!(target_in(&sub, Scope::Project, None).unwrap(), p.join("CLAUDE.md"));
        assert_eq!(target_in(&sub, Scope::All, Some(d.join("g/AGENTS.md"))).unwrap(), d.join("g/AGENTS.md"));
        assert!(target_in(&sub, Scope::All, None).is_err());
        #[cfg(unix)]
        {
            fs::write(d.join("luar.md"), "x").unwrap();
            std::os::unix::fs::symlink(d.join("luar.md"), p.join("AGENTS.md")).unwrap();
            assert!(target_in(&sub, Scope::Project, None).is_err());
        }
    }
}
