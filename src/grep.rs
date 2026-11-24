//! Tool grep: cari isi file (regex) atau daftar file (glob). Hanya membaca, tidak
//! mengikuti symlink, dan melewati yang tersembunyi serta hasil build.

use crate::cancel::Cancel;
use regex_lite::{Regex, RegexBuilder};
use std::fs;
use std::path::Path;
use std::time::{Duration, Instant};

const MAX_HITS: usize = 200;
const MAX_FILES: usize = 50_000;
const MAX_SIZE: u64 = 2 << 20;
const MAX_TIME: Duration = Duration::from_secs(10);
const SKIP: [&str; 6] = ["target", "node_modules", "__pycache__", "vendor", "dist", "venv"];

pub struct Query {
    pattern: Option<Regex>,
    glob: Option<Regex>,
    /// Glob tanpa `/` dicocokkan ke nama file saja.
    glob_name: bool,
}

impl Query {
    pub fn new(pattern: &str, glob: &str, ignore_case: bool) -> Result<Query, String> {
        let (mut pattern, mut glob) = (pattern.trim_matches(['\n', '\r']), glob.trim());
        // Model sering menaruh glob nama file di `pattern`, mis. "*.rs" (bukan regex yang sah).
        if glob.is_empty() && pattern.trim().starts_with('*') && !pattern.trim().contains(' ') {
            (pattern, glob) = ("", pattern.trim());
        }
        if pattern.trim().is_empty() {
            pattern = "";
        }
        if pattern.is_empty() && glob.is_empty() {
            return Err("isi `pattern` (isi file) atau `glob` (nama file)".into());
        }
        let build = |p: &str| RegexBuilder::new(p).case_insensitive(ignore_case).build();
        let pattern = if pattern.is_empty() {
            None
        } else {
            // Model sering memakai gaya grep dasar `a\|b`; regex yang tak sah dicari apa adanya.
            let p = pattern.replace(r"\|", "|");
            Some(build(&p).or_else(|_| build(&regex_lite::escape(pattern))).map_err(|e| e.to_string())?)
        };
        let glob = glob.trim_start_matches("./");
        let re = if glob.is_empty() { None } else { Some(Regex::new(&glob_regex(glob)).map_err(|e| format!("glob: {e}"))?) };
        Ok(Query { pattern, glob: re, glob_name: !glob.contains('/') })
    }

    fn wants(&self, rel: &str) -> bool {
        let s = if self.glob_name { rel.rsplit('/').next().unwrap_or(rel) } else { rel };
        self.glob.as_ref().is_none_or(|g| g.is_match(s))
    }
}

fn glob_regex(g: &str) -> String {
    let mut r = String::from("^");
    let mut it = g.chars().peekable();
    let mut depth = 0;
    while let Some(c) = it.next() {
        match c {
            '*' if it.peek() == Some(&'*') => {
                it.next();
                if it.peek() == Some(&'/') {
                    it.next();
                    r += "(.*/)?";
                } else {
                    r += ".*";
                }
            }
            '*' => r += "[^/]*",
            '?' => r += "[^/]",
            '{' => {
                depth += 1;
                r += "(";
            }
            '}' if depth > 0 => {
                depth -= 1;
                r += ")";
            }
            ',' if depth > 0 => r += "|",
            c => r += &regex_lite::escape(c.encode_utf8(&mut [0; 4])),
        }
    }
    r + &")".repeat(depth) + "$"
}

#[derive(Default)]
pub struct Found {
    pub text: String,
    pub hits: usize,
    pub files: usize,
    /// Alasan berhenti lebih awal.
    pub cut: Option<&'static str>,
    /// File yang namanya cocok dengan pattern; berguna bila isinya tidak ada yang cocok.
    pub names: Vec<String>,
}

struct Walk<'a> {
    q: &'a Query,
    base: &'a Path,
    max: usize,
    cancel: &'a Cancel,
    start: Instant,
    seen: usize,
    out: Found,
}

/// `base` = folder kerja, untuk menulis path relatif. `max` = batas byte hasil.
pub fn run(q: &Query, root: &Path, base: &Path, max: usize, cancel: &Cancel) -> Result<Found, String> {
    let mut w = Walk { q, base, max, cancel, start: Instant::now(), seen: 0, out: Found::default() };
    let m = fs::metadata(root).map_err(|e| format!("{}: {e}", root.display()))?;
    if m.is_dir() {
        w.dir(root, root);
    } else {
        // File yang disebut langsung: glob tidak berlaku.
        w.file(root, true);
    }
    Ok(w.out)
}

impl Walk<'_> {
    fn stop(&mut self) -> bool {
        if self.out.cut.is_none() {
            self.out.cut = if self.cancel.is_set() {
                Some("dibatalkan")
            } else if self.out.hits >= MAX_HITS || self.out.text.len() >= self.max {
                Some("hasil terlalu banyak; persempit pattern, path, atau glob")
            } else if self.seen >= MAX_FILES {
                Some("file terlalu banyak; persempit path")
            } else if self.start.elapsed() > MAX_TIME {
                Some("melewati batas waktu; persempit path")
            } else {
                None
            };
        }
        self.out.cut.is_some()
    }

    fn rel(&self, p: &Path) -> String {
        let s = p.strip_prefix(self.base).unwrap_or(p).display().to_string();
        if cfg!(windows) { s.replace('\\', "/") } else { s }
    }

    fn dir(&mut self, d: &Path, root: &Path) {
        let Ok(rd) = fs::read_dir(d) else { return };
        let mut v: Vec<_> = rd.flatten().collect();
        v.sort_by_key(|e| e.file_name());
        for e in v {
            if self.stop() {
                return;
            }
            self.seen += 1;
            let name = e.file_name();
            let name = name.to_string_lossy();
            // file_type() tidak mengikuti symlink, jadi symlink bukan dir maupun file.
            let Ok(t) = e.file_type() else { continue };
            if name.starts_with('.') || t.is_dir() && SKIP.contains(&name.as_ref()) {
                continue;
            }
            let p = e.path();
            if t.is_dir() {
                self.dir(&p, root);
            } else if t.is_file() {
                let r = p.strip_prefix(root).unwrap_or(&p).display().to_string().replace('\\', "/");
                if self.q.wants(&r) {
                    if self.out.names.len() < 50 && self.q.pattern.as_ref().is_some_and(|re| re.is_match(&r)) {
                        let n = self.rel(&p);
                        self.out.names.push(n);
                    }
                    self.file(&p, false);
                }
            }
        }
    }

    fn file(&mut self, p: &Path, named: bool) {
        let Some(re) = &self.q.pattern else {
            self.out.hits += 1;
            self.out.files += 1;
            self.out.text += &format!("{}\n", self.rel(p));
            return;
        };
        let Ok(m) = fs::metadata(p) else { return };
        if !m.is_file() || m.len() > MAX_SIZE {
            if named {
                self.out.cut = Some("file terlalu besar atau bukan file biasa; pakai bash");
            }
            return;
        }
        let Ok(b) = fs::read(p) else { return };
        if b[..b.len().min(8192)].contains(&0) {
            return;
        }
        let s = String::from_utf8_lossy(&b);
        let rel = self.rel(p);
        let mut any = false;
        for (i, l) in s.lines().enumerate() {
            if i % 256 == 255 && (self.cancel.is_set() || self.start.elapsed() > MAX_TIME) {
                break;
            }
            if !re.is_match(l) {
                continue;
            }
            any = true;
            self.out.hits += 1;
            let l = l.trim_end();
            let cut = l.floor_char_boundary(l.len().min(200));
            self.out.text += &format!("{rel}:{}: {}{}\n", i + 1, &l[..cut], if cut < l.len() { "..." } else { "" });
            if self.out.hits >= MAX_HITS || self.out.text.len() >= self.max {
                break;
            }
        }
        self.out.files += any as usize;
        self.stop();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn dir(n: &str) -> PathBuf {
        let d = Path::new(env!("CARGO_MANIFEST_DIR")).join("target/test").join(n);
        let _ = fs::remove_dir_all(&d);
        fs::create_dir_all(d.join("src/sub")).unwrap();
        fs::canonicalize(d).unwrap()
    }
    fn find(d: &Path, p: &str, g: &str) -> Vec<String> {
        let q = Query::new(p, g, false).unwrap();
        run(&q, d, d, 8000, &Cancel::default()).unwrap().text.lines().map(String::from).collect()
    }

    #[test]
    fn isi_dan_nama() {
        let d = dir("grep");
        fs::write(d.join("a.txt"), "satu\ndua tiga\r\n").unwrap();
        fs::write(d.join("src/b.rs"), "fn dua() {}\n").unwrap();
        fs::write(d.join("src/sub/c.rs"), "fn empat( {}\n").unwrap();
        fs::write(d.join(".env"), "dua=rahasia").unwrap();
        fs::write(d.join("biner"), b"dua\0dua").unwrap();
        fs::create_dir(d.join("target")).unwrap();
        fs::write(d.join("target/x.rs"), "dua").unwrap();
        assert_eq!(find(&d, "dua", ""), ["a.txt:2: dua tiga", "src/b.rs:1: fn dua() {}"]);
        assert_eq!(find(&d, r"satu\|empat", "").len(), 2);
        assert_eq!(find(&d, "satu|empat", "").len(), 2);
        // regex tak sah dicari apa adanya
        assert_eq!(find(&d, "empat(", ""), ["src/sub/c.rs:1: fn empat( {}"]);
        assert_eq!(find(&d, "", "*.rs"), ["src/b.rs", "src/sub/c.rs"]);
        assert_eq!(find(&d, "", "src/*.rs"), ["src/b.rs"]);
        assert_eq!(find(&d, "", "**/*.{rs,txt}").len(), 3);
        assert_eq!(find(&d, "dua", "*.txt"), ["a.txt:2: dua tiga"]);
        assert_eq!(find(&d, "*.rs", "").len(), 2);
        assert_eq!(find(&d, "\n", "\n *.rs"), ["src/b.rs", "src/sub/c.rs"]);
        let q = Query::new(r"\.rs$", "", false).unwrap();
        let r = run(&q, &d, &d, 8000, &Cancel::default()).unwrap();
        assert_eq!((r.hits, r.names.join(" ").as_str()), (0, "src/b.rs src/sub/c.rs"));
        assert!(Query::new("", "", false).is_err() && Query::new(" \n", "", false).is_err());
        let q = Query::new("DUA", "", true).unwrap();
        assert_eq!(run(&q, &d.join("a.txt"), &d, 8000, &Cancel::default()).unwrap().hits, 1);
    }

    #[cfg(unix)]
    #[test]
    fn symlink_tidak_diikuti() {
        let d = dir("greplink");
        fs::write(d.join("rahasia"), "kunci").unwrap();
        fs::create_dir(d.join("kerja")).unwrap();
        std::os::unix::fs::symlink(d.join("rahasia"), d.join("kerja/l")).unwrap();
        std::os::unix::fs::symlink(&d, d.join("kerja/lingkar")).unwrap();
        assert!(find(&d.join("kerja"), "kunci", "").is_empty());
        assert!(find(&d.join("kerja"), "", "*").is_empty());
    }

    #[test]
    fn dibatasi() {
        let d = dir("grepmax");
        fs::write(d.join("f"), "baris panjang sekali\n".repeat(5000)).unwrap();
        let q = Query::new("baris", "", false).unwrap();
        let r = run(&q, &d, &d, 2000, &Cancel::default()).unwrap();
        assert!(r.cut.is_some() && r.text.len() < 2300);
        assert_eq!(run(&q, &d, &d, 1 << 20, &Cancel::default()).unwrap().hits, MAX_HITS);
        let c = Cancel::default();
        c.cancel();
        assert_eq!(run(&q, &d, &d, 2000, &c).unwrap().cut, Some("dibatalkan"));
    }
}
