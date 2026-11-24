//! Instruksi proyek: AGENTS.md (atau CLAUDE.md) dari akar git sampai folder kerja,
//! plus ~/.config/llmy/AGENTS.md. Dibatasi agar context model kecil tidak habis.

use crate::{memory, sys};
use std::fs;
use std::path::{Path, PathBuf};

const MAX: usize = 8_000;

pub struct Rules {
    pub text: String,
    pub files: Vec<String>,
    pub truncated: bool,
}

pub fn global() -> Option<PathBuf> {
    sys::config_dir().map(|p| p.join("AGENTS.md"))
}

/// Akar proyek: folder git terdekat, atau folder kerja.
pub fn top(cwd: &Path) -> &Path {
    cwd.ancestors().find(|d| d.join(".git").exists()).unwrap_or(cwd)
}

pub fn load(cwd: &Path) -> Rules {
    load_from(cwd, global())
}

fn load_from(cwd: &Path, global: Option<PathBuf>) -> Rules {
    let top = top(cwd);
    let mut dirs: Vec<&Path> = cwd.ancestors().take_while(|d| d.starts_with(top)).collect();
    dirs.reverse();
    let mut found: Vec<(PathBuf, String)> = vec![];
    if let Some(g) = global
        && let Ok(s) = fs::read_to_string(&g)
    {
        found.push((g, s));
    }
    for d in dirs {
        for n in ["AGENTS.md", "CLAUDE.md"] {
            let p = d.join(n);
            // Symlink ke luar proyek diabaikan: file ini dibaca tanpa bertanya.
            let Ok(real) = fs::canonicalize(&p) else { continue };
            if !real.starts_with(top) || !real.is_file() {
                continue;
            }
            if let Ok(s) = fs::read_to_string(&real) {
                found.push((p, s));
                break;
            }
        }
    }
    let mut r = Rules { text: String::new(), files: vec![], truncated: false };
    // Yang terdekat dengan folder kerja paling penting: jatah dihitung dari belakang.
    let mut left = MAX;
    let mut parts = vec![];
    for (p, s) in found.iter().rev() {
        let s = memory::front(s.trim());
        if s.is_empty() {
            continue;
        }
        if left == 0 {
            r.truncated = true;
            break;
        }
        let cut = s.floor_char_boundary(left.min(s.len()));
        r.truncated |= cut < s.len();
        left -= cut;
        let name = p.strip_prefix(top).unwrap_or(p).display().to_string().replace('\\', "/");
        parts.push(format!("## {name}\n{}", &s[..cut]));
        r.files.push(name);
    }
    parts.reverse();
    r.files.reverse();
    r.text = parts.join("\n\n");
    r
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn urutan_batas_dan_symlink() {
        let d = Path::new(env!("CARGO_MANIFEST_DIR")).join("target/test/rules");
        let _ = fs::remove_dir_all(&d);
        fs::create_dir_all(d.join("p/.git")).unwrap();
        fs::create_dir_all(d.join("p/a/b")).unwrap();
        let d = fs::canonicalize(d).unwrap();
        fs::write(d.join("AGENTS.md"), "di luar akar").unwrap();
        fs::write(d.join("p/CLAUDE.md"), "akar").unwrap();
        fs::write(d.join("p/a/AGENTS.md"), "dekat").unwrap();
        fs::write(d.join("p/a/CLAUDE.md"), "kalah").unwrap();
        #[cfg(unix)]
        std::os::unix::fs::symlink(d.join("AGENTS.md"), d.join("p/a/b/AGENTS.md")).unwrap();
        let r = load_from(&d.join("p/a/b"), None);
        assert_eq!(r.files, ["CLAUDE.md", "a/AGENTS.md"]);
        assert!(r.text.find("akar").unwrap() < r.text.find("dekat").unwrap());
        assert!(!r.text.contains("kalah") && !r.text.contains("luar") && !r.truncated);
        fs::write(d.join("p/a/AGENTS.md"), "é".repeat(MAX)).unwrap();
        let r = load_from(&d.join("p/a"), None);
        assert!(r.truncated && r.files == ["a/AGENTS.md"]);
    }
}
