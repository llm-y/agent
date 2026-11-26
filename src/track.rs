//! Melacak file yang diubah perintah `bash` (`sed -i`, `mv`, generator): sebelum dan sesudah
//! perintah, folder kerja dipindai (ukuran dan waktu ubah). Isi file kecil disimpan, sehingga
//! perubahan lewat `bash` bisa diverifikasi dan dikembalikan seperti `edit`/`write`.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::SystemTime;

/// Folder yang tidak dipindai: hasil build, dependensi, dan riwayat git.
const SKIP: [&str; 10] = [".git", "target", "node_modules", "dist", "build", "__pycache__", ".venv", "venv", "vendor", ".next"];
const MAX_FILES: usize = 20_000;
/// File lebih besar tidak disimpan isinya: perubahan tetap terlihat, tetapi tak bisa dikembalikan.
const MAX_FILE: u64 = 1 << 20;
const MAX_TOTAL: usize = 64 << 20;

#[derive(Clone)]
struct Entry {
    mtime: Option<SystemTime>,
    len: u64,
    text: Option<Arc<String>>,
}

#[derive(Default, Clone)]
pub struct Tree {
    files: HashMap<PathBuf, Entry>,
    /// Pemindaian berhenti di MAX_FILES.
    pub partial: bool,
}

/// Perubahan satu file: isi sebelum (None = belum ada atau tidak tersimpan) dan sesudah.
pub struct Change {
    pub path: PathBuf,
    pub before: Option<String>,
    /// Isi sebelum diketahui (file baru, atau isinya tersimpan).
    pub known: bool,
    pub after: Option<String>,
}

/// Pindai `root`. Isi file yang tidak berubah sejak `prev` dipakai ulang.
pub fn scan(root: &Path, prev: &Tree) -> Tree {
    let mut t = Tree::default();
    let mut total = 0;
    walk(root, prev, &mut t, &mut total, 0);
    t
}

fn walk(dir: &Path, prev: &Tree, t: &mut Tree, total: &mut usize, depth: usize) {
    let Ok(rd) = std::fs::read_dir(dir) else { return };
    for e in rd.flatten() {
        if t.files.len() >= MAX_FILES {
            t.partial = true;
            return;
        }
        let name = e.file_name();
        let Ok(ft) = e.file_type() else { continue };
        let p = e.path();
        if ft.is_dir() {
            if depth < 20 && !SKIP.iter().any(|s| name == *s) {
                walk(&p, prev, t, total, depth + 1);
            }
            continue;
        }
        if !ft.is_file() {
            continue;
        }
        let Ok(m) = e.metadata() else { continue };
        let (mtime, len) = (m.modified().ok(), m.len());
        let text = match prev.files.get(&p) {
            Some(old) if old.mtime == mtime && old.len == len => old.text.clone(),
            _ if len <= MAX_FILE && *total + len as usize <= MAX_TOTAL => std::fs::read_to_string(&p).ok().map(Arc::new),
            _ => None,
        };
        if let Some(s) = &text {
            *total += s.len();
        }
        t.files.insert(p, Entry { mtime, len, text });
    }
}

/// File yang dibuat, diubah, atau dihapus antara dua pemindaian, urut menurut path.
pub fn diff(a: &Tree, b: &Tree) -> Vec<Change> {
    let mut out = vec![];
    for (p, eb) in &b.files {
        match a.files.get(p) {
            Some(ea) if ea.mtime == eb.mtime && ea.len == eb.len => {}
            Some(ea) => {
                let after = eb.text.as_deref().cloned();
                // Waktu berubah tetapi isinya sama (mis. `touch`): bukan perubahan.
                if ea.text.is_some() && ea.text.as_deref() == eb.text.as_deref() {
                    continue;
                }
                out.push(Change { path: p.clone(), before: ea.text.as_deref().cloned(), known: ea.text.is_some(), after });
            }
            None => out.push(Change { path: p.clone(), before: None, known: true, after: eb.text.as_deref().cloned() }),
        }
    }
    for (p, ea) in &a.files {
        if !b.files.contains_key(p) {
            out.push(Change { path: p.clone(), before: ea.text.as_deref().cloned(), known: ea.text.is_some(), after: None });
        }
    }
    out.sort_by(|x, y| x.path.cmp(&y.path));
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn perubahan_lewat_perintah_terlihat() {
        let d = Path::new(env!("CARGO_MANIFEST_DIR")).join("target/test/track");
        let _ = fs::remove_dir_all(&d);
        fs::create_dir_all(d.join("src")).unwrap();
        fs::create_dir_all(d.join("node_modules")).unwrap();
        fs::write(d.join("src/a.rs"), "lama\n").unwrap();
        fs::write(d.join("b.txt"), "hapus\n").unwrap();
        fs::write(d.join("c.txt"), "tetap\n").unwrap();
        let a = scan(&d, &Tree::default());
        std::thread::sleep(std::time::Duration::from_millis(20));
        fs::write(d.join("src/a.rs"), "baru, lebih panjang\n").unwrap();
        fs::remove_file(d.join("b.txt")).unwrap();
        fs::write(d.join("d.txt"), "baru\n").unwrap();
        fs::write(d.join("node_modules/x.js"), "abaikan").unwrap();
        let b = scan(&d, &a);
        let v = diff(&a, &b);
        let names: Vec<String> = v.iter().map(|c| c.path.strip_prefix(&d).unwrap().to_string_lossy().into_owned()).collect();
        assert_eq!(names, vec!["b.txt", "d.txt", "src/a.rs"]);
        assert_eq!(v[0].before.as_deref(), Some("hapus\n"));
        assert!(v[0].after.is_none());
        assert!(v[1].before.is_none() && v[1].known);
        assert_eq!((v[2].before.as_deref(), v[2].after.as_deref()), (Some("lama\n"), Some("baru, lebih panjang\n")));
    }
}
