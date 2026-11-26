//! Verifikasi otomatis setelah edit, dan ingatan isi file yang sudah gagal.
//!
//! Model kecil sering menganggap edit selesai tanpa membangun ulang, atau mengulang patch
//! yang sama setelah gagal. Keduanya dicegah di kode: setelah langkah yang mengubah file,
//! llmy sendiri menjalankan pemeriksaan proyek; isi file yang terbukti gagal diingat dan
//! ditolak bila ditulis lagi, sampai ada verifikasi yang lulus.

use std::collections::HashSet;
use std::hash::{DefaultHasher, Hash, Hasher};
use std::path::{Path, PathBuf};

/// Berhenti setelah sekian verifikasi gagal berturut-turut tanpa kemajuan.
pub const MAX_STREAK: usize = 4;
const MAX_LINES: usize = 40;
const MAX_BYTES: usize = 4_000;

#[derive(Clone, Debug, PartialEq)]
pub struct Check {
    /// Perintah; untuk pemeriksaan per file, daftar file ditambahkan di belakangnya.
    pub cmd: String,
    /// Folder tempat perintah dijalankan.
    pub dir: PathBuf,
    /// Ekstensi yang memicu pemeriksaan ini; kosong = semua file.
    exts: Vec<&'static str>,
    /// Nama file yang juga memicu (mis. Cargo.toml).
    names: Vec<&'static str>,
    per_file: bool,
    /// Tidak menjalankan kode proyek (mis. py_compile), jadi boleh tanpa tanya.
    pub safe: bool,
}

impl Check {
    fn applies(&self, p: &Path) -> bool {
        if self.exts.is_empty() && self.names.is_empty() {
            return true;
        }
        let ext = p.extension().and_then(|e| e.to_str()).unwrap_or("");
        let name = p.file_name().and_then(|e| e.to_str()).unwrap_or("");
        self.exts.contains(&ext) || self.names.contains(&name)
    }

    /// Untuk pesan sistem.
    pub fn describe(&self) -> String {
        if self.per_file { format!("`{} <changed files>`", self.cmd) } else { format!("`{}`", self.cmd) }
    }

    /// Perintah lengkap untuk file yang diubah, atau None bila tidak ada yang relevan.
    pub fn command(&self, changed: &[PathBuf]) -> Option<String> {
        let hit: Vec<&PathBuf> = changed.iter().filter(|p| self.applies(p) && p.starts_with(&self.dir)).collect();
        if hit.is_empty() {
            return None;
        }
        if !self.per_file {
            return Some(self.cmd.clone());
        }
        let mut files: Vec<String> = hit.iter().filter(|p| p.is_file()).map(|p| quote(&rel(&self.dir, p))).collect();
        files.dedup();
        (!files.is_empty()).then(|| format!("{} {}", self.cmd, files.join(" ")))
    }
}

fn rel(dir: &Path, p: &Path) -> String {
    let r = p.strip_prefix(dir).unwrap_or(p).display().to_string();
    if cfg!(windows) { r.replace('\\', "/") } else { r }
}

/// Kutip untuk bash (juga Git Bash di Windows).
fn quote(s: &str) -> String {
    format!("'{}'", s.replace('\'', r"'\''"))
}

/// Folder dari `cwd` naik sampai akar git (atau hanya `cwd` bila bukan repo git).
fn ancestors(cwd: &Path) -> Vec<PathBuf> {
    let mut v = vec![];
    for d in cwd.ancestors() {
        v.push(d.to_path_buf());
        if d.join(".git").exists() {
            return v;
        }
    }
    v.truncate(1);
    v
}

fn find(cwd: &Path, name: &str) -> Option<PathBuf> {
    ancestors(cwd).into_iter().find(|d| d.join(name).is_file())
}

/// Pemeriksaan bawaan menurut berkas proyek. `custom`: dari `--verify`/`LLMY_VERIFY`
/// (`off` mematikan).
pub fn detect(cwd: &Path, custom: Option<&str>) -> Vec<Check> {
    let plain = |cmd: String, dir: PathBuf, exts: Vec<&'static str>, names: Vec<&'static str>| Check { cmd, dir, exts, names, per_file: false, safe: false };
    match custom.map(str::trim) {
        Some("off" | "none" | "") => return vec![],
        Some(c) => return vec![plain(c.to_string(), cwd.to_path_buf(), vec![], vec![])],
        None => {}
    }
    let mut v = vec![];
    if let Some(d) = find(cwd, "Cargo.toml") {
        v.push(plain("cargo check --quiet --all-targets --message-format short".into(), d, vec!["rs"], vec!["Cargo.toml"]));
    }
    if let Some(d) = find(cwd, "go.mod") {
        v.push(plain("go build ./... && go vet ./...".into(), d, vec!["go"], vec!["go.mod"]));
    }
    if let Some(d) = find(cwd, "package.json") {
        let pkg: serde_json::Value = std::fs::read_to_string(d.join("package.json")).ok().and_then(|s| serde_json::from_str(&s).ok()).unwrap_or_default();
        let ts = vec!["ts", "tsx", "mts", "cts"];
        if pkg["scripts"]["typecheck"].is_string() {
            v.push(plain("npm run --silent typecheck".into(), d, ts, vec!["tsconfig.json"]));
        } else if d.join("tsconfig.json").is_file() {
            v.push(plain("npx --no-install tsc --noEmit".into(), d, ts, vec!["tsconfig.json"]));
        }
    }
    // Python tanpa berkas proyek: cukup periksa sintaks file yang diubah, tanpa menjalankannya.
    let py = if cfg!(windows) { "python" } else { "python3" };
    v.push(Check { cmd: format!("{py} -m py_compile"), dir: cwd.to_path_buf(), exts: vec!["py"], names: vec![], per_file: true, safe: true });
    v
}

/// Baris galat: peringatan dibuang bila ada galat, karena pada build yang gagal hanya mengaburkan.
fn error_lines(out: &str) -> Vec<&str> {
    let has_error = out.lines().any(|l| l.to_ascii_lowercase().contains("error"));
    out.lines()
        .filter(|l| !(has_error && l.to_ascii_lowercase().contains("warning")))
        .filter(|l| !l.trim().is_empty())
        .collect()
}

/// Ukuran kasar banyaknya galat, untuk menilai kemajuan antar-verifikasi.
pub fn weight(out: &str) -> usize {
    error_lines(out).len()
}

/// Pesan galat dulu, dipotong.
pub fn errors_first(out: &str) -> String {
    let mut lines = error_lines(out);
    let total = lines.len();
    lines.truncate(MAX_LINES);
    let mut s = lines.join("\n");
    if s.len() > MAX_BYTES {
        let mut i = MAX_BYTES;
        while !s.is_char_boundary(i) {
            i -= 1;
        }
        s.truncate(i);
    }
    if total > lines.len() {
        s += &format!("\n... {} baris lagi dipotong", total - lines.len());
    }
    s
}

pub fn state_hash(path: &Path, content: &str) -> u64 {
    let mut h = DefaultHasher::new();
    path.hash(&mut h);
    content.hash(&mut h);
    h.finish()
}

/// Isi file yang verifikasinya gagal, dan isi yang ditulis di langkah ini.
#[derive(Default)]
pub struct Memory {
    failed: HashSet<u64>,
    step: Vec<(PathBuf, u64)>,
    pub streak: usize,
    /// Banyaknya galat pada kegagalan terakhir.
    last: Option<usize>,
    /// Galat terakhir, untuk belief mode BDI.
    pub errors: String,
}

impl Memory {
    pub fn tried(&self, path: &Path, content: &str) -> bool {
        self.failed.contains(&state_hash(path, content))
    }

    /// Jumlah tulisan di langkah ini yang belum diverifikasi.
    pub fn pending(&self) -> usize {
        self.step.len()
    }

    pub fn wrote(&mut self, path: &Path, content: &str) {
        self.step.push((path.to_path_buf(), state_hash(path, content)));
    }

    /// File yang diubah di langkah ini (tanpa duplikat), lalu catatan langkah dikosongkan
    /// oleh `passed`/`failed`.
    pub fn changed(&self) -> Vec<PathBuf> {
        let mut v: Vec<PathBuf> = vec![];
        for (p, _) in &self.step {
            if !v.contains(p) {
                v.push(p.clone());
            }
        }
        v
    }

    pub fn passed(&mut self) {
        self.failed.clear();
        self.step.clear();
        self.streak = 0;
        self.last = None;
        self.errors.clear();
    }

    /// Tidak ada verifikasi yang gagal sejak terakhir lulus.
    pub fn clean(&self) -> bool {
        self.last.is_none()
    }

    /// File dikembalikan ke keadaan sebelum intention: status build ikut kembali, tetapi isi
    /// yang sudah gagal tetap ditolak.
    pub fn rollback(&mut self) {
        self.step.clear();
        self.streak = 0;
        self.last = None;
        self.errors.clear();
    }

    /// `weight`: banyaknya galat. Refactor lintas file wajar gagal di tengah jalan; selama
    /// galatnya berkurang, itu kemajuan dan hitungan mulai dari satu lagi.
    pub fn failed(&mut self, weight: usize) {
        self.failed.extend(self.step.drain(..).map(|(_, h)| h));
        self.streak = match self.last {
            Some(prev) if weight < prev => 1,
            _ => self.streak + 1,
        };
        self.last = Some(weight);
    }

    /// Langkah tanpa verifikasi (tidak ada pemeriksaan yang cocok, atau dilewati).
    pub fn skip(&mut self) {
        self.step.clear();
    }

    pub fn reset(&mut self) {
        *self = Memory::default();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn dir(n: &str) -> PathBuf {
        let d = Path::new(env!("CARGO_MANIFEST_DIR")).join("target/test").join(n);
        let _ = fs::remove_dir_all(&d);
        fs::create_dir_all(d.join("sub/.git")).unwrap();
        d
    }

    #[test]
    fn deteksi_menurut_berkas_proyek() {
        let d = dir("verify-deteksi");
        let sub = d.join("sub");
        fs::write(sub.join("Cargo.toml"), "").unwrap();
        fs::create_dir(sub.join("src")).unwrap();
        // Berkas proyek di atas akar git tidak dipakai.
        fs::write(d.join("go.mod"), "").unwrap();
        let v = detect(&sub.join("src"), None);
        assert_eq!(v.len(), 2, "{v:?}");
        assert!(v[0].cmd.starts_with("cargo check") && v[0].dir == sub && !v[0].safe && v[1].safe);
        let rs = sub.join("src/main.rs");
        fs::write(&rs, "").unwrap();
        assert_eq!(v[0].command(std::slice::from_ref(&rs)).as_deref(), Some(v[0].cmd.as_str()));
        assert!(v[0].command(&[sub.join("README.md")]).is_none());
        assert!(v[0].command(&[sub.join("Cargo.toml")]).is_some());
        assert!(v[1].command(std::slice::from_ref(&rs)).is_none());
        let py = sub.join("src/a b'.py");
        fs::write(&py, "").unwrap();
        assert!(v[1].command(&[py, rs]).unwrap().ends_with(r"py_compile 'a b'\''.py'"), "{:?}", v[1]);
        assert!(detect(&sub, Some("off")).is_empty());
        let c = detect(&sub, Some("make test"));
        assert_eq!(c.len(), 1);
        assert_eq!(c[0].command(&[sub.join("x.md")]).as_deref(), Some("make test"));
    }

    #[test]
    fn galat_didahulukan_dan_dipotong() {
        assert_eq!(weight("a: warning: x\nb: error: y\nc: error: z\n\n"), 2);
        let out = "src/a.rs:1:1: warning: unused\nsrc/b.rs:2:3: error[E0425]: cannot find value `x`\n\n[exit 101]";
        assert_eq!(errors_first(out), "src/b.rs:2:3: error[E0425]: cannot find value `x`\n[exit 101]");
        // Go tidak menulis "error": tidak ada yang dibuang.
        assert_eq!(errors_first("./m.go:3:2: undefined: foo\n[exit 1]"), "./m.go:3:2: undefined: foo\n[exit 1]");
        let long: String = (0..100).map(|i| format!("error {i}\n")).collect();
        let e = errors_first(&long);
        assert!(e.lines().count() == MAX_LINES + 1 && e.ends_with("60 baris lagi dipotong"), "{e}");
    }

    #[test]
    fn isi_gagal_diingat_sampai_lulus() {
        let (a, b) = (Path::new("/p/a.rs"), Path::new("/p/b.rs"));
        let mut m = Memory::default();
        m.wrote(a, "1");
        m.wrote(b, "2");
        m.wrote(a, "3");
        assert_eq!(m.changed(), vec![a.to_path_buf(), b.to_path_buf()]);
        m.failed(5);
        assert!(m.tried(a, "1") && m.tried(b, "2") && m.tried(a, "3") && !m.tried(b, "1"));
        assert_eq!((m.streak, m.changed().len()), (1, 0));
        m.wrote(a, "4");
        m.failed(5);
        assert_eq!(m.streak, 2);
        m.failed(6);
        assert_eq!(m.streak, 3);
        // Galat berkurang: kemajuan.
        m.failed(2);
        assert_eq!(m.streak, 1);
        m.failed(2);
        assert_eq!(m.streak, 2);
        m.passed();
        assert!(!m.tried(a, "1") && !m.tried(a, "4") && m.streak == 0);
        m.failed(9);
        assert_eq!(m.streak, 1);
    }
}
