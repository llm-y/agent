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
    /// Ditambahkan setelah daftar file (pemeriksaan per file yang hanya menerima satu file).
    tail: &'static str,
    /// LaTeX: perintahnya dibangun dari file utama (`\documentclass`), bukan file yang diubah.
    tex: bool,
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
        if self.tex {
            return format!("`{} <main .tex file>` (draft, output to a temporary folder)", self.cmd);
        }
        if !self.tail.is_empty() {
            return "`node --check` on each changed file".into();
        }
        if self.per_file { format!("`{} <changed files>`", self.cmd) } else { format!("`{}`", self.cmd) }
    }

    /// Perintah untuk layar dan model: skrip pembungkus pemeriksaan per file diringkas.
    pub fn shown(&self, full: &str) -> String {
        match full.strip_prefix(self.cmd.as_str()).and_then(|r| r.strip_suffix(self.tail)) {
            Some(files) if !self.tail.is_empty() => format!("node --check{files}"),
            _ => full.to_string(),
        }
    }

    /// Perintah lengkap untuk file yang diubah, atau None bila tidak ada yang relevan.
    pub fn command(&self, changed: &[PathBuf]) -> Option<String> {
        let hit: Vec<&PathBuf> = changed.iter().filter(|p| self.applies(p) && p.starts_with(&self.dir)).collect();
        if hit.is_empty() {
            return None;
        }
        if self.tex {
            return self.tex_command(&hit);
        }
        if !self.per_file {
            return Some(self.cmd.clone());
        }
        let mut files: Vec<String> = hit.iter().filter(|p| p.is_file()).map(|p| quote(&rel(&self.dir, p))).collect();
        files.dedup();
        (!files.is_empty()).then(|| format!("{} {}{}", self.cmd, files.join(" "), self.tail))
    }

    /// Kompilasi draft file utama yang memuat file yang diubah. Keluaran (aux, log) ke folder
    /// sementara, jadi folder proyek tetap bersih.
    fn tex_command(&self, hit: &[&PathBuf]) -> Option<String> {
        let roots = tex_roots(&self.dir, hit);
        if roots.is_empty() {
            return None;
        }
        let out = std::env::temp_dir().join(format!("llmy-tex-{}", std::process::id()));
        let mut cmds = vec![];
        for root in roots.iter().take(3) {
            let dir = root.parent().unwrap_or(&self.dir);
            // \include menulis .aux di subfolder yang sama di folder keluaran.
            for sub in tex_dirs(dir) {
                let _ = std::fs::create_dir_all(out.join(sub));
            }
            let _ = std::fs::create_dir_all(&out);
            let src = std::fs::read_to_string(root).unwrap_or_default();
            let engine = if src.contains("!TEX program = lualatex") || src.contains("{luacode}") {
                "lualatex"
            } else if src.contains("!TEX program = xelatex") || src.contains("{fontspec}") || src.contains("{xeCJK}") || src.contains("{polyglossia}") {
                "xelatex"
            } else {
                "pdflatex"
            };
            let name = root.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
            cmds.push(format!(
                "cd {} && {engine} -interaction=nonstopmode -halt-on-error -file-line-error -draftmode -no-shell-escape -output-directory={} {}",
                quote(&dir.display().to_string()), quote(&out.display().to_string()), quote(&name)
            ));
        }
        Some(cmds.join(" && "))
    }
}

fn is_root(p: &Path) -> bool {
    std::fs::read_to_string(p).is_ok_and(|t| t.lines().any(|l| l.trim_start().starts_with("\\documentclass")))
}

/// File utama LaTeX: file yang diubah bila ia sendiri punya `\documentclass`; selain itu file
/// utama di proyek yang menyebut nama file itu (\input, \include, \subfile), atau semuanya.
fn tex_roots(dir: &Path, hit: &[&PathBuf]) -> Vec<PathBuf> {
    let mut roots: Vec<PathBuf> = hit.iter().filter(|p| p.extension().is_some_and(|e| e == "tex") && is_root(p)).map(|p| p.to_path_buf()).collect();
    let rest: Vec<&&PathBuf> = hit.iter().filter(|p| !roots.contains(p)).collect();
    if rest.is_empty() {
        return roots;
    }
    let all: Vec<PathBuf> = crate::bdi::project_files(dir).into_iter().filter(|f| f.ends_with(".tex")).map(|f| dir.join(f)).filter(|p| is_root(p)).take(20).collect();
    let mut mentioned: Vec<PathBuf> = all.iter().filter(|r| {
        let t = std::fs::read_to_string(r).unwrap_or_default();
        rest.iter().any(|p| p.file_stem().is_some_and(|s| t.contains(&*s.to_string_lossy())))
    }).cloned().collect();
    if mentioned.is_empty() {
        mentioned = all;
    }
    for m in mentioned {
        if !roots.contains(&m) {
            roots.push(m);
        }
    }
    roots
}

/// Subfolder yang berisi file .tex, relatif terhadap `dir`.
fn tex_dirs(dir: &Path) -> Vec<PathBuf> {
    let mut v: Vec<PathBuf> = crate::bdi::project_files(dir).into_iter().filter(|f| f.ends_with(".tex"))
        .filter_map(|f| Path::new(&f).parent().map(Path::to_path_buf)).filter(|p| !p.as_os_str().is_empty()).collect();
    v.sort();
    v.dedup();
    v
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

/// Node 22 meloloskan `node --check` pada .js ber-`export` yang sintaksnya rusak (deteksi modul),
/// jadi file juga diperiksa sebagai modul ES lewat stdin; nama `[stdin]` diganti nama filenya.
const JS_TAIL: &str = "; do out=$(node --input-type=module --check < \"$f\" 2>&1) || { printf '%s\\n' \"$out\" | sed \"s|^\\[stdin\\]|$f|\" | grep -v -e '^    at ' -e '^Node.js v'; exit 1; }; node --check \"$f\" || exit $?; done";

/// Pemeriksaan bawaan menurut berkas proyek. `custom`: dari `--verify`/`LLMY_VERIFY`
/// (`off` mematikan).
pub fn detect(cwd: &Path, custom: Option<&str>) -> Vec<Check> {
    let plain = |cmd: String, dir: PathBuf, exts: Vec<&'static str>, names: Vec<&'static str>| Check { cmd, dir, exts, names, per_file: false, safe: false, tail: "", tex: false };
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
    v.push(Check { cmd: format!("{py} -m py_compile"), dir: cwd.to_path_buf(), exts: vec!["py"], names: vec![], per_file: true, safe: true, tail: "", tex: false });
    // JavaScript (mis. croot.js): periksa sintaks tiap file, tanpa menjalankannya.
    v.push(Check {
        cmd: "command -v node >/dev/null || exit 127; for f in".into(), dir: cwd.to_path_buf(), exts: vec!["js", "mjs", "cjs"], names: vec![],
        per_file: true, safe: true, tail: JS_TAIL, tex: false,
    });
    // LaTeX: kompilasi draft tanpa shell escape; keluaran ke folder sementara.
    let root = ancestors(cwd).pop().unwrap_or_else(|| cwd.to_path_buf());
    v.push(Check { cmd: "pdflatex".into(), dir: root, exts: vec!["tex", "sty", "cls"], names: vec![], per_file: false, safe: true, tail: "", tex: true });
    v
}

/// Uji proyek di akhir giliran: lebih lambat dari pemeriksaan per langkah, jadi dijalankan saat
/// model menyatakan selesai.
#[derive(Clone, Debug, PartialEq)]
pub struct Gate {
    pub cmd: String,
    pub dir: PathBuf,
    exts: Vec<&'static str>,
    /// Tidak menjalankan kode proyek (LaTeX tanpa shell escape).
    pub safe: bool,
    tex: bool,
}

impl Gate {
    /// Perintah bila ada file relevan yang berubah.
    pub fn command(&self, changed: &[PathBuf]) -> Option<String> {
        let hit: Vec<&PathBuf> = changed.iter().filter(|p| p.starts_with(&self.dir) && p.extension().and_then(|e| e.to_str()).is_some_and(|e| self.exts.contains(&e))).collect();
        if hit.is_empty() {
            return None;
        }
        if !self.tex {
            return Some(self.cmd.clone());
        }
        // Dua kali kompilasi: rujukan \ref baru terselesaikan di kompilasi kedua.
        let out = std::env::temp_dir().join(format!("llmy-tex-{}", std::process::id()));
        let roots = tex_roots(&self.dir, &hit);
        let mut v = vec![];
        for root in roots.iter().take(3) {
            let dir = root.parent().unwrap_or(&self.dir);
            for sub in tex_dirs(dir) {
                let _ = std::fs::create_dir_all(out.join(sub));
            }
            let _ = std::fs::create_dir_all(&out);
            let name = quote(&root.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default());
            let o = quote(&out.display().to_string());
            let run = format!("pdflatex -interaction=nonstopmode -halt-on-error -file-line-error -draftmode -no-shell-escape -output-directory={o} {name}");
            let log = quote(&out.join("gate.txt").display().to_string());
            v.push(format!("cd {} && {run} > /dev/null 2>&1; {run} > {log} 2>&1 || {{ grep -A2 -E '^(!|.*\\.tex:[0-9]+:)' {log} | head -n 30; exit 1; }}; ! grep -E 'LaTeX Warning: Reference .* undefined' {log}",
                quote(&dir.display().to_string())));
        }
        (!v.is_empty()).then(|| v.join(" && "))
    }
}

/// Uji proyek: `cargo test`, `pytest`/`unittest`, `npm test`/`node --test`, dan LaTeX (rujukan).
pub fn gates(cwd: &Path, custom: Option<&str>) -> Vec<Gate> {
    if matches!(custom.map(str::trim), Some("off" | "none" | "")) {
        return vec![];
    }
    let mut v = vec![];
    let g = |cmd: &str, dir: PathBuf, exts: Vec<&'static str>| Gate { cmd: cmd.into(), dir, exts, safe: false, tex: false };
    if let Some(d) = find(cwd, "Cargo.toml") {
        v.push(g("cargo test --quiet 2>&1", d, vec!["rs", "toml"]));
    }
    let files = crate::bdi::project_files(cwd);
    let py_tests = files.iter().any(|f| {
        let n = f.rsplit('/').next().unwrap_or(f);
        n.ends_with(".py") && (n.starts_with("test_") || n.ends_with("_test.py"))
    });
    if py_tests {
        let pytest = files.iter().any(|f| f.ends_with("conftest.py") || f.ends_with("pytest.ini"))
            || std::fs::read_to_string(cwd.join("pyproject.toml")).is_ok_and(|t| t.contains("[tool.pytest"));
        let py = if cfg!(windows) { "python" } else { "python3" };
        let cmd = if pytest { format!("{py} -m pytest -q -x 2>&1") } else { format!("{py} -m unittest -q 2>&1") };
        v.push(g(&cmd, cwd.to_path_buf(), vec!["py"]));
    }
    let js = vec!["js", "mjs", "cjs", "ts", "tsx"];
    // Tanpa file uji, `npm test` (mis. jest) gagal dengan "No tests found": umpan balik yang menyesatkan.
    let js_tests = files.iter().any(|f| f.contains(".test.") || f.contains(".spec.") || f.contains("_test.") || f.starts_with("test/") || f.contains("/test/") || f.contains("__tests__/"));
    let pkg_test = find(cwd, "package.json").filter(|_| js_tests).and_then(|d| {
        let pkg: serde_json::Value = std::fs::read_to_string(d.join("package.json")).ok().and_then(|s| serde_json::from_str(&s).ok())?;
        let t = pkg["scripts"]["test"].as_str()?;
        (!t.contains("no test specified")).then_some(d)
    });
    if let Some(d) = pkg_test {
        v.push(g("npm test --silent 2>&1", d, js));
    } else if js_tests && files.iter().any(|f| f.ends_with(".js") || f.ends_with(".mjs")) {
        v.push(g("node --test 2>&1", cwd.to_path_buf(), js));
    }
    let root = ancestors(cwd).pop().unwrap_or_else(|| cwd.to_path_buf());
    v.push(Gate { cmd: "pdflatex".into(), dir: root, exts: vec!["tex", "sty", "cls", "bib"], safe: true, tex: true });
    v
}

/// Ringkasan kegagalan uji: baris yang menyebut kegagalan, plus ekor keluaran.
pub fn test_summary(out: &str) -> String {
    static RE: std::sync::OnceLock<regex_lite::Regex> = std::sync::OnceLock::new();
    let re = RE.get_or_init(|| regex_lite::Regex::new(r"(?i)(fail|panick|assert|error|expected|left|right|traceback|file .*line \d|\.(rs|py|js|tex):\d+|undefined|not ok)").unwrap());
    let lines: Vec<&str> = out.lines().filter(|l| !l.trim().is_empty()).collect();
    let tail = lines.len().saturating_sub(12);
    let mut v: Vec<&str> = vec![];
    for (i, l) in lines.iter().enumerate() {
        if (re.is_match(l) || i >= tail) && !v.contains(l) {
            v.push(l);
        }
    }
    let total = v.len();
    v.truncate(MAX_LINES);
    let mut s = v.join("\n");
    if s.len() > MAX_BYTES {
        let mut i = MAX_BYTES;
        while !s.is_char_boundary(i) {
            i -= 1;
        }
        s.truncate(i);
    }
    if total > MAX_LINES {
        s += &format!("\n... {} baris lagi dipotong", total - MAX_LINES);
    }
    s
}

/// Baris galat: peringatan dibuang bila ada galat, karena pada build yang gagal hanya mengaburkan.
fn error_lines(out: &str) -> Vec<&str> {
    // LaTeX mencetak ratusan baris log; galatnya `file.tex:12: pesan` (atau `! pesan`) diikuti `l.12 ...`.
    let lines: Vec<&str> = out.lines().collect();
    let tex = |l: &str| l.starts_with("! ") || l.split_once(".tex:").is_some_and(|(_, r)| r.split(':').next().is_some_and(|n| !n.is_empty() && n.bytes().all(|b| b.is_ascii_digit())));
    if lines.iter().any(|l| tex(l)) {
        let mut v = vec![];
        for (i, l) in lines.iter().enumerate() {
            if tex(l) {
                v.push(*l);
                v.extend(lines[i + 1..].iter().take(3).filter(|x| x.starts_with("l.")).copied());
            }
        }
        return v;
    }
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

/// Lokasi galat (file, baris) dari keluaran pemeriksaan: `src/a.rs:12:5`, `./main.tex:12:`,
/// `lib/app.js:2`, atau `File "a.py", line 3`. Hanya file di dalam `cwd`, paling banyak `max`.
pub fn locations(out: &str, dir: &Path, cwd: &Path, max: usize) -> Vec<(PathBuf, usize)> {
    static RE: std::sync::OnceLock<(regex_lite::Regex, regex_lite::Regex)> = std::sync::OnceLock::new();
    let (generic, py) = RE.get_or_init(|| {
        (
            regex_lite::Regex::new(r"([A-Za-z0-9_./\\-]+\.(?:rs|py|js|mjs|cjs|ts|tsx|go|tex|sty|cls)):(\d+)").unwrap(),
            regex_lite::Regex::new(r#"File "([^"]+)", line (\d+)"#).unwrap(),
        )
    });
    let mut v: Vec<(PathBuf, usize)> = vec![];
    for l in out.lines() {
        // Baris perintah (`$ ...`) bukan keluaran.
        if l.starts_with("$ ") {
            continue;
        }
        for c in generic.captures_iter(l).chain(py.captures_iter(l)) {
            let (Some(f), Some(n)) = (c.get(1), c.get(2).and_then(|n| n.as_str().parse::<usize>().ok())) else { continue };
            let p = dir.join(f.as_str().trim_start_matches("./"));
            let Ok(p) = p.canonicalize() else { continue };
            if !p.starts_with(cwd) || !p.is_file() || n == 0 || v.iter().any(|(q, m)| *q == p && m.abs_diff(n) < 6) {
                continue;
            }
            v.push((p, n));
            if v.len() >= max {
                return v;
            }
        }
    }
    v
}

/// Baris di sekitar lokasi galat, bernomor seperti `read`.
pub fn slice(path: &Path, line: usize, cwd: &Path) -> Option<String> {
    let t = std::fs::read_to_string(path).ok()?;
    let lines: Vec<&str> = t.lines().collect();
    if line > lines.len() {
        return None;
    }
    let (a, b) = (line.saturating_sub(6), (line + 5).min(lines.len()));
    let rel = path.strip_prefix(cwd).unwrap_or(path).display().to_string().replace('\\', "/");
    let mut s = format!("[code around {rel}:{line}]\n");
    for (i, l) in lines[a..b].iter().enumerate() {
        let l: String = l.chars().take(300).collect();
        s += &format!("{}│{l}\n", a + i + 1);
    }
    Some(s)
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
        assert_eq!(v.len(), 4, "{v:?}");
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
    fn javascript_dan_latex() {
        let d = dir("verify-jstex");
        let sub = d.join("sub");
        let v = detect(&sub, None);
        let js = v.iter().find(|c| c.exts.contains(&"js")).unwrap();
        let f = sub.join("app.js");
        fs::write(&f, "").unwrap();
        assert_eq!(js.command(&[f]).unwrap(), format!("command -v node >/dev/null || exit 127; for f in 'app.js'{JS_TAIL}"));
        assert!(js.safe);
        assert_eq!(js.shown(&js.command(&[sub.join("app.js")]).unwrap()), "node --check 'app.js'");
        // Bab di subfolder: yang dikompilasi file utama yang menyebutnya, dari foldernya sendiri.
        let tex = v.iter().find(|c| c.tex).unwrap();
        fs::create_dir_all(sub.join("bab")).unwrap();
        fs::write(sub.join("main.tex"), "% utama\n\\documentclass{article}\n\\begin{document}\\input{bab/satu}\\end{document}\n").unwrap();
        fs::write(sub.join("lain.tex"), "\\documentclass{article}\n").unwrap();
        fs::write(sub.join("bab/satu.tex"), "Halo.\n").unwrap();
        let c = tex.command(&[sub.join("bab/satu.tex")]).unwrap();
        assert!(c.contains("pdflatex -interaction=nonstopmode -halt-on-error -file-line-error -draftmode -no-shell-escape") && c.ends_with("'main.tex'") && !c.contains("lain.tex"), "{c}");
        assert!(std::env::temp_dir().join(format!("llmy-tex-{}/bab", std::process::id())).is_dir());
        fs::write(sub.join("x.tex"), "\\documentclass{article}\\usepackage{fontspec}\n").unwrap();
        assert!(tex.command(&[sub.join("x.tex")]).unwrap().contains("xelatex"));
        assert!(tex.command(&[sub.join("README.md")]).is_none());
        // Galat LaTeX: baris galat dan baris l.N saja, bukan log ratusan baris.
        let log = "This is pdfTeX\n(./main.tex\nLaTeX2e <2024>\n./main.tex:12: Undefined control sequence.\nl.12 \\foo\n           \nNo pages of output.\n[exit 1]";
        assert_eq!(errors_first(log), "./main.tex:12: Undefined control sequence.\nl.12 \\foo");
        assert_eq!(weight(log), 2);
    }

    #[test]
    fn lokasi_galat_dan_potongan() {
        let d = dir("verify-lokasi").canonicalize().unwrap();
        fs::create_dir_all(d.join("src")).unwrap();
        fs::write(d.join("src/a.rs"), (1..=30).map(|i| format!("baris {i}\n")).collect::<String>()).unwrap();
        fs::write(d.join("b.py"), "x = (\n").unwrap();
        let out = "$ cargo check src/a.rs:99\nsrc/a.rs:12:5: error[E0425]: x\nsrc/a.rs:14:1: error: y\n  File \"b.py\", line 1\n/etc/passwd.rs:3: x\nsrc/a.rs:25:1: error: z";
        let v = locations(out, &d, &d, 3);
        assert_eq!(v, vec![(d.join("src/a.rs"), 12), (d.join("b.py"), 1), (d.join("src/a.rs"), 25)]);
        let s = slice(&d.join("src/a.rs"), 12, &d).unwrap();
        assert!(s.starts_with("[code around src/a.rs:12]\n7│baris 7\n") && s.ends_with("17│baris 17\n"), "{s}");
        assert!(slice(&d.join("src/a.rs"), 31, &d).is_none());
    }

    #[test]
    fn uji_akhir_giliran() {
        let d = dir("verify-gerbang");
        let sub = d.join("sub");
        fs::write(sub.join("test_a.py"), "").unwrap();
        fs::write(sub.join("a.py"), "").unwrap();
        fs::create_dir_all(sub.join("test")).unwrap();
        fs::write(sub.join("test/x.test.js"), "").unwrap();
        fs::write(sub.join("app.js"), "").unwrap();
        let g = gates(&sub, None);
        let cmds: Vec<&str> = g.iter().map(|x| x.cmd.as_str()).collect();
        assert!(cmds.contains(&"python3 -m unittest -q 2>&1") || cfg!(windows), "{cmds:?}");
        assert!(cmds.contains(&"node --test 2>&1"), "{cmds:?}");
        let py = g.iter().find(|x| x.cmd.contains("unittest")).unwrap();
        assert!(py.command(&[sub.join("a.py")]).is_some() && py.command(&[sub.join("x.md")]).is_none() && !py.safe);
        fs::write(sub.join("conftest.py"), "").unwrap();
        assert!(gates(&sub, None).iter().any(|x| x.cmd.contains("pytest")));
        fs::write(sub.join("package.json"), r#"{"scripts":{"test":"vitest run"}}"#).unwrap();
        assert!(gates(&sub, None).iter().any(|x| x.cmd == "npm test --silent 2>&1"));
        // Tanpa file uji JS: tidak ada gerbang JS.
        fs::remove_dir_all(sub.join("test")).unwrap();
        assert!(!gates(&sub, None).iter().any(|x| x.cmd.contains("npm") || x.cmd.contains("node --test")));
        assert!(gates(&sub, Some("off")).is_empty());
        let tex = g.iter().find(|x| x.safe).unwrap();
        fs::write(sub.join("main.tex"), "\\documentclass{article}\n").unwrap();
        let c = tex.command(&[sub.join("main.tex")]).unwrap();
        assert!(c.contains("Reference .* undefined") && c.contains("'main.tex'"), "{c}");
        let s = test_summary("running 2 tests\ntest a ... ok\ntest b ... FAILED\nthread 'b' panicked at src/lib.rs:3:5:\nassertion `left == right` failed\n  left: 1\n right: 2\n");
        assert!(s.contains("panicked at src/lib.rs:3:5") && s.contains("right: 2"), "{s}");
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
