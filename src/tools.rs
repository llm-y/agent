//! Empat tool: read, write, edit, bash.
//! `prepare` memvalidasi tanpa efek samping; `run` mengeksekusi setelah izin.

use crate::cancel::{Cancel, kill_group};
use serde_json::{Value, json};
use std::fs;
use std::io::Read;
use std::os::unix::process::CommandExt;
use std::path::{Component, Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

const READ_MAX: usize = 16_000;
const BASH_MAX: usize = 8_000;
const FILE_MAX: u64 = 10 << 20;

pub fn schema() -> Value {
    fn f(name: &str, desc: &str, props: Value, req: &[&str]) -> Value {
        json!({"type":"function","function":{"name":name,"description":desc,
            "parameters":{"type":"object","properties":props,"required":req}}})
    }
    let s = |d: &str| json!({"type":"string","description":d});
    let i = |d: &str| json!({"type":"integer","description":d});
    json!([
        f("read", "Read a text file (with line numbers) or list a directory.",
          json!({"path": s("File or directory path, relative to the working directory"), "offset": i("First line, 1-based"), "limit": i("Number of lines")}), &["path"]),
        f("edit", "Replace exact text in a file. `old` must match exactly, including indentation, and occur once unless replace_all is true. Do not include line numbers.",
          json!({"path": s("File path, relative to the working directory"), "old": s("Exact existing text"), "new": s("Replacement text"), "replace_all": {"type":"boolean"}}), &["path", "old", "new"]),
        f("write", "Create a new file or overwrite a whole file. Prefer edit for existing files.",
          json!({"path": s("File path, relative to the working directory"), "content": s("Full file content")}), &["path", "content"]),
        f("bash", "Run a bash command in the working directory. Use it for search (grep, find), build, test, git.",
          json!({"command": s("Command"), "timeout": i("Seconds, default 120, max 600")}), &["command"]),
    ])
}

pub enum Action {
    Read { path: PathBuf, offset: usize, limit: usize },
    Write { path: PathBuf, old: Option<String>, new: String, is_edit: bool },
    Bash { cmd: String, timeout: u64 },
}

pub struct Prepared {
    pub action: Action,
    pub title: String,
    /// Path di luar folder kerja atau di dalam .git: selalu ditanya.
    pub guarded: bool,
}

fn str_arg(a: &Value, k: &str) -> Result<String, String> {
    match &a[k] {
        Value::String(s) => Ok(s.clone()),
        Value::Null => Err(format!("argumen `{k}` wajib diisi")),
        v => Err(format!("argumen `{k}` harus string, bukan {v}")),
    }
}

fn int_arg(a: &Value, k: &str) -> Option<u64> {
    let v = &a[k];
    v.as_u64().or_else(|| v.as_f64().map(|f| f as u64)).or_else(|| v.as_str()?.trim().parse().ok())
}

/// Path nyata: symlink diikuti sejauh komponennya ada, sisanya dirapikan.
pub fn resolve(cwd: &Path, p: &str) -> Result<PathBuf, String> {
    if p.trim().is_empty() {
        return Err("path kosong".into());
    }
    // Model sering ikut menyalin `@` dari `@file`; pakai apa adanya hanya bila memang ada.
    let p = match p.strip_prefix('@') {
        Some(r) if !r.is_empty() && fs::symlink_metadata(cwd.join(p)).is_err() => r,
        _ => p,
    };
    let home = std::env::var("HOME").unwrap_or_default();
    let p = match p.strip_prefix("~/") {
        Some(r) if !home.is_empty() => Path::new(&home).join(r),
        _ => cwd.join(p),
    };
    let mut out = PathBuf::from("/");
    let mut missing = false;
    for c in p.components() {
        match c {
            Component::ParentDir => {
                out.pop();
            }
            Component::Normal(n) => {
                out.push(n);
                if !missing {
                    match fs::canonicalize(&out) {
                        Ok(r) => out = r,
                        Err(_) if fs::symlink_metadata(&out).is_ok() => {
                            return Err(format!("{}: symlink rusak atau tidak bisa diakses", out.display()));
                        }
                        Err(_) => missing = true,
                    }
                } else if fs::symlink_metadata(&out).is_ok() {
                    // `tidak-ada/../link` kembali ke jalur nyata
                    out = fs::canonicalize(&out).map_err(|e| e.to_string())?;
                }
            }
            _ => {}
        }
    }
    Ok(out)
}

fn guarded(cwd: &Path, p: &Path) -> bool {
    match p.strip_prefix(cwd) {
        Ok(r) => r.components().any(|c| c.as_os_str() == ".git"),
        Err(_) => true,
    }
}

fn show(cwd: &Path, p: &Path) -> String {
    match p.strip_prefix(cwd) {
        Ok(r) if r.as_os_str().is_empty() => ".".into(),
        Ok(r) => r.display().to_string(),
        Err(_) => p.display().to_string(),
    }
}

fn load(p: &Path) -> Result<String, String> {
    let m = fs::metadata(p).map_err(|e| format!("{}: {e}", p.display()))?;
    if !m.is_file() {
        return Err(format!("{} bukan file biasa", p.display()));
    }
    if m.len() > FILE_MAX {
        return Err(format!("{} terlalu besar ({} byte); pakai bash dengan sed/head", p.display(), m.len()));
    }
    let b = fs::read(p).map_err(|e| format!("{}: {e}", p.display()))?;
    String::from_utf8(b).map_err(|_| format!("{} bukan teks UTF-8", p.display()))
}

pub fn prepare(name: &str, a: &Value, cwd: &Path) -> Result<Prepared, String> {
    match name {
        "read" => {
            let path = resolve(cwd, &str_arg(a, "path")?)?;
            if !path.exists() {
                return Err(format!("{} tidak ada. Pakai path relatif terhadap folder kerja", show(cwd, &path)));
            }
            Ok(Prepared {
                title: format!("read {}", show(cwd, &path)),
                guarded: path.strip_prefix(cwd).is_err(),
                action: Action::Read {
                    path,
                    offset: int_arg(a, "offset").unwrap_or(1).max(1) as usize,
                    limit: int_arg(a, "limit").unwrap_or(400).clamp(1, 2000) as usize,
                },
            })
        }
        "write" => {
            let path = resolve(cwd, &str_arg(a, "path")?)?;
            let new = str_arg(a, "content")?;
            let old = if path.exists() { Some(load(&path)?) } else { None };
            if old.as_deref() == Some(new.as_str()) {
                return Err("isi file sudah sama; tidak ada yang berubah".into());
            }
            Ok(Prepared {
                title: format!("write {}", show(cwd, &path)),
                guarded: guarded(cwd, &path),
                action: Action::Write { path, old, new, is_edit: false },
            })
        }
        "edit" => {
            let path = resolve(cwd, &str_arg(a, "path")?)?;
            let g = guarded(cwd, &path);
            let src = load(&path)?;
            let all = a["replace_all"].as_bool().unwrap_or(false) || a["replace_all"] == "true";
            let new = apply_edit(&src, &str_arg(a, "old")?, &str_arg(a, "new")?, all, !g)?;
            Ok(Prepared {
                title: format!("edit {}", show(cwd, &path)),
                guarded: g,
                action: Action::Write { path, old: Some(src), new, is_edit: true },
            })
        }
        "bash" => {
            let cmd = str_arg(a, "command")?;
            if cmd.trim().is_empty() || cmd.contains('\0') {
                return Err("command kosong atau tidak valid".into());
            }
            let timeout = int_arg(a, "timeout").unwrap_or(120).clamp(1, 600);
            Ok(Prepared { title: "bash".into(), guarded: false, action: Action::Bash { cmd, timeout } })
        }
        _ => Err(format!("tool `{name}` tidak ada. Yang tersedia: read, edit, write, bash")),
    }
}

/// `hint`: boleh menyertakan potongan isi file pada pesan error.
pub fn apply_edit(src: &str, old: &str, new: &str, all: bool, hint: bool) -> Result<String, String> {
    if old.is_empty() {
        return Err("`old` kosong. Untuk membuat file baru pakai write".into());
    }
    let (mut old, mut new) = (old.to_string(), new.to_string());
    if !src.contains(&old) && src.contains("\r\n") && !old.contains('\r') {
        old = old.replace('\n', "\r\n");
        new = new.replace('\n', "\r\n");
    }
    if old == new {
        return Err("`old` dan `new` sama".into());
    }
    let hits: Vec<usize> = src.match_indices(&old).map(|(i, _)| i).collect();
    match hits.len() {
        0 => {
            let mut e = String::from("`old` tidak ditemukan di file.");
            let first = old.lines().map(str::trim).find(|l| !l.is_empty()).unwrap_or("");
            if hint && let Some((n, l)) = src.lines().enumerate().find(|(_, l)| !first.is_empty() && l.trim() == first) {
                e += &format!(" Baris mirip ada di {}: {:?}. Samakan spasi/indentasi persis.", n + 1, l);
            } else {
                e += " Baca ulang file dengan read, lalu salin teksnya persis tanpa nomor baris.";
            }
            Err(e)
        }
        1 => Ok(src.replacen(&old, &new, 1)),
        _ if all => Ok(src.replace(&old, &new)),
        n => {
            let at: Vec<String> = hits.iter().take(8)
                .map(|&i| (src[..i].matches('\n').count() + 1).to_string()).collect();
            Err(format!("`old` muncul {n} kali (baris {}). Tambah konteks agar unik, atau replace_all=true.", at.join(", ")))
        }
    }
}

/// Diff satu blok: awalan dan akhiran yang sama dibuang, sisakan 2 baris konteks.
/// Penanda: ' ' konteks, '-' hapus, '+' tambah.
#[allow(clippy::needless_range_loop)]
pub fn diff(old: &str, new: &str) -> Vec<(char, usize, String)> {
    let (a, b): (Vec<&str>, Vec<&str>) = (old.lines().collect(), new.lines().collect());
    let pre = a.iter().zip(&b).take_while(|(x, y)| x == y).count();
    let max = a.len().min(b.len()) - pre;
    let suf = a.iter().rev().zip(b.iter().rev()).take(max).take_while(|(x, y)| x == y).count();
    let mut v = vec![];
    for i in pre.saturating_sub(2)..pre {
        v.push((' ', i + 1, a[i].to_string()));
    }
    for i in pre..a.len() - suf {
        v.push(('-', i + 1, a[i].to_string()));
    }
    for i in pre..b.len() - suf {
        v.push(('+', i + 1, b[i].to_string()));
    }
    for i in b.len() - suf..(b.len() - suf + 2).min(b.len()) {
        v.push((' ', i + 1, b[i].to_string()));
    }
    v
}

pub struct Output {
    /// Untuk model.
    pub text: String,
    /// Untuk layar.
    pub summary: String,
    pub ok: bool,
}

impl Output {
    pub fn err(e: impl Into<String>) -> Self {
        let e = e.into();
        Output { text: format!("Error: {e}"), summary: e, ok: false }
    }
}

pub fn run(p: Prepared, cwd: &Path, cancel: &Cancel) -> Output {
    match p.action {
        Action::Read { path, offset, limit } => read(&path, offset, limit).unwrap_or_else(Output::err),
        Action::Write { path, old, new, is_edit } => write(&path, old, &new, is_edit).unwrap_or_else(Output::err),
        Action::Bash { cmd, timeout } => bash(&cmd, timeout, cwd, cancel).unwrap_or_else(Output::err),
    }
}

fn read(path: &Path, offset: usize, limit: usize) -> Result<Output, String> {
    if path.is_dir() {
        let mut v: Vec<String> = fs::read_dir(path).map_err(|e| e.to_string())?.flatten()
            .map(|e| {
                let d = e.file_type().is_ok_and(|t| t.is_dir());
                format!("{}{}", e.file_name().to_string_lossy(), if d { "/" } else { "" })
            }).collect();
        v.sort();
        let n = v.len();
        if n > 300 {
            v.truncate(300);
            v.push(format!("... {} entri lagi", n - 300));
        }
        let text = if n == 0 { "(folder kosong)".into() } else { v.join("\n") };
        return Ok(Output { text, summary: format!("{n} entri"), ok: true });
    }
    let src = load(path)?;
    let total = src.lines().count();
    if total == 0 {
        return Ok(Output { text: "(file kosong)".into(), summary: "kosong".into(), ok: true });
    }
    if offset > total {
        return Err(format!("offset {offset} melewati akhir file ({total} baris)"));
    }
    let (mut text, mut n) = (String::new(), 0);
    for (i, l) in src.lines().enumerate().skip(offset - 1).take(limit) {
        let l = clip(l, 2000);
        if n > 0 && text.len() + l.len() > READ_MAX {
            break;
        }
        text += &format!("{}\t{}\n", i + 1, l);
        n += 1;
    }
    let end = offset - 1 + n;
    if end < total {
        text += &format!("... {} baris lagi; lanjut dengan offset={}\n", total - end, end + 1);
    }
    let summary = if n == total { format!("{n} baris") } else { format!("baris {offset}-{end} dari {total}") };
    Ok(Output { text, summary, ok: true })
}

/// Potong pada batas karakter.
fn clip(s: &str, max: usize) -> &str {
    if s.len() <= max { s } else { &s[..s.floor_char_boundary(max)] }
}

fn write(path: &Path, old: Option<String>, new: &str, is_edit: bool) -> Result<Output, String> {
    // File bisa berubah selagi menunggu izin; jangan timpa versi yang belum dilihat.
    let now = if path.exists() { Some(load(path)?) } else { None };
    if now != old {
        return Err("file berubah selagi menunggu izin; baca ulang lalu coba lagi".into());
    }
    if let Some(d) = path.parent() {
        fs::create_dir_all(d).map_err(|e| format!("{}: {e}", d.display()))?;
    }
    fs::write(path, new).map_err(|e| format!("{}: {e}", path.display()))?;
    let (mut add, mut del) = (0, 0);
    for (c, _, _) in diff(old.as_deref().unwrap_or(""), new) {
        match c {
            '+' => add += 1,
            '-' => del += 1,
            _ => {}
        }
    }
    let what = if is_edit { "diedit" } else if old.is_some() { "ditimpa" } else { "dibuat" };
    Ok(Output { text: format!("OK: file {what} (+{add} -{del})"), summary: format!("{what} (+{add} -{del})"), ok: true })
}

/// Penampung output berbatas: simpan awal dan akhir, buang tengah.
#[derive(Default)]
struct Capped {
    head: Vec<u8>,
    tail: Vec<u8>,
    dropped: usize,
}

impl Capped {
    fn push(&mut self, mut b: &[u8]) {
        let room = (BASH_MAX / 2).saturating_sub(self.head.len()).min(b.len());
        self.head.extend_from_slice(&b[..room]);
        b = &b[room..];
        self.tail.extend_from_slice(b);
        if self.tail.len() > BASH_MAX {
            let cut = self.tail.len() - BASH_MAX / 2;
            self.tail.drain(..cut);
            self.dropped += cut;
        }
    }

    fn text(&mut self) -> String {
        if self.tail.len() > BASH_MAX / 2 {
            let cut = self.tail.len() - BASH_MAX / 2;
            self.tail.drain(..cut);
            self.dropped += cut;
        }
        if self.dropped == 0 {
            // gabung dulu: batas head/tail bisa membelah satu karakter UTF-8
            let all = [self.head.as_slice(), &self.tail].concat();
            return String::from_utf8_lossy(&all).into_owned();
        }
        format!("{}\n... [{} byte dipotong] ...\n{}", String::from_utf8_lossy(&self.head),
            self.dropped, String::from_utf8_lossy(&self.tail))
    }
}

fn bash(cmd: &str, timeout: u64, cwd: &Path, cancel: &Cancel) -> Result<Output, String> {
    let io = |e: std::io::Error| format!("bash: {e}");
    let (mut rd, wr) = std::io::pipe().map_err(io)?;
    let mut c = Command::new("bash");
    c.arg("-c").arg(cmd).current_dir(cwd).stdin(Stdio::null())
        .stdout(wr.try_clone().map_err(io)?).stderr(wr)
        .env("PAGER", "cat").env("GIT_PAGER", "cat").env("GIT_TERMINAL_PROMPT", "0").env("TERM", "dumb")
        .process_group(0);
    let mut child = c.spawn().map_err(io)?;
    drop(c); // tutup ujung tulis milik kita agar pembaca mendapat EOF
    let pgid = child.id() as i32;
    cancel.set_pgid(pgid);

    let buf = Arc::new(Mutex::new(Capped::default()));
    let (tx, eof) = std::sync::mpsc::channel::<()>();
    let b2 = buf.clone();
    std::thread::spawn(move || {
        let mut t = [0u8; 8192];
        while let Ok(n @ 1..) = rd.read(&mut t) {
            b2.lock().unwrap().push(&t[..n]);
        }
        drop(tx);
    });

    let start = Instant::now();
    let mut timed_out = false;
    let status = loop {
        if let Some(s) = cancel.reap(&mut child).map_err(io)? {
            break s;
        }
        if !timed_out && start.elapsed() > Duration::from_secs(timeout) {
            timed_out = true;
            kill_group(pgid);
        }
        std::thread::sleep(Duration::from_millis(15));
    };
    // Proses latar (`cmd &`) bisa tetap memegang pipe; jangan menunggunya.
    let _ = eof.recv_timeout(Duration::from_millis(300));
    let mut text = buf.lock().unwrap().text();
    let lines = text.lines().count();
    if text.trim().is_empty() {
        text = "(tanpa output)".into();
    }
    let secs = start.elapsed().as_secs_f32();
    if cancel.is_set() {
        text += "\n[dibatalkan pengguna]";
        return Ok(Output { text, summary: "dibatalkan".into(), ok: false });
    }
    if timed_out {
        text += &format!("\n[dihentikan: melewati batas {timeout} detik]");
        return Ok(Output { text, summary: format!("melewati batas {timeout} dtk"), ok: false });
    }
    let code = status.code().unwrap_or(-1);
    if !status.success() {
        text += &format!("\n[exit {code}]");
    }
    Ok(Output { summary: format!("exit {code} · {lines} baris · {secs:.1} dtk"), text, ok: status.success() })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dir(n: &str) -> PathBuf {
        let d = Path::new(env!("CARGO_MANIFEST_DIR")).join("target/test").join(n);
        let _ = fs::remove_dir_all(&d);
        fs::create_dir_all(&d).unwrap();
        fs::canonicalize(d).unwrap()
    }

    #[test]
    fn edit_harus_unik() {
        assert_eq!(apply_edit("a b", "a", "c", false, true).unwrap(), "c b");
        assert!(apply_edit("a a", "a", "c", false, true).unwrap_err().contains("2 kali"));
        assert_eq!(apply_edit("a a", "a", "c", true, true).unwrap(), "c c");
        assert!(apply_edit("  x = 1\n", "x  = 1", "y", false, true).is_err());
        assert!(apply_edit("    foo\n", "\tfoo", "y", false, true).unwrap_err().contains("mirip"));
        assert!(!apply_edit("    foo\n", "\tfoo", "y", false, false).unwrap_err().contains("mirip"));
        assert_eq!(apply_edit("a\r\nb\r\n", "a\nb", "x\ny", false, true).unwrap(), "x\r\ny\r\n");
    }

    #[test]
    fn path_di_luar_dan_symlink_dijaga() {
        let d = dir("path");
        fs::create_dir(d.join("in")).unwrap();
        std::os::unix::fs::symlink("/etc", d.join("in/link")).unwrap();
        std::os::unix::fs::symlink("/tidak/ada", d.join("in/rusak")).unwrap();
        let cwd = d.join("in");
        let g = |p: &str| guarded(&cwd, &resolve(&cwd, p).unwrap());
        assert!(!g("a/b.txt"));
        assert!(g("../x"));
        assert!(g("link/passwd"));
        assert!(g("baru/../link/passwd"));
        assert!(g("baru/../../x"));
        assert!(g(".git/hooks/pre-commit"));
        assert!(g("/etc/hosts"));
        assert!(g("@/etc/hosts") && g("@link/passwd") && g("@../x"));
        assert_eq!(resolve(&cwd, "@a.txt").unwrap(), cwd.join("a.txt"));
        fs::write(cwd.join("@b"), "").unwrap();
        assert_eq!(resolve(&cwd, "@b").unwrap(), cwd.join("@b"));
        assert!(resolve(&cwd, "rusak").is_err());
    }

    #[test]
    fn tulis_ditolak_bila_file_berubah() {
        let d = dir("toctou");
        fs::write(d.join("f"), "satu").unwrap();
        let p = prepare("edit", &json!({"path":"f","old":"satu","new":"dua"}), &d).unwrap();
        fs::write(d.join("f"), "lain").unwrap();
        assert!(!run(p, &d, &Cancel::default()).ok);
        assert_eq!(fs::read_to_string(d.join("f")).unwrap(), "lain");
    }

    #[test]
    fn diff_tidak_panik() {
        for (a, b) in [("", "a"), ("a", ""), ("a\na", "a"), ("a", "a\na"), ("x\ny\nz", "x\nq\nz"), ("a\n", "a")] {
            diff(a, b);
        }
        let d: String = diff("x\ny\nz", "x\nq\nz").iter().map(|t| t.0).collect();
        assert_eq!(d, " -+ ");
    }

    #[test]
    fn bash_exit_timeout_latar_batal() {
        let d = dir("bash");
        let c = Cancel::default();
        let o = bash("echo hai; echo err >&2; exit 3", 5, &d, &c).unwrap();
        assert!(o.text.contains("hai") && o.text.contains("err") && o.text.contains("[exit 3]"));
        let t = Instant::now();
        assert!(bash("sleep 20 & sleep 20", 1, &d, &c).unwrap().summary.contains("batas"));
        assert!(bash("sleep 3 & echo ok", 5, &d, &c).unwrap().text.contains("ok"));
        assert!(t.elapsed() < Duration::from_secs(3));
        let c = Arc::new(Cancel::default());
        let c2 = c.clone();
        std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(200));
            c2.cancel();
        });
        let t = Instant::now();
        assert_eq!(bash("sleep 20", 30, &d, &c).unwrap().summary, "dibatalkan");
        assert!(t.elapsed() < Duration::from_secs(2));
    }

    #[test]
    fn output_besar_dipotong() {
        let d = dir("besar");
        let o = bash("seq 1 200000", 20, &d, &Cancel::default()).unwrap();
        assert!(o.text.len() < BASH_MAX + 200 && o.text.contains("dipotong"));
        assert!(o.text.starts_with("1\n") && o.text.contains("200000"));
    }
}
