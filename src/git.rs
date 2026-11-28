//! Git sebagai bagian harness (2026-11-28): setiap folder kerja adalah repo git. Bila belum, llmy menjalankan
//! `git init` (kecuali di `$HOME` atau `/`). Snapshot memakai repo yang sama tanpa menyentuh milik pengguna:
//! index sementara (`GIT_INDEX_FILE`), `write-tree`, `commit-tree`, lalu `update-ref refs/llmy/<sesi>`. Branch,
//! HEAD, index, `git log`, dan hook pengguna tidak berubah; `git log refs/llmy/<sesi>` menampilkan langkah llmy.

use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

/// Folder build yang tidak ikut snapshot walau tidak ada di `.gitignore`.
const SKIP: [&str; 8] = ["target", "node_modules", "__pycache__", ".venv", "venv", "dist", "build", ".pytest_cache"];

pub struct Repo {
    pub root: PathBuf,
    git_dir: PathBuf,
    /// `refs/llmy/<sesi>`.
    pub reff: String,
    /// Snapshot pertama sesi (isi asli sebelum llmy mengubah apa pun).
    pub base: Option<String>,
    last: Option<String>,
    /// Snapshot dengan uji lulus terbanyak: (lulus, gagal, commit).
    pub best: Option<(usize, usize, String)>,
    /// Skor uji terakhir yang terlihat.
    pub now: Option<(usize, usize)>,
    /// Folder state llmy (sesi, metrik) bila berada di dalam repo: tidak ikut snapshot.
    skip: Option<String>,
}

fn git(dir: &Path, args: &[&str], env: &[(&str, &str)]) -> Result<String, String> {
    let mut c = Command::new("git");
    c.args(["-c", "core.quotepath=off"]).args(args).current_dir(dir).stdin(Stdio::null());
    for (k, v) in env {
        c.env(k, v);
    }
    let o = c.output().map_err(|e| format!("git tidak bisa dijalankan: {e}"))?;
    if o.status.success() {
        Ok(String::from_utf8_lossy(&o.stdout).trim_end().to_string())
    } else {
        Err(String::from_utf8_lossy(&o.stderr).trim().to_string())
    }
}

/// Pesan bila git tidak ada, dengan cara memasangnya.
pub fn missing() -> String {
    let how = if cfg!(windows) {
        "winget install --id Git.Git -e"
    } else if cfg!(target_os = "macos") {
        "xcode-select --install (atau brew install git)"
    } else {
        "sudo apt install git (atau dnf/pacman sesuai distro)"
    };
    format!("llmy membutuhkan git: snapshot, undo, dan pemeriksaan perubahan memakai git. Pasang dengan: {how}")
}

/// Folder yang tidak boleh dijadikan repo otomatis.
fn forbidden(cwd: &Path) -> bool {
    let canon = |p: &Path| std::fs::canonicalize(p).unwrap_or_else(|_| p.to_path_buf());
    cwd.parent().is_none() || crate::sys::home().is_some_and(|h| canon(&h) == canon(cwd))
}

pub enum Open {
    Repo(Repo),
    /// Git ada tetapi folder ini `$HOME` atau `/`: llmy jalan tanpa fitur git.
    Skipped(String),
}

/// Pastikan `cwd` berada di repo git (init bila belum), lalu siapkan ref sesi. Err: git tidak terpasang.
pub fn open(cwd: &Path, session: &str) -> Result<Open, String> {
    git(cwd, &["--version"], &[]).map_err(|_| missing())?;
    // Akar repo diturunkan dari `cwd` (`--show-cdup` = "../../"), bukan dari `--show-toplevel`: di Windows `cwd`
    // hasil canonicalize berawalan `\\?\` sedangkan git menulis `C:/...`, dan path harus cocok satu sama lain.
    let mut up = git(cwd, &["rev-parse", "--show-cdup"], &[]).ok().map(|s| s.matches("..").count());
    // `.git` kosong atau rusak (folder uji, soal benchmark): git naik ke repo induk. Folder ini dimaksudkan
    // sebagai repo sendiri, jadi di-init di sini.
    if cwd.join(".git").is_dir() && up.is_some_and(|n| n > 0) {
        up = None;
    }
    // Folder yang di-ignore repo induknya (mis. `scratch/`) bukan bagian repo itu: snapshot tidak akan memuatnya.
    if up.is_some_and(|n| n > 0) && git(cwd, &["check-ignore", "-q", "."], &[]).is_ok() {
        up = None;
    }
    let root = match up.and_then(|n| cwd.ancestors().nth(n)) {
        Some(r) => r.to_path_buf(),
        None => {
            if forbidden(cwd) {
                return Ok(Open::Skipped(format!(
                    "{} bukan repo git dan tidak dijadikan repo otomatis; snapshot dan undo berbasis git mati. Jalankan llmy di folder proyek.",
                    cwd.display()
                )));
            }
            git(cwd, &["init", "-q"], &[])?;
            cwd.to_path_buf()
        }
    };
    // Repo dotfiles di $HOME: snapshot akan memuat seluruh home.
    if forbidden(&root) {
        return Ok(Open::Skipped(format!("akar repo git adalah {}; snapshot dan undo berbasis git mati. Jalankan llmy di folder proyek.", root.display())));
    }
    let git_dir = PathBuf::from(git(cwd, &["rev-parse", "--absolute-git-dir"], &[])?);
    let reff = format!("refs/llmy/{}", session.chars().filter(|c| c.is_ascii_alphanumeric() || *c == '-').collect::<String>());
    let skip = crate::sys::state_dir().and_then(|s| std::fs::canonicalize(s).ok())
        .and_then(|s| s.strip_prefix(&root).ok().map(|p| p.to_string_lossy().replace('\\', "/")))
        .filter(|s| !s.is_empty());
    let mut r = Repo { root, git_dir, reff, base: None, last: None, best: None, now: None, skip };
    // Sesi yang dilanjutkan: snapshot pertama tetap isi asli.
    if let Ok(tip) = r.git(&["rev-parse", "--verify", "-q", &r.reff], &[]) {
        r.base = r.git(&["rev-list", "--max-parents=0", &tip], &[]).ok().and_then(|s| s.lines().last().map(String::from));
        r.last = Some(tip);
    }
    Ok(Open::Repo(r))
}

impl Repo {
    fn git(&self, args: &[&str], env: &[(&str, &str)]) -> Result<String, String> {
        git(&self.root, args, env)
    }

    /// Simpan keadaan folder kerja sebagai commit di ref sesi. Hanya membaca index pengguna (disalin).
    pub fn snapshot(&mut self, msg: &str) -> Option<String> {
        let tmp = self.git_dir.join(format!("llmy-index-{}", std::process::id()));
        let real = self.git_dir.join("index");
        if real.exists() {
            let _ = std::fs::copy(&real, &tmp);
        } else {
            let _ = std::fs::remove_file(&tmp);
        }
        let idx = tmp.to_string_lossy().into_owned();
        let env = [("GIT_INDEX_FILE", idx.as_str())];
        let mut args: Vec<String> = ["add", "-A", "--", "."].iter().map(|s| s.to_string()).collect();
        args.extend(SKIP.iter().map(|d| format!(":(exclude,glob)**/{d}/**")));
        if let Some(s) = &self.skip {
            args.push(format!(":(exclude){s}"));
        }
        let a: Vec<&str> = args.iter().map(String::as_str).collect();
        let tree = self.git(&a, &env).and_then(|_| self.git(&["write-tree"], &env));
        let _ = std::fs::remove_file(&tmp);
        let tree = tree.ok()?;
        let who = [("GIT_AUTHOR_NAME", "llmy"), ("GIT_AUTHOR_EMAIL", "llmy@localhost"), ("GIT_COMMITTER_NAME", "llmy"), ("GIT_COMMITTER_EMAIL", "llmy@localhost")];
        let mut ct = vec!["commit-tree", tree.as_str(), "-m", msg];
        let parent = self.last.clone();
        if let Some(p) = &parent {
            // Tidak ada yang berubah: tidak perlu commit baru.
            if self.git(&["rev-parse", &format!("{p}^{{tree}}")], &[]).ok().as_deref() == Some(tree.as_str()) {
                return Some(p.clone());
            }
            ct.extend(["-p", p.as_str()]);
        }
        let c = self.git(&ct, &who).ok()?;
        self.git(&["update-ref", &self.reff, &c], &[]).ok()?;
        self.last = Some(c.clone());
        if self.base.is_none() {
            self.base = Some(c.clone());
        }
        Some(c)
    }

    /// Isi mentah `path` di snapshot `commit` (None bila file belum ada saat itu).
    pub fn bytes(&self, commit: &str, path: &Path) -> Option<Vec<u8>> {
        let rel = path.strip_prefix(&self.root).ok()?.to_string_lossy().replace('\\', "/");
        let o = Command::new("git").args(["cat-file", "blob", &format!("{commit}:{rel}")]).current_dir(&self.root)
            .stdin(Stdio::null()).stderr(Stdio::null()).output().ok()?;
        o.status.success().then_some(o.stdout)
    }

    /// Isi teks `path` di snapshot `commit`.
    pub fn show(&self, commit: &str, path: &Path) -> Option<String> {
        String::from_utf8(self.bytes(commit, path)?).ok()
    }

    /// File yang berbeda antara dua snapshot.
    pub fn changed(&self, a: &str, b: &str) -> Vec<PathBuf> {
        self.git(&["diff", "--name-only", "--no-renames", a, b], &[]).unwrap_or_default()
            .lines().filter(|l| !l.is_empty()).map(|l| self.root.join(l)).collect()
    }

    /// Kembalikan `paths` ke isinya di `commit`; file yang belum ada di sana dihapus. Hanya path yang diberikan.
    pub fn restore(&self, commit: &str, paths: &[PathBuf]) -> Vec<PathBuf> {
        let mut done = vec![];
        for p in paths {
            let ok = match self.bytes(commit, p) {
                Some(s) => std::fs::write(p, s).is_ok(),
                None => !p.exists() || std::fs::remove_file(p).is_ok(),
            };
            if ok {
                done.push(p.clone());
            }
        }
        done
    }

    /// Catat skor uji untuk keadaan sekarang; simpan snapshot terbaik.
    pub fn scored(&mut self, score: (usize, usize)) {
        self.now = Some(score);
        let Some(c) = self.snapshot(&format!("uji: {} lulus, {} gagal", score.0, score.1)) else { return };
        if self.best.as_ref().is_none_or(|b| better(score, (b.0, b.1))) {
            self.best = Some((score.0, score.1, c));
        }
    }
}

/// Lebih banyak lulus, atau sama lulus dengan lebih sedikit gagal.
pub fn better(a: (usize, usize), b: (usize, usize)) -> bool {
    a.0 > b.0 || (a.0 == b.0 && a.1 < b.1)
}

/// Jumlah uji (lulus, gagal) dari keluaran cargo test, pytest, jest, node --test, atau unittest.
pub fn score(out: &str) -> Option<(usize, usize)> {
    static RE: std::sync::OnceLock<[regex_lite::Regex; 6]> = std::sync::OnceLock::new();
    let r = RE.get_or_init(|| [
        regex_lite::Regex::new(r"test result: \w+\. (\d+) passed; (\d+) failed").unwrap(),
        regex_lite::Regex::new(r"(?m)^Tests:\s+(?:(\d+) failed, )?(?:\d+ skipped, )?(?:\d+ todo, )?(\d+) passed").unwrap(),
        regex_lite::Regex::new(r"(?m)^=*\s*((?:\d+ (?:passed|failed|errors?|skipped|warnings?|xfailed|xpassed|deselected)(?:, )?)+) in [\d.]+s").unwrap(),
        regex_lite::Regex::new(r"(?m)^# pass (\d+)\s*$").unwrap(),
        regex_lite::Regex::new(r"(?m)^Ran (\d+) tests?").unwrap(),
        regex_lite::Regex::new(r"(?m)^FAILED \((?:failures=(\d+))?(?:, )?(?:errors=(\d+))?").unwrap(),
    ]);
    let n = |m: Option<regex_lite::Match>| m.map_or(0, |m| m.as_str().parse::<usize>().unwrap_or(0));
    // cargo: satu baris per target uji.
    let cargo: Vec<_> = r[0].captures_iter(out).collect();
    if !cargo.is_empty() {
        return Some(cargo.iter().fold((0, 0), |(p, f), c| (p + n(c.get(1)), f + n(c.get(2)))));
    }
    if let Some(c) = r[1].captures(out) {
        return Some((n(c.get(2)), n(c.get(1))));
    }
    if let Some(c) = r[2].captures_iter(out).last() {
        let s = c.get(1).map_or("", |m| m.as_str());
        let k = |w: &str| s.split(", ").filter_map(|x| x.strip_suffix(w)).map(|x| x.trim().parse::<usize>().unwrap_or(0)).sum::<usize>();
        let (p, f) = (k(" passed"), k(" failed") + k(" error") + k(" errors"));
        if p + f > 0 {
            return Some((p, f));
        }
    }
    if let Some(c) = r[3].captures(out) {
        let fail = regex_lite::Regex::new(r"(?m)^# fail (\d+)").ok()?.captures(out).map_or(0, |c| n(c.get(1)));
        return Some((n(c.get(1)), fail));
    }
    if let Some(c) = r[4].captures(out) {
        let total = n(c.get(1));
        let f = r[5].captures(out).map_or(0, |c| n(c.get(1)) + n(c.get(2)));
        return Some((total.saturating_sub(f), f));
    }
    None
}

/// Perintah `bash` yang menjalankan uji (untuk membaca skor dari keluarannya).
pub fn is_test(cmd: &str) -> bool {
    ["cargo test", "pytest", "unittest", "npm test", "npm run test", "node --test", "jest"].iter().any(|k| cmd.contains(k))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn skor_dari_keluaran_uji() {
        assert_eq!(score("test result: ok. 3 passed; 0 failed; 0 ignored\ntest result: FAILED. 2 passed; 1 failed;"), Some((5, 1)));
        assert_eq!(score("Tests:       1 failed, 9 passed, 10 total"), Some((9, 1)));
        assert_eq!(score("Tests:       10 passed, 10 total"), Some((10, 0)));
        assert_eq!(score("=========== 2 failed, 7 passed in 0.12s ===========\n"), Some((7, 2)));
        assert_eq!(score("====== 4 passed in 0.01s ======"), Some((4, 0)));
        assert_eq!(score("..F\n1 failed, 2 passed, 1 warning in 0.05s"), Some((2, 1)));
        assert_eq!(score("# tests 3\n# pass 2\n# fail 1\n"), Some((2, 1)));
        assert_eq!(score("Ran 5 tests in 0.001s\n\nFAILED (failures=2)"), Some((3, 2)));
        assert_eq!(score("Ran 5 tests in 0.001s\n\nOK"), Some((5, 0)));
        assert_eq!(score("error[E0425]: cannot find value"), None);
        assert!(is_test("cd /work && python3 -m pytest -q") && !is_test("ls -la"));
    }

    fn repo(n: &str) -> PathBuf {
        let d = Path::new(env!("CARGO_MANIFEST_DIR")).join("target/test").join(n);
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    #[test]
    fn snapshot_tanpa_menyentuh_milik_pengguna() {
        if git(Path::new("."), &["--version"], &[]).is_err() {
            return;
        }
        let d = repo("git-snapshot");
        // `.git` kosong di dalam repo lain (seperti folder uji): tetap jadi repo sendiri.
        std::fs::create_dir_all(d.join(".git")).unwrap();
        std::fs::write(d.join("a.py"), "A = 1\n").unwrap();
        std::fs::create_dir_all(d.join("target")).unwrap();
        std::fs::write(d.join("target/besar.bin"), "x").unwrap();
        let Open::Repo(mut r) = open(&d, "uji-1").unwrap() else { panic!("harus repo") };
        assert!(d.join(".git/HEAD").is_file() && r.root == d);
        let s0 = r.snapshot("awal").unwrap();
        assert_eq!(r.base.as_deref(), Some(s0.as_str()));
        // Snapshot tidak membuat commit di branch, tidak mengisi index pengguna, dan melewati folder build.
        assert!(git(&d, &["rev-parse", "--verify", "-q", "HEAD"], &[]).is_err());
        assert_eq!(git(&d, &["status", "--porcelain"], &[]).unwrap().lines().filter(|l| !l.starts_with("??")).count(), 0);
        assert!(r.show(&s0, &d.join("target/besar.bin")).is_none());
        std::fs::write(d.join("a.py"), "A = 2\n").unwrap();
        std::fs::write(d.join("b.py"), "B = 1\n").unwrap();
        r.scored((3, 1));
        let best = r.best.clone().unwrap();
        std::fs::write(d.join("a.py"), "A = rusak\n").unwrap();
        r.scored((0, 4));
        assert_eq!(r.best.as_ref().unwrap().2, best.2, "skor lebih buruk tidak menggantikan yang terbaik");
        let kini = r.snapshot("kini").unwrap();
        let ch = r.changed(&best.2, &kini);
        assert_eq!(ch, vec![d.join("a.py")]);
        r.restore(&best.2, &ch);
        assert_eq!(std::fs::read_to_string(d.join("a.py")).unwrap(), "A = 2\n");
        // Kembali ke awal: file baru dihapus.
        r.restore(&s0, &[d.join("b.py")]);
        assert!(!d.join("b.py").exists());
        // Folder yang di-ignore repo induk jadi repo sendiri.
        std::fs::write(d.join(".gitignore"), "abaikan/\n").unwrap();
        std::fs::create_dir_all(d.join("abaikan")).unwrap();
        let Open::Repo(r3) = open(&d.join("abaikan"), "uji-2").unwrap() else { panic!() };
        assert!(r3.root == d.join("abaikan") && d.join("abaikan/.git/HEAD").is_file());
        // Sesi dilanjutkan: base tetap snapshot pertama.
        let Open::Repo(r2) = open(&d, "uji-1").unwrap() else { panic!() };
        assert_eq!(r2.base.as_deref(), Some(s0.as_str()));
    }
}
