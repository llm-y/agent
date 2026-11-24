//! Perbedaan antar sistem operasi, dikumpulkan di satu tempat.

use std::path::PathBuf;
use std::process::Command;

fn abs(k: &str) -> Option<PathBuf> {
    std::env::var_os(k).map(PathBuf::from).filter(|p| p.is_absolute())
}

pub fn home() -> Option<PathBuf> {
    abs("HOME").or_else(|| abs("USERPROFILE"))
}

pub fn state_dir() -> Option<PathBuf> {
    abs("XDG_STATE_HOME")
        .or_else(|| if cfg!(windows) { abs("LOCALAPPDATA") } else { None })
        .or_else(|| home().map(|h| h.join(".local/state")))
        .map(|p| p.join("llmy"))
}

pub fn config_dir() -> Option<PathBuf> {
    abs("XDG_CONFIG_HOME")
        .or_else(|| if cfg!(windows) { abs("APPDATA") } else { None })
        .or_else(|| home().map(|h| h.join(".config")))
        .map(|p| p.join("llmy"))
}

/// Matikan proses beserta turunannya.
#[cfg(unix)]
pub fn kill_tree(pid: u32) {
    // Anak dijalankan sebagai pemimpin grup, jadi pid = pgid.
    // SAFETY: kill() tidak menyentuh memori; pid > 1 sehingga -pid hanya grup itu.
    if pid > 1 {
        unsafe { libc::kill(-(pid as i32), libc::SIGKILL) };
    }
}

#[cfg(windows)]
pub fn kill_tree(pid: u32) {
    let _ = Command::new("taskkill")
        .args(["/T", "/F", "/PID", &pid.to_string()])
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status();
}

#[cfg(unix)]
pub fn shell() -> Command {
    use std::os::unix::process::CommandExt;
    let mut c = Command::new("bash");
    c.process_group(0);
    c
}

/// `bash` di PATH Windows bisa berarti WSL, yang melihat sistem berkas berbeda.
/// Utamakan Git Bash.
#[cfg(windows)]
pub fn shell() -> Command {
    let git = ["ProgramFiles", "ProgramW6432", "LOCALAPPDATA"].iter().filter_map(|k| abs(k)).flat_map(|p| {
        [p.join("Git/bin/bash.exe"), p.join("Programs/Git/bin/bash.exe")]
    });
    let path = abs("LLMY_BASH").into_iter().chain(git).find(|p| p.is_file());
    Command::new(path.unwrap_or_else(|| "bash".into()))
}

/// Sinyal berhenti: jalankan `f`, lalu keluar. Di Windows Ctrl-C sudah sampai ke
/// seluruh proses di konsol yang sama, jadi tidak perlu apa-apa.
#[cfg(unix)]
pub fn on_signal(f: impl FnOnce() + Send + 'static) {
    // Diblokir sebelum thread lain dibuat; std mengosongkan mask untuk proses anak.
    // SAFETY: `set` diinisialisasi sigemptyset sebelum dipakai.
    let set = unsafe {
        let mut set = std::mem::zeroed::<libc::sigset_t>();
        libc::sigemptyset(&mut set);
        for s in [libc::SIGINT, libc::SIGTERM, libc::SIGHUP] {
            libc::sigaddset(&mut set, s);
        }
        libc::pthread_sigmask(libc::SIG_BLOCK, &set, std::ptr::null_mut());
        set
    };
    std::thread::spawn(move || {
        let mut sig = 0;
        // SAFETY: kedua pointer valid selama pemanggilan.
        unsafe { libc::sigwait(&set, &mut sig) };
        f();
        std::process::exit(128 + sig);
    });
}

#[cfg(windows)]
pub fn on_signal(_: impl FnOnce() + Send + 'static) {}

/// Buat folder dan file yang hanya bisa dibaca pemiliknya (di Windows mengikuti
/// izin folder profil pengguna).
pub fn private_dir(p: &std::path::Path) -> std::io::Result<()> {
    let mut b = std::fs::DirBuilder::new();
    #[cfg(unix)]
    std::os::unix::fs::DirBuilderExt::mode(&mut b, 0o700);
    b.recursive(true).create(p)
}

pub fn private_file(p: &std::path::Path) -> std::io::Result<std::fs::File> {
    let mut o = std::fs::OpenOptions::new();
    #[cfg(unix)]
    std::os::unix::fs::OpenOptionsExt::mode(&mut o, 0o600);
    // `read` perlu agar penguncian berfungsi di semua sistem.
    o.read(true).append(true).create_new(true).open(p)
}
