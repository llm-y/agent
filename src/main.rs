mod agent;
mod cancel;
mod editor;
mod http;
mod perm;
mod provider;
mod rules;
mod session;
mod tools;
mod ui;

use agent::{Agent, Cmd, Ev};
use cancel::Cancel;
use perm::Perm;
use provider::Provider;
use session::Session;
use std::io::{IsTerminal, Read, Write};
use std::sync::Arc;
use std::sync::mpsc::channel;

const HELP: &str = "llmy - agent AI CLI yang ringan

Pakai:
  llmy [opsi]              mode interaktif
  llmy [opsi] -p <prompt>  sekali jalan (stdin ikut dibaca bila dipipa)

Opsi:
  -m, --model <nama>   model Ollama      (env LLMY_MODEL, bawaan qwen3.5:9b)
      --url <url>      alamat Ollama     (env LLMY_URL, bawaan http://localhost:11434)
      --ctx <n>        ukuran context    (env LLMY_CTX, bawaan 32768)
      --think          aktifkan thinking
  -c, --continue       lanjutkan sesi terakhir di folder ini
  -r, --resume <id>    lanjutkan sesi tertentu
      --sessions       daftar sesi di folder ini
      --no-session     jangan simpan sesi
      --allow <daftar> tanpa tanya: edit, bash, atau edit,bash
  -p, --print <prompt> sekali jalan
  -h, --help / -V, --version

Instruksi dibaca dari AGENTS.md (atau CLAUDE.md), akar git sampai folder kerja.
Sesi disimpan di ~/.local/state/llmy/sessions, hanya bisa dibaca pemiliknya.
edit berlaku hanya di dalam folder kerja. bash berjalan dengan hak akses Anda.";

struct Opts {
    provider: Provider,
    perm: Perm,
    prompt: Option<String>,
    /// Some(None) = sesi terakhir.
    resume: Option<Option<String>>,
    save: bool,
    list: bool,
}

fn parse() -> Result<Opts, String> {
    let env = |k: &str, d: &str| std::env::var(k).ok().filter(|s| !s.is_empty()).unwrap_or(d.into());
    let mut o = Opts {
        provider: Provider {
            base: env("LLMY_URL", "http://localhost:11434"),
            model: env("LLMY_MODEL", "qwen3.5:9b"),
            ctx: 0,
            think: false,
        },
        perm: Perm::default(),
        prompt: None,
        resume: None,
        save: true,
        list: false,
    };
    let mut ctx = env("LLMY_CTX", "32768");
    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        let mut val = || it.next().ok_or(format!("{a} butuh nilai"));
        match a.as_str() {
            "-h" | "--help" => {
                println!("{HELP}");
                std::process::exit(0);
            }
            "-V" | "--version" => {
                println!("llmy {}", env!("CARGO_PKG_VERSION"));
                std::process::exit(0);
            }
            "-m" | "--model" => o.provider.model = val()?,
            "--url" => o.provider.base = val()?,
            "--ctx" => ctx = val()?,
            "-c" | "--continue" => o.resume = Some(None),
            "-r" | "--resume" => o.resume = Some(Some(val()?)),
            "--sessions" => o.list = true,
            "--no-session" => o.save = false,
            "--think" => o.provider.think = true,
            "-p" | "--print" => o.prompt = Some(val()?),
            "--allow" => {
                for s in val()?.split(',') {
                    match s {
                        "edit" => o.perm.allow_edit = true,
                        "bash" => o.perm.allow_bash = true,
                        _ => return Err(format!("--allow tidak mengenal `{s}`")),
                    }
                }
            }
            _ => return Err(format!("opsi tidak dikenal: {a} (lihat --help)")),
        }
    }
    o.provider.ctx = ctx.parse().ok().filter(|n| *n >= 4096).ok_or("ctx harus angka >= 4096")?;
    Ok(o)
}

fn main() {
    let o = parse().unwrap_or_else(|e| die(&e));
    let cwd = std::env::current_dir().and_then(std::fs::canonicalize).unwrap_or_else(|e| die(&e.to_string()));
    let mut session = if o.save { session::root().map(|r| Session::new(&r, &cwd)) } else { None };
    if o.list {
        return list(session.as_ref());
    }
    let cancel = Arc::new(Cancel::default());
    let (tx, rx) = channel();
    let interactive = o.prompt.is_none();
    if interactive && !(std::io::stdin().is_terminal() && std::io::stdout().is_terminal()) {
        die("mode interaktif butuh terminal; pakai -p <prompt>");
    }
    let history = match (&o.resume, &mut session) {
        (None, _) => vec![],
        (Some(id), Some(s)) => s.resume(id.as_deref()).unwrap_or_else(|e| die(&e)),
        (Some(_), None) => die("--continue/--resume tidak bisa dipakai tanpa penyimpanan sesi"),
    };
    on_signal(cancel.clone(), interactive);
    let agent = Agent::new(o.provider, o.perm, interactive, cwd.clone(), cancel.clone(), tx.clone())
        .with_session(session, history);
    let (ctx, crx) = channel();
    std::thread::spawn(move || agent.serve(crx));
    match o.prompt {
        None => {
            if let Err(e) = ui::run(cwd, cancel, tx, rx, ctx) {
                die(&e.to_string());
            }
        }
        Some(mut p) => {
            if !std::io::stdin().is_terminal() {
                let mut s = String::new();
                let _ = std::io::stdin().read_to_string(&mut s);
                if !s.trim().is_empty() {
                    p = format!("{p}\n\n{s}");
                }
            }
            let _ = ctx.send(Cmd::User(p));
            std::process::exit(print_mode(rx, o.resume.is_some()));
        }
    }
}

fn list(session: Option<&Session>) {
    let v = session.map(|s| s.list()).unwrap_or_default();
    if v.is_empty() {
        println!("Belum ada sesi di folder ini.");
    }
    for e in v {
        let m = e.modified.elapsed().map_or(0, |d| d.as_secs() / 60);
        let age = match m {
            0 => "baru saja".into(),
            1..60 => format!("{m} menit lalu"),
            60..1440 => format!("{} jam lalu", m / 60),
            _ => format!("{} hari lalu", m / 1440),
        };
        println!("{:<22} {age:<15} {}", e.id, ui::sanitize(&e.title));
    }
}

/// Sinyal berhenti: matikan grup proses `bash` yang berjalan, pulihkan terminal, keluar.
/// Diblokir sebelum thread lain dibuat; std mengosongkan mask untuk proses anak.
fn on_signal(cancel: Arc<Cancel>, interactive: bool) {
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
        cancel.cancel();
        if interactive {
            ui::restore();
        }
        std::process::exit(128 + sig);
    });
}

/// Jawaban ke stdout, jejak tool ke stderr.
fn print_mode(rx: std::sync::mpsc::Receiver<Ev>, replay: bool) -> i32 {
    let mut code = 0;
    let mut nl = true;
    let mut old = replay;
    let out = |s: &str| {
        let mut o = std::io::stdout();
        if o.write_all(s.as_bytes()).and_then(|_| o.flush()).is_err() {
            std::process::exit(1);
        }
    };
    for e in rx {
        match e {
            // Status pertama menandai akhir putar ulang riwayat.
            Ev::Status { .. } => old = false,
            Ev::Text(_) if old => {}
            Ev::Text(s) => {
                nl = s.ends_with('\n');
                out(&ui::sanitize(&s));
            }
            Ev::Tool { title, detail } => {
                if !nl {
                    out("\n");
                    nl = true;
                }
                eprintln!("● {}", ui::sanitize(&title));
                for (c, _, l) in detail.iter().filter(|d| d.0 == '$') {
                    eprintln!("  {c} {}", ui::sanitize(l));
                }
            }
            Ev::ToolDone { summary, .. } => eprintln!("  └ {}", ui::sanitize(&summary)),
            Ev::Info(s) => eprintln!("{}", ui::sanitize(&s)),
            Ev::Error(s) => {
                eprintln!("Error: {}", ui::sanitize(&s));
                code = 1;
            }
            Ev::Done => break,
            _ => {}
        }
    }
    if !nl {
        out("\n");
    }
    code
}

fn die(e: &str) -> ! {
    eprintln!("llmy: {e}");
    std::process::exit(2);
}
