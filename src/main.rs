mod agent;
mod anthropic;
mod cancel;
mod editor;
mod grep;
mod http;
mod ollama;
mod openai;
mod perm;
mod provider;
mod rules;
mod session;
mod sys;
mod text;
mod tools;
mod ui;

use agent::{Agent, Cmd, Ev};
use cancel::Cancel;
use perm::Perm;
use provider::{Provider, env};
use session::Session;
use std::io::{IsTerminal, Read, Write};
use std::sync::Arc;
use std::sync::mpsc::channel;

/// Diisi workflow rilis dari tag; build lokal memakai versi di Cargo.toml.
pub const VERSION: &str = match option_env!("LLMY_VERSION") {
    Some(v) => v,
    None => env!("CARGO_PKG_VERSION"),
};

const HELP: &str = "llmy - agent AI CLI yang ringan

Pakai:
  llmy [opsi]              mode interaktif
  llmy [opsi] -p <prompt>  sekali jalan (stdin ikut dibaca bila dipipa)

Opsi:
  -m, --model <nama>   model (env LLMY_MODEL, bawaan qwen3.5:9b)
                         <nama>            Ollama            (alamat: LLMY_URL)
                         anthropic/<nama>  Claude            (kunci: ANTHROPIC_API_KEY)
                         openai/<nama>     OpenAI/kompatibel (kunci: OPENAI_API_KEY,
                                           alamat: OPENAI_BASE_URL)
      --url <url>      alamat untuk model yang dipilih
      --ctx <n>        ukuran context (env LLMY_CTX; bawaan 32768, cloud sampai 200000)
      --tools <cara>   native (bawaan) atau text: tool lewat teks, untuk server
                       yang tidak meneruskan tool calling (env LLMY_TOOLS)
      --think          tampilkan/aktifkan thinking
      --effort <e>     usaha model cloud: low, medium, high, xhigh, max
  -c, --continue       lanjutkan sesi terakhir di folder ini
  -r, --resume <id>    lanjutkan sesi tertentu
      --sessions       daftar sesi di folder ini
      --no-session     jangan simpan sesi
      --allow <daftar> tanpa tanya: edit, bash, atau edit,bash
  -p, --print <prompt> sekali jalan
  -h, --help / -V, --version

Instruksi dibaca dari AGENTS.md (atau CLAUDE.md), akar git sampai folder kerja.
Sesi disimpan di ~/.local/state/llmy/sessions (Windows: %LOCALAPPDATA%\\llmy), hanya bisa dibaca pemiliknya.
edit berlaku hanya di dalam folder kerja. bash berjalan dengan hak akses Anda.";

struct Opts {
    perm: Perm,
    prompt: Option<String>,
    /// Some(None) = sesi terakhir.
    resume: Option<Option<String>>,
    save: bool,
    list: bool,
}

fn parse() -> Result<(Opts, impl FnOnce() -> Result<Provider, String>), String> {
    let mut o = Opts { perm: Perm::default(), prompt: None, resume: None, save: true, list: false };
    let mut model = env("LLMY_MODEL").unwrap_or("qwen3.5:9b".into());
    let mut ctx = env("LLMY_CTX");
    let mut tools = env("LLMY_TOOLS").unwrap_or("native".into());
    let (mut url, mut think, mut effort) = (None, false, None);
    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        let mut val = || it.next().ok_or(format!("{a} butuh nilai"));
        match a.as_str() {
            "-h" | "--help" => {
                println!("{HELP}");
                std::process::exit(0);
            }
            "-V" | "--version" => {
                println!("llmy {VERSION}");
                std::process::exit(0);
            }
            "-m" | "--model" => model = val()?,
            "--url" => url = Some(val()?),
            "--ctx" => ctx = Some(val()?),
            "-c" | "--continue" => o.resume = Some(None),
            "-r" | "--resume" => o.resume = Some(Some(val()?)),
            "--sessions" => o.list = true,
            "--no-session" => o.save = false,
            "--tools" => tools = val()?,
            "--think" => think = true,
            "--effort" => {
                let e = val()?;
                if !["low", "medium", "high", "xhigh", "max"].contains(&e.as_str()) {
                    return Err(format!("--effort tidak mengenal `{e}`"));
                }
                effort = Some(e);
            }
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
    let ctx = ctx.map(|c| c.parse().ok().filter(|n| *n >= 4096).ok_or("ctx harus angka >= 4096")).transpose()?;
    let text = match tools.as_str() {
        "native" => false,
        "text" => true,
        _ => return Err(format!("--tools tidak mengenal `{tools}` (native atau text)")),
    };
    // Dibuat belakangan: `--sessions` tidak butuh kunci API.
    Ok((o, move || {
        let mut p = Provider::new(&model, url, ctx)?;
        (p.think, p.effort, p.text_tools) = (think, effort, text);
        Ok(p)
    }))
}

fn main() {
    let (o, provider) = parse().unwrap_or_else(|e| die(&e));
    let cwd = std::env::current_dir().and_then(std::fs::canonicalize).unwrap_or_else(|e| die(&e.to_string()));
    let mut session = if o.save { session::root().map(|r| Session::new(&r, &cwd)) } else { None };
    if o.list {
        return list(session.as_ref());
    }
    let provider = provider().unwrap_or_else(|e| die(&e));
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
    let c = cancel.clone();
    sys::on_signal(move || {
        c.cancel();
        if interactive {
            ui::restore();
        }
    });
    let agent = Agent::new(provider, o.perm, interactive, cwd.clone(), cancel.clone(), tx.clone())
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
