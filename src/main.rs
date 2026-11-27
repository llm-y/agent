mod agent;
mod anthropic;
mod api;
mod bdi;
mod cancel;
mod detect;
mod editor;
mod gemini;
mod grep;
mod http;
mod memory;
mod metrics;
mod ollama;
mod openai;
mod perm;
mod provider;
mod rules;
mod session;
mod setup;
mod sys;
mod text;
mod tools;
mod track;
mod ui;
mod verify;
mod web;

use agent::{Agent, Cmd, Ev};
use cancel::Cancel;
use detect::Tools;
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
  llmy api [perintah]      atur API beberapa provider (set, list, use, remove)
  llmy stats               ringkasan metrik lokal per giliran (LLMY_METRICS=off mematikan)
  llmy setup               pasang provider gratis lewat menu: Groq, Cerebras, OpenRouter (:free),
                           Cohere, NVIDIA NIM, Cloudflare Workers AI, Z.ai, Gemini, Kilo

Opsi:
  -m, --model <nama>   model (env LLMY_MODEL, bawaan qwen3.5:9b atau profil bawaan)
                         <nama>            Ollama            (alamat: LLMY_URL)
                         anthropic/<nama>  Claude            (kunci: ANTHROPIC_API_KEY)
                         openai/<nama>     OpenAI/kompatibel (kunci: OPENAI_API_KEY,
                                           alamat: OPENAI_BASE_URL)
                         gemini/<nama>     Google Gemini     (kunci: GEMINI_API_KEY,
                                           alamat: GEMINI_BASE_URL)
                         <profil>/<nama>   profil tersimpan dari `llmy api`
      --url <url>      alamat untuk model yang dipilih
      --api-key <key>  kunci API hanya untuk sesi ini (tidak disimpan)
      --ctx <n>        ukuran context (env LLMY_CTX; bawaan 32768, cloud sampai 200000)
      --tools <cara>   auto (bawaan), native, atau text: tool lewat teks, untuk server
                       tanpa tool calling. auto mencoba native dulu (env LLMY_TOOLS)
      --think          tampilkan/aktifkan thinking
      --effort <e>     usaha model cloud: low, medium, high, xhigh, max
  -c, --continue       lanjutkan sesi terakhir di folder ini
  -r, --resume <id>    lanjutkan sesi tertentu
      --sessions       daftar sesi di folder ini
      --no-session     jangan simpan sesi
      --allow <daftar> tanpa tanya: edit, bash, memory, web, atau all (pisahkan dengan koma)
      --verify <cmd>   periksa setelah edit dengan perintah ini (env LLMY_VERIFY); off mematikan.
                       Bawaan: cargo check, go build/vet, tsc, atau py_compile menurut proyek
                       (ditanya seperti bash, kecuali --auto atau --allow bash)
      --mode <alur>    loop (bawaan): satu loop, satu model. bdi: perencana memecah tugas,
                       tiap tujuan dikerjakan pekerja dengan context baru, diverifikasi, dan
                       dikembalikan bila buntu (env LLMY_MODE; interaktif: /mode)
      --plan <model>   model perencana mode bdi, mis. groq/openai/gpt-oss-120b (env LLMY_PLAN)
      --fallback <daftar>  model cadangan, dipisah koma: dipakai bergiliran bila kuota habis
                       atau server bermasalah (env LLMY_FALLBACK)
      --sub <model>    model sub-agent (tool task), mis. model lokal yang cepat (env LLMY_SUB).
                       Bawaan: model utama
      --auto           mode auto: setujui semua tanpa tanya (sama dengan --allow all), kecuali
                       path di luar folder kerja, di .git, atau file rahasia. Interaktif:
                       /auto atau Shift+Tab untuk menyalakan/mematikan
  -p, --print <prompt> sekali jalan
  -h, --help / -V, --version

Instruksi dibaca dari AGENTS.md (atau CLAUDE.md), akar git sampai folder kerja.
Memori: bagian Memory di AGENTS.md proyek, atau ~/.config/llmy/AGENTS.md untuk semua proyek.
API tersimpan di ~/.config/llmy/api.json (Windows: %APPDATA%\\llmy), hanya bisa dibaca pemiliknya.
Sesi disimpan di ~/.local/state/llmy/sessions (Windows: %LOCALAPPDATA%\\llmy), hanya bisa dibaca pemiliknya.
Tool internet (curl, github, openalex, doi, spider) selalu ada; tavily, exa, linkup, firecrawl,
scopus, dan brightdata muncul bila kuncinya ada di env (TAVILY_API_KEY, EXA_API_KEY,
LINKUP_API_KEY, FIRECRAWL_API_KEY, SCOPUS_API_KEY, BRIGHTDATA_API_KEY + BRIGHTDATA_ZONE).
GITHUB_TOKEN opsional untuk github.
edit berlaku hanya di dalam folder kerja. bash berjalan dengan hak akses Anda.";

struct Opts {
    perm: Perm,
    prompt: Option<String>,
    /// Some(None) = sesi terakhir.
    resume: Option<Option<String>>,
    save: bool,
    list: bool,
    verify: Option<String>,
    flow: agent::Flow,
    plan: Option<String>,
    fallback: Vec<String>,
    sub: Option<String>,
}

fn default_model() -> String {
    let cfg = api::Config::load();
    if let Some(def) = cfg.default {
        if let Some(m) = cfg.providers.get(&def).and_then(|p| p.model.as_deref()) {
            return format!("{def}/{m}");
        }
        if def == "ollama" {
            return "qwen3.5:9b".into();
        }
        return def;
    }
    "qwen3.5:9b".into()
}

fn parse() -> Result<(Opts, impl FnOnce() -> Result<Provider, String>), String> {
    let cfg = api::Config::load();
    let mut o = Opts {
        perm: Perm::default(), prompt: None, resume: None, save: true, list: false, verify: None,
        flow: agent::Flow::Loop, plan: env("LLMY_PLAN").or(cfg.plan), fallback: cfg.fallback, sub: env("LLMY_SUB"),
    };
    let mut flow = env("LLMY_MODE").or(cfg.mode).unwrap_or("loop".into());
    if let Some(f) = env("LLMY_FALLBACK") {
        o.fallback = api::split_list(&f);
    }
    let mut model = env("LLMY_MODEL").unwrap_or_else(default_model);
    let mut ctx = env("LLMY_CTX");
    let mut tools = env("LLMY_TOOLS").unwrap_or("auto".into());
    o.verify = env("LLMY_VERIFY");
    let (mut url, mut api_key, mut think, mut effort) = (None, None, false, None);
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
            "--api-key" => api_key = Some(val()?),
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
            "--verify" => o.verify = Some(val()?),
            "--mode" => flow = val()?,
            "--plan" => o.plan = Some(val()?),
            "--sub" => o.sub = Some(val()?),
            "--fallback" => o.fallback = api::split_list(&val()?),
            "--auto" => o.perm.all.store(true, std::sync::atomic::Ordering::Relaxed),
            "--allow" => {
                for s in val()?.split(',') {
                    match s {
                        "edit" => o.perm.allow_edit = true,
                        "bash" => o.perm.allow_bash = true,
                        "memory" => o.perm.allow_memory = true,
                        "web" => o.perm.allow_web = true,
                        "all" => o.perm.all.store(true, std::sync::atomic::Ordering::Relaxed),
                        _ => return Err(format!("--allow tidak mengenal `{s}`")),
                    }
                }
            }
            _ => return Err(format!("opsi tidak dikenal: {a} (lihat --help)")),
        }
    }
    o.flow = agent::Flow::parse(&flow).ok_or(format!("--mode tidak mengenal `{flow}` (loop atau bdi)"))?;
    if o.plan.as_deref() == Some("off") {
        o.plan = None;
    }
    let ctx = ctx.map(|c| c.parse().ok().filter(|n| *n >= 4096).ok_or("ctx harus angka >= 4096")).transpose()?;
    let tools = Tools::parse(&tools).ok_or(format!("--tools tidak mengenal `{tools}` (auto, native, atau text)"))?;
    // Dibuat belakangan: `--sessions` tidak butuh kunci API.
    Ok((o, move || {
        let mut p = Provider::with_options(&model, url, api_key, ctx, Default::default())?;
        (p.think, p.effort) = (think, effort);
        p.set_tools(tools);
        Ok(p)
    }))
}

fn main() {
    let raw_args: Vec<String> = std::env::args().collect();
    let sub = |i: usize| raw_args.get(i).map(String::as_str);
    if sub(1) == Some("setup") || (sub(1) == Some("api") && sub(2) == Some("setup")) {
        std::process::exit(setup::cli());
    }
    if sub(1) == Some("stats") {
        std::process::exit(metrics::cli());
    }
    if raw_args.get(1).map(String::as_str) == Some("api") {
        std::process::exit(api::cli(&raw_args[2..]));
    }
    let (o, provider) = parse().unwrap_or_else(|e| die(&e));
    let cwd = std::env::current_dir().and_then(std::fs::canonicalize).unwrap_or_else(|e| die(&e.to_string()));
    let mut session = if o.save { session::root().map(|r| Session::new(&r, &cwd)) } else { None };
    if o.list {
        return list(session.as_ref());
    }
    let provider = provider().unwrap_or_else(|e| die(&e));
    let planner = o.plan.as_deref().map(|p| provider.switch(p).unwrap_or_else(|e| die(&format!("--plan: {e}"))));
    let subagent = o.sub.as_deref().map(|p| provider.switch(p).unwrap_or_else(|e| die(&format!("--sub: {e}"))));
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
    let auto = o.perm.all.clone();
    let mut agent = Agent::new(provider, o.perm, interactive, cwd.clone(), cancel.clone(), tx.clone())
        .with_verify(o.verify)
        .with_session(session, history);
    (agent.flow, agent.planner, agent.fallback, agent.subagent) = (o.flow, planner, o.fallback, subagent);
    let (ctx, crx) = channel();
    std::thread::spawn(move || agent.serve(crx));
    match o.prompt {
        None => {
            if let Err(e) = ui::run(cwd, cancel, tx, rx, ctx, auto) {
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
            Ev::Info(s) => {
                if !nl {
                    out("\n");
                    nl = true;
                }
                eprintln!("{}", ui::sanitize(&s));
            }
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
