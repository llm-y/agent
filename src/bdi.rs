//! Mode BDI (Belief-Desire-Intention). Perencana memecah tujuan menjadi beberapa desire; tiap
//! desire dikerjakan pekerja dengan context baru (satu intention); kode yang memeriksa hasilnya,
//! memutus intention yang buntu, mengembalikan file, dan mencatat pendekatan yang gagal.
//!
//! Model hanya dipakai untuk tiga hal sempit: dekomposisi (fase 0), deliberasi (fase 3), dan
//! eksekusi (fase 4). Belief, komitmen, dan rollback (fase 1-2) ditentukan kode.

use crate::agent::{self, Agent, End, Ev};
use crate::detect::Tools;
use crate::provider::{Delta, Mode, Provider};
use crate::rules;
use crate::tools;
use serde_json::{Value, json};
use std::collections::BTreeMap;
use std::hash::{DefaultHasher, Hash, Hasher};
use std::path::{Path, PathBuf};

const MAX_DESIRES: usize = 8;
/// Percobaan satu intention sebelum diputus dan file dikembalikan.
const MAX_ATTEMPTS: usize = 3;
/// Berapa kali satu desire boleh dideliberasi ulang setelah intention-nya diputus.
const MAX_TRIES: usize = 2;
const MAP_MAX: usize = 4_000;
const SLICE_LINES: usize = 150;

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Status {
    Pending,
    Achieved,
    /// Ditinggalkan setelah intention-nya diputus berkali-kali.
    Dropped,
}

#[derive(Clone, Debug)]
pub struct Desire {
    pub id: String,
    pub goal: String,
    pub files: Vec<String>,
    pub deps: Vec<String>,
    pub status: Status,
    tries: usize,
    /// Jawaban terakhir pekerja.
    report: String,
}

/// Fakta lingkungan, dari kode, bukan dari model.
#[derive(Default)]
pub struct Beliefs {
    pub map: String,
    pub dirty: Vec<String>,
    /// Galat verifikasi terakhir yang belum beres.
    pub errors: String,
    /// Pendekatan yang sudah buntu (diisi setelah rollback).
    pub negative: Vec<String>,
}

pub struct Intention {
    pub id: String,
    pub files: Vec<String>,
    pub steps: Vec<String>,
    attempts: usize,
    failed: Vec<u64>,
}

const PLAN_SYS: &str = "You are the planner of a coding agent. Split the user's goal into small subgoals in the order they must be done. Each subgoal changes at most one or two files and can be checked on its own. A goal that needs one small change, or only an answer or explanation, is ONE subgoal: do not split it. Use exact file paths from the project file list; a new file may be listed too. Copy names, signatures, and constraints from the goal exactly as written; do not reinterpret them. Do not write code. Reply with JSON only, no other text:\n{\"desires\":[{\"id\":\"D1\",\"goal\":\"what to achieve, one sentence\",\"files\":[\"path\"],\"depends_on\":[]}]}";

const DELIB_SYS: &str = "You are the deliberation step of a coding agent. Choose ONE subgoal to do next from the candidates. Use the list of failed approaches so the same dead end is not tried again. Give the files to change and two to five short steps. Do not write code. Reply with JSON only, no other text:\n{\"id\":\"D1\",\"files\":[\"path\"],\"steps\":[\"step\"]}";

const WORK_RULE: &str = "\n- You are a worker on ONE subgoal of a larger task, given in the user message. Do only that subgoal; the other subgoals are done separately by other workers. When it is done, stop calling tools and say in one or two sentences what you changed.";

impl Agent {
    pub fn bdi(&mut self, goal: String) {
        self.push(json!({"role":"user","content":goal}));
        let mut bel = Beliefs::default();
        self.perceive(&mut bel);
        // Fase 0: dekomposisi.
        let mut desires = match self.decompose(&goal, &bel) {
            Ok(d) => d,
            Err(e) if self.cancel.is_set() => return self.send(Ev::Info(format!("Dihentikan. {e}"))),
            Err(e) => {
                self.send(Ev::Info(format!("Perencana gagal ({e}); tujuan dikerjakan utuh.")));
                vec![single(&goal)]
            }
        };
        self.send(Ev::Info(plan_text(&desires)));
        let mut aborted: Option<String> = None;
        loop {
            if self.cancel.is_set() {
                aborted = Some("dihentikan".into());
                break;
            }
            let ready: Vec<usize> = (0..desires.len()).filter(|&i| {
                desires[i].status == Status::Pending
                    && desires[i].deps.iter().all(|d| desires.iter().any(|x| &x.id == d && x.status == Status::Achieved) || !desires.iter().any(|x| &x.id == d))
            }).collect();
            if ready.is_empty() {
                break;
            }
            self.perceive(&mut bel);
            // Fase 3: deliberasi. Satu kandidat tanpa kegagalan tidak perlu ditanyakan.
            let mut it = self.deliberate(&goal, &desires, &ready, &bel);
            let i = desires.iter().position(|d| d.id == it.id).unwrap_or(ready[0]);
            self.send(Ev::Info(format!("▶ {}: {}{}", desires[i].id, desires[i].goal,
                if it.steps.is_empty() { String::new() } else { format!("\n  {}", it.steps.join(" → ")) })));
            // Fase 4 berulang, dengan fase 1-2 di antaranya.
            self.snap = Some(BTreeMap::new());
            let mut note = String::new();
            let verdict = loop {
                let (end, report) = self.work(&goal, &desires, i, &it, &bel, &note);
                desires[i].report = report;
                let changed = self.changed();
                match end {
                    End::Stop => break Verdict::Abort("dihentikan atau izin ditolak".into()),
                    End::Error(e) => break Verdict::Abort(e),
                    End::Done if self.vmem.clean() => {
                        if changed.is_empty() && !desires[i].files.is_empty() && it.attempts == 0 {
                            it.attempts += 1;
                            note = "No file was changed. If this subgoal truly needs no change, say why in one sentence. Otherwise make the change now.".into();
                            continue;
                        }
                        break Verdict::Achieved;
                    }
                    end => {
                        it.attempts += 1;
                        let why = match end {
                            End::Stuck(e) => e,
                            _ => "checks still fail".into(),
                        };
                        let h = state_hash(&changed);
                        let repeat = it.failed.contains(&h);
                        it.failed.push(h);
                        bel.errors = self.vmem.errors.clone();
                        if repeat || it.attempts >= MAX_ATTEMPTS {
                            break Verdict::Drop(why);
                        }
                        self.send(Ev::Info(format!("↺ {}: belum berhasil ({}/{MAX_ATTEMPTS}); dicoba lagi dengan galatnya.", desires[i].id, it.attempts)));
                        note = format!("The previous attempt did not finish: {why}. Fix the remaining errors.");
                    }
                }
            };
            let d = &mut desires[i];
            match verdict {
                Verdict::Achieved => {
                    d.status = Status::Achieved;
                    bel.errors.clear();
                    self.snap = None;
                    self.send(Ev::Info(format!("✓ {} tercapai", d.id)));
                }
                Verdict::Drop(why) => {
                    let n = self.rollback();
                    d.tries += 1;
                    let errs: String = self.vmem.errors.lines().take(6).collect::<Vec<_>>().join(" | ");
                    bel.negative.push(format!("- {} ({}): {} {}", d.id, if it.steps.is_empty() { d.goal.clone() } else { it.steps.join("; ") },
                        why.chars().take(160).collect::<String>(), errs.chars().take(400).collect::<String>()));
                    self.vmem.rollback();
                    bel.errors.clear();
                    if d.tries >= MAX_TRIES {
                        d.status = Status::Dropped;
                    }
                    self.send(Ev::Info(format!("✗ {}: intention diputus, {n} file dikembalikan{}", d.id,
                        if d.status == Status::Dropped { "; tujuan ini ditinggalkan" } else { "; direncanakan ulang" })));
                }
                Verdict::Abort(e) => {
                    self.snap = None;
                    aborted = Some(e);
                    break;
                }
            }
        }
        self.snap = None;
        self.finish(&desires, aborted);
    }

    /// Fase 1: perbarui belief dari lingkungan.
    fn perceive(&self, bel: &mut Beliefs) {
        bel.map = repo_map(&self.cwd);
        bel.dirty = git(&self.cwd, &["status", "--porcelain"]).lines().take(30).map(|l| l.trim().to_string()).collect();
    }

    fn planner(&self) -> Provider {
        let mut p = self.planner.clone().unwrap_or_else(|| self.provider.clone());
        (p.think, p.temperature, p.json) = (false, Some(0.1), true);
        // Tanpa tool: tak perlu deteksi tool calling.
        p.set_tools(Tools::Native);
        p
    }

    /// Satu panggilan JSON ke perencana; diulang sekali bila jawabannya bukan JSON.
    /// Perencana yang kehabisan kuota digantikan model utama.
    fn ask_json(&mut self, sys: &str, user: String, work: &'static str) -> Result<Value, String> {
        let mut p = self.planner();
        let mut msgs = vec![json!({"role":"system","content":sys}), json!({"role":"user","content":user})];
        let mut swapped = false;
        for attempt in 0..3 {
            let tx = self.tx.clone();
            let r = p.chat(&msgs, &Value::Null, Mode::Summary, &self.cancel, &mut |d| {
                if let Delta::Work(_) | Delta::Think = d {
                    let _ = tx.send(Ev::Work(work));
                }
            });
            match r {
                Ok(r) => {
                    if let Some(v) = parse_json(&r.content) {
                        return Ok(v);
                    }
                    if attempt > 0 {
                        return Err("jawaban perencana bukan JSON".into());
                    }
                    msgs.push(json!({"role":"assistant","content":r.content}));
                    msgs.push(json!({"role":"user","content":"That was not valid JSON. Reply with the JSON object only."}));
                }
                // Perencana terpisah yang gagal digantikan model utama; model utama yang
                // kehabisan kuota digantikan cadangannya.
                Err(e) if !swapped && self.planner.is_some() && !self.cancel.is_set() => {
                    self.send(Ev::Info(format!("Perencana {} gagal ({}); pakai {}.", p.label(), e.chars().take(120).collect::<String>(), self.provider.label())));
                    swapped = true;
                    let keep = std::mem::take(&mut self.planner);
                    p = self.planner();
                    self.planner = keep;
                }
                Err(e) if p.exhausted() && !self.cancel.is_set() && self.use_fallback(&e) => {
                    swapped = true;
                    let keep = std::mem::take(&mut self.planner);
                    p = self.planner();
                    self.planner = keep;
                }
                Err(e) => return Err(e),
            }
        }
        Err("jawaban perencana bukan JSON".into())
    }

    fn decompose(&mut self, goal: &str, bel: &Beliefs) -> Result<Vec<Desire>, String> {
        self.send(Ev::Work("Merencanakan"));
        let rules = rules::load(&self.cwd).text;
        let rules: String = rules.chars().take(2_000).collect();
        let user = format!(
            "Goal:\n{goal}\n\nEarlier conversation:\n{}\n\nProject files:\n{}\n\nUncommitted changes:\n{}\n\nProject instructions:\n{}",
            self.recent(), bel.map, or_none(&bel.dirty.join("\n")), or_none(&rules)
        );
        let v = self.ask_json(PLAN_SYS, user, "Merencanakan")?;
        let list = v["desires"].as_array().or(v.as_array()).ok_or("tidak ada daftar desires")?;
        let mut out: Vec<Desire> = vec![];
        for (n, d) in list.iter().take(MAX_DESIRES).enumerate() {
            let goal = d["goal"].as_str().or(d["description"].as_str()).unwrap_or("").trim().to_string();
            if goal.is_empty() {
                continue;
            }
            let id = d["id"].as_str().filter(|s| !s.trim().is_empty()).map_or(format!("D{}", n + 1), |s| s.trim().to_string());
            if out.iter().any(|x| x.id == id) {
                continue;
            }
            out.push(Desire { id, goal, files: strings(&d["files"]), deps: strings(&d["depends_on"]), status: Status::Pending, tries: 0, report: String::new() });
        }
        if out.is_empty() {
            return Err("rencana kosong".into());
        }
        // Satu tujuan = permintaan itu sendiri, kata per kata.
        if out.len() == 1 {
            out[0].goal = goal.trim().to_string();
        }
        // Ketergantungan hanya ke desire sebelumnya: tidak ada siklus.
        for i in 0..out.len() {
            let before: Vec<String> = out[..i].iter().map(|d| d.id.clone()).collect();
            out[i].deps.retain(|d| before.contains(d));
        }
        Ok(out)
    }

    fn deliberate(&mut self, goal: &str, desires: &[Desire], ready: &[usize], bel: &Beliefs) -> Intention {
        let first = &desires[ready[0]];
        let plain = Intention { id: first.id.clone(), files: first.files.clone(), steps: vec![], attempts: 0, failed: vec![] };
        if ready.len() == 1 && bel.negative.is_empty() {
            return plain;
        }
        let cand: Vec<String> = ready.iter().map(|&i| {
            let d = &desires[i];
            format!("- {}: {} (files: {})", d.id, d.goal, d.files.join(", "))
        }).collect();
        let user = format!(
            "Goal:\n{goal}\n\nDone:\n{}\n\nCandidates:\n{}\n\nFailed approaches (do not repeat):\n{}\n\nProject files:\n{}",
            or_none(&done(desires)), cand.join("\n"), or_none(&bel.negative.join("\n")), bel.map
        );
        match self.ask_json(DELIB_SYS, user, "Menimbang") {
            Ok(v) => {
                let id = v["id"].as_str().or(v["selected_desire_id"].as_str()).unwrap_or("").trim().to_string();
                let Some(&i) = ready.iter().find(|&&i| desires[i].id == id) else { return plain };
                let files = strings(&v["files"]);
                Intention {
                    id,
                    files: if files.is_empty() { desires[i].files.clone() } else { files },
                    steps: strings(&v["steps"]).into_iter().take(6).collect(),
                    attempts: 0,
                    failed: vec![],
                }
            }
            Err(e) => {
                if !self.cancel.is_set() {
                    self.send(Ev::Info(format!("Deliberasi gagal ({e}); pakai urutan rencana.")));
                }
                plain
            }
        }
    }

    /// Fase 4: pekerja dengan context baru, tool dan izin yang sama. Riwayat pekerja tidak
    /// masuk sesi; yang masuk hanya ringkasan di akhir.
    fn work(&mut self, goal: &str, desires: &[Desire], i: usize, it: &Intention, bel: &Beliefs, note: &str) -> (End, String) {
        let d = &desires[i];
        // Permintaan pengguna tetap acuan: parafrase perencana bisa menghilangkan detail
        // (mis. `r: &Rect` menjadi "accept &Rect", lalu pekerja membuat method).
        let mut u = if desires.len() == 1 {
            format!("Task:\n{goal}\n")
        } else {
            format!("The user's request. Follow its exact names, signatures, and constraints:\n{goal}\n\nYour part now, subgoal {}: {}\n", d.id, d.goal)
        };
        if !it.files.is_empty() {
            u += &format!("Files: {}\n", it.files.join(", "));
        }
        if !it.steps.is_empty() {
            u += &format!("Plan:\n{}\n", it.steps.iter().enumerate().map(|(n, s)| format!("{}. {s}", n + 1)).collect::<Vec<_>>().join("\n"));
        }
        let fin = done(desires);
        if !fin.is_empty() {
            u += &format!("\nAlready done by other workers:\n{fin}\n");
        }
        if !bel.negative.is_empty() {
            u += &format!("\nApproaches that already failed and were rolled back; do not repeat them:\n{}\n", bel.negative.join("\n"));
        }
        if !bel.errors.is_empty() {
            u += &format!("\nCurrent check errors:\n{}\n", bel.errors);
        }
        if !note.is_empty() {
            u += &format!("\n{note}\n");
        }
        u += &self.slices(&it.files);
        let rules = rules::load(&self.cwd).text;
        let sys = agent::system_prompt(&self.cwd, &rules, &self.checks) + WORK_RULE;
        let main = std::mem::replace(&mut self.msgs, vec![json!({"role":"system","content":sys})]);
        let session = self.session.take();
        let (used, counted) = (self.used, self.counted);
        (self.used, self.counted, self.sub) = (0, 0, true);
        let mut wp = self.provider.clone();
        wp.temperature = Some(0.2);
        let keep = std::mem::replace(&mut self.provider, wp);
        let end = self.run(u);
        // Cadangan yang dipilih pekerja tetap berlaku.
        let mut p = std::mem::replace(&mut self.provider, keep);
        if p.label() != self.provider.label() {
            p.temperature = None;
            self.provider = p;
        }
        let report = self.msgs.iter().rev().find(|m| m["role"] == "assistant" && m["content"].as_str().is_some_and(|s| !s.trim().is_empty()))
            .and_then(|m| m["content"].as_str()).unwrap_or("").trim().to_string();
        self.msgs = main;
        self.session = session;
        (self.used, self.counted, self.sub) = (used, counted, false);
        self.send(Ev::EndText);
        (end, report)
    }

    /// Isi file target yang pendek, bernomor seperti `read`, agar pekerja tidak perlu membaca dulu.
    fn slices(&self, files: &[String]) -> String {
        let mut out = String::new();
        for f in files.iter().take(3) {
            let Ok(p) = tools::prepare("read", &json!({"path": f, "limit": SLICE_LINES}), &self.cwd) else { continue };
            if p.guarded {
                continue;
            }
            let o = tools::run(p, &self.cwd, self.provider.ctx, &self.cancel);
            if o.ok && o.text.len() < 12_000 {
                out += &format!("\nCurrent content of {f}:\n{}\n", o.text);
            }
        }
        out
    }

    /// File yang isinya berbeda dari awal intention.
    fn changed(&self) -> Vec<(PathBuf, Option<String>)> {
        self.snap.iter().flatten().filter_map(|(p, old)| {
            let now = std::fs::read_to_string(p).ok();
            (&now != old).then(|| (p.clone(), now))
        }).collect()
    }

    /// Kembalikan file ke isi sebelum intention. File baru dihapus.
    fn rollback(&mut self) -> usize {
        let snap = self.snap.take().unwrap_or_default();
        let mut n = 0;
        for (p, old) in snap {
            let now = std::fs::read_to_string(&p).ok();
            if now == old {
                continue;
            }
            let r = match &old {
                Some(s) => std::fs::write(&p, s),
                None => std::fs::remove_file(&p),
            };
            match r {
                Ok(()) => n += 1,
                Err(e) => self.send(Ev::Error(format!("Gagal mengembalikan {}: {e}", p.display()))),
            }
        }
        n
    }

    /// Pesan pengguna dan jawaban terakhir di percakapan utama, untuk perencana.
    fn recent(&self) -> String {
        let v: Vec<String> = self.msgs.iter().skip(1).filter(|m| matches!(m["role"].as_str(), Some("user" | "assistant")))
            .filter_map(|m| Some(format!("{}: {}", m["role"].as_str()?, m["content"].as_str()?.chars().take(400).collect::<String>())))
            .filter(|s| !s.ends_with(": "))
            .collect();
        let v = &v[v.len().saturating_sub(5)..v.len().saturating_sub(1)];
        or_none(&v.join("\n"))
    }

    fn finish(&mut self, desires: &[Desire], aborted: Option<String>) {
        let ok = desires.iter().filter(|d| d.status == Status::Achieved).count();
        let mut s = format!("BDI: {ok}/{} tujuan tercapai.\n", desires.len());
        for d in desires {
            let mark = match d.status {
                Status::Achieved => "✓",
                Status::Dropped => "✗",
                Status::Pending => "·",
            };
            s += &format!("{mark} {}: {}\n", d.id, d.goal);
        }
        self.send(Ev::Text(s.clone()));
        self.send(Ev::EndText);
        // Jawaban pekerja sudah tampil; riwayat utama butuh isinya untuk giliran berikutnya.
        let reports: Vec<String> = desires.iter().filter(|d| !d.report.is_empty()).map(|d| format!("{}: {}", d.id, d.report)).collect();
        if !reports.is_empty() {
            s += &format!("\n{}", reports.join("\n"));
        }
        self.push(json!({"role":"assistant","content":s}));
        match aborted {
            Some(e) if self.cancel.is_set() => self.send(Ev::Info(format!("Dihentikan ({e})."))),
            Some(e) => self.send(Ev::Error(format!("BDI berhenti: {e}"))),
            None if ok < desires.len() => self.send(Ev::Error(format!("{} tujuan tidak tercapai.", desires.len() - ok))),
            None => {}
        }
    }
}

enum Verdict {
    Achieved,
    Drop(String),
    Abort(String),
}

fn single(goal: &str) -> Desire {
    Desire { id: "D1".into(), goal: goal.into(), files: vec![], deps: vec![], status: Status::Pending, tries: 0, report: String::new() }
}

fn plan_text(d: &[Desire]) -> String {
    let mut s = format!("Rencana: {} tujuan", d.len());
    for x in d {
        s += &format!("\n  {}: {}", x.id, x.goal);
        if !x.files.is_empty() {
            s += &format!(" [{}]", x.files.join(", "));
        }
        if !x.deps.is_empty() {
            s += &format!(" (setelah {})", x.deps.join(", "));
        }
    }
    s
}

fn done(desires: &[Desire]) -> String {
    desires.iter().filter(|d| d.status == Status::Achieved).map(|d| {
        let r: String = d.report.chars().take(200).collect();
        format!("- {}: {}{}", d.id, d.goal, if r.is_empty() { String::new() } else { format!(" ({r})") })
    }).collect::<Vec<_>>().join("\n")
}

fn or_none(s: &str) -> String {
    if s.trim().is_empty() { "(none)".into() } else { s.into() }
}

fn strings(v: &Value) -> Vec<String> {
    match v {
        Value::Array(a) => a.iter().filter_map(|x| x.as_str().map(|s| s.trim().to_string())).filter(|s| !s.is_empty()).collect(),
        Value::String(s) if !s.trim().is_empty() => vec![s.trim().to_string()],
        _ => vec![],
    }
}

/// JSON dari jawaban model: bisa dibungkus ``` atau diberi teks di sekitarnya.
pub fn parse_json(s: &str) -> Option<Value> {
    let s = s.trim();
    if let Ok(v) = serde_json::from_str::<Value>(s) {
        return v.is_object().then_some(v);
    }
    let (a, b) = (s.find('{')?, s.rfind('}')?);
    if b <= a {
        return None;
    }
    serde_json::from_str::<Value>(&s[a..=b]).ok().filter(Value::is_object)
}

fn state_hash(changed: &[(PathBuf, Option<String>)]) -> u64 {
    let mut h = DefaultHasher::new();
    changed.hash(&mut h);
    h.finish()
}

fn git(cwd: &Path, args: &[&str]) -> String {
    std::process::Command::new("git").args(args).current_dir(cwd).stdin(std::process::Stdio::null())
        .stderr(std::process::Stdio::null()).output().ok().filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).into_owned()).unwrap_or_default()
}

/// Daftar file proyek dengan nama simbol tingkat atas; dari `git ls-files`, atau berjalan
/// sendiri tanpa folder tersembunyi dan folder build.
pub fn repo_map(cwd: &Path) -> String {
    let mut files: Vec<String> = git(cwd, &["ls-files", "--cached", "--others", "--exclude-standard"]).lines().map(String::from).collect();
    if files.is_empty() {
        walk(cwd, cwd, &mut files, 0);
    }
    files.sort();
    files.truncate(400);
    let re = regex_lite::Regex::new(r"^\s*(?:pub(?:\([a-z]+\))?\s+)?(?:export\s+)?(?:async\s+)?(?:fn|struct|enum|trait|impl|class|def|func|function|interface|type)\s+([A-Za-z_][A-Za-z0-9_]*)").unwrap();
    let mut out = String::new();
    for f in &files {
        let mut line = f.clone();
        let src = matches!(Path::new(f).extension().and_then(|e| e.to_str()), Some("rs" | "go" | "py" | "js" | "ts" | "tsx" | "jsx" | "java" | "kt" | "c" | "h" | "cpp" | "cs" | "rb" | "php" | "swift"));
        if src
            && let Ok(t) = std::fs::read_to_string(cwd.join(f))
            && t.len() < 300_000
        {
            let mut names: Vec<&str> = t.lines().filter_map(|l| re.captures(l).and_then(|c| c.get(1)).map(|m| m.as_str())).collect();
            names.dedup();
            let more = names.len() > 12;
            names.truncate(12);
            if !names.is_empty() {
                line += &format!(": {}{}", names.join(", "), if more { ", ..." } else { "" });
            }
        }
        if out.len() + line.len() > MAP_MAX {
            out += &format!("... ({} file lagi)\n", files.len() - out.lines().count());
            break;
        }
        out += &line;
        out.push('\n');
    }
    or_none(&out)
}

fn walk(root: &Path, dir: &Path, out: &mut Vec<String>, depth: usize) {
    let Ok(rd) = std::fs::read_dir(dir) else { return };
    for e in rd.flatten() {
        if out.len() >= 400 {
            return;
        }
        let name = e.file_name().to_string_lossy().into_owned();
        if name.starts_with('.') || matches!(name.as_str(), "target" | "node_modules" | "dist" | "build" | "__pycache__" | "vendor") {
            continue;
        }
        let p = e.path();
        if e.file_type().is_ok_and(|t| t.is_dir()) {
            if depth < 6 {
                walk(root, &p, out, depth + 1);
            }
        } else if let Ok(r) = p.strip_prefix(root) {
            out.push(r.to_string_lossy().replace('\\', "/"));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn json_dari_jawaban_model() {
        assert_eq!(parse_json("{\"a\":1}").unwrap()["a"], 1);
        assert_eq!(parse_json("```json\n{\"desires\":[]}\n```").unwrap()["desires"], json!([]));
        assert_eq!(parse_json("Here is the plan: {\"id\":\"D2\"} done").unwrap()["id"], "D2");
        assert!(parse_json("tidak ada").is_none());
        assert!(parse_json("[1,2]").is_none());
    }

    #[test]
    fn peta_proyek_berisi_simbol() {
        let d = Path::new(env!("CARGO_MANIFEST_DIR")).join("target/test/bdi-peta");
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(d.join("src")).unwrap();
        std::fs::create_dir_all(d.join("target")).unwrap();
        std::fs::write(d.join("src/lib.rs"), "pub struct Rect;\npub fn area(r: &Rect) -> u32 { 0 }\nfn util() {}\n").unwrap();
        std::fs::write(d.join("target/x.rs"), "fn hidden() {}").unwrap();
        let m = repo_map(&d);
        assert!(m.contains("src/lib.rs: Rect, area, util"), "{m}");
        assert!(!m.contains("hidden"), "{m}");
    }

    #[test]
    fn ketergantungan_hanya_ke_belakang() {
        let s = strings(&json!(["D1", " ", 3, "D2"]));
        assert_eq!(s, vec!["D1", "D2"]);
        assert_eq!(strings(&json!("src/a.rs")), vec!["src/a.rs"]);
    }
}
