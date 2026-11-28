//! Metrik lokal per giliran (`metrics.jsonl` di folder state llmy): angka saja, tanpa isi
//! percakapan, nama file, atau path. Untuk melihat pola kegagalan pemakaian nyata.
//! `LLMY_METRICS=off` mematikan; `llmy stats` meringkas.

use crate::sys;
use serde_json::{Value, json};
use std::io::Write;
use std::path::PathBuf;
use std::time::Instant;

#[derive(Default)]
pub struct Turn {
    pub steps: u32,
    pub tool_fail: u32,
    pub check_pass: u32,
    pub check_fail: u32,
    pub gate_pass: u32,
    pub gate_fail: u32,
    pub subagents: u32,
    pub repaired_args: u32,
    pub loose_edits: u32,
    /// Peringatan penjaga antarmuka dan siklus panggilan (E13).
    pub iface_warn: u32,
    pub iface_block: u32,
    pub cycles: u32,
    /// Kembali ke snapshot git dengan uji lulus terbanyak.
    pub restored: u32,
    /// Jumlah panggilan model dan total token yang diproses (masukan + keluaran, semua panggilan).
    pub calls: u32,
    pub tokens_total: u64,
    pub start: Option<Instant>,
}

pub fn file() -> Option<PathBuf> {
    sys::state_dir().map(|d| d.join("metrics.jsonl"))
}

pub fn enabled() -> bool {
    crate::provider::env("LLMY_METRICS").is_none_or(|v| v != "off")
}

/// Tulis satu baris. Gagal menulis tidak mengganggu kerja.
pub fn write(t: &Turn, model: &str, flow: &str, end: &str, tokens: u64) {
    if !enabled() {
        return;
    }
    let Some(f) = file() else { return };
    if let Some(d) = f.parent() {
        let _ = sys::private_dir(d);
    }
    let secs = t.start.map_or(0.0, |s| s.elapsed().as_secs_f64());
    let v = json!({
        "t": std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_secs()),
        "v": crate::VERSION, "model": model, "flow": flow, "end": end, "secs": (secs * 10.0).round() / 10.0, "tokens": tokens,
        "steps": t.steps, "tool_fail": t.tool_fail, "check_pass": t.check_pass, "check_fail": t.check_fail,
        "gate_pass": t.gate_pass, "gate_fail": t.gate_fail, "subagents": t.subagents,
        "repaired_args": t.repaired_args, "loose_edits": t.loose_edits, "iface_warn": t.iface_warn, "iface_block": t.iface_block, "cycles": t.cycles, "restored": t.restored, "calls": t.calls, "tokens_total": t.tokens_total,
    });
    let open = std::fs::OpenOptions::new().create(true).append(true).open(&f);
    if let Ok(mut h) = open {
        let _ = writeln!(h, "{v}");
    }
}

/// `llmy stats`: ringkasan per model.
pub fn cli() -> i32 {
    let Some(f) = file() else {
        eprintln!("llmy stats: folder state tidak ditemukan");
        return 2;
    };
    let rows: Vec<Value> = std::fs::read_to_string(&f).unwrap_or_default().lines().filter_map(|l| serde_json::from_str(l).ok()).collect();
    if rows.is_empty() {
        println!("Belum ada metrik ({}).{}", f.display(), if enabled() { "" } else { " LLMY_METRICS=off." });
        return 0;
    }
    let mut models: Vec<String> = rows.iter().filter_map(|r| r["model"].as_str().map(String::from)).collect();
    models.sort();
    models.dedup();
    println!("{:<34} {:>6} {:>7} {:>7} {:>9} {:>9} {:>8}", "model", "giliran", "selesai", "langkah", "tool gagal", "cek gagal", "uji gagal");
    for m in models {
        let r: Vec<&Value> = rows.iter().filter(|r| r["model"] == m.as_str()).collect();
        let n = r.len() as f64;
        let sum = |k: &str| r.iter().map(|x| x[k].as_f64().unwrap_or(0.0)).sum::<f64>();
        let done = r.iter().filter(|x| x["end"] == "Done").count();
        println!("{:<34} {:>6} {:>6.0}% {:>7.1} {:>9.1} {:>9.1} {:>8.1}", m.chars().take(34).collect::<String>(), r.len(),
            100.0 * done as f64 / n, sum("steps") / n, sum("tool_fail") / n, sum("check_fail") / n, sum("gate_fail") / n);
    }
    println!("\n{} giliran tercatat di {}", rows.len(), f.display());
    0
}
