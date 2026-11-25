//! Ollama native (/api/chat). Sengaja bukan /v1: template model lama bisa salah
//! mengurutkan pesan tanpa error.

use crate::cancel::Cancel;
use crate::http::{self, Fail, Req};
use crate::provider::{Delta, Mode, Provider, Reply};
use serde_json::{Value, json};

pub fn chat(p: &Provider, msgs: &[Value], tools: &Value, mode: Mode, cancel: &Cancel, on: &mut dyn FnMut(Delta)) -> Result<Reply, Fail> {
    let think = p.think && mode == Mode::Turn;
    match once(p, msgs, tools, mode, Some(think), cancel, on) {
        // Model tanpa thinking menolak field `think`; penolakan datang sebelum token apa pun.
        Err(e) if e.msg.contains("does not support thinking") => once(p, msgs, tools, mode, None, cancel, on),
        r => r,
    }
}

fn once(
    p: &Provider, msgs: &[Value], tools: &Value, mode: Mode, think: Option<bool>, cancel: &Cancel, on: &mut dyn FnMut(Delta),
) -> Result<Reply, Fail> {
    let msgs: Vec<Value> = msgs.iter().map(|m| {
        let mut m = m.clone();
        if let Some(o) = m.as_object_mut() {
            o.shift_remove("blocks");
            o.shift_remove("parts");
            o.shift_remove("reasoning");
            // Argumen yang bukan JSON sah (dari provider lain) ditolak Ollama.
            for c in o.get_mut("tool_calls").and_then(Value::as_array_mut).into_iter().flatten() {
                if !c["function"]["arguments"].is_object() {
                    c["function"]["arguments"] = json!({});
                }
            }
        }
        m
    }).collect();
    let mut body = json!({"model": p.model, "messages": msgs, "stream": true, "options": {"num_ctx": p.ctx}});
    if mode == Mode::Turn && tools.is_array() {
        body["tools"] = tools.clone();
    }
    if mode == Mode::Summary {
        body["options"]["num_predict"] = json!(1200);
    }
    if let Some(t) = think {
        body["think"] = json!(t);
    }
    let mut out = Reply::default();
    let mut done = false;
    let headers = p.headers()?;
    let req = Req { base: &p.base, method: "POST", path: "/api/chat", headers: &headers, body: &body.to_string() };
    http::request(&req, cancel, &mut |l| {
        let v: Value = serde_json::from_str(l).map_err(|e| format!("JSON model: {e}"))?;
        if let Some(e) = v["error"].as_str() {
            return Err(e.to_string());
        }
        let m = &v["message"];
        if let Some(s) = m["thinking"].as_str().filter(|s| !s.is_empty()) {
            out.thinking.push_str(s);
            on(Delta::Think);
        }
        if let Some(s) = m["content"].as_str().filter(|s| !s.is_empty()) {
            out.content.push_str(s);
            on(Delta::Text(s));
        }
        if let Some(c) = m["tool_calls"].as_array() {
            out.calls.extend(c.iter().cloned());
        }
        if v["done"].as_bool() == Some(true) {
            done = true;
            out.done_reason = v["done_reason"].as_str().unwrap_or("").into();
            out.tokens = v["prompt_eval_count"].as_u64().unwrap_or(0) + v["eval_count"].as_u64().unwrap_or(0);
        }
        Ok(())
    })?;
    if !done {
        return Err("koneksi terputus sebelum jawaban selesai".to_string().into());
    }
    Ok(out)
}
