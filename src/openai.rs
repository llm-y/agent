//! Chat Completions ala OpenAI, juga untuk server yang kompatibel.

use crate::cancel::Cancel;
use crate::http::{self, Fail, Req};
use crate::provider::{Delta, Mode, Provider, Reply, sse};
use serde_json::{Value, json};

/// Riwayat netral -> format OpenAI. Hasil tool dipasangkan menurut urutan.
fn convert(msgs: &[Value]) -> Vec<Value> {
    let mut out = vec![];
    let mut wait: std::collections::VecDeque<String> = Default::default();
    for (i, m) in msgs.iter().enumerate() {
        match m["role"].as_str() {
            Some("assistant") => {
                let mut o = json!({"role":"assistant","content":m["content"]});
                let calls: Vec<Value> = m["tool_calls"].as_array().into_iter().flatten().enumerate().map(|(j, c)| {
                    let id = format!("call_{i}_{j}");
                    wait.push_back(id.clone());
                    let f = &c["function"];
                    let args = match &f["arguments"] {
                        Value::String(s) => s.clone(),
                        Value::Null => "{}".into(),
                        v => v.to_string(),
                    };
                    json!({"id":id,"type":"function","function":{"name":f["name"],"arguments":args}})
                }).collect();
                if !calls.is_empty() {
                    o["tool_calls"] = json!(calls);
                } else if m["content"].as_str().is_none_or(str::is_empty) {
                    continue;
                }
                out.push(o);
            }
            Some("tool") => {
                if let Some(id) = wait.pop_front() {
                    out.push(json!({"role":"tool","tool_call_id":id,"content":m["content"]}));
                }
            }
            _ => out.push(json!({"role":m["role"],"content":m["content"]})),
        }
    }
    out
}

pub fn chat(p: &Provider, msgs: &[Value], tools: &Value, mode: Mode, cancel: &Cancel, on: &mut dyn FnMut(Delta)) -> Result<Reply, Fail> {
    let mut body = json!({
        "model": p.model, "messages": convert(msgs), "stream": true,
        "stream_options": {"include_usage": true},
    });
    if mode == Mode::Turn {
        body["tools"] = tools.clone();
    }
    if let Some(e) = &p.effort {
        body["reasoning_effort"] = json!(e);
    }
    let mut out = Reply::default();
    let mut calls: Vec<(String, String)> = vec![];
    let mut finish = String::new();
    let mut plain = String::new();
    let headers = p.headers()?;
    let req = Req { base: &p.base, method: "POST", path: "/chat/completions", headers: &headers, body: &body.to_string() };
    http::request(&req, cancel, &mut |l| {
        let Some(d) = sse(l) else {
            // Server yang mengabaikan `stream` membalas JSON biasa; simpan untuk pesan error.
            if plain.len() < 500 {
                plain.push_str(l);
            }
            return Ok(());
        };
        let v: Value = serde_json::from_str(d).map_err(|e| format!("JSON model: {e}"))?;
        if !v["error"].is_null() {
            return Err(v["error"]["message"].as_str().unwrap_or("kesalahan server").to_string());
        }
        if let Some(n) = v["usage"]["total_tokens"].as_u64() {
            out.tokens = n;
        }
        let c = &v["choices"][0];
        let m = &c["delta"];
        for k in ["reasoning_content", "reasoning"] {
            if let Some(s) = m[k].as_str().filter(|s| !s.is_empty()) {
                out.thinking.push_str(s);
                on(Delta::Think);
            }
        }
        if let Some(s) = m["content"].as_str().filter(|s| !s.is_empty()) {
            out.content.push_str(s);
            on(Delta::Text(s));
        }
        for (pos, t) in m["tool_calls"].as_array().into_iter().flatten().enumerate() {
            // Sebagian server tidak mengirim index.
            let i = t["index"].as_u64().map_or_else(|| calls.len().saturating_sub(1).max(pos), |i| i as usize);
            if i > 128 {
                return Err("terlalu banyak tool call".into());
            }
            if calls.len() <= i {
                calls.resize(i + 1, Default::default());
            }
            calls[i].0.push_str(t["function"]["name"].as_str().unwrap_or(""));
            calls[i].1.push_str(t["function"]["arguments"].as_str().unwrap_or(""));
        }
        if let Some(f) = c["finish_reason"].as_str() {
            finish = f.into();
        }
        Ok(())
    })?;
    if finish.is_empty() {
        let why = if plain.is_empty() { "koneksi terputus sebelum jawaban selesai".into() } else { format!("balasan tidak dikenal: {plain}") };
        return Err(Fail::from(why));
    }
    out.done_reason = match finish.as_str() {
        "length" => "length",
        "content_filter" => "refusal",
        _ => "stop",
    }.into();
    // thinking bukan bagian riwayat di format ini
    out.thinking.clear();
    out.calls = calls.into_iter().filter(|c| !c.0.is_empty()).map(|(name, args)| {
        // JSON tak sah dibiarkan sebagai teks; pemanggil menolaknya sebagai argumen salah.
        let a = if args.trim().is_empty() { json!({}) } else { serde_json::from_str(&args).unwrap_or(Value::String(args)) };
        json!({"function":{"name":name,"arguments":a}})
    }).collect();
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn konversi_memasangkan_tool() {
        let m = [
            json!({"role":"system","content":"s"}),
            json!({"role":"user","content":"u"}),
            json!({"role":"assistant","content":"","blocks":[1],"tool_calls":[
                {"id":"x","function":{"name":"read","arguments":{"path":"a"}}},{"function":{"name":"bash","arguments":"{}"}}]}),
            json!({"role":"tool","content":"satu"}),
            json!({"role":"tool","content":"dua"}),
            json!({"role":"assistant","content":""}),
        ];
        let o = convert(&m);
        assert_eq!(o.len(), 5);
        assert_eq!(o[2]["tool_calls"][0]["function"]["arguments"], "{\"path\":\"a\"}");
        assert_eq!(o[2]["tool_calls"][1]["id"], o[4]["tool_call_id"]);
        assert_eq!(o[3]["content"], "satu");
        assert!(o[2]["blocks"].is_null());
    }
}
