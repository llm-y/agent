//! Sesi sebagai JSONL: satu baris per perubahan, jadi crash paling banyak
//! kehilangan baris terakhir. Pesan sistem tidak disimpan (dibuat ulang).
//!
//! Baris: {"cwd":..} kepala, {"m":pesan} tambah, {"set":[pesan]} ganti semua.

use serde_json::{Value, json};
use crate::sys;
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

pub struct Session {
    dir: PathBuf,
    cwd: PathBuf,
    file: Option<File>,
}

pub struct Entry {
    pub id: String,
    pub modified: SystemTime,
    pub title: String,
}

pub fn root() -> Option<PathBuf> {
    sys::state_dir().map(|p| p.join("sessions"))
}

/// Nama folder yang terbaca + hash FNV-1a agar path berbeda tidak bertabrakan.
fn slug(cwd: &Path) -> String {
    let s = cwd.to_string_lossy();
    let h = sys::fnv(s.as_bytes());
    let name = cwd.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
    let name: String = name.chars().filter(|c| c.is_ascii_alphanumeric() || *c == '-').take(40).collect();
    format!("{name}-{h:016x}")
}

impl Session {
    pub fn new(root: &Path, cwd: &Path) -> Session {
        Session { dir: root.join(slug(cwd)), cwd: cwd.into(), file: None }
    }

    /// Mulai file baru pada penulisan berikutnya.
    pub fn reset(&mut self) {
        // Lepas eksplisit: proses `bash` yang sedang di-fork bisa memegang salinan fd sesaat.
        if let Some(f) = self.file.take() {
            let _ = f.unlock();
        }
    }

    pub fn list(&self) -> Vec<Entry> {
        let mut v: Vec<Entry> = fs::read_dir(&self.dir).into_iter().flatten().flatten()
            .filter_map(|e| {
                let id = e.file_name().to_str()?.strip_suffix(".jsonl")?.to_string();
                let title = match fs::read(e.path()) {
                    Ok(raw) => load(&raw, &self.cwd)?.iter()
                        .find(|m| m["role"] == "user")
                        .and_then(|m| m["content"].as_str())
                        .map(|s| s.split_whitespace().collect::<Vec<_>>().join(" ").chars().take(60).collect())?,
                    // Di Windows kunci bersifat wajib: sesi yang aktif tak bisa dibaca.
                    Err(_) => "(sedang dipakai)".to_string(),
                };
                Some(Entry { id, modified: e.metadata().ok()?.modified().ok()?, title })
            })
            .collect();
        v.sort_by(|a, b| b.modified.cmp(&a.modified).then(b.id.cmp(&a.id)));
        v
    }

    /// Lanjutkan sesi `id`, atau yang terakhir dipakai di folder ini.
    pub fn resume(&mut self, id: Option<&str>) -> Result<Vec<Value>, String> {
        let id = match id {
            Some(i) if !i.is_empty() && i.bytes().all(|b| b.is_ascii_digit() || b == b'-') => i.to_string(),
            Some(i) => return Err(format!("id sesi tidak valid: {i}")),
            None => self.list().into_iter().next().ok_or("belum ada sesi di folder ini")?.id,
        };
        let path = self.dir.join(format!("{id}.jsonl"));
        let mut f = OpenOptions::new().read(true).append(true).open(&path)
            .map_err(|e| format!("sesi {id}: {e}"))?;
        lock(&f)?;
        let mut raw = vec![];
        f.read_to_end(&mut raw).map_err(|e| e.to_string())?;
        let mut msgs = load(&raw, &self.cwd).ok_or(format!("sesi {id} rusak atau milik folder lain"))?;
        if raw.last() != Some(&b'\n') {
            // Baris terakhir terpotong: pisahkan agar baris berikutnya tetap terbaca.
            f.seek(SeekFrom::End(0)).and_then(|_| f.write_all(b"\n")).map_err(|e| e.to_string())?;
        }
        self.file = Some(f);
        if repair(&mut msgs) {
            self.set(&msgs)?;
        }
        Ok(msgs)
    }

    fn open(&mut self) -> Result<&mut File, String> {
        if self.file.is_none() {
            let io = |e: std::io::Error| e.to_string();
            sys::private_dir(&self.dir).map_err(io)?;
            let ms = SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |d| d.as_millis());
            let path = self.dir.join(format!("{ms}-{}.jsonl", std::process::id()));
            let mut f = sys::private_file(&path).map_err(io)?;
            lock(&f)?;
            f.write_all(format!("{}\n", json!({"v":1,"cwd":self.cwd.to_string_lossy()})).as_bytes()).map_err(io)?;
            self.file = Some(f);
        }
        Ok(self.file.as_mut().unwrap())
    }

    fn line(&mut self, v: Value) -> Result<(), String> {
        self.open()?.write_all(format!("{v}\n").as_bytes()).map_err(|e| e.to_string())
    }

    pub fn push(&mut self, m: &Value) -> Result<(), String> {
        self.line(json!({"m": m}))
    }

    pub fn set(&mut self, msgs: &[Value]) -> Result<(), String> {
        self.line(json!({"set": msgs}))
    }
}

impl Drop for Session {
    fn drop(&mut self) {
        self.reset();
    }
}

fn lock(f: &File) -> Result<(), String> {
    match f.try_lock() {
        Err(fs::TryLockError::WouldBlock) => Err("sesi sedang dipakai proses lain".into()),
        _ => Ok(()), // sebagian sistem berkas tak mendukung kunci; tetap jalan
    }
}

fn load(raw: &[u8], cwd: &Path) -> Option<Vec<Value>> {
    let mut lines = raw.split(|b| *b == b'\n');
    let head: Value = serde_json::from_slice(lines.next()?).ok()?;
    if head["cwd"].as_str()? != cwd.to_string_lossy() {
        return None;
    }
    let mut msgs = vec![];
    for v in lines.filter_map(|l| serde_json::from_slice::<Value>(l).ok()) {
        if let Some(a) = v["set"].as_array() {
            msgs = a.clone();
        } else if v["m"].is_object() {
            msgs.push(v["m"].clone());
        }
    }
    Some(msgs)
}

/// Pastikan setiap tool_call punya hasil, dan tak ada hasil tanpa panggilan.
pub fn repair(msgs: &mut Vec<Value>) -> bool {
    let mut out: Vec<Value> = vec![];
    let mut wait: Vec<Value> = vec![];
    let mut changed = false;
    let close = |out: &mut Vec<Value>, wait: &mut Vec<Value>| {
        let n = wait.len();
        for c in wait.drain(..) {
            let mut m = json!({"role":"tool","tool_name":c["function"]["name"],"content":"Error: terputus sebelum selesai"});
            if c["id"].is_string() {
                m["tool_call_id"] = c["id"].clone();
            }
            out.push(m);
        }
        n > 0
    };
    for m in msgs.drain(..) {
        if m["role"] == "tool" {
            if wait.is_empty() {
                changed = true;
                continue;
            }
            wait.remove(0);
        } else {
            changed |= close(&mut out, &mut wait);
            wait = m["tool_calls"].as_array().cloned().unwrap_or_default();
        }
        out.push(m);
    }
    changed |= close(&mut out, &mut wait);
    *msgs = out;
    changed
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dir(n: &str) -> PathBuf {
        let d = Path::new(env!("CARGO_MANIFEST_DIR")).join("target/test").join(n);
        let _ = fs::remove_dir_all(&d);
        d
    }
    fn u(s: &str) -> Value {
        json!({"role":"user","content":s})
    }

    #[test]
    fn simpan_lanjut_dan_kunci() {
        let (r, cwd) = (dir("sesi"), Path::new("/a/b"));
        let mut s = Session::new(&r, cwd);
        assert!(s.resume(None).is_err());
        s.push(&u("satu")).unwrap();
        s.push(&u("dua")).unwrap();
        let mut t = Session::new(&r, cwd);
        assert!(t.resume(None).unwrap_err().contains("dipakai"));
        drop(s);
        assert_eq!(t.resume(None).unwrap(), [u("satu"), u("dua")]);
        t.set(&[u("ringkas")]).unwrap();
        t.push(&u("tiga")).unwrap();
        let id = t.list()[0].id.clone();
        drop(t);
        assert_eq!(Session::new(&r, cwd).resume(Some(&id)).unwrap(), [u("ringkas"), u("tiga")]);
        assert!(Session::new(&r, Path::new("/x/b")).resume(None).is_err());
        assert!(Session::new(&r, cwd).resume(Some("../x")).is_err());
        #[cfg(unix)]
        hanya_pemilik(&r, cwd);
    }

    #[cfg(unix)]
    fn hanya_pemilik(r: &Path, cwd: &Path) {
        use std::os::unix::fs::PermissionsExt;
        let d = r.join(slug(cwd));
        let f = fs::read_dir(&d).unwrap().next().unwrap().unwrap();
        assert_eq!(f.metadata().unwrap().permissions().mode() & 0o777, 0o600);
        assert_eq!(fs::metadata(d).unwrap().permissions().mode() & 0o777, 0o700);
    }

    #[test]
    fn baris_terpotong_diabaikan() {
        let (r, cwd) = (dir("potong"), Path::new("/c"));
        let mut s = Session::new(&r, cwd);
        s.push(&u("satu")).unwrap();
        s.file.as_mut().unwrap().write_all(b"{\"m\":{\"ro").unwrap();
        drop(s);
        let mut s = Session::new(&r, cwd);
        assert_eq!(s.resume(None).unwrap(), [u("satu")]);
        s.push(&u("dua")).unwrap();
        drop(s);
        assert_eq!(Session::new(&r, cwd).resume(None).unwrap(), [u("satu"), u("dua")]);
    }

    #[test]
    fn repair_memasangkan() {
        let call = json!({"role":"assistant","content":"","tool_calls":[{"id":"a","function":{"name":"read"}},{"id":"b","function":{"name":"bash"}}]});
        let res = json!({"role":"tool","content":"ok"});
        let mut m = vec![u("x"), res.clone(), call.clone(), res.clone(), u("y"), call, res.clone(), res];
        assert!(repair(&mut m));
        let roles: Vec<_> = m.iter().map(|m| m["role"].as_str().unwrap()).collect();
        assert_eq!(roles, ["user", "assistant", "tool", "tool", "user", "assistant", "tool", "tool"]);
        assert_eq!(m[3]["tool_call_id"], "b");
        assert!(!repair(&mut m));
    }
}
