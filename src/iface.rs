//! Penjaga antarmuka: nama publik file (export JS, `pub` Rust, `def`/`class` Python) dibandingkan dengan
//! isinya di awal giliran. Model kecil menulis ulang stub dan merusak antarmuka yang di-import uji: di E10
//! (Aider Polyglot) 17 dari 43 soal JavaScript yang gagal mengganti `export class X` menjadi
//! `module.exports = X`. Pesan peringatan saja tidak cukup: qwen3.5:9b mengabaikannya 10 kali (uji 2026-11-28), jadi
//! ESM -> CommonJS lewat `write`/`edit` ditolak (`to_cjs`), dan `write` utuh atas file yang belum dibaca ditolak oleh
//! `Agent` (ide `_stale_overwrite_blocker` Hermes Agent). Nama hilang dan signature berubah tetap peringatan.
//! `LLMY_IFACE=off` mematikan semuanya.

use std::collections::BTreeMap;
use std::path::Path;
use std::sync::OnceLock;

#[derive(Default, Debug)]
pub struct Api {
    /// Nama publik -> signature tiap kemunculan.
    names: BTreeMap<String, Vec<String>>,
    esm: bool,
    cjs: bool,
}

struct Re {
    decl: regex_lite::Regex,
    list: regex_lite::Regex,
    cjs: regex_lite::Regex,
}

fn re() -> &'static Re {
    static R: OnceLock<Re> = OnceLock::new();
    R.get_or_init(|| Re {
        decl: regex_lite::Regex::new(r"^\s*export\s+(?:default\s+)?(?:async\s+)?(?:function\s*\*?|class|const|let|var)\s*([A-Za-z_$][\w$]*)").unwrap(),
        list: regex_lite::Regex::new(r"^\s*export\s*\{([^}]*)\}").unwrap(),
        cjs: regex_lite::Regex::new(r"(?m)^\s*(?:module\.)?exports(?:\.[A-Za-z_$][\w$]*)?\s*=").unwrap(),
    })
}

fn sig(l: &str) -> String {
    let h = l.trim().split(" {").next().unwrap_or("").trim_end_matches(['{', ':', ' ', ';']);
    h.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Antarmuka publik, atau None untuk bahasa yang tidak diperiksa.
pub fn api(path: &Path, text: &str) -> Option<Api> {
    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
    let mut a = Api::default();
    let mut add = |n: &str, s: String| a.names.entry(n.to_string()).or_default().push(s);
    match ext {
        "js" | "mjs" | "cjs" | "ts" | "jsx" | "tsx" => {
            let r = re();
            for l in text.lines() {
                if let Some(c) = r.decl.captures(l) {
                    add(&c[1], sig(l));
                } else if let Some(c) = r.list.captures(l) {
                    for item in c[1].split(',') {
                        let n = item.split(" as ").last().unwrap_or("").trim();
                        if !n.is_empty() {
                            add(n, String::new());
                        }
                    }
                } else if l.trim_start().starts_with("export default") {
                    add("default", String::new());
                }
            }
            a.esm = text.lines().any(|l| l.trim_start().starts_with("export "));
            a.cjs = r.cjs.is_match(text);
        }
        "rs" => {
            for (_, s, n) in crate::bdi::defs_in("x.rs", text) {
                if s.starts_with("pub ") {
                    add(&n, sig(&s));
                }
            }
        }
        "py" => {
            for (_, s, n) in crate::bdi::defs_in("x.py", text) {
                if (!n.starts_with('_') || n == "__init__") && (s.starts_with("def ") || s.starts_with("async def ") || s.starts_with("class ")) {
                    add(&n, sig(&s));
                }
            }
        }
        _ => return None,
    }
    Some(a)
}

/// `new` mengganti export ESM `old` dengan CommonJS.
pub fn to_cjs(path: &Path, old: &str, new: &str) -> bool {
    matches!((api(path, old), api(path, new)), (Some(o), Some(n)) if o.esm && !n.esm && n.cjs)
}

/// Galat untuk model saat `to_cjs`: dengan cara menguji modul ESM dari shell.
pub fn cjs_error(path: &Path, old: &str) -> String {
    let name = api(path, old).and_then(|a| a.names.keys().find(|k| *k != "default").cloned()).unwrap_or_else(|| "name".into());
    let file = path.file_name().map_or_else(String::new, |f| f.to_string_lossy().into_owned());
    format!(
        "rejected: this converts the file's ES module exports (`export ...`) to CommonJS (`module.exports`). Keep `export`: \
         tests and other files import it with `import {{ {name} }} from './{file}'`. To try it from the shell use an ES module, \
         e.g. node --input-type=module -e \"import {{ {name} }} from './{file}'; console.log({name})\" (or a test file ending in .mjs), \
         not require()."
    )
}

/// Catatan untuk model bila `new` menghilangkan nama publik, mengubah signature, atau gaya export dari `old`.
pub fn check(path: &Path, old: &str, new: &str) -> Option<String> {
    let (o, n) = (api(path, old)?, api(path, new)?);
    if o.names.is_empty() {
        return None;
    }
    let mut parts = vec![];
    let removed: Vec<&str> = o.names.iter()
        .filter(|(k, v)| n.names.get(*k).map_or(0, Vec::len) < v.len())
        .map(|(k, _)| k.as_str()).collect();
    if !removed.is_empty() {
        let more = if removed.len() > 6 { format!(" and {} more", removed.len() - 6) } else { String::new() };
        parts.push(format!("removed or renamed: {}{more}", removed.iter().take(6).map(|s| format!("`{s}`")).collect::<Vec<_>>().join(", ")));
    }
    let changed: Vec<String> = o.names.iter()
        .filter_map(|(k, v)| match (v.as_slice(), n.names.get(k).map(Vec::as_slice)) {
            ([a], Some([b])) if !a.is_empty() && !b.is_empty() && a != b => Some(format!("`{a}` -> `{b}`")),
            _ => None,
        })
        .take(3).collect();
    if !changed.is_empty() {
        parts.push(format!("signature changed: {}", changed.join("; ")));
    }
    if o.esm && !n.esm && n.cjs {
        parts.push("the file used ES module `export` and now uses CommonJS `module.exports`/`exports`, so `import { ... } from` this file breaks".into());
    }
    if parts.is_empty() {
        return None;
    }
    Some(format!(
        "\n\n[llmy interface check: compared with this file at the start of the turn, {}. Tests and other files depend on the original public interface. Unless the user asked for this change, restore the original names, signatures, and export style.]",
        parts.join("; ")
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn c(p: &str, a: &str, b: &str) -> Option<String> {
        check(Path::new(p), a, b)
    }

    #[test]
    fn js_esm_ke_commonjs_ketahuan() {
        let stub = "export class GradeSchool {\n  roster() {\n    throw new Error('x');\n  }\n}\n";
        let cjs = "class GradeSchool {\n  roster() { return {}; }\n}\nmodule.exports = GradeSchool;\n";
        let m = c("grade-school.js", stub, cjs).unwrap();
        assert!(m.contains("CommonJS") && m.contains("`GradeSchool`"), "{m}");
        // Implementasi yang mempertahankan export: diam.
        assert_eq!(c("grade-school.js", stub, "export class GradeSchool {\n  roster() { return {}; }\n}\n"), None);
        // `export { a as b }` dan `export const`.
        assert_eq!(c("a.js", "export const recite = (a, b) => {\n};\n", "const f = (a, b) => [];\nexport { f as recite };\n"), None);
        assert!(c("a.js", "export const recite = () => {};\nexport function x() {}\n", "export const recite = () => [];\n").unwrap().contains("`x`"));
        let p = Path::new("grade-school.js");
        assert!(to_cjs(p, stub, cjs) && !to_cjs(p, stub, stub) && !to_cjs(p, cjs, cjs));
        assert!(cjs_error(p, stub).contains("import { GradeSchool } from './grade-school.js'"));
    }

    #[test]
    fn rust_dan_python_signature() {
        let m = c("src/lib.rs", "pub fn area(r: &Rect) -> u32 {\n    todo!()\n}\nfn util() {}\n", "pub fn area(r: Rect) -> u32 {\n    0\n}\n").unwrap();
        assert!(m.contains("`pub fn area(r: &Rect) -> u32` -> `pub fn area(r: Rect) -> u32`") && !m.contains("util"), "{m}");
        assert_eq!(c("src/lib.rs", "pub fn area(r: &Rect) -> u32 {\n    todo!()\n}\n", "pub fn area(r: &Rect) -> u32 {\n    r.w * r.h\n}\n"), None);
        let py = "class PhoneNumber:\n    def __init__(self, number):\n        pass\n\n    def area_code(self):\n        pass\n\ndef _helper():\n    pass\n";
        let m = c("phone_number.py", py, "class Phone:\n    def __init__(self, number):\n        pass\n").unwrap();
        assert!(m.contains("`PhoneNumber`") && m.contains("`area_code`") && !m.contains("_helper"), "{m}");
        // Bahasa lain dan file tanpa nama publik: tidak diperiksa.
        assert_eq!(c("main.go", "func A() {}", ""), None);
        assert_eq!(c("a.py", "x = 1\n", "y = 2\n"), None);
    }
}
