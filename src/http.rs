//! HTTP/1.1 minimal, dengan TLS (rustls). Tidak mengikuti pengalihan, jadi kunci API
//! tidak pernah terkirim ke host lain.

use crate::cancel::Cancel;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpStream, ToSocketAddrs};
use std::time::Duration;

const MAX_LINE: u64 = 8 << 20;

/// Batas diam untuk https; Ollama lokal boleh lama memuat model, jadi tanpa batas.
const IDLE: Duration = Duration::from_secs(300);

pub struct Req<'a> {
    pub base: &'a str,
    pub method: &'a str,
    pub path: &'a str,
    pub headers: &'a [(&'a str, String)],
    pub body: &'a str,
}

#[derive(Debug)]
pub struct Fail {
    /// 0 = gagal sebelum ada status HTTP.
    pub status: u16,
    pub msg: String,
    pub retry_after: Option<u64>,
    /// Server meminta agar tidak diulang.
    pub no_retry: bool,
    /// Tujuan pengalihan (3xx); tidak diikuti di sini.
    pub location: Option<String>,
}

impl From<String> for Fail {
    fn from(msg: String) -> Self {
        Fail { status: 0, msg, retry_after: None, no_retry: false, location: None }
    }
}

struct Target {
    tls: bool,
    addr: String,
    host: String,
    prefix: String,
}

fn parse_base(base: &str) -> Result<Target, String> {
    let (tls, rest) = match base.strip_prefix("https://") {
        Some(r) => (true, r),
        None => (false, base.strip_prefix("http://").unwrap_or(base)),
    };
    let (host, prefix) = match rest.find('/') {
        Some(i) => (&rest[..i], rest[i..].trim_end_matches('/')),
        None => (rest, ""),
    };
    if host.is_empty() || host.contains(|c: char| c.is_whitespace() || c.is_control() || c == '@') {
        return Err(format!("URL tidak valid: {base}"));
    }
    let after_v6 = host.rfind(']').map_or(host, |i| &host[i..]);
    let port = if tls { 443 } else { 80 };
    let addr = if after_v6.contains(':') { host.to_string() } else { format!("{host}:{port}") };
    Ok(Target { tls, addr, host: host.into(), prefix: prefix.into() })
}

/// Apakah `base` hanya menuju komputer ini.
pub fn is_local(base: &str) -> bool {
    parse_base(base).is_ok_and(|t| {
        let h = t.addr.rsplit_once(':').map_or(t.addr.as_str(), |x| x.0);
        matches!(h, "localhost" | "[::1]") || h.parse::<std::net::Ipv4Addr>().is_ok_and(|a| a.is_loopback())
    })
}

/// Apakah `base` menunjuk komputer lain di jaringan lokal (alamat privat atau nama lokal).
pub fn is_lan(base: &str) -> bool {
    use std::net::{Ipv4Addr, Ipv6Addr};
    parse_base(base).is_ok_and(|t| {
        let h = t.addr.rsplit_once(':').map_or(t.addr.as_str(), |x| x.0);
        if let Some(v6) = h.strip_prefix('[').and_then(|s| s.strip_suffix(']')) {
            return v6.parse::<Ipv6Addr>().is_ok_and(|a| a.is_unique_local() || a.is_unicast_link_local());
        }
        if let Ok(a) = h.parse::<Ipv4Addr>() {
            let o = a.octets();
            // 100.64/10: CGNAT, dipakai Tailscale.
            return a.is_private() || a.is_link_local() || (o[0] == 100 && (64..128).contains(&o[1]));
        }
        let h = h.to_ascii_lowercase();
        // Nama berawalan angka bisa berarti alamat IP dalam bentuk lain.
        h.starts_with(|c: char| c.is_ascii_alphabetic())
            && h != "localhost"
            && (!h.contains('.') || [".local", ".lan", ".home.arpa", ".internal"].iter().any(|s| h.ends_with(s)))
    })
}

pub fn is_tls(base: &str) -> bool {
    base.starts_with("https://")
}

trait Duplex: Read + Write {}
impl<T: Read + Write> Duplex for T {}

#[cfg(feature = "tls")]
fn wrap_tls(sock: TcpStream, t: &Target) -> Result<Box<dyn Duplex>, String> {
    use std::sync::{Arc, OnceLock};
    static CFG: OnceLock<Arc<rustls::ClientConfig>> = OnceLock::new();
    let cfg = CFG.get_or_init(|| {
        let mut roots = rustls::RootCertStore::empty();
        roots.extend(webpki_roots::TLS_SERVER_ROOTS.iter().cloned());
        let provider = Arc::new(rustls::crypto::ring::default_provider());
        let c = rustls::ClientConfig::builder_with_provider(provider)
            .with_safe_default_protocol_versions()
            .expect("versi TLS bawaan")
            .with_root_certificates(roots)
            .with_no_client_auth();
        Arc::new(c)
    });
    let name = t.addr.rsplit_once(':').map_or(t.addr.as_str(), |x| x.0).to_string();
    let name = rustls::pki_types::ServerName::try_from(name).map_err(|e| format!("nama host: {e}"))?;
    let conn = rustls::ClientConnection::new(cfg.clone(), name).map_err(|e| format!("TLS: {e}"))?;
    Ok(Box::new(rustls::StreamOwned::new(conn, sock)))
}

#[cfg(not(feature = "tls"))]
fn wrap_tls(_: TcpStream, _: &Target) -> Result<Box<dyn Duplex>, String> {
    Err("dibangun tanpa TLS; https tidak tersedia".into())
}

/// Pembaca body: chunked, content-length, atau sampai koneksi ditutup.
enum Body<R: BufRead> {
    Chunked { r: R, left: u64, done: bool },
    Plain(std::io::Take<R>),
}

impl<R: BufRead> Read for Body<R> {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        match self {
            Body::Plain(r) => r.read(buf),
            Body::Chunked { done: true, .. } => Ok(0),
            Body::Chunked { r, left, done } => {
                let bad = |m: &str| std::io::Error::new(std::io::ErrorKind::InvalidData, m.to_string());
                if *left == 0 {
                    let mut l = String::new();
                    r.take(1024).read_line(&mut l)?;
                    if l.trim().is_empty() {
                        return Err(bad("koneksi terputus"));
                    }
                    let hex = l.trim().split(';').next().unwrap_or("");
                    *left = u64::from_str_radix(hex, 16).map_err(|_| bad("chunk tidak valid"))?;
                    if *left == 0 {
                        *done = true;
                        return Ok(0);
                    }
                }
                let max = buf.len().min((*left).min(usize::MAX as u64) as usize);
                let n = r.read(&mut buf[..max])?;
                if n == 0 && max > 0 {
                    return Err(bad("koneksi terputus"));
                }
                *left -= n as u64;
                if *left == 0 {
                    let mut crlf = [0u8; 2];
                    r.read_exact(&mut crlf)?;
                }
                Ok(n)
            }
        }
    }
}

/// Kirim request; `on_line` dipanggil untuk tiap baris body.
/// Status selain 2xx dikembalikan sebagai Fail berisi isi body.
pub fn request(
    req: &Req,
    cancel: &Cancel,
    on_line: &mut dyn FnMut(&str) -> Result<(), String>,
) -> Result<(), Fail> {
    let t = parse_base(req.base)?;
    let addrs = t.addr.to_socket_addrs().map_err(|e| format!("alamat {}: {e}", t.addr))?;
    let mut last = format!("alamat {} tidak ditemukan", t.addr);
    let mut sock = None;
    for a in addrs {
        match TcpStream::connect_timeout(&a, Duration::from_secs(5)) {
            Ok(s) => {
                sock = Some(s);
                break;
            }
            Err(e) => last = format!("tidak bisa terhubung ke {}: {e}", req.base),
        }
    }
    let sock = sock.ok_or(last)?;
    let _ = sock.set_nodelay(true);
    cancel.set_sock(sock.try_clone().ok());
    let r = exchange(sock, &t, req, on_line);
    cancel.set_sock(None);
    if cancel.is_set() { Err("dibatalkan".to_string().into()) } else { r }
}

fn exchange(
    sock: TcpStream,
    t: &Target,
    req: &Req,
    on_line: &mut dyn FnMut(&str) -> Result<(), String>,
) -> Result<(), Fail> {
    let io = |e: std::io::Error| Fail::from(format!("koneksi: {e}"));
    let mut s: Box<dyn Duplex> = if t.tls {
        let _ = sock.set_read_timeout(Some(IDLE));
        let _ = sock.set_write_timeout(Some(IDLE));
        wrap_tls(sock, t)?
    } else {
        Box::new(sock)
    };
    let mut head = format!(
        "{} {}{} HTTP/1.1\r\nHost: {}\r\nUser-Agent: llmy\r\nContent-Length: {}\r\nConnection: close\r\n",
        req.method, t.prefix, req.path, t.host, req.body.len()
    );
    if !req.headers.iter().any(|(k, _)| k.eq_ignore_ascii_case("content-type")) {
        head += "Content-Type: application/json\r\n";
    }
    for (k, v) in req.headers {
        if k.is_empty() || k.contains(|c: char| !c.is_ascii_graphic() || c == ':') {
            return Err(format!("nama header {k:?} tidak sah").into());
        }
        if v.contains(|c: char| c.is_control() || !c.is_ascii()) {
            return Err(format!("nilai header {k} mengandung karakter tidak sah").into());
        }
        head += &format!("{k}: {v}\r\n");
    }
    head += "\r\n";
    s.write_all(head.as_bytes()).map_err(io)?;
    s.write_all(req.body.as_bytes()).map_err(io)?;
    s.flush().map_err(io)?;
    read_response(BufReader::new(s), on_line)
}

fn read_response<R: BufRead>(
    mut r: R,
    on_line: &mut dyn FnMut(&str) -> Result<(), String>,
) -> Result<(), Fail> {
    let io = |e: std::io::Error| {
        Fail::from(match e.kind() {
            std::io::ErrorKind::WouldBlock | std::io::ErrorKind::TimedOut => "server diam terlalu lama".to_string(),
            _ => format!("koneksi: {e}"),
        })
    };
    let mut line = String::new();
    (&mut r).take(8192).read_line(&mut line).map_err(io)?;
    let status: u16 = line
        .split_whitespace()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .ok_or("balasan HTTP tidak valid".to_string())?;
    let (mut chunked, mut len) = (false, u64::MAX);
    let (mut retry_after, mut no_retry, mut location) = (None, false, None);
    for _ in 0..200 {
        line.clear();
        (&mut r).take(16384).read_line(&mut line).map_err(io)?;
        let l = line.trim();
        if l.is_empty() {
            break;
        }
        if let Some((k, v)) = l.split_once(':') {
            match k.to_ascii_lowercase().as_str() {
                "transfer-encoding" => chunked = v.to_ascii_lowercase().contains("chunked"),
                "retry-after" => retry_after = v.trim().parse().ok(),
                "x-should-retry" => no_retry = v.trim() == "false",
                "content-length" => len = v.trim().parse().unwrap_or(u64::MAX),
                "location" => location = Some(v.trim().to_string()),
                _ => {}
            }
        }
    }
    let body = if chunked {
        Body::Chunked { r, left: 0, done: false }
    } else {
        Body::Plain(r.take(len))
    };
    let mut body = BufReader::new(body);
    if !(200..300).contains(&status) {
        let mut b = Vec::new();
        let _ = body.take(64 << 10).read_to_end(&mut b);
        let msg = format!("HTTP {status}: {}", error_text(&String::from_utf8_lossy(&b)));
        return Err(Fail { status, msg, retry_after, no_retry, location });
    }
    let mut buf = Vec::new();
    loop {
        buf.clear();
        let n = (&mut body).take(MAX_LINE).read_until(b'\n', &mut buf).map_err(io)?;
        if n == 0 {
            return Ok(());
        }
        let s = String::from_utf8_lossy(&buf);
        if !s.trim().is_empty() {
            on_line(s.trim())?;
        }
    }
}

/// Ambil pesan dari badan error JSON: {"error": "..."} atau {"error": {"message": "..."}}.
fn error_text(b: &str) -> String {
    let v: serde_json::Value = serde_json::from_str(b).unwrap_or_default();
    let e = &v["error"];
    let m = e["message"].as_str().or(e.as_str()).unwrap_or(b.trim());
    let kind = e["details"]["error_code"].as_str().map(|c| format!(" ({c})")).unwrap_or_default();
    m.chars().take(1000).collect::<String>() + &kind
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(raw: &str) -> (Result<(), Fail>, Vec<String>) {
        let mut v = vec![];
        let r = read_response(raw.as_bytes(), &mut |l| {
            v.push(l.to_string());
            Ok(())
        });
        (r, v)
    }

    #[test]
    fn chunked_terpotong_di_tengah_baris() {
        let (r, v) = run("HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n3\r\n{\"a\r\n6\r\n\":1}\n{\r\n2\r\n}\n\r\n0\r\n\r\n");
        assert!(r.is_ok());
        assert_eq!(v, ["{\"a\":1}", "{}"]);
    }

    #[test]
    fn chunked_putus_adalah_error() {
        let (r, v) = run("HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n4\r\n{}\n\n\r\n");
        assert!(r.is_err());
        assert_eq!(v, ["{}"]);
    }

    #[test]
    fn status_error_memuat_body() {
        let (r, _) = run("HTTP/1.1 400 Bad\r\nContent-Length: 3\r\n\r\nabc");
        assert_eq!(r.unwrap_err().msg, "HTTP 400: abc");
        let (r, _) = run("HTTP/1.1 429 X\r\nRetry-After: 7\r\nx-should-retry: false\r\n\r\n{\"error\":{\"message\":\"pelan\"}}");
        let e = r.unwrap_err();
        assert_eq!((e.status, e.msg.as_str(), e.retry_after, e.no_retry), (429, "HTTP 429: pelan", Some(7), true));
    }

    #[test]
    fn base_url() {
        let t = parse_base("http://localhost:11434/x/").unwrap();
        assert_eq!((t.addr.as_str(), t.prefix.as_str()), ("localhost:11434", "/x"));
        assert_eq!(parse_base("http://[::1]").unwrap().addr, "[::1]:80");
        let t = parse_base("https://a.b/v1").unwrap();
        assert!(t.tls && t.addr == "a.b:443" && t.prefix == "/v1");
        assert!(parse_base("https://user@a").is_err() && parse_base("http://a b").is_err());
        for l in ["http://localhost:1", "http://127.0.0.1", "http://[::1]:9"] {
            assert!(is_local(l), "{l}");
        }
        for l in ["http://localhost.evil.com", "http://127.0.0.1.evil.com", "https://a.b", "http://10.0.0.1"] {
            assert!(!is_local(l), "{l}");
        }
        for l in ["http://10.0.0.1", "http://192.168.1.5:8000/v1", "http://172.31.0.1", "http://100.100.1.1", "http://gpu:8000", "http://gpu.local/v1", "http://[fd00::1]:1", "http://[fe80::1]"] {
            assert!(is_lan(l) && !is_local(l), "{l}");
        }
        for l in ["http://localhost", "http://127.0.0.1", "http://8.8.8.8", "http://172.32.0.1", "http://100.128.0.1", "http://a.b", "http://x.local.evil.com", "http://134744072", "http://0x8080808", "http://[2001:db8::1]"] {
            assert!(!is_lan(l), "{l}");
        }
    }
}
