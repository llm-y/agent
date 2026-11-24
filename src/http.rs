//! HTTP/1.1 minimal di atas TcpStream. Cukup untuk Ollama, tanpa TLS.

use crate::cancel::Cancel;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpStream, ToSocketAddrs};
use std::time::Duration;

const MAX_LINE: u64 = 8 << 20;

struct Target {
    addr: String,
    host: String,
    prefix: String,
}

fn parse_base(base: &str) -> Result<Target, String> {
    if base.starts_with("https://") {
        return Err("URL https belum didukung; pakai http://".into());
    }
    let rest = base.strip_prefix("http://").unwrap_or(base);
    let (host, prefix) = match rest.find('/') {
        Some(i) => (&rest[..i], rest[i..].trim_end_matches('/')),
        None => (rest, ""),
    };
    if host.is_empty() || host.contains(|c: char| c.is_whitespace() || c.is_control()) {
        return Err(format!("URL tidak valid: {base}"));
    }
    let after_v6 = host.rfind(']').map_or(host, |i| &host[i..]);
    let addr = if after_v6.contains(':') { host.to_string() } else { format!("{host}:80") };
    Ok(Target { addr, host: host.into(), prefix: prefix.into() })
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
/// Status selain 2xx dikembalikan sebagai Err berisi isi body.
pub fn request(
    base: &str,
    method: &str,
    path: &str,
    body: &str,
    cancel: &Cancel,
    on_line: &mut dyn FnMut(&str) -> Result<(), String>,
) -> Result<(), String> {
    let t = parse_base(base)?;
    let addrs = t.addr.to_socket_addrs().map_err(|e| format!("alamat {}: {e}", t.addr))?;
    let mut last = format!("alamat {} tidak ditemukan", t.addr);
    let mut sock = None;
    for a in addrs {
        match TcpStream::connect_timeout(&a, Duration::from_secs(5)) {
            Ok(s) => {
                sock = Some(s);
                break;
            }
            Err(e) => last = format!("tidak bisa terhubung ke {base}: {e}"),
        }
    }
    let sock = sock.ok_or(last)?;
    let _ = sock.set_nodelay(true);
    cancel.set_sock(sock.try_clone().ok());
    let r = exchange(sock, &t, method, path, body, on_line);
    cancel.set_sock(None);
    if cancel.is_set() { Err("dibatalkan".into()) } else { r }
}

fn exchange(
    mut sock: TcpStream,
    t: &Target,
    method: &str,
    path: &str,
    body: &str,
    on_line: &mut dyn FnMut(&str) -> Result<(), String>,
) -> Result<(), String> {
    let io = |e: std::io::Error| format!("koneksi: {e}");
    let head = format!(
        "{method} {}{path} HTTP/1.1\r\nHost: {}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        t.prefix, t.host, body.len()
    );
    sock.write_all(head.as_bytes()).map_err(io)?;
    sock.write_all(body.as_bytes()).map_err(io)?;
    read_response(BufReader::new(sock), on_line)
}

fn read_response<R: BufRead>(
    mut r: R,
    on_line: &mut dyn FnMut(&str) -> Result<(), String>,
) -> Result<(), String> {
    let io = |e: std::io::Error| format!("koneksi: {e}");
    let mut line = String::new();
    (&mut r).take(8192).read_line(&mut line).map_err(io)?;
    let status: u16 = line
        .split_whitespace()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .ok_or("balasan HTTP tidak valid")?;
    let (mut chunked, mut len) = (false, u64::MAX);
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
                "content-length" => len = v.trim().parse().unwrap_or(u64::MAX),
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
        return Err(format!("HTTP {status}: {}", String::from_utf8_lossy(&b).trim()));
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

#[cfg(test)]
mod tests {
    use super::*;

    fn run(raw: &str) -> (Result<(), String>, Vec<String>) {
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
        assert_eq!(r.unwrap_err(), "HTTP 400: abc");
    }

    #[test]
    fn base_url() {
        let t = parse_base("http://localhost:11434/x/").unwrap();
        assert_eq!((t.addr.as_str(), t.prefix.as_str()), ("localhost:11434", "/x"));
        assert_eq!(parse_base("http://[::1]").unwrap().addr, "[::1]:80");
        assert!(parse_base("https://a").is_err());
    }
}
