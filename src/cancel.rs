//! Pembatalan giliran: memutus soket HTTP dan membunuh grup proses `bash`.

use std::net::{Shutdown, TcpStream};
use std::process::{Child, ExitStatus};
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};

#[derive(Default)]
pub struct Cancel {
    flag: AtomicBool,
    sock: Mutex<Option<TcpStream>>,
    pgid: Mutex<Option<i32>>,
}

pub fn kill_group(pgid: i32) {
    // SAFETY: kill() tidak menyentuh memori; pgid > 0 sehingga -pgid hanya grup itu.
    if pgid > 1 {
        unsafe { libc::kill(-pgid, libc::SIGKILL) };
    }
}

impl Cancel {
    pub fn is_set(&self) -> bool {
        self.flag.load(Ordering::SeqCst)
    }

    pub fn reset(&self) {
        self.flag.store(false, Ordering::SeqCst);
    }

    pub fn cancel(&self) {
        self.flag.store(true, Ordering::SeqCst);
        if let Some(s) = self.sock.lock().unwrap().as_ref() {
            let _ = s.shutdown(Shutdown::Both);
        }
        if let Some(p) = *self.pgid.lock().unwrap() {
            kill_group(p);
        }
    }

    /// Daftarkan soket aktif. Bila sudah dibatalkan lebih dulu, langsung diputus.
    pub fn set_sock(&self, s: Option<TcpStream>) {
        let mut g = self.sock.lock().unwrap();
        *g = s;
        if self.is_set()
            && let Some(s) = g.as_ref()
        {
            let _ = s.shutdown(Shutdown::Both);
        }
    }

    pub fn set_pgid(&self, pgid: i32) {
        let mut g = self.pgid.lock().unwrap();
        *g = Some(pgid);
        if self.is_set() {
            kill_group(pgid);
        }
    }

    /// try_wait di bawah kunci, supaya pgid tidak pernah dibunuh setelah dipanen
    /// (nomornya bisa dipakai ulang oleh proses lain).
    pub fn reap(&self, child: &mut Child) -> std::io::Result<Option<ExitStatus>> {
        let mut g = self.pgid.lock().unwrap();
        let r = child.try_wait();
        if !matches!(r, Ok(None)) {
            *g = None;
        }
        r
    }
}
