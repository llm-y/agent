//! Pembatalan giliran: memutus soket HTTP dan membunuh grup proses `bash`.

use crate::sys::kill_tree;
use std::net::{Shutdown, TcpStream};
use std::process::{Child, ExitStatus};
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};

#[derive(Default)]
pub struct Cancel {
    flag: AtomicBool,
    sock: Mutex<Option<TcpStream>>,
    pid: Mutex<Option<u32>>,
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
        if let Some(p) = *self.pid.lock().unwrap() {
            kill_tree(p);
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

    pub fn set_pid(&self, pid: u32) {
        let mut g = self.pid.lock().unwrap();
        *g = Some(pid);
        if self.is_set() {
            kill_tree(pid);
        }
    }

    /// try_wait di bawah kunci, supaya pid tidak pernah dibunuh setelah dipanen
    /// (nomornya bisa dipakai ulang oleh proses lain).
    pub fn reap(&self, child: &mut Child) -> std::io::Result<Option<ExitStatus>> {
        let mut g = self.pid.lock().unwrap();
        let r = child.try_wait();
        if !matches!(r, Ok(None)) {
            *g = None;
        }
        r
    }
}
