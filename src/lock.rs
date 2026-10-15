// Advisory cross-process locks (flock). The lock is held for as long as the
// returned File is alive and is released by the OS when the process dies.
// docs: https://man7.org/linux/man-pages/man2/flock.2.html

use crate::error::{Error, Result};
use std::fs::File;
use std::os::unix::io::AsRawFd;
use std::path::Path;

fn open(path: &Path) -> Result<File> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(false)
        .open(path)
        .map_err(Error::Io)
}

/// Take the lock without waiting. None when another process holds it.
pub fn try_exclusive(path: &Path) -> Result<Option<File>> {
    let file = open(path)?;
    if unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) } == 0 {
        return Ok(Some(file));
    }
    let err = std::io::Error::last_os_error();
    if err.kind() == std::io::ErrorKind::WouldBlock {
        return Ok(None);
    }
    Err(Error::Io(err))
}

/// Take the lock, waiting for the current holder.
pub fn exclusive(path: &Path) -> Result<File> {
    let file = open(path)?;
    if unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX) } != 0 {
        return Err(Error::Io(std::io::Error::last_os_error()));
    }
    Ok(file)
}

/// Like `exclusive`, but says why on stderr when it has to wait.
pub fn exclusive_or_wait(path: &Path, waiting_for: &str) -> Result<File> {
    if let Some(file) = try_exclusive(path)? {
        return Ok(file);
    }
    eprintln!("waiting for {waiting_for}…");
    exclusive(path)
}
