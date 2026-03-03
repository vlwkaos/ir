// Per-collection SQLite connection management.
// sqlite-vec is registered as a static extension before any connection opens.
// docs: https://alexgarcia.xyz/sqlite-vec/rust.html

pub mod fts;
pub mod schema;
pub mod vectors;

use crate::error::Result;
use rusqlite::ffi::sqlite3_auto_extension;
use rusqlite::{Connection, OpenFlags};
use std::path::Path;
use std::sync::Once;

static SQLITE_VEC_INIT: Once = Once::new();

/// Register the sqlite-vec extension exactly once per process.
/// Safe to call from multiple threads or call sites.
pub fn ensure_sqlite_vec() {
    SQLITE_VEC_INIT.call_once(|| {
        // SAFETY: sqlite3_auto_extension is idempotent and the function pointer
        // is valid for the lifetime of the process. Once is used to prevent
        // double-registration.
        unsafe {
            sqlite3_auto_extension(Some(std::mem::transmute(
                sqlite_vec::sqlite3_vec_init as *const (),
            )));
        }
    });
}

pub struct CollectionDb {
    pub name: String,
    conn: Connection,
}

impl CollectionDb {
    /// Open (or create) a collection DB at the given path with read-write access.
    pub fn open(name: impl Into<String>, db_path: &Path) -> Result<Self> {
        if let Some(parent) = db_path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        ensure_sqlite_vec();

        let conn = Connection::open_with_flags(
            db_path,
            OpenFlags::SQLITE_OPEN_READ_WRITE | OpenFlags::SQLITE_OPEN_CREATE,
        )?;

        configure(&conn)?;

        let name = name.into();
        schema::init(&conn, &name)?;

        Ok(Self { name, conn })
    }

    /// Open an existing collection DB immutable/read-only.
    /// Uses `immutable=1` URI mode to bypass WAL/shm entirely — safe in read-only filesystems
    /// and sandboxed environments where write access to the data directory is unavailable.
    pub fn open_readonly(name: impl Into<String>, db_path: &Path) -> Result<Self> {
        ensure_sqlite_vec();

        // ! immutable=1 bypasses WAL read-mark writes (shm file) — required when the
        // ! calling process lacks write access to the collections directory (e.g. sandbox).
        let uri = format!(
            "file://{}?immutable=1",
            db_path.to_string_lossy().replace(' ', "%20")
        );

        let conn = Connection::open_with_flags(
            uri,
            OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_URI,
        )?;

        conn.execute_batch("PRAGMA cache_size = -64000;")?;

        Ok(Self {
            name: name.into(),
            conn,
        })
    }

    pub fn conn(&self) -> &Connection {
        &self.conn
    }
}

fn configure(conn: &Connection) -> Result<()> {
    conn.execute_batch(
        "PRAGMA journal_mode = WAL;
         PRAGMA busy_timeout = 5000;
         PRAGMA synchronous  = NORMAL;
         PRAGMA cache_size   = -64000;
         PRAGMA foreign_keys = ON;",
    )?;
    Ok(())
}
