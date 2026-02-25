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

pub struct CollectionDb {
    pub name: String,
    conn: Connection,
}

impl CollectionDb {
    /// Open (or create) a collection DB at the given path.
    pub fn open(name: impl Into<String>, db_path: &Path) -> Result<Self> {
        if let Some(parent) = db_path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        // Register sqlite-vec before opening any connection.
        // docs: https://alexgarcia.xyz/sqlite-vec/rust.html
        unsafe {
            sqlite3_auto_extension(Some(std::mem::transmute(
                sqlite_vec::sqlite3_vec_init as *const (),
            )));
        }

        let conn = Connection::open_with_flags(
            db_path,
            OpenFlags::SQLITE_OPEN_READ_WRITE | OpenFlags::SQLITE_OPEN_CREATE,
        )?;

        configure(&conn)?;

        let name = name.into();
        schema::init(&conn, &name)?;

        Ok(Self { name, conn })
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
