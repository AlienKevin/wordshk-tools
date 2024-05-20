pub struct SqliteDb {
    pool: r2d2::Pool<r2d2_sqlite::SqliteConnectionManager>,
}

impl SqliteDb {
    pub fn new<P>(db_path: P) -> Self
    where
        P: AsRef<std::path::Path>,
    {
        let manager = r2d2_sqlite::SqliteConnectionManager::file(db_path);
        let pool = r2d2::Pool::new(manager).unwrap();
        Self { pool }
    }

    pub fn conn(&self) -> r2d2::PooledConnection<r2d2_sqlite::SqliteConnectionManager> {
        self.pool.get().unwrap()
    }
}
