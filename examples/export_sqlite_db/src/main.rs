use r2d2_sqlite::SqliteConnectionManager;
use std::time::Instant;
use wordshk_tools::{
    app_api::Api,
    dict::EntryId,
    jyutping::Romanization,
    rich_dict::RichEntry,
    search::{self, rich_dict_to_variants_map, RichDictLike, Script},
};

const APP_TMP_DIR: &str = "./app_tmp";

fn main() {
    // export_sqlite_db();
    test_sqlite_search();
}

fn export_sqlite_db() {
    let api = unsafe { Api::load(APP_TMP_DIR, Romanization::Jyutping) };
    api.export_dict_as_sqlite_db("dict.db").unwrap();
}

struct SqliteRichDict {
    pool: r2d2::Pool<SqliteConnectionManager>,
}

impl SqliteRichDict {
    fn new(dict_path: &str) -> Self {
        let manager = SqliteConnectionManager::file(dict_path);
        let pool = r2d2::Pool::new(manager).unwrap();
        Self { pool }
    }

    fn conn(&self) -> r2d2::PooledConnection<SqliteConnectionManager> {
        self.pool.get().unwrap()
    }
}

impl RichDictLike for SqliteRichDict {
    fn get_entry(&self, entry_id: EntryId) -> RichEntry {
        use rkyv::Deserialize;

        let conn = self.conn();
        let mut stmt = conn
            .prepare("SELECT entry_rkyv FROM rich_dict WHERE id = ?")
            .unwrap();
        let entry_rkyv_bytes: Vec<u8> = stmt.query_row([entry_id], |row| row.get(0)).unwrap();
        let entry: RichEntry = unsafe { rkyv::archived_root::<RichEntry>(&entry_rkyv_bytes[..]) }
            .deserialize(&mut rkyv::Infallible)
            .unwrap();
        entry
    }

    fn get_ids(&self) -> Vec<EntryId> {
        let conn = self.conn();
        let mut stmt = conn.prepare("SELECT id FROM rich_dict").unwrap();
        stmt.query_map([], |row| row.get(0))
            .unwrap()
            .map(|id| id.unwrap())
            .collect()
    }
}

fn test_sqlite_search() {
    let dict = SqliteRichDict::new("dict.db");
    let variants_map = rich_dict_to_variants_map(&dict);
    let start_time = Instant::now();
    let results = search::variant_search(&dict, &variants_map, "好", Script::Simplified);
    println!("{:?}", results);
    println!("{:?}", start_time.elapsed());
}
