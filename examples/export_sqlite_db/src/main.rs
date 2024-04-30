use std::time::Instant;
use wordshk_tools::{
    app_api::Api,
    dict::EntryId,
    jyutping::Romanization,
    rich_dict::RichEntry,
    search::{self, rich_dict_to_variants_map, RichDictLike, Script, SqliteRichDict},
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

fn test_sqlite_search() {
    let dict = SqliteRichDict::new("dict.db");
    let variants_map = rich_dict_to_variants_map(&dict);
    let start_time = Instant::now();
    let results = search::variant_search(&dict, &variants_map, "好", Script::Simplified);
    println!("{:?}", results);
    println!("{:?}", start_time.elapsed());
}
