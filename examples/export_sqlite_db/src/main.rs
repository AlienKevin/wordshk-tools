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
    export_sqlite_db(true);
    // test_sqlite_search();
}

fn export_sqlite_db(regenerate_api_from_csv: bool) {
    let api = if regenerate_api_from_csv {
        std::fs::create_dir(APP_TMP_DIR).ok();
        unsafe {
            Api::new(
                APP_TMP_DIR,
                include_str!("../../wordshk.csv"),
                Romanization::Jyutping,
            )
        }
    } else {
        unsafe { Api::load(APP_TMP_DIR, Romanization::Jyutping) }
    };

    let dict_path = std::path::Path::new(APP_TMP_DIR).join("dict.db");
    if std::fs::metadata(&dict_path).is_ok() {
        std::fs::remove_file(&dict_path).unwrap();
    }
    api.export_dict_as_sqlite_db(&dict_path, "3.2.3+26")
        .unwrap();
}

fn test_sqlite_search() {
    let dict = SqliteRichDict::new("dict.db");
    let variants_map = rich_dict_to_variants_map(&dict);
    let start_time = Instant::now();
    let results = search::variant_search(&dict, &variants_map, "好", Script::Simplified);
    println!("{:?}", results);
    println!("{:?}", start_time.elapsed());
}
