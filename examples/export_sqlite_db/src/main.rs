use std::time::Instant;
use wordshk_tools::{
    app_api::Api,
    dict::EntryId,
    jyutping::Romanization,
    rich_dict::RichEntry,
    search::{self, RichDictLike, Script},
    sqlite_db::SqliteDb,
};

const APP_TMP_DIR: &str = "./app_tmp";

fn main() {
    // export_sqlite_db(true);
    // test_sqlite_search();
    show_pr_index_sizes();
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
    api.export_dict_as_sqlite_db(&dict_path, "3.2.4+27")
        .unwrap();
}

fn test_sqlite_search() {
    let dict = SqliteDb::new(&std::path::Path::new(APP_TMP_DIR).join("dict.db"));
    let start_time = Instant::now();
    let results = search::variant_search(&dict, &dict, "好", Script::Simplified);
    println!("{:?}", results);
    println!("{:?}", start_time.elapsed());

    let start_time = Instant::now();
    let results = search::variant_search(&dict, &dict, "苹", Script::Simplified);
    println!("{:?}", results);
    println!("{:?}", start_time.elapsed());

    let start_time = Instant::now();
    let results = search::variant_search(&dict, &dict, "苹", Script::Traditional);
    println!("{:?}", results);
    println!("{:?}", start_time.elapsed());

    let start_time = Instant::now();
    let results = search::variant_search(&dict, &dict, "蘋", Script::Simplified);
    println!("{:?}", results);
    println!("{:?}", start_time.elapsed());

    let start_time = Instant::now();
    let results = search::variant_search(&dict, &dict, "蘋", Script::Traditional);
    println!("{:?}", results);
    println!("{:?}", start_time.elapsed());

    let start_time = Instant::now();
    let results = search::english_search(&dict, &dict, "lucky", Script::Simplified);
    println!("{:?}", results);
    println!("{:?}", start_time.elapsed());

    let start_time = Instant::now();
    let results = search::pr_search(
        &dict,
        &dict,
        "mingmei",
        Script::Traditional,
        Romanization::Jyutping,
    );
    println!("{:?}", results);
    println!("{:?}", start_time.elapsed());
}

fn show_pr_index_sizes() {
    let dict = SqliteDb::new(&std::path::Path::new(APP_TMP_DIR).join("dict.db"));
    let pr_indices = wordshk_tools::pr_index::pr_indices_into_fst(
        wordshk_tools::pr_index::generate_pr_indices(&dict, Romanization::Jyutping),
    );
    println!(
        "pr_indices.fst_size_in_bytes: {}",
        pr_indices.fst_size_in_bytes()
    );
    println!(
        "pr_indices.locations_size_in_bytes: {}",
        pr_indices.locations_size_in_bytes()
    );
}
