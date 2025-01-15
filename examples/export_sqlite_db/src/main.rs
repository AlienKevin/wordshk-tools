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
    // export_sqlite_db();
    test_sqlite_search();
    // show_pr_index_sizes();
}

fn export_sqlite_db() {
    std::fs::create_dir(APP_TMP_DIR).ok();
    let api = Api::new(include_str!("../../wordshk.csv"));

    let dict_path = std::path::Path::new(APP_TMP_DIR).join("dict.db");
    if std::fs::metadata(&dict_path).is_ok() {
        std::fs::remove_file(&dict_path).unwrap();
    }
    api.export_dict_as_sqlite_db(&dict_path, "3.4.1+34")
        .unwrap();
}

fn test_sqlite_search() {
    let dict = SqliteDb::new(&std::path::Path::new(APP_TMP_DIR).join("dict.db"));
    let start_time = Instant::now();
    let results = search::variant_search(&dict, &dict, "好", Script::Simplified);
    // println!("{:?}", results);
    println!("{:?}", start_time.elapsed());

    let start_time = Instant::now();
    let results = search::variant_search(&dict, &dict, "苹", Script::Simplified);
    // println!("{:?}", results);
    println!("{:?}", start_time.elapsed());

    let start_time = Instant::now();
    let results = search::variant_search(&dict, &dict, "苹", Script::Traditional);
    // println!("{:?}", results);
    println!("{:?}", start_time.elapsed());

    let start_time = Instant::now();
    let results = search::variant_search(&dict, &dict, "蘋", Script::Simplified);
    // println!("{:?}", results);
    println!("{:?}", start_time.elapsed());

    let start_time = Instant::now();
    let results = search::variant_search(&dict, &dict, "蘋", Script::Traditional);
    // println!("{:?}", results);
    println!("{:?}", start_time.elapsed());

    let start_time = Instant::now();
    let results = search::variant_search(&dict, &dict, "医返", Script::Simplified);
    // println!("{:?}", results);
    println!("{:?}", start_time.elapsed());

    let start_time = Instant::now();
    let results = search::english_search(&dict, &dict, "lucky", Script::Simplified);
    // println!("{:?}", results);
    println!("{:?}", start_time.elapsed());

    let start_time = Instant::now();
    let results = search::english_search(&dict, &dict, "thank you", Script::Simplified);
    // println!("{:?}", results);
    println!("{:?}", start_time.elapsed());

    let start_time = Instant::now();
    let results = search::pr_search(
        &dict,
        &dict,
        "mingmei",
        Script::Traditional,
        Romanization::Jyutping,
    );
    // println!("{:?}", results);
    println!("{:?}", start_time.elapsed());

    let start_time = Instant::now();
    let results = search::pr_search(
        &dict,
        &dict,
        "ming4mei4",
        Script::Traditional,
        Romanization::Jyutping,
    );
    // println!("{:?}", results);
    println!("{:?}", start_time.elapsed());

    // Edge case where a long query would trigger TooManyStates(10000) during Levenshtein::new if not handled correctly
    let start_time = Instant::now();
    let results = search::pr_search(
        &dict,
        &dict,
        "keoi5dei6so2ngo5wai4e6aa3jan1wai6bei3ngoi3jan4fei4po4si3dong1zan1",
        Script::Traditional,
        Romanization::Jyutping,
    );
    // println!("{:?}", results);
    println!("{:?}", start_time.elapsed());

    let start_time = Instant::now();
    let results = search::eg_search(&dict, "閂水喉", Script::Traditional);
    println!("{:?}", start_time.elapsed());
    println!("{:?}", results);

    let start_time = Instant::now();
    let results = search::eg_search(&dict, "佢話", Script::Traditional);
    println!("{:?}", start_time.elapsed());
    println!("{:?}", results);

    let start_time = Instant::now();
    let results = search::eg_search(&dict, "我", Script::Traditional);
    println!("{:?}", start_time.elapsed());
    // println!("{:?}", results);
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
