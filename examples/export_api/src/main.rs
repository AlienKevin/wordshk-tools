use wordshk_tools::{app_api::Api, jyutping::Romanization, search::pr_search};

const APP_TMP_DIR: &str = "./app_tmp";

fn main() {
    // std::fs::create_dir(APP_TMP_DIR).ok();
    // generate_api_json();
    test_pr_search();
}

fn generate_api_json() {
    let _ = Api::new(
        APP_TMP_DIR,
        include_str!("../../wordshk.csv"),
        Romanization::Jyutping,
    );
}

fn test_pr_search() {
    use flate2::read::GzDecoder;
    use flate2::Compression;
    use serde::Deserialize;
    use std::io::prelude::*;

    let mut api_decompressor = GzDecoder::new(&include_bytes!("../app_tmp/api.json")[..]);
    let mut api_str = String::new();
    api_decompressor.read_to_string(&mut api_str).unwrap();
    let mut api: Api = serde_json::from_str(&api_str).unwrap();

    let mut pr_indices_decompressor =
        GzDecoder::new(&include_bytes!("../app_tmp/pr_indices.json")[..]);
    let mut pr_indices_str = String::new();
    pr_indices_decompressor
        .read_to_string(&mut pr_indices_str)
        .unwrap();
    let pr_indices = serde_json::from_str(&pr_indices_str).unwrap();

    println!(
        "result: {:?}",
        pr_search(
            &pr_indices,
            &api.dict,
            "ming baak",
            Romanization::Jyutping,
        )
    );
}
