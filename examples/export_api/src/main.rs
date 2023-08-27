use rmp_serde::Deserializer;
use std::io::Cursor;
use wordshk_tools::{app_api::Api, jyutping::Romanization, search::pr_search};
use xxhash_rust::xxh3::xxh3_64;

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
        GzDecoder::new(&include_bytes!("../app_tmp/pr_indices.msgpack")[..]);
    let mut pr_indices_bytes = Vec::new();
    pr_indices_decompressor
        .read_to_end(&mut pr_indices_bytes)
        .unwrap();
    let pr_indices = rmp_serde::from_slice(&pr_indices_bytes[..])
        .expect("Failed to deserialize pr_indices from msgpack format");

    println!(
        "result: {:?}",
        pr_search(&pr_indices, &api.dict, "hou coi", Romanization::Jyutping,)
    );

    println!(
        "{:?}",
        pr_indices.space[0].get(&xxh3_64("hou coi".as_bytes()))
    );
}
