use rmp_serde::Deserializer;
use std::io::Cursor;
use wordshk_tools::{app_api::Api, jyutping::Romanization, search::pr_search};
use xxhash_rust::xxh3::xxh3_64;

const APP_TMP_DIR: &str = "./app_tmp";

fn main() {
    // std::fs::create_dir(APP_TMP_DIR).ok();
    // generate_api_json();
    test_yale_search();
}

fn generate_api_json() {
    let _ = Api::new(
        APP_TMP_DIR,
        include_str!("../../wordshk.csv"),
        Romanization::Yale,
    );
}

fn test_jyutping_search() {
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
        "result: {:?}\n\n",
        pr_search(&pr_indices, &api.dict, "hou coi", Romanization::Jyutping,)
    );

    println!(
        "result: {:?}\n\n",
        pr_search(&pr_indices, &api.dict, "ho coi", Romanization::Jyutping,)
    );

    println!(
        "result: {:?}\n\n",
        pr_search(&pr_indices, &api.dict, "hou choi", Romanization::Jyutping,)
    );

    println!(
        "result: {:?}\n\n",
        pr_search(&pr_indices, &api.dict, "hou2 coi", Romanization::Jyutping,)
    );

    println!(
        "result: {:?}\n\n",
        pr_search(&pr_indices, &api.dict, "hou5 coi2", Romanization::Jyutping,)
    );

    println!(
        "result: {:?}\n\n",
        pr_search(&pr_indices, &api.dict, "hou2 coi2", Romanization::Jyutping,)
    );

    println!(
        "result: {:?}\n\n",
        pr_search(&pr_indices, &api.dict, "houcoi", Romanization::Jyutping,)
    );

    println!(
        "result: {:?}\n\n",
        pr_search(&pr_indices, &api.dict, "hocoi", Romanization::Jyutping,)
    );

    println!(
        "result: {:?}\n\n",
        pr_search(&pr_indices, &api.dict, "houchoi", Romanization::Jyutping,)
    );

    println!(
        "result: {:?}\n\n",
        pr_search(&pr_indices, &api.dict, "hou2coi", Romanization::Jyutping,)
    );

    println!(
        "result: {:?}\n\n",
        pr_search(&pr_indices, &api.dict, "hou5coi2", Romanization::Jyutping,)
    );

    println!(
        "result: {:?}\n\n",
        pr_search(&pr_indices, &api.dict, "hou2coi2", Romanization::Jyutping,)
    );
}

fn test_yale_search() {
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

    // println!(
    //     "result: {:?}\n\n",
    //     pr_search(&pr_indices, &api.dict, "hou choi", Romanization::Yale,)
    // );

    // println!(
    //     "result: {:?}\n\n",
    //     pr_search(&pr_indices, &api.dict, "ho choi", Romanization::Yale,)
    // );

    // println!(
    //     "result: {:?}\n\n",
    //     pr_search(&pr_indices, &api.dict, "hou coi", Romanization::Yale,)
    // );

    // println!(
    //     "result: {:?}\n\n",
    //     pr_search(&pr_indices, &api.dict, "houchoi", Romanization::Yale,)
    // );

    // println!(
    //     "result: {:?}\n\n",
    //     pr_search(&pr_indices, &api.dict, "hochoi", Romanization::Yale,)
    // );

    // println!(
    //     "result: {:?}\n\n",
    //     pr_search(&pr_indices, &api.dict, "houcoi", Romanization::Yale,)
    // );

    println!(
        "result: {:?}\n\n",
        pr_search(&pr_indices, &api.dict, "ou dei lei", Romanization::Yale,)
    );
}
