use rmp_serde::Deserializer;
use std::collections::BTreeMap;
use std::io::Cursor;
use wordshk_tools::{
    app_api::Api,
    jyutping::{
        is_standard_jyutping, jyutping_to_yale, JyutPing, LaxJyutPingSegment, Romanization,
    },
    rich_dict::{RichLine, RubySegment},
    search::pr_search,
};
use xxhash_rust::xxh3::xxh3_64;

const APP_TMP_DIR: &str = "./app_tmp";

fn main() {
    // std::fs::create_dir(APP_TMP_DIR).ok();
    // let api = generate_api_json();
    // test_yale_search();
    // generate_jyutping_to_yale(&api);
    compare_yale();
}

fn compare_yale() {
    let jyutping_to_yale_json =
        std::fs::read_to_string(std::path::Path::new(APP_TMP_DIR).join("jyutping_to_yale.json"))
            .expect("Unable to read file");
    let jyutping_to_yale: BTreeMap<String, String> =
        serde_json::from_str(&jyutping_to_yale_json).expect("Error parsing the JSON");

    let yyzd: BTreeMap<String, String> =
        read_yydz_romanization("yyzd_romanization.tsv").expect("Failed to read yydz romanization");

    let (unique_to_jyutping_to_yale, unique_to_yyzd, differing_values) =
        compare_btreemaps(&jyutping_to_yale, &yyzd);

    println!(
        "Keys unique to jyutping_to_yale: {:?}",
        unique_to_jyutping_to_yale
    );
    println!("Keys unique to yyzd: {:?}", unique_to_yyzd);
    // println!("Keys with differing values: {:?}", differing_values);
    for (jyutping_to_yale, yyzd) in differing_values {
        println!("{}", format!("{yyzd} -> {jyutping_to_yale}"));
    }
}

fn compare_btreemaps(
    a: &BTreeMap<String, String>,
    b: &BTreeMap<String, String>,
) -> (Vec<String>, Vec<String>, Vec<(String, String)>) {
    let mut unique_to_a = Vec::new();
    let mut unique_to_b = Vec::new();
    let mut differing_values = Vec::new();

    for (key, value) in a.iter() {
        match b.get(key) {
            Some(b_value) if b_value == value => {}
            Some(b_value) => differing_values.push((value.clone(), b_value.clone())),
            None => unique_to_a.push(key.clone()),
        }
    }

    for key in b.keys() {
        if !a.contains_key(key) {
            unique_to_b.push(key.clone());
        }
    }

    (unique_to_a, unique_to_b, differing_values)
}

fn read_yydz_romanization<P: AsRef<std::path::Path>>(
    file_path: P,
) -> Result<BTreeMap<String, String>, Box<dyn std::error::Error>> {
    // Open the TSV file
    let file = std::fs::File::open(file_path)?;

    // Create a CSV reader with a tab as a delimiter
    let mut rdr = csv::ReaderBuilder::new().delimiter(b'\t').from_reader(file);

    // Create an empty BTreeMap
    let mut map = BTreeMap::new();

    // Skip the header
    rdr.records().next();

    // Read each record
    for result in rdr.records() {
        let record = result?;
        let key = record.get(0).ok_or("No first column")?.to_string();
        let value = record.get(2).ok_or("No third column")?.to_string();
        map.insert(key, value);
    }

    Ok(map)
}

fn generate_jyutping_to_yale(api: &Api) {
    use std::collections::HashSet;

    let mut jyutpings: HashSet<String> = HashSet::new();
    for entry in api.dict.values() {
        for variant in &entry.variants.0 {
            for prs in &variant.prs.0 {
                for jyutping in &prs.0 {
                    if let LaxJyutPingSegment::Standard(jyutping) = jyutping {
                        jyutpings.insert(jyutping.to_string());
                    }
                }
            }
        }
        for def in &entry.defs {
            for eg in &def.egs {
                if let Some(zho) = &eg.zho {
                    if let RichLine::Ruby(line) = zho {
                        for seg in line {
                            match seg {
                                RubySegment::Word(_, prs) => {
                                    jyutpings.extend(prs.clone());
                                }
                                RubySegment::LinkedWord(words) => {
                                    for (_, prs) in words {
                                        jyutpings.extend(prs.clone());
                                    }
                                }
                                _ => {
                                    // do nothing
                                }
                            }
                        }
                    }
                }
                if let Some(yue) = &eg.yue {
                    if let RichLine::Ruby(line) = yue {
                        for seg in line {
                            match seg {
                                RubySegment::Word(_, prs) => {
                                    jyutpings.extend(prs.clone());
                                }
                                RubySegment::LinkedWord(words) => {
                                    for (_, prs) in words {
                                        jyutpings.extend(prs.clone());
                                    }
                                }
                                _ => {
                                    // do nothing
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    let mut map: BTreeMap<String, String> = BTreeMap::new();

    for jyutping in &jyutpings {
        if is_standard_jyutping(jyutping) {
            map.insert(jyutping.clone(), jyutping_to_yale(jyutping));
        }
    }

    let json = serde_json::to_string_pretty(&map)
        .expect("Failed to serialize jyutping 2 yale mapping to JSON");
    let output_path = std::path::Path::new(APP_TMP_DIR).join("jyutping_to_yale.json");
    std::fs::write(output_path, json).expect("Failed to write jyutping 2 yale JSON to file");
}

fn generate_api_json() -> Api {
    let api = Api::new(
        APP_TMP_DIR,
        include_str!("../../wordshk.csv"),
        Romanization::Yale,
    );
    api
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

    println!("Loaded pr_indices and api");

    println!(
        "result: {:?}\n\n",
        pr_search(&pr_indices, &api.dict, "hou choi", Romanization::Yale,)
    );

    println!(
        "result: {:?}\n\n",
        pr_search(&pr_indices, &api.dict, "ho choi", Romanization::Yale,)
    );

    println!(
        "result: {:?}\n\n",
        pr_search(&pr_indices, &api.dict, "hou coi", Romanization::Yale,)
    );

    println!(
        "result: {:?}\n\n",
        pr_search(&pr_indices, &api.dict, "houchoi", Romanization::Yale,)
    );

    println!(
        "result: {:?}\n\n",
        pr_search(&pr_indices, &api.dict, "hochoi", Romanization::Yale,)
    );

    println!(
        "result: {:?}\n\n",
        pr_search(&pr_indices, &api.dict, "houcoi", Romanization::Yale,)
    );

    println!(
        "result: {:?}\n\n",
        pr_search(&pr_indices, &api.dict, "bok laam wui", Romanization::Yale,)
    );

    println!(
        "result: {:?}\n\n",
        pr_search(&pr_indices, &api.dict, "ming mei", Romanization::Yale,)
    );
}
