use rmp_serde::Deserializer;
use std::collections::BTreeMap;
use std::io::Cursor;
use wordshk_tools::{
    app_api::Api,
    jyutping::{
        is_standard_jyutping, jyutping_to_yale, JyutPing, LaxJyutPingSegment, Romanization,
    },
    rich_dict::{RichLine, RubySegment},
    search::{eg_search, pr_search, EgSearchRank, Script},
};
use xxhash_rust::xxh3::xxh3_64;

const APP_TMP_DIR: &str = "./app_tmp";

fn main() {
    // std::fs::create_dir(APP_TMP_DIR).ok();
    // let api = generate_api_json();
    // test_jyutping_search();
    // test_yale_search();
    // generate_jyutping_to_yale(&api);
    // compare_yale();

    // get_disyllabic_prs_shorter_than(8);

    test_eg_search();
}

fn test_eg_search() {
    use itertools::Itertools;

    let api = Api::load(APP_TMP_DIR);

    // simulate a mobile processor
    rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build_global()
        .unwrap();

    let start_time = std::time::Instant::now();
    let (query_normalized, results) = eg_search(&api.dict, "唔明白", 12, Script::Traditional);
    println!(
        "{}",
        results
            .iter()
            .map(
                |EgSearchRank {
                     id,
                     def_index,
                     eg_index,
                     eg_length,
                 }| {
                    api.dict[id].defs[*def_index].egs[*eg_index]
                        .yue
                        .as_ref()
                        .unwrap()
                        .clone()
                },
            )
            .join("\n")
    );
    println!("Search took {:?}", start_time.elapsed());
}

fn get_disyllabic_prs_shorter_than(characters: usize) {
    use itertools::FoldWhile::{Continue, Done};
    use itertools::Itertools;
    use std::collections::{HashMap, HashSet};

    let api = Api::load(APP_TMP_DIR);

    let mut prs: HashMap<String, HashSet<(String, Vec<u8>)>> = HashMap::new();
    for entry in api.dict.values() {
        let variant = &entry.variants.0.first().unwrap();
        let pr = &variant.prs.0.first().unwrap();
        let variant_str = &variant.word;
        let jyutpings =
            pr.0.iter()
                .fold_while(None, |mut jyutpings, pr| match pr {
                    LaxJyutPingSegment::Standard(jyutping) => match jyutpings {
                        None => Continue(Some(vec![jyutping])),
                        Some(mut jyutpings) => {
                            jyutpings.push(jyutping);
                            Continue(Some(jyutpings))
                        }
                    },
                    LaxJyutPingSegment::Nonstandard(_) => Done(None),
                })
                .into_inner();
        match jyutpings {
            None => continue,
            Some(jyutpings) => {
                if jyutpings.len() == 2 && variant_str.chars().all(wordshk_tools::unicode::is_cjk) {
                    let pr_str = jyutpings
                        .iter()
                        .map(|pr| pr.to_string_without_tone())
                        .join(" ");
                    let tones: Vec<u8> = jyutpings
                        .iter()
                        .map(|pr| pr.tone.unwrap().to_string().parse::<u8>().unwrap())
                        .collect();
                    if pr_str.len() <= characters + 1 {
                        prs.entry(pr_str)
                            .and_modify(|variants| {
                                variants.insert((variant_str.clone(), tones.clone()));
                            })
                            .or_insert(HashSet::from_iter([(variant_str.clone(), tones)]));
                    }
                }
            }
        }
    }
    println!("Found {} unique disyllabic prs", prs.len());
    std::fs::write(
        std::path::Path::new(APP_TMP_DIR).join("disyllabic.json"),
        serde_json::to_string(&prs).expect("Failed to serialize prs to JSON"),
    )
    .expect("Failed to write JSON to file");
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

    let romanization = Romanization::Jyutping;

    let test_pr_search = |expected: &str, query: &str, expected_score: usize| {
        println!("query: {}", query);
        let result = pr_search(&pr_indices, &api.dict, query, romanization);
        println!("result: {:?}", result);
        assert!(result
            .iter()
            .any(|rank| rank.pr == expected && rank.score == expected_score));
    };

    test_pr_search("hou2 coi2", "hou2 coi2", 100);
    test_pr_search("hou2 coi2", "hou2coi2", 100);
    test_pr_search("hou2 coi2", "hou2 coi3", 99);
    test_pr_search("hou2 coi2", "hou2coi3", 99);
    test_pr_search("hou2 coi2", "hou coi", 100);
    test_pr_search("hou2 coi2", "houcoi", 100);
    test_pr_search("hou2 coi2", "ho coi", 99);
    test_pr_search("hou2 coi2", "hochoi", 98);
    test_pr_search("hou2 coi2", "hocoi", 99);
    test_pr_search("hou2 coi2", "hou choi", 99);
    test_pr_search("hou2 coi2", "houchoi", 99);

    test_pr_search("bok3 laam5 wui2", "bok laam wui", 100);
    test_pr_search("bok3 laam5 wui2", "boklaamwui", 100);
    test_pr_search("bok3 laam5 wui2", "bok laahm wui", 99);
    test_pr_search("bok3 laam5 wui2", "boklaahmwui", 99);
    test_pr_search("bok3 laam5 wui2", "bok3 laam5 wui2", 100);
    test_pr_search("bok3 laam5 wui2", "bok3laam5wui2", 100);
    test_pr_search("bok3 laam5 wui2", "bok3 laam5 wui3", 99);
    test_pr_search("bok3 laam5 wui2", "bok3laam5wui3", 99);
    test_pr_search("bok3 laam5 wui2", "bok3 laam5 wui5", 99);
    test_pr_search("bok3 laam5 wui2", "bok3laam5wui5", 99);

    test_pr_search("ming4 mei4", "ming mei", 100);
    test_pr_search("ming4 mei4", "mingmei", 100);
    test_pr_search("ming4 mei4", "ming4 mei3", 99);
    test_pr_search("ming4 mei4", "ming4mei3", 99);
    test_pr_search("ming4 mei4", "ming4 mei4", 100);
    test_pr_search("ming4 mei4", "ming4mei4", 100);
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

    let romanization = Romanization::Yale;

    let test_pr_search = |expected: &str, query: &str, expected_score: usize| {
        println!("query: {}", query);
        let result = pr_search(&pr_indices, &api.dict, query, romanization);
        println!("result: {:?}", result);
        assert!(result
            .iter()
            .any(|rank| rank.pr == expected && rank.score == expected_score));
    };

    test_pr_search("hou2 coi2", "hóu chói", 100);
    test_pr_search("hou2 coi2", "hóu choi", 99);
    test_pr_search("hou2 coi2", "hou choi", 100);
    test_pr_search("hou2 coi2", "ho choi", 99);
    test_pr_search("hou2 coi2", "hou coi", 99);
    test_pr_search("hou2 coi2", "houcoi", 99);

    test_pr_search("bok3 laam5 wui2", "bok laam wui", 100);
    test_pr_search("bok3 laam5 wui2", "boklaamwui", 100);
    test_pr_search("bok3 laam5 wui2", "bok laahm wui", 99);
    test_pr_search("bok3 laam5 wui2", "boklaahmwui", 99);
    test_pr_search("bok3 laam5 wui2", "bok láahm wúi", 100);
    test_pr_search("bok3 laam5 wui2", "bokláahmwúi", 100);
    test_pr_search("bok3 laam5 wui2", "bok láahm wui", 99);
    test_pr_search("bok3 laam5 wui2", "bokláahmwui", 99);
    test_pr_search("bok3 laam5 wui2", "bok láahm wúih", 99);
    test_pr_search("bok3 laam5 wui2", "bokláahmwúih", 99);

    test_pr_search("ming4 mei4", "ming mei", 100);
    test_pr_search("ming4 mei4", "mihng mei", 99);
    test_pr_search("ming4 mei4", "mìhng mèih", 100);
    test_pr_search("ming4 mei4", "mìhngmèih", 100);
    test_pr_search("ming4 mei4", "mìhng mèi", 99);
    test_pr_search("ming4 mei4", "mìhngmèi", 99);

    test_pr_search("mei6", "meih", 100);
    test_pr_search("jat6 jat6", "yaht yaht", 100);
    test_pr_search("jat6 jat6", "yaht yat", 99);

    test_pr_search("jyun4 cyun4", "yun chyun", 100);
    test_pr_search("jyun4 cyun4", "yùhn chyùhn", 100);
    test_pr_search("jyun4 cyun4", "yùhn chyuhn", 99);
    test_pr_search("jyun4 cyun4", "yuhn chyùhn", 99);
}
