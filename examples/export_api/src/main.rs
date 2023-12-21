use fastembed::{EmbeddingBase, EmbeddingModel, FlagEmbedding, InitOptions};
use itertools::Itertools;
use regex::Regex;
use rmp_serde::Deserializer;
use serde::Deserialize;
use std::collections::BTreeMap;
use std::io::Cursor;
use wordshk_tools::{
    app_api::Api,
    dict::clause_to_string,
    jyutping::{
        is_standard_jyutping, jyutping_to_yale, JyutPing, LaxJyutPingSegment, Romanization,
    },
    rich_dict::{RichDict, RichLine, RubySegment},
    search::{
        eg_search, pr_search, rich_dict_to_variants_map, variant_search, EgSearchRank, Script,
    },
    unicode::to_hk_safe_variant,
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

    // test_eg_search();

    // test_variant_search();

    let model = get_embedding_model();
    // test_calculate_embedding(&model);
    map_hbl_to_wordshk(&model);
}

#[derive(Deserialize)]
struct HBLRecord {
    #[serde(rename = "#")]
    index: u64,
    key: String,
    #[serde(rename = "char")]
    characters: String,
    #[serde(rename = "hbl")]
    hbl_scale: String,
    #[serde(rename = "cat")]
    category: Option<String>,
    edb: String,
    #[serde(rename = "jp")]
    jyutping: String,
    eng: String,
    cmn: Option<String>,
    urd: Option<String>,
    urd_: Option<String>,
    nep: Option<String>,
    hin: Option<String>,
    hin_: Option<String>,
    ind: Option<String>,
}

fn parse_category(cat: &str) -> String {
    let name = cat.split("__").nth(1).unwrap();
    match name {
        "v_adj" => "adj".to_string(),
        "v_loc" => "loc".to_string(),
        _ => name.split("_").next().unwrap().to_string(),
    }
}

fn get_embedding_model() -> FlagEmbedding {
    let model: FlagEmbedding = FlagEmbedding::try_new(Default::default()).unwrap();

    // List all supported models
    // dbg!(FlagEmbedding::list_supported_models());

    let model: FlagEmbedding = FlagEmbedding::try_new(InitOptions {
        model_name: EmbeddingModel::BGESmallENV15,
        show_download_message: true,
        ..Default::default()
    })
    .unwrap();

    model
}

fn test_calculate_embedding(model: &FlagEmbedding) {
    let e1 = &model.embed(vec!["query: (measure)"], None).unwrap()[0];

    let e2 = &model.embed(vec!["query: the most general measure word; used for people, round objects, abstract things, such as questions / ideas, etc."], None).unwrap()[0];

    let e3 = &model.embed(vec!["query: used before nouns (including nouns that usually take a different measure word, abstract nouns and uncountable nouns) to highlight a specific type of the thing in question"], None).unwrap()[0];

    let e4 = &model
        .embed(
            vec!["query: dollar; used for non-rounded quantities of money"],
            None,
        )
        .unwrap()[0];

    let e5 = &model.embed(vec!["query: ten thousand dollars; when it represents ten thousand, it is seldom preceded by the numerals 1-9 but instead by numerals ≥10; other restrictions include that it cannot be followed by 半 bun3 (a half), 幾 gei2 (several), nor pair with the suffixes 水 seoi2 (money) or 嘢 je5 (things)"], None).unwrap()[0];

    let e6 = &model.embed(vec!["query: pronunciation of the particle 㗎 (gaa3) before particles with an -o final due to assimilation"], None).unwrap()[0];

    let e7 = &model.embed(vec!["query: used to indicate that the action is relatively easy, relaxing and even enjoyable; added in predicate-object constructions, usually pronounced unstressed"], None).unwrap()[0];

    println!("{}", acap::cos::cosine_similarity(e1, e2));
    println!("{}", acap::cos::cosine_similarity(e1, e3));
    println!("{}", acap::cos::cosine_similarity(e1, e4));
    println!("{}", acap::cos::cosine_similarity(e1, e5));
    println!("{}", acap::cos::cosine_similarity(e1, e6));
    println!("{}", acap::cos::cosine_similarity(e1, e7));
}

fn calculate_embedding(sent: &str, model: &FlagEmbedding) -> Vec<f32> {
    model
        .embed(vec![format!("query: {}", sent)], None)
        .unwrap()
        .into_iter()
        .next()
        .unwrap()
}

fn map_hbl_to_wordshk(model: &FlagEmbedding) {
    let api = Api::load(APP_TMP_DIR);

    // Read CSV
    let mut reader = csv::ReaderBuilder::new()
        .from_path("hbl_words.csv")
        .unwrap();

    let mut category_sets = std::collections::HashSet::new();

    let hbl_category_to_pos: std::collections::HashMap<&str, &str> =
        std::collections::HashMap::from_iter([
            ("gram", "介詞"),
            ("v", "動詞"),
            ("adj", "形容詞"),
            ("loc", "方位詞"),
            ("n", "名詞"),
            ("intj", "感嘆詞"),
            ("sfp", "助詞"),
            ("pron", "代詞"),
            ("adv", "副詞"),
            ("dem", "代詞"),
            ("num", "數詞"),
            ("expr", "語句"),
            ("conj", "連詞"),
            ("mw", "量詞"),
            ("aff", "詞綴"),
            // freq is unclear so it's omitted
        ]);

    for record in reader.deserialize() {
        let record: HBLRecord = record.unwrap();
        let mut pos = None;
        if let Some(cat) = record.category {
            let cat = parse_category(&cat);
            category_sets.insert(cat.clone());
            pos = hbl_category_to_pos.get(cat.as_str());
        }

        let key_safe = to_hk_safe_variant(&record.key);
        let mut variant_matches = vec![];
        let mut strict_matches = vec![];
        for entry in api.dict.values().sorted_by_key(|entry| entry.id) {
            for variant in &entry.variants.0 {
                if variant.word == key_safe {
                    variant_matches.push(entry.id);
                    // if let Some(pos) = pos {
                    // if entry.poses.contains(&pos.to_string())
                    let matched_def_indices = entry
                        .defs
                        .iter()
                        .enumerate()
                        .filter_map(|(i, def)| {
                            def.eng
                                .as_ref()
                                .map(|eng| {
                                    let matched = Regex::new(
                                        format!(r"\b{}\b", regex::escape(&record.eng)).as_str(),
                                    )
                                    .unwrap()
                                    .is_match(&clause_to_string(eng));
                                    if matched {
                                        Some(i)
                                    } else {
                                        None
                                    }
                                })
                                .unwrap_or(None)
                        })
                        .collect::<Vec<_>>();
                    if !matched_def_indices.is_empty() {
                        strict_matches.push((entry.id, matched_def_indices));
                    }
                    break;
                }
            }
        }

        fn strict_match_to_string(m: &(usize, Vec<usize>)) -> String {
            let (id, def_indices) = m;
            format!(r#"{{"id": {}, "def_indices": {:?}}}"#, id, def_indices)
        }

        fn strict_matches_to_string(ms: &[(usize, Vec<usize>)]) -> String {
            format!(
                "[{}]",
                ms.iter().map(|m| strict_match_to_string(m)).join(",")
            )
        }

        if strict_matches.len() == 1 {
            if strict_matches[0].1.len() > 1 {
                println!(
                    r#"{{"hbl_id": {}, "word": "{}", "defs": {}, "mapping_type": "single_entry_multiple_defs"}}"#,
                    record.index,
                    record.key,
                    strict_matches_to_string(&strict_matches)
                );
            } else {
                println!(
                    r#"{{"hbl_id": {}, "word": "{}", "defs": {}, "mapping_type": "single_entry_single_def"}}"#,
                    record.index,
                    record.key,
                    strict_matches_to_string(&strict_matches)
                );
            }
        } else if variant_matches.len() > 1 {
            let query_emb = calculate_embedding(&record.eng, &model);
            // Must be above this threshold to be considered a match
            let mut best_score = 0f32;
            let mut best_match = None;
            for id in &variant_matches {
                let entry = &api.dict[&id];
                for (i, def) in entry.defs.iter().enumerate() {
                    if let Some(eng) = def.eng.as_ref() {
                        let sent = clause_to_string(eng);
                        let parts = sent.split(";").map(|part| part.trim());
                        for part in parts {
                            let part_emb = calculate_embedding(part, &model);
                            let score = acap::cos::cosine_similarity(&query_emb, &part_emb);
                            if score > best_score {
                                best_score = score;
                                best_match = Some((id, i));
                            }
                        }
                    }
                }
            }
            let inferred_def = if let Some((id, def_index)) = best_match {
                strict_matches_to_string(&[(*id, vec![def_index])])
            } else {
                "null".to_string()
            };
            let inferred_similarity = best_score;
            if strict_matches.is_empty() {
                println!(
                    r#"{{"hbl_id": {}, "word": "{}", "defs": {:?}, "mapping_type": "multiple_entries_unknown_def", "inferred_def": {}, "inferred_similarity": {}}}"#,
                    record.index, record.key, variant_matches, inferred_def, inferred_similarity
                );
            } else {
                println!(
                    r#"{{"hbl_id": {}, "word": "{}", "defs": {}, "mapping_type": "multiple_entries_multiple_defs", "inferred_def": {}, "inferred_similarity": {}}}"#,
                    record.index,
                    record.key,
                    strict_matches_to_string(&strict_matches),
                    inferred_def,
                    inferred_similarity
                );
            }
        } else if variant_matches.is_empty() {
            println!(
                r#"{{"hbl_id": {}, "word": "{}", "defs": [], "mapping_type": "no_entry"}}"#,
                record.index, record.key,
            );
        } else {
            // variant_matches.len() == 1 && strict_matches.len() != 1
            if strict_matches.is_empty() {
                let m = variant_matches[0];
                if api.dict.get(&m).unwrap().defs.len() == 1 {
                    println!(
                        r#"{{"hbl_id": {}, "word": "{}", "defs": {}, "type": "single_entry_single_def"}}"#,
                        record.index,
                        record.key,
                        strict_matches_to_string(&[(m, vec![0])])
                    );
                } else {
                    println!(
                        r#"{{"hbl_id": {}, "word": "{}", "defs": {:?}, "mapping_type": "single_entry_unknown_def"}}"#,
                        record.index, record.key, variant_matches
                    );
                }
            } else {
                println!(
                    r#"{{"hbl_id": {}, "word": "{}", "defs": {:?}, "mapping_type": "single_entry_multiple_defs"}}"#,
                    record.index, record.key, strict_matches
                );
            }
        }
    }
    // Print categories
    // println!("{:?}", category_sets);
}

fn test_variant_search() {
    let api = Api::load(APP_TMP_DIR);
    let variants_map = rich_dict_to_variants_map(&api.dict);
    let results = variant_search(&variants_map, "苹果", Script::Traditional);
    println!("{:?}", results.len());
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
    let (query_found, results) = eg_search(&api.dict, "嚟呢度", 10, Script::Traditional);
    println!("Query found: {:?}", query_found);
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

    let mut dict_decompressor = GzDecoder::new(&include_bytes!("../app_tmp/dict.json")[..]);
    let mut dict_str = String::new();
    dict_decompressor.read_to_string(&mut dict_str).unwrap();
    let dict: RichDict = serde_json::from_str(&dict_str).unwrap();

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
        let result = pr_search(&pr_indices, &dict, query, romanization);
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

    let mut dict_decompressor = GzDecoder::new(&include_bytes!("../app_tmp/dict.json")[..]);
    let mut dict_str = String::new();
    dict_decompressor.read_to_string(&mut dict_str).unwrap();
    let dict: RichDict = serde_json::from_str(&dict_str).unwrap();

    let mut pr_indices_decompressor =
        GzDecoder::new(&include_bytes!("../app_tmp/pr_indices.msgpack")[..]);
    let mut pr_indices_bytes = Vec::new();
    pr_indices_decompressor
        .read_to_end(&mut pr_indices_bytes)
        .unwrap();
    let pr_indices = rmp_serde::from_slice(&pr_indices_bytes[..])
        .expect("Failed to deserialize pr_indices from msgpack format");

    println!("Loaded pr_indices and dict");

    let romanization = Romanization::Yale;

    let test_pr_search = |expected: &str, query: &str, expected_score: usize| {
        println!("query: {}", query);
        let result = pr_search(&pr_indices, &dict, query, romanization);
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
