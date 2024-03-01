use itertools::Itertools;
use rand::rngs::StdRng;
use rand::{seq::SliceRandom, SeedableRng};
use regex::Regex;
use serde::{Deserialize, Serialize};
use serde_json;
use std::collections::BTreeMap;
use std::collections::HashMap;
use std::collections::HashSet;
use std::fs::File;
use std::io::Cursor;
use std::io::{self, BufRead};
use wordshk_tools::{
    app_api::Api,
    dict::clause_to_string,
    jyutping::{
        is_standard_jyutping, jyutping_to_yale, JyutPing, LaxJyutPing, LaxJyutPingSegment,
        LaxJyutPings, Romanization,
    },
    rich_dict::{RichDict, RichLine, RubySegment},
    search::{
        eg_search, pr_search, rich_dict_to_variants_map, variant_search, EgSearchRank, Script,
    },
    unicode::to_hk_safe_variant,
};

const APP_TMP_DIR: &str = "./app_tmp";

fn main() {
    std::fs::create_dir(APP_TMP_DIR).ok();
    let api = unsafe { generate_api_json() };

    // test_english_embedding_search();

    // generate_english_vocab();

    // test_pr_search_ranking();

    // test_jyutping_search();
    // test_yale_search();

    // generate_jyutping_to_yale(&api);
    // compare_yale();

    // get_disyllabic_prs_shorter_than(8);

    // test_eg_search();

    // test_variant_search();

    // let model = get_embedding_model();
    // test_calculate_embedding(&model);
    // map_hbl_to_wordshk(&model);
    // sample_mappings();

    // get_bisyllabic_words();
}

unsafe fn generate_api_json() -> Api {
    let api = Api::new(
        APP_TMP_DIR,
        include_str!("../../wordshk.csv"),
        Romanization::Yale,
    );
    api
}

fn get_bisyllabic_words() {
    use rkyv::Deserialize;

    let api = unsafe { Api::load(APP_TMP_DIR, Romanization::Jyutping) };
    let mut bisyllabic_words = vec![];
    for entry in unsafe { api.dict() }.values() {
        for variant in entry.variants.0.iter() {
            for pr in variant.prs.0.iter() {
                let pr: LaxJyutPing = pr.deserialize(&mut rkyv::Infallible).unwrap();
                if pr
                    .0
                    .iter()
                    .all(|jyutping| matches!(jyutping, LaxJyutPingSegment::Standard(_)))
                    && pr.0.len() == 2
                {
                    bisyllabic_words.push(pr.to_string());
                }
            }
        }
    }
    bisyllabic_words.sort();
    bisyllabic_words.dedup();
    println!("{:?}", bisyllabic_words);
}

#[cfg(feature = "embedding-search")]
fn test_english_embedding_search() -> anyhow::Result<()> {
    use finalfusion::prelude::*;
    use rkyv::Deserialize;
    use std::io::BufReader;
    use std::io::Read;
    use std::path::Path;
    use wordshk_tools::dict::Clause;
    use wordshk_tools::english_index::EnglishSearchRank;
    use wordshk_tools::search::english_embedding_search;

    let romanization = Romanization::Jyutping;
    let api = unsafe { Api::load(APP_TMP_DIR, romanization) };

    let mut reader = BufReader::new(File::open(
        Path::new(APP_TMP_DIR).join("english_embeddings.fifu"),
    )?);
    let phrase_embeddings = Embeddings::<VocabWrap, StorageWrap>::mmap_embeddings(&mut reader)?;

    let result = english_embedding_search(&phrase_embeddings, unsafe { api.dict() }, "beverages");

    for EnglishSearchRank {
        entry_id,
        def_index,
        matched_eng,
        score,
    } in result
    {
        let entry = unsafe { api.dict() }.get(&entry_id).unwrap();
        let variant = &entry.variants.0[0].word;
        let def = &entry.defs[def_index as usize];
        let eng = def.eng.as_ref().unwrap();
        let eng: Clause = eng.deserialize(&mut rkyv::Infallible).unwrap();
        let eng = clause_to_string(&eng);
        println!("{entry_id}\t{variant}\t{def_index}\t{score}\t{eng}");
    }

    Ok(())
}

/*
fn sample_mappings() -> Result<(), serde_json::Error> {
    // Read mappings.jsonl line by line
    let file = File::open("mappings_pos_pr_eng.jsonl").expect("Unable to open mappings.jsonl");
    let reader = io::BufReader::new(file);

    let mut lines = reader
        .lines()
        .map(|line| line.expect("Unable to read line"))
        .collect::<Vec<String>>();

    // Shuffle lines with a seeded RNG
    let mut rng = StdRng::seed_from_u64(1);
    lines.shuffle(&mut rng);

    let mut samples = [vec![], vec![], vec![], vec![], vec![], vec![]];

    let limit_per_type = 10;

    for line in lines {
        let mapping: Mapping = serde_json::from_str(&line)?;
        let mapping_type_index = match mapping {
            Mapping::single_entry_single_def { .. } => 0,
            Mapping::single_entry_multiple_defs { .. } => 1,
            Mapping::single_entry_unknown_def { .. } => 2,
            Mapping::multiple_entries_multiple_defs { .. } => 3,
            Mapping::multiple_entries_unknown_def { .. } => 4,
            Mapping::no_entry { .. } => 5,
        };
        if samples[mapping_type_index].len() < limit_per_type {
            samples[mapping_type_index].push(mapping);
        }
    }

    let api = Api::load(APP_TMP_DIR);

    let mut hbl_records = HashMap::new();

    let mut reader = csv::ReaderBuilder::new()
        .from_path("hbl_words.csv")
        .unwrap();

    for record in reader.deserialize() {
        let record: HBLRecord = record.unwrap();
        hbl_records.insert(record.index, record);
    }

    // Print samples
    for (mapping_type_index, sample) in samples.iter().enumerate() {
        match mapping_type_index {
            0 => println!("single_entry_single_def"),
            1 => println!("single_entry_multiple_defs"),
            2 => println!("single_entry_unknown_def"),
            3 => println!("multiple_entries_multiple_defs"),
            4 => println!("multiple_entries_unknown_def"),
            5 => println!("no_entry"),
            _ => unreachable!(),
        }
        for mapping in sample {
            match mapping {
                Mapping::single_entry_single_def {
                    hbl_id,
                    entry_id,
                    def_index,
                } => {
                    let hbl_record = hbl_records.get(hbl_id).unwrap();
                    let entry = &api.dict[&(*entry_id as usize)];
                    let def = &entry.defs[*def_index as usize];
                    println!(
                        "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
                        hbl_record.index,
                        hbl_record.key,
                        hbl_record
                            .category
                            .as_ref()
                            .map(|cat| parse_category(cat))
                            .unwrap_or("".to_string()),
                        entry.poses.join(","),
                        hbl_record.jyutping,
                        entry.variants.0[0].prs,
                        hbl_record.eng,
                        clause_to_string(&def.eng.as_ref().unwrap()),
                    );
                }
                Mapping::single_entry_multiple_defs {
                    hbl_id,
                    entry_id,
                    def_indices,
                    inferred_def_index,
                    inferred_similarity,
                } => {
                    let hbl_record = hbl_records.get(hbl_id).unwrap();
                    let entry = &api.dict[&(*entry_id as usize)];
                    println!(
                        "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
                        hbl_record.index,
                        hbl_record.key,
                        hbl_record
                            .category
                            .as_ref()
                            .map(|cat| parse_category(cat))
                            .unwrap_or("".to_string()),
                        entry.poses.join(","),
                        hbl_record.jyutping,
                        entry.variants.0[0].prs,
                        hbl_record.eng,
                        inferred_def_index
                            .map(|def_index| clause_to_string(
                                &entry.defs[def_index as usize].eng.as_ref().unwrap()
                            ))
                            .unwrap_or("".to_string()),
                        inferred_similarity,
                    );
                }
                Mapping::single_entry_unknown_def {
                    hbl_id,
                    entry_id,
                    inferred_def_index,
                    inferred_similarity,
                } => {
                    let hbl_record = hbl_records.get(hbl_id).unwrap();
                    let entry = &api.dict[&(*entry_id as usize)];
                    println!(
                        "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
                        hbl_record.index,
                        hbl_record.key,
                        hbl_record
                            .category
                            .as_ref()
                            .map(|cat| parse_category(cat))
                            .unwrap_or("".to_string()),
                        entry.poses.join(","),
                        hbl_record.jyutping,
                        entry.variants.0[0].prs,
                        hbl_record.eng,
                        inferred_def_index
                            .map(|def_index| clause_to_string(
                                &entry.defs[def_index as usize].eng.as_ref().unwrap()
                            ))
                            .unwrap_or("".to_string()),
                        inferred_similarity,
                    );
                }
                Mapping::multiple_entries_multiple_defs {
                    hbl_id,
                    inferred_def,
                    inferred_similarity,
                    ..
                }
                | Mapping::multiple_entries_unknown_def {
                    hbl_id,
                    inferred_def,
                    inferred_similarity,
                    ..
                } => {
                    let hbl_record = hbl_records.get(hbl_id).unwrap();
                    let entry = inferred_def
                        .as_ref()
                        .map(|inferred_def| &api.dict[&(inferred_def.entry_id as usize)]);
                    let def = entry.as_ref().and_then(|entry| {
                        inferred_def
                            .as_ref()
                            .map(|inferred_def| &entry.defs[inferred_def.def_index as usize])
                    });
                    println!(
                        "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
                        hbl_record.index,
                        hbl_record.key,
                        hbl_record
                            .category
                            .as_ref()
                            .map(|cat| parse_category(cat))
                            .unwrap_or("".to_string()),
                        entry
                            .map(|entry| entry.poses.join(","))
                            .unwrap_or("".to_string()),
                        hbl_record.jyutping,
                        entry
                            .as_ref()
                            .map(|entry| &entry.variants.0[0].prs)
                            .unwrap_or(&LaxJyutPings(vec![])),
                        hbl_record.eng,
                        def.as_ref()
                            .map(|def| clause_to_string(&def.eng.as_ref().unwrap()))
                            .unwrap_or("".to_string()),
                        inferred_similarity,
                    );
                }
                Mapping::no_entry { hbl_id } => {
                    let hbl_record = hbl_records.get(hbl_id).unwrap();
                    println!(
                        "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
                        hbl_record.index,
                        hbl_record.key,
                        hbl_record
                            .category
                            .as_ref()
                            .map(|cat| parse_category(cat))
                            .unwrap_or("".to_string()),
                        "",
                        hbl_record.jyutping,
                        "",
                        hbl_record.eng,
                        "",
                        "",
                    );
                }
            }
        }
    }

    Ok(())
}

#[derive(Serialize, Deserialize, Debug)]
struct MappingDef {
    entry_id: u64,
    def_index: u64,
}

#[derive(Debug)]
struct MappingDefGroup {
    entry_id: u64,
    def_indices: Vec<u64>,
}

#[derive(Serialize, Deserialize, Debug)]
enum Mapping {
    single_entry_single_def {
        hbl_id: u64,
        entry_id: u64,
        def_index: u64,
    },
    single_entry_multiple_defs {
        hbl_id: u64,
        entry_id: u64,
        def_indices: Vec<u64>,
        inferred_def_index: Option<u64>,
        inferred_similarity: f32,
    },
    single_entry_unknown_def {
        hbl_id: u64,
        entry_id: u64,
        inferred_def_index: Option<u64>,
        inferred_similarity: f32,
    },
    multiple_entries_multiple_defs {
        hbl_id: u64,
        defs: Vec<MappingDef>,
        inferred_def: Option<MappingDef>,
        inferred_similarity: f32,
    },
    multiple_entries_unknown_def {
        hbl_id: u64,
        entry_ids: Vec<u64>,
        inferred_def: Option<MappingDef>,
        inferred_similarity: f32,
    },
    no_entry {
        hbl_id: u64,
    },
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

fn infer_def(
    dict: &RichDict,
    eng: &str,
    variant_matches: &[u64],
    model: &FlagEmbedding,
) -> (Option<MappingDef>, f32) {
    let query_emb = calculate_embedding(&eng, &model);
    // Must be above this threshold to be considered a match
    let mut inferred_similarity = 0f32;
    let mut inferred_def = None;
    for id in variant_matches {
        let entry = &dict[&(*id as usize)];
        for (i, def) in entry.defs.iter().enumerate() {
            if let Some(eng) = def.eng.as_ref() {
                let sent = clean_eng(&clause_to_string(eng));
                let parts = sent.split(";").map(|part| part.trim());
                for part in parts {
                    let part_emb = calculate_embedding(part, &model);
                    let similarity = acap::cos::cosine_similarity(&query_emb, &part_emb);
                    if similarity > inferred_similarity {
                        inferred_similarity = similarity;
                        inferred_def = Some(MappingDef {
                            entry_id: *id,
                            def_index: i as u64,
                        });
                        if inferred_similarity == 1.0 {
                            return (inferred_def, inferred_similarity);
                        }
                    }
                }
            }
        }
    }
    (inferred_def, inferred_similarity)
}

fn clean_eng(eng: &str) -> String {
    let pat = Regex::new(r#"literally\b.+"#).unwrap();
    eng.split(";")
        .filter(|part| !pat.is_match(part.trim()))
        .join(";")
}

fn map_hbl_to_wordshk(model: &FlagEmbedding) {
    let api = Api::load(APP_TMP_DIR);

    // Read CSV
    let mut reader = csv::ReaderBuilder::new()
        .from_path("hbl_words.csv")
        .unwrap();

    let mut category_sets = std::collections::HashSet::new();

    let hbl_category_to_poses: std::collections::HashMap<&str, Vec<&str>> =
        std::collections::HashMap::from_iter([
            ("gram", vec!["介詞"]),
            ("v", vec!["動詞"]),
            ("adj", vec!["形容詞", "動詞"]),
            ("n", vec!["名詞", "語素"]),
            ("intj", vec!["感嘆詞"]),
            ("sfp", vec!["助詞"]),
            ("pron", vec!["代詞"]),
            ("adv", vec!["副詞"]),
            ("dem", vec!["代詞"]),
            ("num", vec!["數詞"]),
            ("expr", vec!["語句"]),
            ("conj", vec!["連詞"]),
            ("mw", vec!["量詞"]),
            ("aff", vec!["詞綴"]),
            // freq is unclear so it's omitted
        ]);

    for record in reader.deserialize() {
        let record: HBLRecord = record.unwrap();
        let mut poses = None;

        let cat = if record.eng == "(measure)" {
            Some("cat__mw")
        } else if record.eng == "(particle)" {
            Some("cat__sfp")
        } else {
            record.category.as_deref()
        };
        if let Some(cat) = cat {
            let cat = parse_category(&cat);
            category_sets.insert(cat.clone());
            poses = hbl_category_to_poses.get(cat.as_str());
        }

        let mut skip_eng_check = poses == Some(&vec!["量詞"]) || poses == Some(&vec!["助詞"]);

        let key_safe = to_hk_safe_variant(&record.key);
        let mut variant_matches = vec![];
        let mut strict_matches = vec![];
        for entry in api.dict().values().sorted_by_key(|entry| entry.id) {
            for variant in &entry.variants.0 {
                if variant.word == key_safe {
                    variant_matches.push(entry.id as u64);
                    // If poses is specified, only consider entries containing at least one shared pos
                    if let Some(poses) = poses {
                        if !entry.poses.iter().any(|pos| poses.contains(&pos.as_str())) {
                            continue;
                        }
                    }
                    // Check if pr matches
                    // Handles alternative prs using contains like: 牙刷 ngaa4-caat2[ngaa4-caat3]
                    let pr_matches = variant
                        .prs
                        .0
                        .iter()
                        .any(|pr| record.jyutping.contains(&pr.to_string().replace(' ', "-")));
                    if !pr_matches {
                        continue;
                    }
                    let matched_def_indices = if skip_eng_check {
                        entry
                            .defs
                            .iter()
                            .enumerate()
                            .map(|(i, _)| i as u64)
                            .collect::<Vec<_>>()
                    } else {
                        entry
                            .defs
                            .iter()
                            .enumerate()
                            .filter_map(|(i, def)| {
                                def.eng
                                    .as_ref()
                                    .map(|eng| {
                                        let eng = clean_eng(&clause_to_string(eng));
                                        let matched = Regex::new(
                                            format!(r"\b{}\b", regex::escape(&record.eng)).as_str(),
                                        )
                                        .unwrap()
                                        .is_match(&eng);
                                        if matched {
                                            Some(i as u64)
                                        } else {
                                            None
                                        }
                                    })
                                    .unwrap_or(None)
                            })
                            .collect::<Vec<_>>()
                    };
                    if !matched_def_indices.is_empty() {
                        strict_matches.push(MappingDefGroup {
                            entry_id: entry.id as u64,
                            def_indices: matched_def_indices,
                        });
                    }
                    break;
                }
            }
        }

        let mapping = if strict_matches.len() == 1 {
            assert!(variant_matches.len() >= 1);
            if strict_matches[0].def_indices.len() > 1 {
                let (inferred_def, inferred_similarity) = if skip_eng_check {
                    (None, 0.0)
                } else {
                    infer_def(
                        &api.dict(),
                        &record.eng,
                        &[strict_matches[0].entry_id],
                        &model,
                    )
                };
                Mapping::single_entry_multiple_defs {
                    hbl_id: record.index,
                    entry_id: strict_matches[0].entry_id,
                    def_indices: strict_matches[0].def_indices.clone(),
                    inferred_def_index: inferred_def.map(|def| def.def_index),
                    inferred_similarity,
                }
            } else {
                Mapping::single_entry_single_def {
                    hbl_id: record.index,
                    entry_id: strict_matches[0].entry_id,
                    def_index: strict_matches[0].def_indices[0],
                }
            }
        } else if variant_matches.len() > 1 {
            let (inferred_def, inferred_similarity) = if skip_eng_check {
                (None, 0.0)
            } else {
                infer_def(&api.dict(), &record.eng, &variant_matches, &model)
            };
            if strict_matches.is_empty() {
                Mapping::multiple_entries_unknown_def {
                    hbl_id: record.index,
                    entry_ids: variant_matches,
                    inferred_def,
                    inferred_similarity,
                }
            } else {
                assert!(strict_matches.len() > 1);
                Mapping::multiple_entries_multiple_defs {
                    hbl_id: record.index,
                    defs: strict_matches
                        .iter()
                        .flat_map(
                            |MappingDefGroup {
                                 entry_id,
                                 def_indices,
                             }| {
                                def_indices.iter().map(|def_index| MappingDef {
                                    entry_id: *entry_id,
                                    def_index: *def_index,
                                })
                            },
                        )
                        .collect(),
                    inferred_def,
                    inferred_similarity,
                }
            }
        } else if variant_matches.is_empty() {
            Mapping::no_entry {
                hbl_id: record.index,
            }
        } else {
            assert!(variant_matches.len() == 1);
            // strict_matches.len() cannot be greater than 1 because variant_matches.len() == 1
            assert!(strict_matches.is_empty());

            let entry_id = variant_matches[0] as u64;
            if api.dict().get(&(entry_id as usize)).unwrap().defs.len() == 1 {
                Mapping::single_entry_single_def {
                    hbl_id: record.index,
                    entry_id,
                    def_index: 0,
                }
            } else {
                let (inferred_def, inferred_similarity) = if skip_eng_check {
                    (None, 0.0)
                } else {
                    infer_def(&api.dict(), &record.eng, &variant_matches, &model)
                };
                Mapping::single_entry_unknown_def {
                    hbl_id: record.index,
                    entry_id,
                    inferred_def_index: inferred_def.map(|def| def.def_index),
                    inferred_similarity,
                }
            }
        };
        println!("{}", serde_json::to_string(&mapping).unwrap());
    }
    // Print categories
    // println!("{:?}", category_sets);
}

fn test_variant_search() {
    let api = Api::load(APP_TMP_DIR);
    let variants_map = rich_dict_to_variants_map(&api.dict());
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
    let (query_found, results) = eg_search(&api.dict(), "嚟呢度", 10, Script::Traditional);
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
                    api.dict()[id].defs[*def_index].egs[*eg_index]
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
    for entry in api.dict().values() {
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
    for entry in api.dict().values() {
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
*/

fn test_pr_search_ranking() {
    use wordshk_tools::search::MatchedSegment;

    let romanization = Romanization::Jyutping;
    let script = Script::Traditional;
    let api = unsafe { Api::load(APP_TMP_DIR, romanization) };
    let variants_map = wordshk_tools::search::rich_dict_to_variants_map(unsafe { &api.dict() });

    let results = pr_search(
        &api.fst_pr_indices,
        unsafe { &api.dict() },
        &variants_map,
        "gau1",
        script,
        romanization,
    );

    println!("results: {:?}", results);
}

fn test_jyutping_search() {
    use wordshk_tools::search::MatchedSegment;

    let romanization = Romanization::Jyutping;
    let script = Script::Traditional;
    let api = unsafe { Api::load(APP_TMP_DIR, romanization) };
    let variants_map = wordshk_tools::search::rich_dict_to_variants_map(unsafe { &api.dict() });

    macro_rules! test_pr_search {
        ($expected:expr, $query:expr, $expected_score:expr) => {{
            // println!("query: {}", $query);
            let result = pr_search(
                &api.fst_pr_indices,
                unsafe { &api.dict() },
                &variants_map,
                $query,
                script,
                romanization,
            );
            // println!("result: {:?}", result);
            assert!(result
                .iter()
                .any(|rank| { rank.jyutping == $expected && rank.score == $expected_score }));
        }};
    }

    test_pr_search!("hou2 coi2", "hou2 coi2", 100);
    test_pr_search!("hou2 coi2", "hou2coi2", 100);
    test_pr_search!("hou2 coi2", "hou2 coi3", 99);
    test_pr_search!("hou2 coi2", "hou2coi3", 99);
    test_pr_search!("hou2 coi2", "hou coi", 100);
    test_pr_search!("hou2 coi2", "houcoi", 100);
    test_pr_search!("hou2 coi2", "ho coi", 99);
    test_pr_search!("hou2 coi2", "hocoi", 99);
    test_pr_search!("hou2 coi2", "hou choi", 99);
    test_pr_search!("hou2 coi2", "houchoi", 99);

    test_pr_search!("bok3 laam5 wui2", "bok laam wui", 100);
    test_pr_search!("bok3 laam5 wui2", "boklaamwui", 100);
    test_pr_search!("bok3 laam5 wui2", "bok laahm wui", 99);
    test_pr_search!("bok3 laam5 wui2", "boklaahmwui", 99);
    test_pr_search!("bok3 laam5 wui2", "bok3 laam5 wui2", 100);
    test_pr_search!("bok3 laam5 wui2", "bok3laam5wui2", 100);
    test_pr_search!("bok3 laam5 wui2", "bok3 laam5 wui3", 99);
    test_pr_search!("bok3 laam5 wui2", "bok3laam5wui3", 99);
    test_pr_search!("bok3 laam5 wui2", "bok3 laam5 wui5", 99);
    test_pr_search!("bok3 laam5 wui2", "bok3laam5wui5", 99);

    test_pr_search!("ming4 mei4", "ming mei", 100);
    test_pr_search!("ming4 mei4", "mingmei", 100);
    test_pr_search!("ming4 mei4", "ming4 mei3", 99);
    test_pr_search!("ming4 mei4", "ming4mei3", 99);
    test_pr_search!("ming4 mei4", "ming4 mei4", 100);
    test_pr_search!("ming4 mei4", "ming4mei4", 100);

    test_pr_search!("ming4 mei6", "ming4 mei6", 100);

    println!("All Jyutping search tests passed!");
}

fn test_yale_search() {
    use wordshk_tools::search::MatchedSegment;

    let romanization = Romanization::Yale;
    let script = Script::Traditional;
    let api = unsafe { Api::load(APP_TMP_DIR, romanization) };
    let variants_map = wordshk_tools::search::rich_dict_to_variants_map(unsafe { &api.dict() });

    macro_rules! test_pr_search {
        ($expected:expr, $query:expr, $expected_score:expr) => {{
            // println!("query: {}", $query);
            let result = pr_search(
                &api.fst_pr_indices,
                unsafe { &api.dict() },
                &variants_map,
                $query,
                script,
                romanization,
            );
            // println!("result: {:?}", result);
            assert!(result
                .iter()
                .any(|rank| { rank.jyutping == $expected && rank.score == $expected_score }));
        }};
    }

    test_pr_search!("hou2 coi2", "hóu chói", 100);
    test_pr_search!("hou2 coi2", "hóu choi", 99);
    test_pr_search!("hou2 coi2", "hou choi", 100);
    test_pr_search!("hou2 coi2", "ho choi", 99);
    test_pr_search!("hou2 coi2", "hou coi", 99);
    test_pr_search!("hou2 coi2", "houcoi", 99);

    test_pr_search!("bok3 laam5 wui2", "bok laam wui", 100);
    test_pr_search!("bok3 laam5 wui2", "boklaamwui", 100);
    test_pr_search!("bok3 laam5 wui2", "bok laahm wui", 99);
    test_pr_search!("bok3 laam5 wui2", "boklaahmwui", 99);
    test_pr_search!("bok3 laam5 wui2", "bok láahm wúi", 100);
    test_pr_search!("bok3 laam5 wui2", "bokláahmwúi", 100);
    test_pr_search!("bok3 laam5 wui2", "bok láahm wui", 99);
    test_pr_search!("bok3 laam5 wui2", "bokláahmwui", 99);
    test_pr_search!("bok3 laam5 wui2", "bok láahm wúih", 99);
    test_pr_search!("bok3 laam5 wui2", "bokláahmwúih", 99);

    test_pr_search!("ming4 mei4", "ming mei", 100);
    test_pr_search!("ming4 mei4", "mihng mei", 99);
    test_pr_search!("ming4 mei4", "mìhng mèih", 100);
    test_pr_search!("ming4 mei4", "mìhngmèih", 100);
    test_pr_search!("ming4 mei4", "mìhng mèi", 99);
    test_pr_search!("ming4 mei4", "mìhngmèi", 99);

    test_pr_search!("ming4 mei6", "mìhng meih", 100);

    test_pr_search!("mei6", "meih", 100);
    test_pr_search!("jat6 jat6", "yaht yaht", 100);
    test_pr_search!("jat6 jat6", "yaht yat", 99);

    test_pr_search!("jyun4 cyun4", "yun chyun", 100);
    test_pr_search!("jyun4 cyun4", "yùhn chyùhn", 100);
    test_pr_search!("jyun4 cyun4", "yùhn chyuhn", 99);
    test_pr_search!("jyun4 cyun4", "yuhn chyùhn", 99);

    println!("All Yale search tests passed!");
}
