use std::collections::{BTreeSet, HashMap, HashSet};
use wordshk_tools::dict::{clause_to_string, Dict, Entry};
use wordshk_tools::parse::parse_dict;
use wordshk_tools::rich_dict;

fn main() {
    static DATA_FILE: &'static str = include_str!("../../wordshk.csv");
    let dict = parse_dict(DATA_FILE.as_bytes()).expect("Failed to parse dict");
    // show_colloquial_words(&dict);
    // show_tagged_words(&dict, &["爭議"]);
    // get_eng_definitions(&dict);
    // get_formal_words(&dict);
    // get_jyutping_egs(&dict);
    println!("{:?}", get_all_variants(&dict));
}

fn get_all_variants(dict: &Dict) -> BTreeSet<String> {
    dict.values()
        .flat_map(|entry| get_variants(entry))
        .collect()
}

fn get_jyutping_egs(dict: &Dict) {
    let rich_dict = rich_dict::enrich_dict(
        &dict,
        &rich_dict::EnrichDictOptions {
            remove_dead_links: true,
        },
    );

    let mut sentences = vec![];

    use kdam::tqdm;
    for entry in tqdm!(rich_dict.values()) {
        for def in &entry.defs {
            for eg in &def.egs {
                if let Some(line) = &eg.yue {
                    if let rich_dict::RichLine::Ruby(line) = &line {
                        let mut sentence = "".to_string();
                        let mut prs = vec![];
                        for segment in line {
                            match segment {
                                rich_dict::RubySegment::Punc(punc) => {
                                    sentence.push_str(&punc);
                                }
                                rich_dict::RubySegment::Word(word, word_prs) => {
                                    for (_, text) in &word.0 {
                                        sentence.push_str(&text);
                                    }
                                    prs.append(&mut word_prs.clone());
                                }
                                rich_dict::RubySegment::LinkedWord(words) => {
                                    for (word, word_prs) in words {
                                        for (_, text) in &word.0 {
                                            sentence.push_str(&text);
                                        }
                                        prs.append(&mut word_prs.clone());
                                    }
                                }
                            }
                        }
                        if sentence.contains("\n") {
                            println!("Found multiline sentence: {sentence}");
                        }
                        sentences.push((sentence, prs.join(" ")));
                    }
                }
            }
        }
    }

    let file = std::fs::File::create("jyutping_sentences.txt").unwrap();
    let mut writer = std::io::BufWriter::new(file);
    use std::io::Write;
    sentences.sort_by(|a, b| a.0.len().cmp(&b.0.len()));
    for (sentence, prs) in sentences {
        writer
            .write((sentence + "\n" + &prs + "\n\n").as_bytes())
            .unwrap();
    }
    writer.flush().unwrap();
}

fn get_formal_words(dict: &Dict) {
    use std::fs::File;
    use std::io::BufWriter;
    use std::io::Write;

    let mut formal_words = HashSet::new();
    for entry in dict.values() {
        if entry.labels.contains(&"書面語".to_string()) {
            formal_words.extend(entry.variants.0.iter().map(|variant| variant.word.clone()));
        }
    }
    let file = File::create("formal_words.txt").unwrap();
    let mut writer = BufWriter::new(file);
    for word in formal_words {
        writer.write((word + "\n").as_bytes()).unwrap();
    }
    writer.flush().unwrap();
}

fn get_eng_definitions(dict: &Dict) {
    use itertools::Itertools;
    use std::fs::File;
    use std::io::BufWriter;
    use std::io::Write;

    let mut defs: HashMap<String, HashSet<String>> = HashMap::new();
    for entry in dict.values() {
        let variants: String = entry
            .variants
            .0
            .iter()
            .map(|variant| variant.word.clone())
            .join("/");
        for def in &entry.defs {
            if let Some(eng) = &def.eng {
                defs.entry(variants.clone())
                    .and_modify(|defs| {
                        defs.insert(clause_to_string(&eng));
                    })
                    .or_insert(HashSet::from([clause_to_string(&eng)]));
            }
        }
    }
    let file = File::create("eng_definitions.json").unwrap();
    let mut writer = BufWriter::new(file);
    serde_json::to_writer(&mut writer, &defs).unwrap();
    writer.flush().unwrap();
}

fn show_colloquial_words(dict: &Dict) {
    let mut words = HashSet::new();
    // let mut poses = HashSet::new();
    for entry in dict.values() {
        if entry.labels.contains(&"書面語".to_string())
            || entry.labels.contains(&"專名".to_string())
            || entry.labels.contains(&"大陸".to_string())
            || entry.labels.contains(&"術語".to_string())
            || entry.labels.contains(&"錯字".to_string())
            || entry.labels.contains(&"文言".to_string())
            || entry.labels.contains(&"舊式".to_string())
            || has_tag(entry, &["黃賭毒", "粗俗"])
        {
            continue;
        }
        // poses.insert(entry.poses.first().unwrap().clone());
        if entry.labels.contains(&"口語".to_string()) || entry.labels.contains(&"潮語".to_string())
        {
            words.insert((get_variants(&entry), get_pos(&entry), true));
        } else {
            words.insert((get_variants(&entry), get_pos(&entry), false));
        }
    }
    for (word, pos, is_colloquial) in words {
        if is_colloquial {
            println!("{}\t{}\tcolloquial", word.join(","), pos);
        } else {
            println!("{}\t{}", word.join(","), pos);
        }
    }
    // for pos in poses {
    //     println!("{}", pos);
    // }
}

fn get_pos(entry: &Entry) -> &'static str {
    convert_zh_pos_to_en(entry.poses.first().unwrap())
}

// https://gist.github.com/hscspring/c985355e0814f01437eaf8fd55fd7998
// Convert words.hk POS tag to Jieba standard
fn convert_zh_pos_to_en(zh_pos: &str) -> &'static str {
    let map: HashMap<&str, &str> = HashMap::from_iter([
        ("方位詞", "f"),
        ("動詞", "v"),
        ("擬聲詞", "o"),
        ("感嘆詞", "e"),
        ("代詞", "r"),
        ("副詞", "d"),
        ("詞綴", "?"),
        ("名詞", "n"),
        ("連詞", "c"),
        ("量詞", "q"),
        ("助詞", "u"),
        ("區別詞", "b"),
        ("數詞", "m"),
        ("形容詞", "a"),
        ("介詞", "p"),
        ("語句", "l"),
        ("語素", "g"),
    ]);

    map.get(zh_pos).unwrap()
}

fn show_tagged_words(dict: &Dict, tags: &[&'static str]) {
    // Show offensive words in the dictionary
    let mut tagged_words = vec![];
    'outer: for entry in dict.values() {
        if has_tag(entry, &tags) {
            for word in get_variants(&entry) {
                // Filter words that are ambiguously used as a non-offensive word
                for other_entry in dict.values() {
                    if other_entry.id != entry.id
                        && other_entry
                            .variants
                            .0
                            .iter()
                            .any(|variant| variant.word == *word)
                    {
                        continue 'outer;
                    }
                }
                tagged_words.push(word.clone());
            }
        }
    }
    for word in tagged_words {
        println!("{}", word);
    }
}

fn has_tag(entry: &Entry, tags: &[&'static str]) -> bool {
    tags.iter()
        .any(|tag| entry.labels.contains(&tag.to_string()))
}

fn get_variants(entry: &Entry) -> Vec<String> {
    entry
        .variants
        .0
        .iter()
        .map(|variant| variant.word.clone())
        .collect()
}
