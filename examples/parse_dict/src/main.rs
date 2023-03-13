use wordshk_tools::dict::{Dict, Entry};
use wordshk_tools::parse::parse_dict;
use std::collections::{HashMap, HashSet};

fn main() {
    static DATA_FILE: &'static str = include_str!("../../wordshk.csv");
    let dict = parse_dict(DATA_FILE.as_bytes()).expect("Failed to parse dict");
    show_colloquial_words(&dict);
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
            || is_offensive(entry)
        {
            continue;
        }
        // poses.insert(entry.poses.first().unwrap().clone());
        if entry.labels.contains(&"口語".to_string()) || entry.labels.contains(&"潮語".to_string())
        {
            words.insert((
                get_variants(&entry),
                get_pos(&entry),
                true,
            ));
        } else {
            words.insert((
                get_variants(&entry),
                get_pos(&entry),
                false,
            ));
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

fn show_offensive(dict: &Dict) {
    // Show offensive words in the dictionary
    let mut offensive_words = vec![];
    'outer: for entry in dict.values() {
        if is_offensive(entry) {
            for word in get_variants(&entry) {
                // Filter words that are ambiguously used as a non-offensive word
                for other_entry in dict.values() {
                    if other_entry.id != entry.id
                        && other_entry
                            .variants
                            .0
                            .iter()
                            .any(|variant| variant.word == *word)
                        && !is_offensive(other_entry)
                    {
                        continue 'outer;
                    }
                }
                offensive_words.push(word.clone());
            }
        }
    }
    for word in offensive_words {
        println!("{}", word);
    }
}

fn is_offensive(entry: &Entry) -> bool {
    entry.labels.contains(&"黃賭毒".to_string()) || entry.labels.contains(&"粗俗".to_string())
}

fn get_variants(entry: &Entry) -> Vec<String> {
    entry
        .variants
        .0
        .iter()
        .map(|variant| variant.word.clone())
        .collect()
}
