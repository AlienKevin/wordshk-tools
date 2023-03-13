use wordshk_tools::dict::{Dict, Entry};
use wordshk_tools::parse::parse_dict;

fn main() {
    static DATA_FILE: &'static str = include_str!("../../wordshk.csv");
    let dict = parse_dict(DATA_FILE.as_bytes()).expect("Failed to parse dict");
    show_colloquial_words(&dict);
}

fn show_colloquial_words(dict: &Dict) {
    let mut words = std::collections::HashSet::new();
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
        if entry.labels.contains(&"口語".to_string()) || entry.labels.contains(&"潮語".to_string())
        {
            words.insert((get_variants(&entry), true));
        } else {
            words.insert((get_variants(&entry), false));
        }
    }
    for (word, is_colloquial) in words {
        if is_colloquial {
            println!("{}\tcolloquial", word.join(","));
        } else {
            println!("{}", word.join(","));
        }
    }
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
    entry.variants.0.iter().map(|variant| variant.word.clone()).collect()
}
