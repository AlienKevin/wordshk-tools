use super::dict::clause_to_string;
use super::rich_dict::{RichDict, RichEntry};
use super::unicode;
use itertools::Itertools;
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::collections::hash_map::HashMap;
use std::collections::hash_set::HashSet;

pub type EnglishIndex = HashMap<String, Vec<EnglishIndexData>>;

#[derive(Serialize, Deserialize, Clone)]
pub struct EnglishIndexData {
    #[serde(rename = "e")]
    pub entry_id: usize,

    #[serde(rename = "d")]
    pub def_index: usize,

    #[serde(rename = "s")]
    pub score: usize,
}

// Use reverse ordering so BTreeSet sorts in descending order according to scores
impl Ord for EnglishIndexData {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        other
            .score
            .partial_cmp(&self.score) // we don't have edge cases like NaN, Inf, etc.
            .unwrap_or(other.entry_id.cmp(&self.entry_id))
    }
}

impl PartialOrd for EnglishIndexData {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        other.score.partial_cmp(&self.score)
    }
}

impl PartialEq for EnglishIndexData {
    fn eq(&self, other: &Self) -> bool {
        other.score == self.score
    }
}

impl Eq for EnglishIndexData {}

type Counter = HashMap<String, u32>;

pub fn generate_english_index(dict: &RichDict) -> EnglishIndex {
    let mut counter: Counter = HashMap::new();
    dict.iter().for_each(|(_, entry)| {
        let mut repeated_terms = HashSet::new();
        // for _, terms in indexer.tokenize_word(word):
        tokenize_entry(entry).iter().for_each(|terms_of_defs| {
            terms_of_defs.iter().for_each(|(_, terms)| {
                terms.iter().for_each(|term| {
                    let term = unicode::american_english_stem(term);
                    if repeated_terms.contains(&term) {
                        return;
                    }
                    repeated_terms.insert(term.to_string());
                    *counter.entry(term).or_insert(0) += 1;
                });
            });
        });
    });

    let mut index = HashMap::new();

    dict.iter().for_each(|(_, entry)| {
        index_entry(&counter, entry, &mut index);
    });

    index.retain(|phrase, index_data| {
        index_data.sort();
        // Remove stem words because not used
        // in english_search
        !phrase.starts_with('!')
    });

    index
}

/// Source: zidin/indexer.py
/// This method should only filter out cruft and invalid input, add related
/// terms (eg. from thesaurus), but should not perform normalization.
/// Normalization should be done in a function that is common to this indexer
/// and the search view.
fn process_phrase(phrase: &str) -> Option<String> {
    // Remove multilines (not really expected), trim spaces and normalize to
    // lower so that our strings will match regardless of case below.
    let mut phrase = phrase.replace(['\r', '\n'], " ").trim().to_lowercase();

    // Remove parentheses
    phrase = Regex::new(r#"\(.*\)"#)
        .unwrap()
        .replace_all(&phrase, "")
        .into_owned();
    // Remove non-content words
    phrase = Regex::new(r#"literally[:,]\s*"#)
        .unwrap()
        .replace_all(&phrase, "")
        .into_owned()
        .replace("literally means", "")
        .replace("literally \"", "")
        .replace("literally '", "");

    // Convert colons into spaces
    phrase = phrase.replace(':', " ");

    // Remove multiple spaces
    phrase = phrase.split_whitespace().join(" ");

    match phrase.as_str() {
        "x" | "xxx" | "asdf" => None,
        _ => Some(phrase),
    }
}

/// Source: zidin/indexer.py
fn weight(term: &str, counter: &Counter) -> f32 {
    let stem = unicode::american_english_stem(term);
    (1.0 / ((*counter.get(stem.as_str()).unwrap() as f32).ln() + 1.0)).powi(2)
}

/// Source: zidin/indexer.py
fn score_for_term(term: &str, splitted_phrase: &[String], counter: &Counter) -> f32 {
    return weight(term, counter)
        / splitted_phrase
            .iter()
            .map(|term| weight(term, counter))
            .sum::<f32>();
}

/// Returns a normalized phrase and a normalized splitted phrase with
/// individual tokens
fn tokenize_entry(entry: &RichEntry) -> Vec<Vec<(String, Vec<String>)>> {
    entry
        .defs
        .iter()
        .filter_map(|def| {
            def.eng.as_ref().map(|eng| {
                clause_to_string(eng)
                    // Split with semicolon
                    .split(';')
                    .filter_map(|phrase| {
                        if let Some(phrase) = process_phrase(phrase) {
                            // Split the phrase into different keywords
                            let splitted_phrase = phrase
                                .split(' ')
                                .map(unicode::normalize_english_word_for_search_index)
                                .collect();
                            Some((
                                unicode::normalize_english_word_for_search_index(&phrase),
                                splitted_phrase,
                            ))
                        } else {
                            None
                        }
                    })
                    .collect()
            })
        })
        .collect()
}

fn mark_stem(stem: &str) -> String {
    format!("!{stem}")
}

fn insert_to_index(term: &str, index_data: EnglishIndexData, index: &mut EnglishIndex) {
    match index.get(&term.to_string()).and_then(|entries| {
        entries
            .iter()
            .position(|entry| entry.entry_id == index_data.entry_id)
    }) {
        Some(entry_index) => {
            let entries = index.entry(term.to_string()).or_insert(vec![]);
            if entries[entry_index].score < index_data.score {
                entries[entry_index] = index_data;
            }
        }
        None => {
            let entries = index.entry(term.to_string()).or_insert(vec![]);
            entries.push(index_data);
        }
    }
}

fn index_entry(counter: &Counter, entry: &RichEntry, index: &mut EnglishIndex) {
    // Map of term to scores -- we need to postprocess them
    let mut scores: HashMap<String, Vec<f32>> = HashMap::new();

    // For phrase, splitted_phrase in tokenize_word(word):
    tokenize_entry(entry)
        .iter()
        .enumerate()
        .for_each(|(def_index, phrases)| {
            phrases.iter().for_each(|(phrase, splitted_phrase)| {
                if splitted_phrase.len() <= 7 {
                    // Score = 100 for exact phrase match
                    insert_to_index(
                        phrase,
                        EnglishIndexData {
                            entry_id: entry.id,
                            def_index,
                            score: 100,
                        },
                        index,
                    );
                }

                splitted_phrase.iter().for_each(|term| {
                    let score = score_for_term(term, splitted_phrase, counter);

                    // Don't repeat the whole phrase
                    if splitted_phrase.len() > 1 {
                        scores.entry(term.to_string()).or_insert(vec![]).push(score);
                    }

                    let stem = unicode::american_english_stem(term);
                    scores.entry(mark_stem(&stem)).or_insert(vec![]).push(score);
                });
                scores.iter().for_each(|(term_stem, score_list)| {
                    // the uncombined original score_list is roughly a real number from 0-1
                    // with higher score a higher match. To combine two scores with same
                    // key, x and y, each we interpret it as a probability and consider the
                    // probability it is NOT a match on BOTH counts, and calculate it by
                    // 1 - (1-x)(1-y)
                    let score = (1.0
                        - score_list
                            .iter()
                            .fold(1.0, |total, score| total * (1.0 - score)))
                        * 100.0;

                    if score >= 40.0 {
                        // This includes the stems
                        insert_to_index(
                            term_stem,
                            EnglishIndexData {
                                entry_id: entry.id,
                                def_index,
                                score: score as usize,
                            },
                            index,
                        );
                    }
                });
            });
        });
}
