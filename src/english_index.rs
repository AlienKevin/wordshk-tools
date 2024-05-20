use crate::dict::EntryId;
use crate::rich_dict::{RichDict, RichEntry};
use crate::search::{get_entry_frequency, MatchedSegment};
use crate::sqlite_db::SqliteDb;

use super::dict::clause_to_string;
use super::unicode;
use itertools::Itertools;
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::collections::hash_map::HashMap;
use std::collections::hash_set::HashSet;

pub type EnglishIndex = HashMap<String, Vec<EnglishIndexData>>;

#[ouroboros::self_referencing]
pub struct EnglishIndexLikeIteratorRows<'conn> {
    stmt: rusqlite::Statement<'conn>,
    #[borrows(mut stmt)]
    #[covariant]
    rows: rusqlite::Rows<'this>,
}

#[ouroboros::self_referencing]
pub struct EnglishIndexLikeIterator {
    conn: r2d2::PooledConnection<r2d2_sqlite::SqliteConnectionManager>,
    #[borrows(conn)]
    #[covariant]
    rows: EnglishIndexLikeIteratorRows<'this>,
}

pub trait EnglishIndexLike: Sync + Send {
    fn get(&self, phrase: &str) -> Option<Vec<EnglishIndexData>>;

    fn iter(&self) -> EnglishIndexLikeIterator;
}

impl Iterator for EnglishIndexLikeIterator {
    type Item = String;

    fn next(&mut self) -> Option<Self::Item> {
        self.with_rows_mut(|rows| {
            rows.with_rows_mut(|rows| {
                rows.next()
                    .ok()
                    .and_then(|row| row.map(|row| row.get(0).unwrap()))
            })
        })
    }
}

impl EnglishIndexLike for SqliteDb {
    fn get(&self, phrase: &str) -> Option<Vec<EnglishIndexData>> {
        let conn = self.conn();
        let mut stmt = conn
            .prepare("SELECT english_index_data FROM english_index WHERE phrase = ?")
            .unwrap();
        let english_index_data_string: Option<String> =
            stmt.query_row([phrase], |row| row.get(0)).ok();
        english_index_data_string.map(|string| serde_json::from_str(&string).unwrap())
    }

    fn iter(&self) -> EnglishIndexLikeIterator {
        let conn = self.conn();
        EnglishIndexLikeIteratorTryBuilder {
            conn,
            rows_builder: |c| {
                EnglishIndexLikeIteratorRowsTryBuilder {
                    stmt: c.prepare("SELECT phrase FROM english_index").unwrap(),
                    rows_builder: |s| s.query([]),
                }
                .try_build()
            },
        }
        .try_build()
        .unwrap()
    }
}

#[derive(Serialize, Deserialize, Clone)]
pub struct EnglishIndexData {
    #[serde(rename = "e")]
    pub entry_id: EntryId,

    #[serde(rename = "d")]
    pub def_index: usize,

    #[serde(rename = "s")]
    pub score: usize,
}

#[derive(Debug, Clone)]
pub struct EnglishSearchRank {
    pub entry_id: EntryId,
    pub def_len: usize,
    pub def_index: usize,
    pub variant: String,
    pub score: usize,
    pub frequency_count: usize,
    pub matched_eng: Vec<MatchedSegment>,
}

impl Ord for EnglishSearchRank {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        other
            .score
            .partial_cmp(&self.score) // we don't have edge cases like NaN, Inf, etc.
            .unwrap_or(
                self.frequency_count
                    .cmp(&other.frequency_count)
                    .then(other.variant.cmp(&self.variant))
                    .then_with(|| {
                        get_entry_frequency(self.entry_id).cmp(&get_entry_frequency(other.entry_id))
                    })
                    .then(self.def_len.cmp(&other.def_len))
                    .then(other.entry_id.cmp(&self.entry_id))
                    .then(other.def_index.cmp(&self.def_index)),
            )
    }
}
impl PartialOrd for EnglishSearchRank {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(other.cmp(&self))
    }
}

impl PartialEq for EnglishSearchRank {
    fn eq(&self, other: &Self) -> bool {
        other.score == self.score
    }
}

impl Eq for EnglishSearchRank {}

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
        Some(other.cmp(&self))
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
    for (_, entry) in dict.iter() {
        let mut repeated_terms = HashSet::new();
        // for _, terms in indexer.tokenize_word(word):
        split_entry_eng_defs_into_phrases(entry, unicode::normalize_english_word_for_search_index)
            .iter()
            .for_each(|(_, phrases)| {
                phrases.iter().for_each(|phrase| {
                    let splitted_phrase = phrase
                        .split_whitespace()
                        .map(|s| s.to_string())
                        .collect_vec();
                    splitted_phrase.iter().for_each(|term| {
                        let term = unicode::american_english_stem(term);
                        if repeated_terms.contains(&term) {
                            return;
                        }
                        repeated_terms.insert(term.to_string());
                        *counter.entry(term).or_insert(0) += 1;
                    });
                });
            });
    }

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
        "" => None,
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

/// Returns a normalized phrase with
/// individual tokens
pub fn split_entry_eng_defs_into_phrases(
    entry: &RichEntry,
    normalize: fn(&str) -> String,
) -> Vec<(usize, Vec<String>)> {
    entry
        .defs
        .iter()
        .enumerate()
        .filter_map(|(def_index, def)| {
            def.eng.as_ref().and_then(|eng| {
                let phrases = clause_to_string(&eng)
                    // Split with semicolon
                    .split(';')
                    .filter_map(|s| process_phrase(&normalize(s)))
                    .collect_vec();
                if phrases.is_empty() {
                    None
                } else {
                    Some((def_index, phrases))
                }
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
    split_entry_eng_defs_into_phrases(entry, unicode::normalize_english_word_for_search_index)
        .iter()
        .for_each(|(def_index, phrases)| {
            phrases.iter().for_each(|phrase| {
                let splitted_phrase = phrase
                    .split_whitespace()
                    .map(|s| s.to_string())
                    .collect_vec();
                if splitted_phrase.len() <= 7 {
                    // Score = 100 for exact phrase match
                    insert_to_index(
                        phrase,
                        EnglishIndexData {
                            entry_id: entry.id,
                            def_index: *def_index,
                            score: 100,
                        },
                        index,
                    );
                }

                splitted_phrase.iter().for_each(|term| {
                    let score = score_for_term(term, &splitted_phrase, counter);

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
                                def_index: *def_index,
                                score: score as usize,
                            },
                            index,
                        );
                    }
                });
            });
        });
}
