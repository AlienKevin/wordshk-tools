use crate::dict::{clause_to_string, Clause, EntryId};
use crate::eg_index::EgIndexLike;
use crate::jyutping::{parse_jyutpings, remove_yale_diacritics, LaxJyutPing};
use crate::lihkg_frequencies::LIHKG_FREQUENCIES;
use crate::mandarin_variant_index::MandarinVariantIndexLike;
use crate::pr_index::{FstPrIndicesLike, FstSearchResult, PrLocation, MAX_DELETIONS};
use crate::rich_dict::{RichVariant, RichVariants};
use crate::sqlite_db::SqliteDb;
use crate::variant_index::VariantIndexLike;

use super::charlist::CHARLIST;
use super::dict::{Variant, Variants};
use super::english_index::{EnglishIndexData, EnglishIndexLike, EnglishSearchRank};
use super::iconic_simps::ICONIC_SIMPS;
use super::iconic_trads::ICONIC_TRADS;
use super::jyutping::{LaxJyutPings, Romanization};
use super::rich_dict::RichEntry;
use super::unicode;
use super::word_frequencies::WORD_FREQUENCIES;
use core::fmt;
use itertools::Itertools;
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use regex::Regex;
use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::collections::{BinaryHeap, HashSet};
use strsim::{generic_levenshtein, levenshtein, normalized_levenshtein};
use unicode_normalization::UnicodeNormalization;

/// Max score is 100
type Score = usize;

const MAX_SCORE: Score = 100;

type Index = usize;

#[derive(Copy, Clone, PartialEq, Eq)]
pub enum Script {
    Simplified,
    Traditional,
}

impl fmt::Display for Script {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Script::Simplified => write!(f, "simp"),
            Script::Traditional => write!(f, "trad"),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ComboVariant {
    pub word_trad: String,
    pub word_simp: String,
    pub prs: LaxJyutPings,
}

pub type ComboVariants = Vec<ComboVariant>;

pub fn create_combo_variants(
    trad_variants: &Variants,
    simp_variant_strings: &Vec<String>,
) -> ComboVariants {
    trad_variants
        .0
        .iter()
        .enumerate()
        .map(|(variant_index, Variant { word, prs, .. })| ComboVariant {
            word_trad: word.clone(),
            word_simp: simp_variant_strings[variant_index].clone(),
            prs: prs.clone(),
        })
        .collect()
}

#[derive(Clone, Eq, PartialEq, Debug, serde::Serialize, serde::Deserialize)]
pub struct MatchedSegment {
    pub segment: String,
    pub matched: bool,
}

#[derive(Clone, Eq, PartialEq, Debug)]
pub struct PrSearchRank {
    pub id: EntryId,
    pub def_len: usize,
    // (variant_index, pr_index)
    pub variant_indices: Vec<(Index, Index)>,
    pub variants: Vec<String>,
    pub jyutping: String,
    pub matched_pr: Vec<MatchedSegment>,
    pub num_matched_initial_chars: u32,
    pub num_matched_final_chars: u32,
    pub score: Score,
    pub frequency_count: Index,
}

impl Ord for PrSearchRank {
    fn cmp(&self, other: &Self) -> Ordering {
        self.score
            .cmp(&other.score)
            .then(
                self.num_matched_initial_chars
                    .cmp(&other.num_matched_initial_chars),
            )
            .then(
                self.num_matched_final_chars
                    .cmp(&other.num_matched_final_chars),
            )
            .then(other.jyutping.cmp(&self.jyutping))
            .then(self.frequency_count.cmp(&other.frequency_count))
            .then(other.variants.cmp(&self.variants))
            .then_with(|| get_entry_frequency(self.id).cmp(&get_entry_frequency(other.id)))
            .then(self.def_len.cmp(&other.def_len))
            .then(other.id.cmp(&self.id))
    }
}

impl PartialOrd for PrSearchRank {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

pub trait RichDictLike: Sync + Send {
    fn get_entry(&self, id: EntryId) -> Option<RichEntry>;
    fn get_entries(&self, ids: &[EntryId]) -> HashMap<EntryId, RichEntry>;
    fn get_ids(&self) -> Vec<EntryId>;
}

fn pick_variant(variant: &RichVariant, script: Script) -> Variant {
    match script {
        Script::Simplified => Variant {
            word: variant.word_simp.clone(),
            prs: variant.prs.clone(),
        },
        Script::Traditional => Variant {
            word: variant.word.clone(),
            prs: variant.prs.clone(),
        },
    }
}

impl RichDictLike for SqliteDb {
    fn get_entry(&self, entry_id: EntryId) -> Option<RichEntry> {
        let conn = self.conn();
        let mut stmt = conn
            .prepare("SELECT entry FROM rich_dict WHERE id = ?")
            .ok()?;
        let entry_str: String = stmt.query_row([entry_id], |row| row.get(0)).ok()?;
        serde_json::from_str(&entry_str).ok()
    }

    fn get_entries(&self, entry_ids: &[EntryId]) -> HashMap<EntryId, RichEntry> {
        let batch_size = 100;
        let entry_ids_batches: Vec<&[EntryId]> = entry_ids.chunks(batch_size).collect();

        // Process batches in parallel using Rayon
        let result: HashMap<EntryId, RichEntry> = entry_ids_batches
            .par_iter()
            .map(|batch| {
                let conn = self.conn();
                let placeholders = batch.iter().map(|_| "?").collect::<Vec<_>>().join(", ");
                let query = format!(
                    "SELECT id, entry FROM rich_dict WHERE id IN ({})",
                    placeholders
                );

                let mut stmt = match conn.prepare(&query) {
                    Ok(stmt) => stmt,
                    Err(_) => return HashMap::new(),
                };

                let rows = match stmt.query_map(rusqlite::params_from_iter(*batch), |row| {
                    let id = row.get::<_, EntryId>(0)?;
                    let entry_str = row.get::<_, String>(1)?;
                    let entry = serde_json::from_str(&entry_str).map_err(|serde_err| {
                        rusqlite::Error::FromSqlConversionFailure(
                            1, // hint
                            rusqlite::types::Type::Text,
                            Box::new(serde_err),
                        )
                    })?;
                    Ok((id, entry))
                }) {
                    Ok(rows) => rows,
                    Err(_) => return HashMap::new(),
                };

                let mut batch_entries = HashMap::new();
                for row in rows {
                    if let Ok((id, entry)) = row {
                        batch_entries.insert(id, entry);
                    }
                }

                batch_entries
            })
            .reduce(HashMap::new, |mut acc, batch| {
                acc.extend(batch);
                acc
            });

        result
    }

    fn get_ids(&self) -> Vec<EntryId> {
        let conn = self.conn();
        let mut stmt = conn.prepare("SELECT id FROM rich_dict").unwrap();
        stmt.query_map([], |row| row.get(0))
            .unwrap()
            .map(|id| id.unwrap())
            .collect()
    }
}

pub fn pick_variants(variants: &RichVariants, script: Script) -> Variants {
    Variants(
        variants
            .0
            .iter()
            .map(|variant| pick_variant(variant, script))
            .collect(),
    )
}

pub trait VariantMapLike {
    fn get(&self, query: &str, script: Script) -> Option<EntryId>;
}

impl VariantMapLike for SqliteDb {
    fn get(&self, query: &str, script: Script) -> Option<EntryId> {
        let conn = self.conn();
        let mut stmt = conn
            .prepare(&format!(
                "SELECT entry_id FROM variant_map_{script} WHERE variant = ?"
            ))
            .unwrap();
        stmt.query_row([query], |row| row.get(0)).ok()
    }
}

pub(crate) fn get_entry_frequency(entry_id: EntryId) -> u8 {
    *WORD_FREQUENCIES.get(&entry_id).unwrap_or(&50)
}

fn get_max_frequency_count(variants: &RichVariants) -> usize {
    *variants
        .0
        .iter()
        .max_by(|variant1, variant2| {
            LIHKG_FREQUENCIES
                .get(&variant1.word)
                .unwrap_or(&0)
                .cmp(LIHKG_FREQUENCIES.get(&variant2.word).unwrap_or(&0))
        })
        .map(|most_frequent_variant| {
            LIHKG_FREQUENCIES
                .get(&most_frequent_variant.word)
                .unwrap_or(&0)
        })
        .unwrap_or(&0)
}

fn regex_remove(input: &str, pattern: &str) -> (String, Vec<(usize, String)>) {
    let re = Regex::new(pattern).unwrap();
    let mut removed_substrings = Vec::new();

    // Convert the input string to a vector of characters for proper Unicode handling
    let input_chars: Vec<char> = input.chars().collect();

    for mat in re.find_iter(input) {
        let start = input[..mat.start()].chars().count();
        let end = input[..mat.end()].chars().count();
        let matched_str: String = input_chars[start..end].iter().collect();

        // Store the removal details
        removed_substrings.push((start, matched_str.clone()));
    }

    let result_string = re.replace_all(input, "").to_string();

    (result_string, removed_substrings)
}

#[cfg(test)]
mod test_regex_remove {
    #[test]
    fn test_regex_remove_space() {
        let input = "hello world";
        let pattern = " ";
        let (result, insertions) = super::regex_remove(input, pattern);
        assert_eq!(result, "helloworld");
        assert_eq!(insertions, vec![(5, " ".to_string())]);
    }

    #[test]
    fn test_regex_remove_tone() {
        let input = "hello1 world2";
        let pattern = "[123456]";
        let (result, insertions) = super::regex_remove(input, pattern);
        assert_eq!(result, "hello world");
        assert_eq!(
            insertions,
            vec![(5, "1".to_string()), (12, "2".to_string())]
        );
    }

    #[test]
    fn test_regex_remove_space_and_tone() {
        let input = "hello1 world2";
        let pattern = "[123456 ]";
        let (result, insertions) = super::regex_remove(input, pattern);
        assert_eq!(result, "helloworld");
        assert_eq!(
            insertions,
            vec![
                (5, "1".to_string()),
                (6, " ".to_string()),
                (12, "2".to_string())
            ]
        );
    }

    #[test]
    fn test_regex_remove_unicode() {
        let input = "你😊好世界";
        let pattern = "😊";
        let (result, insertions) = super::regex_remove(input, pattern);
        assert_eq!(result, "你好世界");
        assert_eq!(insertions, vec![(1, "😊".to_string())]);
    }
}

fn insert_multiple(input: &str, mut insertions: Vec<(usize, String)>) -> String {
    // Sort insertions by index
    insertions.sort_by_key(|&(index, _)| index);

    let input_chars: Vec<char> = input.chars().collect();

    let mut result = String::new();
    let mut last_index = 0;

    for (index, insert_str) in insertions {
        // Push the part of the input string before the current index
        result.push_str(&input_chars[last_index..index].iter().collect::<String>());
        // Push the insertion string
        result.push_str(&insert_str);
        // Update the last index
        last_index = index;
    }

    // Push the remaining part of the input string
    result.push_str(&input_chars[last_index..].iter().collect::<String>());

    result
}

#[cfg(test)]
mod test_insert_multiple {
    #[test]
    fn test_insert_multiple() {
        let input = "helloworld";
        let insertions = vec![
            (5, "1".to_string()),
            (5, " ".to_string()),
            (10, "2".to_string()),
        ];
        let result = super::insert_multiple(input, insertions);
        assert_eq!(result, "hello1 world2");
    }

    #[test]
    fn test_insert_multiple_unicode() {
        let input = "你😊好世界";
        let insertions = vec![(3, "🌍".to_string()), (5, "🌍".to_string())];
        let result = super::insert_multiple(input, insertions);
        assert_eq!(result, "你😊好🌍世界🌍");
    }
}

fn apply_insertions(
    pr_variant_insertions: Vec<(usize, String)>,
    matched_pr: Vec<MatchedSegment>,
) -> Vec<MatchedSegment> {
    let mut insertions_grouped_by_segments: BTreeMap<usize, Vec<(usize, String)>> = BTreeMap::new();

    let mut offset = 0;

    for (segment_index, MatchedSegment { segment, .. }) in matched_pr.iter().enumerate() {
        let segment_chars: Vec<char> = segment.chars().collect();
        let segment_len = segment_chars.len();

        for (insertion_index, insertion_string) in pr_variant_insertions.iter() {
            if *insertion_index >= offset && *insertion_index <= offset + segment_len {
                let insertion = (insertion_index - offset, insertion_string.clone());
                insertions_grouped_by_segments
                    .entry(segment_index)
                    .and_modify(|entry| entry.push(insertion.clone()))
                    .or_insert(vec![insertion]);
                offset += insertion_string.chars().count();
                continue;
            }
        }
        offset += segment_len;
    }

    let mut result_matched_pr = vec![];
    for (segment_index, MatchedSegment { segment, matched }) in matched_pr.iter().enumerate() {
        if let Some(insertions) = insertions_grouped_by_segments.get(&segment_index) {
            if *matched {
                let mut last_index = 0;
                let segment_chars: Vec<char> = segment.chars().collect();
                for (insertion_index, insertion_string) in insertions {
                    if *insertion_index > last_index {
                        result_matched_pr.push(MatchedSegment {
                            segment: segment_chars[last_index..*insertion_index].iter().collect(),
                            matched: true,
                        });
                    }
                    result_matched_pr.push(MatchedSegment {
                        segment: insertion_string.clone(),
                        matched: false,
                    });
                    last_index = *insertion_index;
                }
                if last_index < segment_chars.len() {
                    result_matched_pr.push(MatchedSegment {
                        segment: segment_chars[last_index..].iter().collect(),
                        matched: true,
                    });
                }
            } else {
                result_matched_pr.push(MatchedSegment {
                    segment: insert_multiple(segment, insertions.clone()),
                    matched: *matched,
                });
            }
        } else {
            result_matched_pr.push(MatchedSegment {
                segment: segment.to_string(),
                matched: *matched,
            });
        };
    }

    result_matched_pr
}

#[cfg(test)]
mod test_apply_insertions {
    #[test]
    fn test_apply_insertions_edge() {
        let pr_variant_insertions = vec![
            (5, "1".to_string()),
            (6, " ".to_string()),
            (12, "2".to_string()),
        ];
        let matched_pr = vec![
            super::MatchedSegment {
                segment: "hello".to_string(),
                matched: true,
            },
            super::MatchedSegment {
                segment: "world".to_string(),
                matched: false,
            },
        ];
        let result = super::apply_insertions(pr_variant_insertions, matched_pr);
        assert_eq!(
            result,
            vec![
                super::MatchedSegment {
                    segment: "hello".to_string(),
                    matched: true
                },
                super::MatchedSegment {
                    segment: "1".to_string(),
                    matched: false
                },
                super::MatchedSegment {
                    segment: " ".to_string(),
                    matched: false
                },
                super::MatchedSegment {
                    segment: "world2".to_string(),
                    matched: false
                },
            ]
        );
    }

    #[test]
    fn test_apply_insertions_middle() {
        let pr_variant_insertions = vec![
            (2, "1".to_string()),
            (6, "2".to_string()),
            (7, " ".to_string()),
            (10, "3".to_string()),
        ];
        let matched_pr = vec![
            super::MatchedSegment {
                segment: "hello".to_string(),
                matched: true,
            },
            super::MatchedSegment {
                segment: "world".to_string(),
                matched: false,
            },
        ];
        let result = super::apply_insertions(pr_variant_insertions, matched_pr);
        assert_eq!(
            result,
            vec![
                super::MatchedSegment {
                    segment: "he".to_string(),
                    matched: true
                },
                super::MatchedSegment {
                    segment: "1".to_string(),
                    matched: false
                },
                super::MatchedSegment {
                    segment: "llo".to_string(),
                    matched: true
                },
                super::MatchedSegment {
                    segment: "2".to_string(),
                    matched: false
                },
                super::MatchedSegment {
                    segment: " ".to_string(),
                    matched: false
                },
                super::MatchedSegment {
                    segment: "wo3rld".to_string(),
                    matched: false
                },
            ]
        );
    }
}

pub fn pr_search(
    pr_indices: &dyn FstPrIndicesLike,
    dict: &dyn RichDictLike,
    query: &str,
    script: Script,
    romanization: Romanization,
) -> BinaryHeap<PrSearchRank> {
    let query = unicode::normalize(query);

    if query.is_empty() || query.chars().any(unicode::is_cjk) {
        return BinaryHeap::new();
    }

    let mut ranks = BTreeMap::new();

    fn to_yale(s: &str) -> String {
        parse_jyutpings(s)
            .unwrap()
            .into_iter()
            .map(|jyutping| jyutping.to_yale())
            .join(" ")
            .nfd()
            .collect::<String>()
    }

    fn lookup_index(
        query: &str,
        search: impl FnOnce(&str) -> FstSearchResult,
        dict: &dyn RichDictLike,
        script: Script,
        romanization: Romanization,
        ranks: &mut BTreeMap<EntryId, PrSearchRank>,
        pr_variant_generator: fn(&str) -> (String, Vec<(usize, String)>),
    ) {
        let search_result = search(&query);

        match &search_result {
            FstSearchResult::Prefix(result) | FstSearchResult::Levenshtein(result) => {
                for &PrLocation {
                    entry_id,
                    variant_index,
                    pr_index,
                } in result
                {
                    // Decompose tone marks for Yale
                    let query = query.nfd().collect::<String>();

                    let entry = dict.get_entry(entry_id).unwrap();
                    let jyutping: &LaxJyutPing =
                        &entry.variants.0[variant_index as Index].prs.0[pr_index as Index];
                    let jyutping = jyutping.to_string();
                    let (pr_variant, pr_variant_insertions) = pr_variant_generator(&jyutping);
                    let distance = match search_result {
                        FstSearchResult::Prefix(_) => {
                            if pr_variant.starts_with(&query) {
                                pr_variant.chars().count() - query.chars().count()
                            } else {
                                usize::MAX
                            }
                        }
                        FstSearchResult::Levenshtein(_) => levenshtein(&query, &pr_variant),
                    };
                    if match search_result {
                        FstSearchResult::Prefix(_) => distance < usize::MAX,
                        FstSearchResult::Levenshtein(_) => distance <= MAX_DELETIONS,
                    } {
                        let matched_pr = match search_result {
                            FstSearchResult::Prefix(_) => vec![
                                MatchedSegment {
                                    segment: query.to_string(),
                                    matched: true,
                                },
                                MatchedSegment {
                                    segment: pr_variant.strip_prefix(&query).unwrap().to_string(),
                                    matched: false,
                                },
                            ],
                            FstSearchResult::Levenshtein(_) => diff_prs(&query, &pr_variant),
                        };

                        let matched_pr = apply_insertions(pr_variant_insertions, matched_pr);

                        let mut at_initial = true;
                        let mut num_matched_initial_chars = 0;
                        let mut num_matched_final_chars = 0;
                        static VOWELS: [char; 5] = ['a', 'e', 'i', 'o', 'u'];
                        for MatchedSegment { segment, matched } in &matched_pr {
                            for c in segment.chars() {
                                match c {
                                    ' ' => {
                                        at_initial = true;
                                    }
                                    _ if VOWELS.contains(&c) => {
                                        at_initial = false;
                                    }
                                    _ if at_initial && *matched => {
                                        num_matched_initial_chars += 1;
                                    }
                                    _ if !at_initial
                                        && *matched
                                        && !VOWELS.contains(&c)
                                        && c.is_ascii_alphabetic() =>
                                    {
                                        num_matched_final_chars += 1;
                                    }
                                    _ => {}
                                }
                            }
                        }
                        let frequency_count = get_max_frequency_count(&entry.variants);
                        let def_len = entry.defs.len();
                        let variant = pick_variant(
                            entry.variants.0.get(variant_index as usize).unwrap(),
                            script,
                        )
                        .word;
                        let score = MAX_SCORE - distance;
                        match ranks.entry(entry_id) {
                            std::collections::btree_map::Entry::Occupied(mut rank) => {
                                let rank = rank.get_mut();
                                if score > rank.score {
                                    rank.score = score;
                                    rank.variant_indices =
                                        vec![(variant_index as Index, pr_index as Index)];
                                    rank.variants = vec![variant];
                                    rank.jyutping = jyutping;
                                    rank.matched_pr = matched_pr;
                                    rank.num_matched_initial_chars = num_matched_initial_chars;
                                    rank.num_matched_final_chars = num_matched_final_chars;
                                    rank.frequency_count = frequency_count;
                                } else if score == rank.score
                                    && jyutping == rank.jyutping
                                    && !rank.variants.contains(&variant)
                                {
                                    rank.variant_indices
                                        .push((variant_index as Index, pr_index as Index));
                                    rank.variants.push(variant);
                                }
                            }
                            std::collections::btree_map::Entry::Vacant(rank) => {
                                rank.insert(PrSearchRank {
                                    id: entry_id,
                                    def_len,
                                    variant_indices: vec![(
                                        variant_index as Index,
                                        pr_index as Index,
                                    )],
                                    variants: vec![variant],
                                    jyutping,
                                    matched_pr,
                                    num_matched_initial_chars,
                                    num_matched_final_chars,
                                    score,
                                    frequency_count,
                                });
                            }
                        }
                    }
                }
            }
        }
    }

    const SPACE_REGEX: &'static str = " ";

    match romanization {
        Romanization::Jyutping => {
            const TONES: [char; 6] = ['1', '2', '3', '4', '5', '6'];
            const TONE_REGEX: &'static str = "[123456]";
            const TONE_AND_SPACE_REGEX: &'static str = "[123456 ]";

            if query.contains(TONES) && query.contains(' ') {
                lookup_index(
                    &query,
                    |query| pr_indices.search(true, query, romanization),
                    dict,
                    script,
                    romanization,
                    &mut ranks,
                    |s| (s.to_string(), vec![]),
                );
            } else if query.contains(TONES) {
                lookup_index(
                    &query,
                    |query| pr_indices.search(true, query, romanization),
                    dict,
                    script,
                    romanization,
                    &mut ranks,
                    |s| regex_remove(s, SPACE_REGEX),
                );
            } else if query.contains(' ') {
                lookup_index(
                    &query,
                    |query| pr_indices.search(false, query, romanization),
                    dict,
                    script,
                    romanization,
                    &mut ranks,
                    |s| regex_remove(s, TONE_REGEX),
                );
            } else {
                lookup_index(
                    &query,
                    |query| pr_indices.search(false, query, romanization),
                    dict,
                    script,
                    romanization,
                    &mut ranks,
                    |s| regex_remove(s, TONE_AND_SPACE_REGEX),
                );
            }
        }
        Romanization::Yale => {
            const TONE_REGEX: &'static str = "[\u{0300}\u{0301}\u{0304}]|h\\b";
            const TONE_AND_SPACE_REGEX: &'static str = "[\u{0300}\u{0301}\u{0304} ]|h\\b";

            let has_tone = remove_yale_diacritics(&query) != query;

            if has_tone && query.contains(' ') {
                lookup_index(
                    &query,
                    |query| pr_indices.search(true, query, romanization),
                    dict,
                    script,
                    romanization,
                    &mut ranks,
                    |s| (to_yale(s), vec![]),
                );
            } else if has_tone {
                lookup_index(
                    &query,
                    |query| pr_indices.search(true, query, romanization),
                    dict,
                    script,
                    romanization,
                    &mut ranks,
                    |s| regex_remove(&to_yale(s), SPACE_REGEX),
                );
            } else if query.contains(' ') {
                lookup_index(
                    &query,
                    |query| pr_indices.search(true, query, romanization),
                    dict,
                    script,
                    romanization,
                    &mut ranks,
                    |s| (to_yale(s), vec![]),
                );

                lookup_index(
                    &query,
                    |query| pr_indices.search(false, query, romanization),
                    dict,
                    script,
                    romanization,
                    &mut ranks,
                    |s| regex_remove(&to_yale(s), &TONE_REGEX),
                );
            } else {
                lookup_index(
                    &query,
                    |query| pr_indices.search(true, query, romanization),
                    dict,
                    script,
                    romanization,
                    &mut ranks,
                    |s| regex_remove(&to_yale(s), SPACE_REGEX),
                );

                lookup_index(
                    &query,
                    |query| pr_indices.search(false, query, romanization),
                    dict,
                    script,
                    romanization,
                    &mut ranks,
                    |s| regex_remove(&to_yale(s), &TONE_AND_SPACE_REGEX),
                );
            }
        }
    }

    ranks.into_values().collect()
}

#[derive(Debug, Clone, Eq, PartialEq, Default)]
pub struct MatchedInfix {
    pub prefix: String,
    pub query: String,
    pub suffix: String,
}

impl std::fmt::Display for MatchedInfix {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}{}{}", self.prefix, self.query, self.suffix)
    }
}

#[derive(Clone, Eq, PartialEq, Debug)]
pub struct VariantSearchRank {
    pub id: EntryId,
    pub def_len: usize,
    pub variant_index: Index,
    pub variant: String,
    pub occurrence_index: Index,
    pub length_diff: usize,
    pub matched_variant: MatchedInfix,
    pub frequency_count: usize,
}

impl Ord for VariantSearchRank {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .occurrence_index
            .cmp(&self.occurrence_index)
            .then(other.length_diff.cmp(&self.length_diff))
            .then(self.frequency_count.cmp(&other.frequency_count))
            .then(other.variant.cmp(&self.variant))
            .then_with(|| get_entry_frequency(self.id).cmp(&get_entry_frequency(other.id)))
            .then(self.def_len.cmp(&other.def_len))
            .then(other.id.cmp(&self.id))
    }
}

impl PartialOrd for VariantSearchRank {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

fn word_levenshtein(a: &Vec<&str>, b: &Vec<&str>) -> usize {
    if a.is_empty() && b.is_empty() {
        return 0;
    }
    generic_levenshtein(a, b)
}

fn score_variant_query(
    variant_in_query_script: &str,
    variant_in_target_script: &str,
    query_normalized: &str,
) -> (Index, Score, MatchedInfix) {
    let variant_normalized = &unicode::normalize(variant_in_query_script)[..];
    // The query has to be fully contained by the variant
    let occurrence_index = match variant_normalized.find(query_normalized) {
        Some(i) => i,
        None => return (usize::MAX, usize::MAX, MatchedInfix::default()),
    };
    let length_diff: usize = variant_normalized.chars().count() - query_normalized.chars().count();
    let variant_normalized_original = &unicode::normalize(variant_in_target_script)[..];
    assert_eq!(variant_normalized.len(), variant_normalized_original.len());
    let matched_variant = MatchedInfix {
        prefix: variant_normalized_original[..occurrence_index].to_string(),
        query: variant_normalized_original
            [occurrence_index..occurrence_index + query_normalized.len()]
            .to_string(),
        suffix: variant_normalized_original[occurrence_index + query_normalized.len()..]
            .to_string(),
    };
    (occurrence_index, length_diff, matched_variant)
}

pub fn variant_search(
    dict: &dyn RichDictLike,
    variant_index: &dyn VariantIndexLike,
    query: &str,
    script: Script,
) -> BinaryHeap<VariantSearchRank> {
    let mut ranks = BinaryHeap::new();
    let query_normalized = &unicode::to_hk_safe_variant(&unicode::normalize(query))[..];
    let query_script = if query_normalized.chars().any(|c| ICONIC_SIMPS.contains(&c)) {
        // query contains iconic simplified characters
        Script::Simplified
    } else if query_normalized.chars().any(|c| ICONIC_TRADS.contains(&c)) {
        Script::Traditional
    } else {
        script
    };

    let entry_ids = query_normalized
        .chars()
        .fold(BTreeSet::new(), |mut entry_ids, c| {
            entry_ids.extend(
                variant_index
                    .get(c, query_script)
                    .unwrap_or(BTreeSet::new()),
            );
            entry_ids
        });
    for id in entry_ids {
        let entry = dict.get_entry(id).unwrap();
        let frequency_count = get_max_frequency_count(&entry.variants);
        entry
            .variants
            .0
            .iter()
            .enumerate()
            .for_each(|(variant_index, variant)| {
                let (occurrence_index, length_diff, matched_variant) = score_variant_query(
                    &pick_variant(variant, query_script).word,
                    &pick_variant(variant, script).word,
                    &query_normalized,
                );
                if occurrence_index < usize::MAX {
                    ranks.push(VariantSearchRank {
                        id,
                        def_len: entry.defs.len(),
                        variant_index,
                        variant: pick_variant(variant, script).word,
                        occurrence_index,
                        length_diff,
                        matched_variant,
                        frequency_count,
                    });
                }
            });
    }
    ranks
}

pub fn mandarin_variant_search(
    dict: &dyn RichDictLike,
    mandarin_variant_index: &dyn MandarinVariantIndexLike,
    query: &str,
) -> BinaryHeap<VariantSearchRank> {
    let mut ranks = BinaryHeap::new();
    // For now, query is normalized to simplified
    let query_normalized =
        fast2s::convert(&unicode::to_hk_safe_variant(&unicode::normalize(query))[..]);

    let entry_ids = query_normalized
        .chars()
        .fold(BTreeSet::new(), |mut entry_ids, c| {
            entry_ids.extend(mandarin_variant_index.get(c).unwrap_or(BTreeSet::new()));
            entry_ids
        });
    for id in entry_ids {
        let entry = dict.get_entry(id).unwrap();
        let frequency_count = get_max_frequency_count(&entry.variants);
        entry
            .mandarin_variants
            .0
            .iter()
            .enumerate()
            .for_each(|(variant_index, variant)| {
                let (occurrence_index, length_diff, matched_variant) =
                    score_variant_query(&variant.word_simp, &variant.word_simp, &query_normalized);
                if occurrence_index < usize::MAX {
                    ranks.push(VariantSearchRank {
                        id,
                        def_len: entry.defs.len(),
                        variant_index,
                        variant: variant.word_simp.clone(),
                        occurrence_index,
                        length_diff,
                        matched_variant,
                        frequency_count,
                    });
                }
            });
    }
    ranks
}

#[derive(Clone, Eq, PartialEq, Debug)]
pub struct EgSearchRank {
    pub id: EntryId,
    pub def_index: Index,
    pub eg_index: Index,
    pub eg_length: usize,
    pub matched_eg: MatchedInfix,
}

impl Ord for EgSearchRank {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .eg_length
            .cmp(&self.eg_length)
            .then_with(|| get_entry_frequency(self.id).cmp(&get_entry_frequency(other.id)))
            .then(other.def_index.cmp(&self.def_index))
            .then(other.eg_index.cmp(&self.eg_index))
    }
}

impl PartialOrd for EgSearchRank {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

pub fn eg_search(
    eg_index: &dyn EgIndexLike,
    query: &str,
    script: Script,
) -> BinaryHeap<EgSearchRank> {
    let query_normalized = unicode::to_hk_safe_variant(&unicode::normalize(query));
    let query_script = if query_normalized.chars().any(|c| ICONIC_SIMPS.contains(&c)) {
        // query contains iconic simplified characters
        Script::Simplified
    } else if query_normalized.chars().any(|c| ICONIC_TRADS.contains(&c)) {
        Script::Traditional
    } else {
        script
    };

    if query_normalized.is_empty() {
        return BinaryHeap::new();
    }

    use std::sync::Arc;
    use std::sync::Mutex;

    let ranks: Arc<Mutex<BinaryHeap<_>>> = Arc::new(Mutex::new(BinaryHeap::new()));

    use rayon::iter::IntoParallelRefIterator;
    use rayon::iter::ParallelIterator;

    let eg_locations = query_normalized.chars().skip(1).fold(
        eg_index.get_egs_by_character(query_normalized.chars().next().unwrap()),
        |eg_locations, c| {
            eg_locations
                .intersection(&eg_index.get_egs_by_character(c))
                .cloned()
                .collect()
        },
    );

    eg_locations
        .par_iter()
        .for_each(|((entry_id, def_index, eg_index), eg_yue, eg_yue_simp)| {
            let get_line_in_script = |script| match script {
                Script::Traditional => eg_yue.clone(),
                Script::Simplified => eg_yue_simp.clone(),
            };
            let line: String = get_line_in_script(query_script);
            let line_len = line.chars().count();
            if let Some(first_index) = line.find(&query_normalized) {
                let line = if query_script == script {
                    line
                } else {
                    get_line_in_script(script)
                };
                let matched_eg = MatchedInfix {
                    prefix: line[..first_index].to_string(),
                    query: line[first_index..first_index + query_normalized.len()].to_string(),
                    suffix: line[first_index + query_normalized.len()..].to_string(),
                };
                ranks.lock().unwrap().push(EgSearchRank {
                    id: *entry_id,
                    def_index: *def_index,
                    eg_index: *eg_index,
                    eg_length: line_len,
                    matched_eg,
                });
            }
        });

    Arc::try_unwrap(ranks).unwrap().into_inner().unwrap()
}

pub struct CombinedSearchRank {
    pub variant: BinaryHeap<VariantSearchRank>,
    pub mandarin_variant: BinaryHeap<VariantSearchRank>,
    pub pr: BinaryHeap<PrSearchRank>,
    pub english: BinaryHeap<EnglishSearchRank>,
    pub eg: BinaryHeap<EgSearchRank>,
}

// Auto recognize the type of the query
pub fn combined_search<D>(
    dict: &D,
    query: &str,
    script: Script,
    romanization: Romanization,
) -> CombinedSearchRank
where
    D: RichDictLike
        + VariantIndexLike
        + MandarinVariantIndexLike
        + FstPrIndicesLike
        + EnglishIndexLike
        + EgIndexLike,
{
    let variants_ranks = variant_search(dict, dict, query, script);
    let mandarin_variants_ranks = mandarin_variant_search(dict, dict, query);
    let pr_ranks = pr_search(dict, dict, query, script, romanization);
    let english_results = english_search(dict, dict, query, script);
    let eg_results = eg_search(dict, query, script);

    CombinedSearchRank {
        variant: variants_ranks,
        mandarin_variant: mandarin_variants_ranks,
        pr: pr_ranks,
        english: english_results,
        eg: eg_results,
    }
}

#[derive(Clone, Eq, PartialEq, Debug)]
struct EngSegment {
    segment: String,
    is_word: bool,
}

fn segment_eng_words(input: &str) -> Vec<EngSegment> {
    let mut segments = Vec::new();
    let mut last_match_end = 0;

    // Regular expression to match words (including those with internal apostrophes and hyphens)
    let word_re = regex::Regex::new(r"\b[a-zA-Z]+(?:-[a-zA-Z]+)*(?:(?:'[a-zA-Z]*)?|\b)").unwrap();

    for mat in word_re.find_iter(input) {
        let match_start = mat.start();
        let match_end = mat.end();

        // Handle any non-word segment before the current word
        if match_start > last_match_end {
            let non_word_segment = &input[last_match_end..match_start];
            segments.push(EngSegment {
                segment: non_word_segment.to_string(),
                is_word: false,
            });
        }

        // Add the current word segment
        let word_segment = mat.as_str().to_string();
        segments.push(EngSegment {
            segment: word_segment,
            is_word: true,
        });

        last_match_end = match_end;
    }

    // Handle any trailing non-word segment
    if last_match_end < input.len() {
        let trailing_segment = &input[last_match_end..];
        segments.push(EngSegment {
            segment: trailing_segment.to_string(),
            is_word: false,
        });
    }

    segments
}

#[cfg(test)]
mod test_segment_eng_words {
    use super::{segment_eng_words, EngSegment};

    #[test]
    fn test_empty_string() {
        let input = "";
        assert_eq!(segment_eng_words(input), vec![]);
    }

    #[test]
    fn test_only_words() {
        let input = "Hello world";
        assert_eq!(
            segment_eng_words(input),
            vec![
                EngSegment {
                    segment: "Hello".to_string(),
                    is_word: true
                },
                EngSegment {
                    segment: " ".to_string(),
                    is_word: false
                },
                EngSegment {
                    segment: "world".to_string(),
                    is_word: true
                },
            ]
        );
    }

    #[test]
    fn test_hyphens() {
        let input = "- This is a good-for-nothing bullet";
        assert_eq!(
            segment_eng_words(input),
            vec![
                EngSegment {
                    segment: "- ".to_string(),
                    is_word: false
                },
                EngSegment {
                    segment: "This".to_string(),
                    is_word: true
                },
                EngSegment {
                    segment: " ".to_string(),
                    is_word: false
                },
                EngSegment {
                    segment: "is".to_string(),
                    is_word: true
                },
                EngSegment {
                    segment: " ".to_string(),
                    is_word: false
                },
                EngSegment {
                    segment: "a".to_string(),
                    is_word: true
                },
                EngSegment {
                    segment: " ".to_string(),
                    is_word: false
                },
                EngSegment {
                    segment: "good-for-nothing".to_string(),
                    is_word: true
                },
                EngSegment {
                    segment: " ".to_string(),
                    is_word: false
                },
                EngSegment {
                    segment: "bullet".to_string(),
                    is_word: true
                },
            ]
        );
    }

    #[test]
    fn test_apostrophes() {
        let input = "- it's comin'";
        assert_eq!(
            segment_eng_words(input),
            vec![
                EngSegment {
                    segment: "- ".to_string(),
                    is_word: false
                },
                EngSegment {
                    segment: "it's".to_string(),
                    is_word: true
                },
                EngSegment {
                    segment: " ".to_string(),
                    is_word: false
                },
                EngSegment {
                    segment: "comin'".to_string(),
                    is_word: true
                },
            ]
        );
    }

    #[test]
    fn test_only_non_word_characters() {
        let input = " ,.!?";
        assert_eq!(
            segment_eng_words(input),
            vec![EngSegment {
                segment: " ,.!?".to_string(),
                is_word: false
            },]
        );
    }

    #[test]
    fn test_words_and_non_word_characters() {
        let input = "Hello, world 啦!";
        assert_eq!(
            segment_eng_words(input),
            vec![
                EngSegment {
                    segment: "Hello".to_string(),
                    is_word: true
                },
                EngSegment {
                    segment: ", ".to_string(),
                    is_word: false
                },
                EngSegment {
                    segment: "world".to_string(),
                    is_word: true
                },
                EngSegment {
                    segment: " 啦!".to_string(),
                    is_word: false
                },
            ]
        );
    }
}

// Allow multiple matches and normalize segments before matching
fn match_eng_words(sentence: &str, query_normalized: &str) -> Vec<MatchedSegment> {
    let segments = segment_eng_words(sentence);
    let query_normalized = segment_eng_words(query_normalized);

    if segments.is_empty() {
        return vec![];
    }

    if query_normalized.is_empty() {
        return vec![MatchedSegment {
            segment: segments.into_iter().map(|seg| seg.segment).join(""),
            matched: false,
        }];
    }

    let segments_normalized = segments
        .iter()
        .map(|seg| {
            if seg.is_word {
                EngSegment {
                    segment: unicode::normalize_english_word_for_search_index(&seg.segment),
                    is_word: true,
                }
            } else {
                seg.clone()
            }
        })
        .collect::<Vec<_>>();
    let mut current_segment_index = 0;
    let mut matched_segments = Vec::new();
    while current_segment_index + query_normalized.len() <= segments.len() {
        let segs_normalized = &segments_normalized
            [current_segment_index..current_segment_index + query_normalized.len()];
        if segs_normalized == query_normalized {
            matched_segments.push(MatchedSegment {
                segment: segments
                    [current_segment_index..current_segment_index + query_normalized.len()]
                    .iter()
                    .map(|seg| seg.segment.clone())
                    .join(""),
                matched: true,
            });
            current_segment_index += query_normalized.len();
        } else {
            matched_segments.push(MatchedSegment {
                segment: segments[current_segment_index].segment.clone(),
                matched: false,
            });
            current_segment_index += 1;
        }
    }
    if current_segment_index < segments.len() {
        matched_segments.push(MatchedSegment {
            segment: segments[current_segment_index..segments.len()]
                .iter()
                .map(|seg| seg.segment.clone())
                .join(""),
            matched: false,
        });
    }
    matched_segments
}

#[cfg(test)]
mod test_match_eng_words {
    use super::{match_eng_words, unicode, MatchedSegment};

    #[test]
    fn test_empty_sentence_and_query() {
        let sentence = "";
        let query = "";
        assert_eq!(match_eng_words(sentence, query), vec![]);
    }

    #[test]
    fn test_empty_sentence() {
        let sentence = "";
        let query = "test";
        assert_eq!(match_eng_words(sentence, query), vec![]);
    }

    #[test]
    fn test_empty_query() {
        let sentence = "Hello world";
        let query = "";
        assert_eq!(
            match_eng_words(sentence, query),
            vec![MatchedSegment {
                segment: "Hello world".to_string(),
                matched: false
            },]
        );
    }

    #[test]
    fn test_no_match() {
        let sentence = "Hello world";
        let query = "test";
        assert_eq!(
            match_eng_words(sentence, query),
            vec![
                MatchedSegment {
                    segment: "Hello".to_string(),
                    matched: false
                },
                MatchedSegment {
                    segment: " ".to_string(),
                    matched: false
                },
                MatchedSegment {
                    segment: "world".to_string(),
                    matched: false
                },
            ]
        );
    }

    #[test]
    fn test_exact_match() {
        let sentence = "Hello world";
        let query = "hello";
        assert_eq!(
            match_eng_words(sentence, &query),
            vec![
                MatchedSegment {
                    segment: "Hello".to_string(),
                    matched: true
                },
                MatchedSegment {
                    segment: " ".to_string(),
                    matched: false
                },
                MatchedSegment {
                    segment: "world".to_string(),
                    matched: false
                },
            ]
        );
    }

    #[test]
    fn test_exact_full_match() {
        let sentence = "Hello world";
        let query = "hello world";
        assert_eq!(
            match_eng_words(sentence, &query),
            vec![MatchedSegment {
                segment: "Hello world".to_string(),
                matched: true
            }]
        );
    }

    #[test]
    fn test_broken_full_match() {
        let sentence = "Hello, world";
        let query = "hello world";
        assert_eq!(
            match_eng_words(sentence, &query),
            vec![
                MatchedSegment {
                    segment: "Hello".to_string(),
                    matched: false
                },
                MatchedSegment {
                    segment: ", world".to_string(),
                    matched: false
                }
            ]
        );
    }

    #[test]
    fn test_apostrophe_match() {
        let sentence = "Hello world! What's up, y'all!";
        let query = unicode::normalize_english_word_for_search_index("what's up");
        assert_eq!(
            match_eng_words(sentence, &query),
            vec![
                MatchedSegment {
                    segment: "Hello".to_string(),
                    matched: false
                },
                MatchedSegment {
                    segment: " ".to_string(),
                    matched: false
                },
                MatchedSegment {
                    segment: "world".to_string(),
                    matched: false
                },
                MatchedSegment {
                    segment: "! ".to_string(),
                    matched: false
                },
                MatchedSegment {
                    segment: "What's up".to_string(),
                    matched: true
                },
                MatchedSegment {
                    segment: ", ".to_string(),
                    matched: false
                },
                MatchedSegment {
                    segment: "y'all!".to_string(),
                    matched: false
                }
            ]
        );
    }

    #[test]
    fn test_hyphen_and_apostrophe_match() {
        let sentence = "Hello world! good-for-nothing' comin' y'all!";
        let query = unicode::normalize_english_word_for_search_index("good-for-nothing'");
        assert_eq!(
            match_eng_words(sentence, &query),
            vec![
                MatchedSegment {
                    segment: "Hello".to_string(),
                    matched: false
                },
                MatchedSegment {
                    segment: " ".to_string(),
                    matched: false
                },
                MatchedSegment {
                    segment: "world".to_string(),
                    matched: false
                },
                MatchedSegment {
                    segment: "! ".to_string(),
                    matched: false
                },
                MatchedSegment {
                    segment: "good-for-nothing'".to_string(),
                    matched: true
                },
                MatchedSegment {
                    segment: " ".to_string(),
                    matched: false
                },
                MatchedSegment {
                    segment: "comin'".to_string(),
                    matched: false
                },
                MatchedSegment {
                    segment: " ".to_string(),
                    matched: false
                },
                MatchedSegment {
                    segment: "y'all".to_string(),
                    matched: false
                },
                MatchedSegment {
                    segment: "!".to_string(),
                    matched: false
                }
            ]
        );
    }

    #[test]
    fn test_partial_match() {
        let sentence = "Hello world";
        let query = "ello";
        assert_eq!(
            match_eng_words(sentence, query),
            vec![
                MatchedSegment {
                    segment: "Hello".to_string(),
                    matched: false
                },
                MatchedSegment {
                    segment: " ".to_string(),
                    matched: false
                },
                MatchedSegment {
                    segment: "world".to_string(),
                    matched: false
                },
            ]
        );
    }

    #[test]
    fn test_multiple_matches() {
        let sentence = "Hello world, hello again";
        let query = "hello";
        assert_eq!(
            match_eng_words(sentence, query),
            vec![
                MatchedSegment {
                    segment: "Hello".to_string(),
                    matched: true
                },
                MatchedSegment {
                    segment: " ".to_string(),
                    matched: false
                },
                MatchedSegment {
                    segment: "world".to_string(),
                    matched: false
                },
                MatchedSegment {
                    segment: ", ".to_string(),
                    matched: false
                },
                MatchedSegment {
                    segment: "hello".to_string(),
                    matched: true
                },
                MatchedSegment {
                    segment: " ".to_string(),
                    matched: false
                },
                MatchedSegment {
                    segment: "again".to_string(),
                    matched: false,
                }
            ]
        );
    }

    #[test]
    fn test_case_insensitivity_and_normalization() {
        let sentence = "Hello, WORLD!";
        let query = "world";
        assert_eq!(
            match_eng_words(sentence, query),
            vec![
                MatchedSegment {
                    segment: "Hello".to_string(),
                    matched: false
                },
                MatchedSegment {
                    segment: ", ".to_string(),
                    matched: false
                },
                MatchedSegment {
                    segment: "WORLD".to_string(),
                    matched: true
                },
                MatchedSegment {
                    segment: "!".to_string(),
                    matched: false
                },
            ]
        );
    }
}

pub fn english_search(
    english_index: &dyn EnglishIndexLike,
    dict: &dyn RichDictLike,
    query: &str,
    script: Script,
) -> BinaryHeap<EnglishSearchRank> {
    if query.chars().any(unicode::is_cjk) {
        return BinaryHeap::default();
    }

    let query = unicode::normalize_english_word_for_search_index(query);
    let results = english_index
        .get(query.as_str())
        .unwrap_or(fuzzy_english_search(english_index, &[query.clone()]));
    results
        .into_iter()
        .filter_map(
            |EnglishIndexData {
                 entry_id,
                 def_index,
                 score,
             }| {
                let entry = dict.get_entry(entry_id)?;
                let def = &entry.defs[def_index as usize];
                let eng: Clause = def.eng.as_ref().unwrap().clone();
                let eng = clause_to_string(&eng);
                let matched_eng = match_eng_words(&eng, &query);
                let frequency_count = get_max_frequency_count(&entry.variants);
                Some(EnglishSearchRank {
                    entry_id,
                    def_len: entry.defs.len(),
                    def_index,
                    variant: pick_variant(entry.variants.0.first().unwrap(), script).word,
                    score,
                    frequency_count,
                    matched_eng,
                })
            },
        )
        .collect()
}

fn fuzzy_english_search<'a>(
    english_index: &'a dyn EnglishIndexLike,
    queries: &[String],
) -> Vec<EnglishIndexData> {
    english_index
        .iter()
        .fold(
            (60, None), // must have a score of at least 60 out of 100
            |(max_score, max_entries), phrase| {
                let (mut next_max_score, mut next_max_entries) = (max_score, max_entries);
                queries.iter().for_each(|query| {
                    let current_score = score_english_query(query, &phrase);
                    if current_score > max_score {
                        (next_max_score, next_max_entries) =
                            (current_score, english_index.get(&phrase))
                    }
                });
                (next_max_score, next_max_entries)
            },
        )
        .1
        .unwrap_or(vec![])
}

// Reference: https://www.oracle.com/webfolder/technetwork/data-quality/edqhelp/Content/processor_library/matching/comparisons/word_match_percentage.htm
fn word_match_percent(a: &Vec<&str>, b: &Vec<&str>) -> Score {
    let max_word_length = a.len().max(b.len());
    ((max_word_length - word_levenshtein(a, b)) as f64 / (max_word_length as f64) * 100.0).round()
        as Score
}

fn score_english_query(query: &str, phrase: &str) -> Score {
    // multi-words
    if unicode::is_multi_word(query) || unicode::is_multi_word(phrase) {
        word_match_percent(
            &query.split_ascii_whitespace().collect(),
            &phrase.split_ascii_whitespace().collect(),
        )
    }
    // single word
    // score is scaled down from a multiple of 100% to 80% to better match
    // the score of multi-word word match percent
    else {
        (normalized_levenshtein(query, phrase) * 80.0).round() as Score
    }
}

pub fn get_char_jyutpings(query: char) -> Option<Vec<String>> {
    CHARLIST
        .get(&query)
        .map(|prs| prs.iter().map(|pr| pr.to_string()).collect())
}

pub fn diff_prs(query: &str, reference: &str) -> Vec<MatchedSegment> {
    let source = reference.chars().rev().collect::<Vec<_>>();
    let target = query.chars().rev().collect::<Vec<_>>();

    let (_, mat) = levenshtein_diff::distance(&source, &target);
    let edits = levenshtein_diff::generate_edits(&source, &target, &mat).unwrap();
    let hidden_indices: Vec<usize> = edits
        .iter()
        .filter_map(|edit| match edit {
            // Edits are 1-indexed, perfect for reversed strings
            // Ignore insertions because they are not part of the reference string
            levenshtein_diff::Edit::Insert(..) => None,
            levenshtein_diff::Edit::Delete(index)
            | levenshtein_diff::Edit::Substitute(index, ..) => Some(source.len() - index),
        })
        .collect();
    let hidden_intervals = merge_intervals(&hidden_indices);
    segment_string(reference, &hidden_intervals)
        .into_iter()
        .map(|MatchedSegment { segment, matched }| MatchedSegment {
            segment,
            matched: !matched,
        })
        .collect()
}

// Segment an input string into matched and unmatched segments
fn segment_string(input: &str, intervals: &[(usize, usize)]) -> Vec<MatchedSegment> {
    let chars: Vec<_> = input.chars().collect();
    let mut result = Vec::new();
    let mut current_index = 0;

    for &(start, end) in intervals {
        // Add the segment before the current interval (if any)
        if current_index < start {
            let segment = &chars[current_index..start];
            result.push(MatchedSegment {
                segment: segment.iter().collect(),
                matched: false,
            });
        }

        // Add the current interval
        let segment = &chars[start..=end];
        result.push(MatchedSegment {
            segment: segment.iter().collect(),
            matched: true,
        });

        current_index = end + 1;
    }

    // Add the remaining part of the string (if any)
    if current_index < chars.len() {
        let segment = &chars[current_index..];
        result.push(MatchedSegment {
            segment: segment.iter().collect(),
            matched: false,
        });
    }

    result
}

#[cfg(test)]
mod test_segment_string {
    use super::{segment_string, MatchedSegment};

    #[test]
    fn test_empty_string() {
        assert_eq!(segment_string("", &vec![]), vec![]);
    }

    #[test]
    fn test_no_intervals() {
        let input = "こんにちは";
        assert_eq!(
            segment_string(input, &[]),
            vec![MatchedSegment {
                segment: input.to_string(),
                matched: false
            }]
        );
    }

    #[test]
    fn test_full_interval() {
        let input = "こんにちは";
        assert_eq!(
            segment_string(input, &[(0, 4)]),
            vec![MatchedSegment {
                segment: input.to_string(),
                matched: true
            }]
        );
    }

    #[test]
    fn test_single_character_interval() {
        let input = "こんにちは";
        assert_eq!(
            segment_string(input, &[(2, 2)]),
            vec![
                MatchedSegment {
                    segment: "こん".to_string(),
                    matched: false
                },
                MatchedSegment {
                    segment: "に".to_string(),
                    matched: true
                },
                MatchedSegment {
                    segment: "ちは".to_string(),
                    matched: false
                },
            ]
        );
    }

    #[test]
    fn test_multiple_intervals() {
        let input = "こんにちは世界";
        assert_eq!(
            segment_string(input, &[(0, 1), (3, 6)]),
            vec![
                MatchedSegment {
                    segment: "こん".to_string(),
                    matched: true
                },
                MatchedSegment {
                    segment: "に".to_string(),
                    matched: false
                },
                MatchedSegment {
                    segment: "ちは世界".to_string(),
                    matched: true
                },
            ]
        );
    }
}

fn merge_intervals(indices: &[usize]) -> Vec<(usize, usize)> {
    if indices.is_empty() {
        return Vec::new();
    }

    let mut intervals = Vec::new();
    let mut start = indices[0];
    let mut end = indices[0];

    for &index in indices.iter().skip(1) {
        if index == end + 1 {
            end = index;
        } else {
            intervals.push((start, end));
            start = index;
            end = index;
        }
    }

    intervals.push((start, end));
    intervals
}

#[cfg(test)]
mod test_merge_intervals {
    use super::merge_intervals;

    #[test]
    fn test_empty_vector() {
        let vec: Vec<usize> = vec![];
        assert_eq!(merge_intervals(&vec), vec![]);
    }

    #[test]
    fn test_single_element() {
        let vec = vec![1];
        assert_eq!(merge_intervals(&vec), vec![(1, 1)]);
    }

    #[test]
    fn test_non_consecutive_numbers() {
        let vec = vec![1, 3, 5, 7, 9];
        assert_eq!(
            merge_intervals(&vec),
            vec![(1, 1), (3, 3), (5, 5), (7, 7), (9, 9)]
        );
    }

    #[test]
    fn test_consecutive_numbers() {
        let vec = vec![1, 2, 3, 4, 5];
        assert_eq!(merge_intervals(&vec), vec![(1, 5)]);
    }

    #[test]
    fn test_mixed_numbers() {
        let vec = vec![1, 2, 4, 5, 7];
        assert_eq!(merge_intervals(&vec), vec![(1, 2), (4, 5), (7, 7)]);
    }
}
