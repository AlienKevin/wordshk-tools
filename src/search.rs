use crate::pr_index::{PrIndex, PrIndices, PrLocation};

use super::charlist::CHARLIST;
use super::dict::{Variant, Variants};
use super::english_index::{EnglishIndex, EnglishIndexData};
use super::iconic_simps::ICONIC_SIMPS;
use super::iconic_trads::ICONIC_TRADS;
use super::jyutping::{LaxJyutPings, Romanization};
use super::rich_dict::{RichDict, RichEntry};
use super::unicode;
use super::word_frequencies::WORD_FREQUENCIES;
use std::cmp::Ordering;
use std::collections::HashMap;
use std::collections::{BinaryHeap, HashSet};
use strsim::{generic_levenshtein, normalized_levenshtein};
use thesaurus;
use unicode_segmentation::UnicodeSegmentation;
use xxhash_rust::xxh3::xxh3_64;

/// Max score is 100
type Score = usize;

const MAX_SCORE: Score = 100;

type Index = usize;

#[derive(Copy, Clone)]
pub enum Script {
    Simplified,
    Traditional,
}

/// A Map from entry ID to variants and simplified variants
pub type VariantsMap = HashMap<usize, ComboVariants>;

#[derive(Debug, Clone, PartialEq)]
pub struct ComboVariant {
    pub word_trad: String,
    pub word_simp: String,
    pub prs: LaxJyutPings,
}

pub type ComboVariants = Vec<ComboVariant>;

pub fn rich_dict_to_variants_map(dict: &RichDict) -> VariantsMap {
    dict.iter()
        .map(|(id, entry)| {
            (
                *id,
                create_combo_variants(&entry.variants, &entry.variants_simp),
            )
        })
        .collect()
}

pub fn create_combo_variants(
    trad_variants: &Variants,
    simp_variant_strings: &[String],
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

fn normalize_score(s: f64) -> Score {
    (s * MAX_SCORE as f64).round() as usize
}

#[derive(Clone, Eq, PartialEq, Debug)]
pub struct PrSearchRank {
    pub id: usize,
    pub variant_index: Index,
    pub pr_index: Index,
    pub pr: String,
    pub score: Score,
}

impl Ord for PrSearchRank {
    fn cmp(&self, other: &Self) -> Ordering {
        self.score.cmp(&other.score).then(other.pr.cmp(&self.pr))
    }
}

impl PartialOrd for PrSearchRank {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

fn pick_variant(variant: &ComboVariant, script: Script) -> Variant {
    match script {
        Script::Simplified => Variant {
            word: variant.word_simp.clone(),
            prs: variant.prs.clone(),
        },
        Script::Traditional => Variant {
            word: variant.word_trad.clone(),
            prs: variant.prs.clone(),
        },
    }
}

pub fn pick_variants(variants: &ComboVariants, script: Script) -> Variants {
    Variants(
        variants
            .iter()
            .map(|combo| pick_variant(combo, script))
            .collect(),
    )
}

pub fn get_entry_id(variants_map: &VariantsMap, query: &str, script: Script) -> Option<usize> {
    variants_map.iter().find_map(|(id, variants)| {
        if pick_variants(variants, script)
            .to_words_set()
            .contains(query)
        {
            Some(*id)
        } else {
            None
        }
    })
}

pub fn get_entry_group(dict: &RichDict, id: &usize) -> Vec<RichEntry> {
    let query_entry = dict.get(id).unwrap();
    sort_entry_group(
        dict.iter()
            .filter_map(|(_, entry)| {
                if query_entry
                    .variants
                    .to_words_set()
                    .intersection(&entry.variants.to_words_set())
                    .next()
                    .is_some()
                {
                    Some(entry.clone())
                } else {
                    None
                }
            })
            .collect(),
    )
}

fn sort_entry_group(entry_group: Vec<RichEntry>) -> Vec<RichEntry> {
    let mut general = vec![];
    let mut vulgar = vec![];

    entry_group.iter().for_each(|entry| {
        if entry.labels.contains(&"粗俗".to_string()) || entry.labels.contains(&"俚語".to_string())
        {
            vulgar.push(entry);
        } else {
            general.push(entry);
        }
    });

    sort_entries_by_frequency(&mut general);
    sort_entries_by_frequency(&mut vulgar);

    general.append(&mut vulgar);
    general.iter().map(|entry| (*entry).clone()).collect()
}

fn sort_entries_by_frequency(entries: &mut [&RichEntry]) {
    entries.sort_by(|a, b| {
        get_entry_frequency(a.id)
            .cmp(&get_entry_frequency(b.id))
            .reverse()
            .then(a.defs.len().cmp(&b.defs.len()).reverse())
    });
}

fn get_entry_frequency(entry_id: usize) -> u8 {
    *WORD_FREQUENCIES.get(&(entry_id as u32)).unwrap_or(&50)
}

pub fn pr_search(
    pr_indices: &PrIndices,
    dict: &RichDict,
    query: &str,
    romanization: Romanization,
) -> BinaryHeap<PrSearchRank> {
    let mut ranks = BinaryHeap::new();
    let query = unicode::normalize(query);
    fn lookup_index(
        query: &str,
        deletions: usize,
        index: &PrIndex,
        dict: &RichDict,
        ranks: &mut BinaryHeap<PrSearchRank>,
    ) {
        if deletions < query.chars().count() {
            for (query_variant, added_deletions) in
                crate::pr_index::generate_deletion_neighborhood(&query, deletions)
            {
                if let Some(locations) = index.get(&xxh3_64(query_variant.as_bytes())) {
                    for PrLocation {
                        entry_id,
                        variant_index,
                        pr_index,
                    } in locations
                    {
                        ranks.push(PrSearchRank {
                            id: *entry_id,
                            variant_index: *variant_index as Index,
                            pr_index: *pr_index as Index,
                            pr: dict.get(&entry_id).unwrap().variants.0[*variant_index as Index]
                                .prs
                                .0[*pr_index as Index]
                                .to_string(),
                            score: MAX_SCORE - deletions - added_deletions,
                        });
                    }
                }
            }
        }
    }
    match romanization {
        Romanization::Jyutping => {
            if query.contains(&['1', '2', '3', '4', '5', '6']) && query.contains(" ") {
                for (deletions, index) in pr_indices.tone_and_space.iter().enumerate() {
                    lookup_index(&query, deletions, index, dict, &mut ranks);
                }
            } else if query.contains(&['1', '2', '3', '4', '5', '6']) {
                for (deletions, index) in pr_indices.tone.iter().enumerate() {
                    lookup_index(&query, deletions, index, dict, &mut ranks);
                }
            } else if query.contains(' ') {
                for (deletions, index) in pr_indices.space.iter().enumerate() {
                    lookup_index(&query, deletions, index, dict, &mut ranks);
                }
            } else {
                for (deletions, index) in pr_indices.none.iter().enumerate() {
                    lookup_index(&query, deletions, index, dict, &mut ranks);
                }
            }
        }
        Romanization::Yale => {
            if query.contains(' ') {
                for (deletions, index) in pr_indices.space.iter().enumerate() {
                    lookup_index(&query, deletions, index, dict, &mut ranks);
                }
                for (deletions, index) in pr_indices.tone_and_space.iter().enumerate() {
                    lookup_index(&query, deletions, index, dict, &mut ranks);
                }
            } else {
                for (deletions, index) in pr_indices.none.iter().enumerate() {
                    lookup_index(&query, deletions, index, dict, &mut ranks);
                }
                for (deletions, index) in pr_indices.tone.iter().enumerate() {
                    lookup_index(&query, deletions, index, dict, &mut ranks);
                }
            }
        }
    }

    // deduplicate ranks
    let mut seen_ids = HashSet::new();
    ranks
        .into_sorted_vec()
        .into_iter()
        .rev()
        .filter(|rank| {
            if seen_ids.contains(&rank.id) {
                false
            } else {
                seen_ids.insert(rank.id)
            }
        })
        .collect()
}

#[derive(Copy, Clone, Eq, PartialEq)]
pub struct VariantSearchRank {
    pub id: usize,
    pub variant_index: Index,
    pub occurrence_index: Index,
    pub levenshtein_score: Score,
}

impl Ord for VariantSearchRank {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .occurrence_index
            .cmp(&self.occurrence_index)
            .then_with(|| self.levenshtein_score.cmp(&other.levenshtein_score))
    }
}

impl PartialOrd for VariantSearchRank {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

fn normalized_unicode_levenshtein(a: &str, b: &str) -> f64 {
    if a.is_empty() && b.is_empty() {
        return 1.0;
    }
    let a_graphemes = UnicodeSegmentation::graphemes(a, true).collect::<Vec<&str>>();
    let a_len = a_graphemes.len();
    let b_graphemes = UnicodeSegmentation::graphemes(b, true).collect::<Vec<&str>>();
    let b_len = b_graphemes.len();
    1.0 - (generic_levenshtein(&a_graphemes, &b_graphemes) as f64) / (a_len.max(b_len) as f64)
}

fn word_levenshtein(a: &Vec<&str>, b: &Vec<&str>) -> usize {
    if a.is_empty() && b.is_empty() {
        return 0;
    }
    generic_levenshtein(a, b)
}

// source: https://stackoverflow.com/a/35907071/6798201
fn find_subsequence<T>(haystack: &[T], needle: &[T]) -> Option<usize>
where
    for<'a> &'a [T]: PartialEq,
{
    haystack
        .windows(needle.len())
        .position(|window| window == needle)
}

fn score_variant_query(
    entry_variant: &ComboVariant,
    query_normalized: &str,
    query_script: Script,
) -> (Index, Score) {
    let entry_variant_normalized =
        &unicode::normalize(&pick_variant(entry_variant, query_script).word)[..];
    let variant_graphemes =
        UnicodeSegmentation::graphemes(entry_variant_normalized, true).collect::<Vec<&str>>();
    let query_graphemes =
        UnicodeSegmentation::graphemes(query_normalized, true).collect::<Vec<&str>>();
    let occurrence_index = match find_subsequence(&variant_graphemes, &query_graphemes) {
        Some(i) => i,
        None => usize::MAX,
    };
    let levenshtein_score = normalize_score(normalized_unicode_levenshtein(
        entry_variant_normalized,
        query_normalized,
    ));
    (occurrence_index, levenshtein_score)
}

pub fn variant_search(
    variants_map: &VariantsMap,
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
    variants_map.iter().for_each(|(id, variants)| {
        variants
            .iter()
            .enumerate()
            .for_each(|(variant_index, variant)| {
                let (occurrence_index, levenshtein_score) =
                    score_variant_query(variant, query_normalized, query_script);
                if occurrence_index < usize::MAX || levenshtein_score >= 80 {
                    ranks.push(VariantSearchRank {
                        id: *id,
                        variant_index,
                        occurrence_index,
                        levenshtein_score,
                    });
                }
            });
    });
    ranks
}

pub enum CombinedSearchRank {
    Variant(BinaryHeap<VariantSearchRank>),
    Pr(BinaryHeap<PrSearchRank>),
    All(
        BinaryHeap<VariantSearchRank>,
        BinaryHeap<PrSearchRank>,
        Vec<EnglishIndexData>,
    ),
}

// Auto recognize the type of the query
pub fn combined_search(
    variants_map: &VariantsMap,
    pr_indices: &PrIndices,
    english_index: &EnglishIndex,
    dict: &RichDict,
    query: &str,
    script: Script,
    romanization: Romanization,
) -> CombinedSearchRank {
    // if the query has CJK characters, it can only be a variant
    if query.chars().any(unicode::is_cjk) {
        return CombinedSearchRank::Variant(variant_search(variants_map, query, script));
    }

    // otherwise if the query doesn't have a very strong feature,
    // it can be a variant, a jyutping or an english phrase
    let query_normalized = &unicode::to_hk_safe_variant(&unicode::normalize(query))[..];
    let query_script = if query_normalized.chars().any(|c| ICONIC_SIMPS.contains(&c)) {
        // query contains iconic simplified characters
        Script::Simplified
    } else if query_normalized.chars().any(|c| ICONIC_TRADS.contains(&c)) {
        Script::Traditional
    } else {
        script
    };
    let variants_ranks = variant_search(variants_map, query, query_script);
    let pr_ranks = pr_search(pr_indices, dict, query, romanization);
    let english_results = english_search(english_index, query);

    CombinedSearchRank::All(variants_ranks, pr_ranks, english_results)
}

pub fn english_search(english_index: &EnglishIndex, query: &str) -> Vec<EnglishIndexData> {
    let query = unicode::normalize_english_word_for_search_index(query);
    let default_results = vec![];
    let results = english_index
        .get(&query)
        .unwrap_or(fuzzy_english_search(
            english_index,
            &[query.clone()],
            &default_results,
        ))
        .to_vec();
    if results.is_empty() {
        let synonyms = thesaurus::synonyms(query);
        if synonyms.is_empty() {
            results
        } else {
            synonyms
                .iter()
                .fold(
                    None,
                    |results: Option<&Vec<EnglishIndexData>>, word| match (
                        english_index.get(&word.to_string()),
                        results,
                    ) {
                        (Some(current_results), Some(results)) => {
                            if current_results[0].score > results[0].score {
                                Some(current_results)
                            } else {
                                Some(results)
                            }
                        }
                        (Some(current_results), None) => Some(current_results),
                        _ => results,
                    },
                )
                .unwrap_or(fuzzy_english_search(
                    english_index,
                    &synonyms,
                    &default_results,
                ))
                .to_vec()
        }
    } else {
        results
    }
}

fn fuzzy_english_search<'a>(
    english_index: &'a EnglishIndex,
    queries: &[String],
    default_results: &'a Vec<EnglishIndexData>,
) -> &'a Vec<EnglishIndexData> {
    english_index
        .iter()
        .fold(
            (60, default_results), // must have a score of at least 60 out of 100
            |(max_score, max_entries), (phrase, entries)| {
                let (mut next_max_score, mut next_max_entries) = (max_score, max_entries);
                queries.iter().for_each(|query| {
                    let current_score = score_english_query(query, phrase);
                    if current_score > max_score {
                        (next_max_score, next_max_entries) = (current_score, entries)
                    }
                });
                (next_max_score, next_max_entries)
            },
        )
        .1
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
