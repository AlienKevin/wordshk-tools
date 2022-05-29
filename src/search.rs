use super::dict::Variants;
use super::jyutping::{
    parse_pr, JyutPing, JyutPingCoda, JyutPingInitial, JyutPingNucleus, LaxJyutPing,
    LaxJyutPingSegment,
};
use super::rich_dict::{get_simplified_variants, RichDict, RichEntry};
use super::unicode;
use super::word_frequencies::WORD_FREQUENCIES;
use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::collections::HashMap;
use strsim::{generic_levenshtein, normalized_levenshtein};
use unicode_segmentation::UnicodeSegmentation;

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

pub struct ComboVariants {
    pub traditional: Variants,
    pub simplified: Variants,
}

pub fn rich_dict_to_variants_map(dict: &RichDict) -> VariantsMap {
    dict.iter()
        .map(|(id, entry)| {
            (
                *id,
                ComboVariants {
                    traditional: entry.variants.clone(),
                    simplified: get_simplified_variants(&entry.variants, &entry.variants_simp),
                },
            )
        })
        .collect()
}

/// Manners of articulation of initials
///
/// source: <https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.148.6501&rep=rep1&type=pdf>
/// (page 9)
#[derive(Debug, PartialEq)]
enum InitialCategories {
    Plosive,
    Nasal,
    Fricative,
    Affricate,
    TongueRolled,
    SemiVowel,
}

/// Classify initials based on their manner of articulation
///
/// source: https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.148.6501&rep=rep1&type=pdf
/// (page 7)
fn classify_initial(initial: &JyutPingInitial) -> InitialCategories {
    use InitialCategories::*;
    use JyutPingInitial::*;
    match initial {
        B | P | D | T | G | K | Gw | Kw => Plosive,
        M | N | Ng => Nasal,
        S | F | H => Fricative,
        C | Z => Affricate,
        L => TongueRolled,
        W | J => SemiVowel,
    }
}

/// Categories of nucleus
type NucleusCategories = (VowelBackness, VowelHeight, VowelRoundedness);

#[derive(Debug, PartialEq)]
enum VowelBackness {
    Front,
    Central,
    Back,
}

#[derive(Debug, PartialEq)]
enum VowelHeight {
    Close,
    Mid,
    Open,
}

#[derive(Debug, PartialEq)]
enum VowelRoundedness {
    Rounded,
    Unrounded,
}

/// Classify nucleus based on their backness, height, and roundedness
///
/// source: <https://www.wikiwand.com/en/Cantonese_phonology#/Vowels_and_terminals>
fn classify_nucleus(nucleus: &JyutPingNucleus) -> NucleusCategories {
    use JyutPingNucleus::*;
    use VowelBackness::*;
    use VowelHeight::*;
    use VowelRoundedness::*;
    match nucleus {
        I => (Front, Close, Unrounded),
        Yu => (Front, Close, Rounded),
        E => (Front, Mid, Unrounded),
        Oe | Eo => (Front, Mid, Rounded),
        A | Aa => (Central, Open, Unrounded),
        U => (Back, Close, Rounded),
        O => (Back, Mid, Rounded),
    }
}

/// Categories of coda
#[derive(Debug, PartialEq)]
enum CodaCategories {
    Stop,
    Nasal,
    VowelI,
    VowelU,
}

/// Classify coda based on whether they are stops, nasals, or vowels
fn classify_coda(coda: &JyutPingCoda) -> CodaCategories {
    use CodaCategories::*;
    use JyutPingCoda::*;
    match coda {
        P | T | K => Stop,
        M | N | Ng => Nasal,
        I => VowelI,
        U => VowelU,
    }
}

/// Weighted jyutping comparison
///
/// Highest score when identical: 100
///
/// Score is split into four parts:
/// * Initial: 40
/// * Nucleus: 32
/// * Coda: 24
/// * Tone: 4
///
pub fn compare_jyutping(pr1: &JyutPing, pr2: &JyutPing) -> Score {
    if pr1 == pr2 {
        100
    } else {
        (if pr1.initial == pr2.initial {
            40
        } else if let (Some(i1), Some(i2)) = (pr1.initial.as_ref(), pr2.initial.as_ref()) {
            if classify_initial(&i1) == classify_initial(&i2) {
                24
            } else {
                0
            }
        } else {
            0
        }) + (if pr1.nucleus == pr2.nucleus {
            32
        } else if let (Some(n1), Some(n2)) = (pr1.nucleus.as_ref(), pr2.nucleus.as_ref()) {
            let ((backness1, height1, roundedness1), (backness2, height2, roundedness2)) =
                (classify_nucleus(&n1), classify_nucleus(&n2));
            if backness1 == backness2 && height1 == height2 && roundedness1 == roundedness2 {
                32 - 4
            } else {
                32 - 3
                    - (if backness1 == backness2 { 0 } else { 4 })
                    - (if height1 == height2 { 0 } else { 4 })
                    - (if roundedness1 == roundedness2 { 0 } else { 3 })
            }
        } else {
            0
        }) + (if pr1.coda == pr2.coda {
            24
        } else if let (Some(i1), Some(i2)) = (pr1.coda.as_ref(), pr2.coda.as_ref()) {
            if classify_coda(&i1) == classify_coda(&i2) {
                18
            } else {
                0
            }
        } else {
            0
        }) + (if pr1.tone == pr2.tone { 4 } else { 0 })
    }
}

fn compare_string(s1: &str, s2: &str) -> Score {
    normalize_score(normalized_levenshtein(s1, s2))
}

fn normalize_score(s: f64) -> Score {
    (s * MAX_SCORE as f64).round() as usize
}

pub fn compare_lax_jyutping_segment(pr1: &LaxJyutPingSegment, pr2: &LaxJyutPingSegment) -> Score {
    use LaxJyutPingSegment::*;
    match (pr1, pr2) {
        (Standard(pr1), Standard(pr2)) => compare_jyutping(pr1, pr2),
        (Standard(pr1), Nonstandard(pr2)) => compare_string(&pr1.to_string(), &pr2),
        (Nonstandard(pr1), Standard(pr2)) => compare_string(pr1, &pr2.to_string()),
        (Nonstandard(pr1), Nonstandard(pr2)) => compare_string(pr1, pr2),
    }
}

fn compare_lax_jyutping(pr1: &LaxJyutPing, pr2: &LaxJyutPing) -> Score {
    if pr1.0.len() != pr2.0.len() {
        0
    } else {
        let score: Score = pr1
            .0
            .iter()
            .zip(&pr2.0)
            .map(|(pr1, pr2)| compare_lax_jyutping_segment(pr1, &pr2))
            .sum();
        score / pr1.0.len()
    }
}

fn score_pr_query(entry_pr: &LaxJyutPing, query: &LaxJyutPing) -> (Score, Index) {
    entry_pr.0.windows(query.0.len()).enumerate().fold(
        (0, 0),
        |(max_score, max_index), (i, seg)| {
            let score = compare_lax_jyutping(entry_pr, query);
            if score > max_score {
                (score, i)
            } else {
                (max_score, max_index)
            }
        },
    )
}

#[derive(Copy, Clone, Eq, PartialEq)]
pub struct PrSearchRank {
    pub id: usize,
    pub variant_index: Index,
    pub pr_index: Index,
    pub score: Score,
    pub pr_start_index: Index,
}

impl Ord for PrSearchRank {
    fn cmp(&self, other: &Self) -> Ordering {
        self.score
            .cmp(&other.score)
            .then_with(|| other.pr_start_index.cmp(&self.pr_start_index))
    }
}

impl PartialOrd for PrSearchRank {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

pub fn pick_variants(variants: &ComboVariants, script: Script) -> &Variants {
    match script {
        Script::Simplified => &variants.simplified,
        Script::Traditional => &variants.traditional,
    }
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
    let query_entry = dict.get(&id).unwrap();
    sort_entry_group(
        dict.iter()
            .filter_map(|(_, entry)| {
                if query_entry
                    .variants
                    .to_words_set()
                    .intersection(&entry.variants.to_words_set())
                    .next()
                    != None
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

fn sort_entries_by_frequency(entries: &mut Vec<&RichEntry>) {
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

pub fn pr_search(variants_map: &VariantsMap, query: &str) -> BinaryHeap<PrSearchRank> {
    let mut ranks = BinaryHeap::new();
    if query.is_ascii() {
        let query = parse_pr(query);
        variants_map.iter().for_each(|(id, variants)| {
            variants
                .traditional
                .0
                .iter()
                .enumerate()
                .for_each(|(variant_index, variant)| {
                    let (score, pr_start_index, pr_index) = variant.prs.0.iter().enumerate().fold(
                        (0, 0, 0),
                        |(max_score, max_pr_start_index, max_pr_index), (pr_index, pr)| {
                            let (score, pr_start_index) = score_pr_query(pr, &query);
                            if score > max_score {
                                (score, pr_start_index, pr_index)
                            } else {
                                (max_score, max_pr_start_index, max_pr_index)
                            }
                        },
                    );
                    ranks.push(PrSearchRank {
                        id: *id,
                        variant_index,
                        pr_index,
                        score,
                        pr_start_index,
                    });
                });
        });
    }
    ranks
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

// source: https://stackoverflow.com/a/35907071/6798201
fn find_subsequence<T>(haystack: &[T], needle: &[T]) -> Option<usize>
where
    for<'a> &'a [T]: PartialEq,
{
    haystack
        .windows(needle.len())
        .position(|window| window == needle)
}

fn score_variant_query(entry_variant: &str, query: &str, script: Script) -> (Index, Score) {
    let entry_variant_normalized = &unicode::normalize(entry_variant)[..];
    let query_normalized = &match script {
        Script::Simplified => unicode::to_simplified,
        Script::Traditional => unicode::to_traditional,
    }(&unicode::to_hk_safe_variant(&unicode::normalize(query)))[..];
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
    variants_map.iter().for_each(|(id, variants)| {
        pick_variants(variants, script)
            .0
            .iter()
            .enumerate()
            .for_each(|(variant_index, variant)| {
                let (occurrence_index, levenshtein_score) =
                    score_variant_query(&variant.word, query, script);
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

pub fn combined_search(
    variants_map: &VariantsMap,
    query: &str,
    script: Script,
) -> (BinaryHeap<VariantSearchRank>, BinaryHeap<PrSearchRank>) {
    let mut variants_ranks = BinaryHeap::new();
    let mut pr_ranks = BinaryHeap::new();
    let pr_query = if query.is_ascii() {
        Some(parse_pr(query))
    } else {
        None
    };
    variants_map.iter().for_each(|(id, variants)| {
        pick_variants(variants, script)
            .0
            .iter()
            .enumerate()
            .for_each(|(variant_index, variant)| {
                let (occurrence_index, levenshtein_score) =
                    score_variant_query(&variant.word, query, script);
                if occurrence_index < usize::MAX || levenshtein_score >= 80 {
                    variants_ranks.push(VariantSearchRank {
                        id: *id,
                        variant_index,
                        occurrence_index,
                        levenshtein_score,
                    });
                }

                match &pr_query {
                    Some(query) => {
                        let (score, pr_start_index, pr_index) =
                            variant.prs.0.iter().enumerate().fold(
                                (0, 0, 0),
                                |(max_score, max_pr_start_index, max_pr_index), (pr_index, pr)| {
                                    let (score, pr_start_index) = score_pr_query(pr, query);
                                    if score > max_score {
                                        (score, pr_start_index, pr_index)
                                    } else {
                                        (max_score, max_pr_start_index, max_pr_index)
                                    }
                                },
                            );
                        pr_ranks.push(PrSearchRank {
                            id: *id,
                            variant_index,
                            pr_index,
                            score,
                            pr_start_index,
                        });
                    }
                    None => {
                        // do nothing
                    }
                }
            });
    });
    (variants_ranks, pr_ranks)
}
