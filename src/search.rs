use super::*;
use strsim::{normalized_levenshtein, generic_levenshtein};
use std::collections::BinaryHeap;
use std::cmp::Ordering;
use unicode_segmentation::UnicodeSegmentation;

/// Max score is 25
type Score = usize;

const MAX_SCORE: Score = 25;

type Index = usize;

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
    Vowel,
}

/// Classify coda based on whether they are stops, nasals, or vowels
fn classify_coda(coda: &JyutPingCoda) -> CodaCategories {
    use CodaCategories::*;
    use JyutPingCoda::*;
    match coda {
        P | T | K => Stop,
        M | N | Ng => Nasal,
        I | U => Vowel,
    }
}

/// Weighted jyutping comparison
///
/// Highest score when identical: 25
///
/// Score is split into four parts:
/// * Initial: 10
/// * Nucleus: 8
/// * Coda: 6
/// * Tone: 1
///
pub fn compare_jyutping(pr1: &JyutPing, pr2: &JyutPing) -> Score {
    if pr1 == pr2 {
        10 + 8 + 6 + 1 // 25
    } else {
        (if pr1.initial == pr2.initial {
            10
        } else if let (Some(i1), Some(i2)) = (pr1.initial.as_ref(), pr2.initial.as_ref()) {
            if classify_initial(&i1) == classify_initial(&i2) {
                6
            } else {
                0
            }
        } else {
            0
        }) + (if pr1.nucleus == pr2.nucleus {
            8
        } else {
            let ((backness1, height1, roundedness1), (backness2, height2, roundedness2)) =
                (classify_nucleus(&pr1.nucleus), classify_nucleus(&pr2.nucleus));
            (if backness1 == backness2 { 2 } else { 0 })
                + (if height1 == height2 { 2 } else { 0 })
                + (if roundedness1 == roundedness2 { 1 } else { 0 })
        }) + (if pr1.coda == pr2.coda {
            6
        } else if let (Some(i1), Some(i2)) = (pr1.coda.as_ref(), pr2.coda.as_ref()) {
            if classify_coda(&i1) == classify_coda(&i2) {
                4
            } else {
                0
            }
        } else {
            0
        }) + (if pr1.tone == pr2.tone { 1 } else { 0 })
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
        (Standard(pr1), Standard(pr2)) =>
            compare_jyutping(pr1, pr2),
        (Standard(pr1), Nonstandard(pr2)) =>
            compare_string(&jyutping_to_string(pr1), &pr2),
        (Nonstandard(pr1), Standard(pr2)) =>
            compare_string(pr1, &jyutping_to_string(pr2)),
        (Nonstandard(pr1), Nonstandard(pr2)) =>
            compare_string(pr1, pr2),
    }
}

fn compare_lax_jyutping(pr1: &LaxJyutPing, pr2: &LaxJyutPing) -> Score {
    if pr1.len() != pr2.len() {
        0
    } else {
        let score: Score = pr1.iter().zip(pr2).map(|(pr1, pr2)|
            compare_lax_jyutping_segment(pr1, pr2)
        ).sum();
        score / pr1.len()
    }
}

fn score_pr_query(entry_pr: &LaxJyutPing, query: &LaxJyutPing) -> (Score, Index) {
    entry_pr
        .windows(query.len())
        .enumerate()
        .fold((0, 0), |(max_score, max_index), (i, seg)| {
            let score = compare_lax_jyutping(entry_pr, query);
            if score > max_score {
                (score, i)
            } else {
                (max_score, max_index)
            }
        })
}

#[derive(Copy, Clone, Eq, PartialEq)]
pub struct PrSearchResult {
    pub id: usize,
    pub variant_index: Index,
    pub score: Score,
    pub pr_index: Index,
}

impl Ord for PrSearchResult {
    fn cmp(&self, other: &Self) -> Ordering {
        self.score.cmp(&other.score)
            .then_with(|| other.pr_index.cmp(&self.pr_index))
    }
}

impl PartialOrd for PrSearchResult {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

pub fn pr_search(dict: &Dict, query: &LaxJyutPing) -> BinaryHeap<PrSearchResult> {
    let mut results = BinaryHeap::new();
    dict.iter().for_each(|(id, entry)| {
        entry.variants.iter().enumerate().for_each(|(variant_index, variant)| {
            variant.prs.iter().for_each(|pr| {
                let (score, pr_index) = score_pr_query(pr, query);
                results.push(PrSearchResult {
                    id: *id,
                    variant_index,
                    score,
                    pr_index,
                });
            });
        });
    });
    results
}

#[derive(Copy, Clone, Eq, PartialEq)]
pub struct VariantSearchResult {
    pub id: usize,
    pub variant_index: Index,
    pub occurrence_index: Index,
    pub levenshtein_score: Score,
}

impl Ord for VariantSearchResult {
    fn cmp(&self, other: &Self) -> Ordering {
        other.occurrence_index.cmp(&self.occurrence_index)
            .then_with(|| self.levenshtein_score.cmp(&other.levenshtein_score))
    }
}

impl PartialOrd for VariantSearchResult {
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
    haystack.windows(needle.len()).position(|window| window == needle)
}

fn score_variant_query(entry_variant: &str, query: &str) -> (Index, Score) {
    let variant_graphemes = UnicodeSegmentation::graphemes(entry_variant, true).collect::<Vec<&str>>();
    let query_graphemes = UnicodeSegmentation::graphemes(query, true).collect::<Vec<&str>>();
    let occurrence_index =
        match find_subsequence(&variant_graphemes, &query_graphemes) {
            Some(i) => i,
            None => usize::MAX,
        };
    let levenshtein_score = normalize_score(normalized_unicode_levenshtein(entry_variant, query));
    (occurrence_index, levenshtein_score)
}

pub fn variant_search(dict: &Dict, query: &str) -> BinaryHeap<VariantSearchResult> {
    let mut results = BinaryHeap::new();
    dict.iter().for_each(|(id, entry)| {
        entry.variants.iter().enumerate().for_each(|(variant_index, variant)| {
            let (occurrence_index, levenshtein_score) = score_variant_query(&variant.word, query);
            results.push(VariantSearchResult {
                id: *id,
                variant_index,
                occurrence_index,
                levenshtein_score,
            });
        });
    });
    results
}
