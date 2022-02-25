use super::dict::{
    Dict, JyutPing, JyutPingCoda, JyutPingInitial, JyutPingNucleus, LaxJyutPing, LaxJyutPingSegment,
};
use super::rich_dict::RichDict;
use super::unicode;
use lazy_static::lazy_static;
use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::collections::HashMap;
use strsim::{generic_levenshtein, normalized_levenshtein};
use unicode_segmentation::UnicodeSegmentation;

/// Max score is 100
type Score = usize;

const MAX_SCORE: Score = 100;

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

pub fn pr_search(dict: &RichDict, query: &LaxJyutPing) -> BinaryHeap<PrSearchRank> {
    let mut ranks = BinaryHeap::new();
    dict.iter().for_each(|(id, entry)| {
        entry
            .variants
            .0
            .iter()
            .enumerate()
            .for_each(|(variant_index, variant)| {
                let (score, pr_start_index, pr_index) = variant.prs.0.iter().enumerate().fold(
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
                ranks.push(PrSearchRank {
                    id: *id,
                    variant_index,
                    pr_index,
                    score,
                    pr_start_index,
                });
            });
    });
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

fn score_variant_query(entry_variant: &str, query: &str) -> (Index, Score) {
    let variant_graphemes =
        UnicodeSegmentation::graphemes(entry_variant, true).collect::<Vec<&str>>();
    let query_graphemes = UnicodeSegmentation::graphemes(query, true).collect::<Vec<&str>>();
    let occurrence_index = match find_subsequence(&variant_graphemes, &query_graphemes) {
        Some(i) => i,
        None => usize::MAX,
    };
    let levenshtein_score = normalize_score(normalized_unicode_levenshtein(entry_variant, query));
    (occurrence_index, levenshtein_score)
}

pub fn variant_search(dict: &RichDict, query: &str) -> BinaryHeap<VariantSearchRank> {
    let query_safe = &convert_to_hk_safe_variant(query);
    let mut ranks = BinaryHeap::new();
    dict.iter().for_each(|(id, entry)| {
        entry
            .variants
            .0
            .iter()
            .enumerate()
            .for_each(|(variant_index, variant)| {
                let (occurrence_index, levenshtein_score) =
                    score_variant_query(&variant.word, query_safe);
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

lazy_static! {
    static ref HONG_KONG_VARIANT_MAP_SAFE: HashMap<char, char> = {
        HashMap::from([
            ('倂', '併'),
            ('僞', '偽'),
            ('兌', '兑'),
            ('册', '冊'),
            ('删', '刪'),
            ('匀', '勻'),
            ('滙', '匯'),
            ('叄', '叁'),
            ('吿', '告'),
            ('啟', '啓'),
            ('塡', '填'),
            ('姗', '姍'),
            ('媼', '媪'),
            ('嬀', '媯'),
            ('幷', '并'),
            ('悅', '悦'),
            ('悳', '惪'),
            ('慍', '愠'),
            ('戶', '户'),
            ('抛', '拋'),
            ('挩', '捝'),
            ('㨂', '揀'),
            ('搵', '揾'),
            ('敎', '教'),
            ('敓', '敚'),
            ('旣', '既'),
            ('曁', '暨'),
            ('栅', '柵'),
            ('梲', '棁'),
            ('槪', '概'),
            ('榲', '榅'),
            ('氳', '氲'),
            ('汙', '污'),
            ('没', '沒'),
            ('洩', '泄'),
            ('涗', '涚'),
            ('溫', '温'),
            ('潙', '溈'),
            ('潨', '潀'),
            ('溼', '濕'),
            ('爲', '為'),
            ('熅', '煴'),
            ('床', '牀'),
            ('奬', '獎'),
            ('眞', '真'),
            ('衆', '眾'),
            ('硏', '研'),
            ('稅', '税'),
            ('緖', '緒'),
            ('縕', '緼'),
            ('駡', '罵'),
            ('羣', '群'),
            ('脫', '脱'),
            ('膃', '腽'),
            ('蔥', '葱'),
            ('蒕', '蒀'),
            ('蔿', '蒍'),
            ('葯', '藥'),
            ('蘊', '藴'),
            ('蛻', '蜕'),
            ('衛', '衞'),
            ('裡', '裏'),
            ('說', '説'),
            ('艷', '豔'),
            ('轀', '輼'),
            ('醞', '醖'),
            ('鉤', '鈎'),
            ('銳', '鋭'),
            ('錬', '鍊'),
            ('鎭', '鎮'),
            ('銹', '鏽'),
            ('閱', '閲'),
            ('鷄', '雞'),
            ('鰮', '鰛'),
            ('麵', '麪'),
        ])
    };
}

/// Returns a 'HK variant' of the characters of the input text. The input is
/// assumed to be Chinese traditional. This variant list confirms to most
/// expectations of how characters should be written in Hong Kong, but does not
/// necessarily conform to any rigid standard. It may be fine tuned by editors
/// of words.hk. This is the "safe" version that is probably less controversial.
fn convert_to_hk_safe_variant(variant: &str) -> String {
    unicode::to_graphemes(variant)
        .iter()
        .map(|g| {
            if unicode::test_g(unicode::is_cjk, g) {
                if g.chars().count() == 1 {
                    if let Some(c) = g.chars().next() {
                        return match HONG_KONG_VARIANT_MAP_SAFE.get(&c) {
                            Some(s) => s.to_string(),
                            None => g.to_string(),
                        };
                    }
                }
            }
            return g.to_string();
        })
        .collect::<Vec<String>>()
        .join("")
}
