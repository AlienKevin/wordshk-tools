use crate::dict::{clause_to_string, Clause, EntryId};
use crate::jyutping::{parse_jyutpings, remove_yale_diacritics, LaxJyutPing};
use crate::lihkg_frequencies::LIHKG_FREQUENCIES;
use crate::pr_index::{FstPrIndicesLike, PrLocation, MAX_DELETIONS};
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
use fst::automaton::Levenshtein;
use itertools::Itertools;
use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet};
use std::collections::{BinaryHeap, HashSet};
use strsim::{generic_levenshtein, levenshtein, normalized_levenshtein};

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
    fn get_entry(&self, id: EntryId) -> RichEntry;
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
    fn get_entry(&self, entry_id: EntryId) -> RichEntry {
        let conn = self.conn();
        let mut stmt = conn
            .prepare("SELECT entry FROM rich_dict WHERE id = ?")
            .unwrap();
        let entry_str: String = stmt.query_row([entry_id], |row| row.get(0)).unwrap();
        serde_json::from_str(&entry_str).unwrap()
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

pub fn get_entry_group(dict: &dyn RichDictLike, id: EntryId) -> Vec<RichEntry> {
    let entry = dict.get_entry(id);
    let query_word_set: HashSet<&str> = entry.variants.to_words_set();
    sort_entry_group(
        dict.get_ids()
            .iter()
            .filter_map(|id| {
                let entry = dict.get_entry(*id);
                let current_word_set: HashSet<&str> = entry.variants.to_words_set();
                if query_word_set
                    .intersection(&current_word_set)
                    .next()
                    .is_some()
                {
                    Some(dict.get_entry(*id).clone())
                } else {
                    None
                }
            })
            .collect(),
    )
}

fn sort_entry_group(mut entry_group: Vec<RichEntry>) -> Vec<RichEntry> {
    entry_group.sort_by(|a, b| {
        get_entry_frequency(a.id)
            .cmp(&get_entry_frequency(b.id))
            .reverse()
            .then(a.defs.len().cmp(&b.defs.len()).reverse())
            .then(a.id.cmp(&b.id))
    });
    entry_group
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

pub fn pr_search(
    pr_indices: &dyn FstPrIndicesLike,
    dict: &dyn RichDictLike,
    query: &str,
    script: Script,
    romanization: Romanization,
) -> BinaryHeap<PrSearchRank> {
    let query = unicode::normalize(query);

    if query.is_empty() {
        return BinaryHeap::new();
    }

    let mut ranks = BTreeMap::new();

    fn to_yale(s: &str) -> String {
        parse_jyutpings(s)
            .unwrap()
            .into_iter()
            .map(|jyutping| jyutping.to_yale())
            .join(" ")
    }

    fn lookup_index(
        query: &str,
        search: impl FnOnce(Levenshtein) -> BTreeSet<PrLocation>,
        dict: &dyn RichDictLike,
        script: Script,
        romanization: Romanization,
        ranks: &mut BTreeMap<EntryId, PrSearchRank>,
        pr_variant_generator: fn(&str) -> String,
    ) {
        let query_no_space = query.replace(' ', "");
        let max_deletions = (query_no_space.chars().count() - 1).min(MAX_DELETIONS);
        let lev = Levenshtein::new(&query_no_space, max_deletions as u32).unwrap();

        for PrLocation {
            entry_id,
            variant_index,
            pr_index,
        } in search(lev)
        {
            let jyutping: &LaxJyutPing = &dict.get_entry(entry_id).variants.0
                [variant_index as Index]
                .prs
                .0[pr_index as Index];
            let jyutping = jyutping.to_string();
            let pr_variant = pr_variant_generator(&jyutping);
            let distance = levenshtein(&query, &pr_variant);
            if distance <= 1 {
                let matched_pr = match romanization {
                    Romanization::Jyutping => diff_prs(query, &jyutping),
                    Romanization::Yale => {
                        use unicode_normalization::UnicodeNormalization;
                        let yale = to_yale(&jyutping).nfd().collect::<String>();
                        let query = query.nfd().collect::<String>();
                        diff_prs(&query, &yale)
                    }
                };
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
                let frequency_count = get_max_frequency_count(&dict.get_entry(entry_id).variants);
                let def_len = dict.get_entry(entry_id).defs.len();
                let variant = pick_variant(
                    dict.get_entry(entry_id)
                        .variants
                        .0
                        .get(variant_index as usize)
                        .unwrap(),
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
                            variant_indices: vec![(variant_index as Index, pr_index as Index)],
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

    match romanization {
        Romanization::Jyutping => {
            const TONES: [char; 6] = ['1', '2', '3', '4', '5', '6'];

            if query.contains(TONES) && query.contains(' ') {
                lookup_index(
                    &query,
                    |query| pr_indices.search(true, query, romanization),
                    dict,
                    script,
                    romanization,
                    &mut ranks,
                    |s| s.to_string(),
                );
            } else if query.contains(TONES) {
                lookup_index(
                    &query,
                    |query| pr_indices.search(true, query, romanization),
                    dict,
                    script,
                    romanization,
                    &mut ranks,
                    |s| s.replace(' ', ""),
                );
            } else if query.contains(' ') {
                lookup_index(
                    &query,
                    |query| pr_indices.search(false, query, romanization),
                    dict,
                    script,
                    romanization,
                    &mut ranks,
                    |s| s.replace(TONES, ""),
                );
            } else {
                lookup_index(
                    &query,
                    |query| pr_indices.search(false, query, romanization),
                    dict,
                    script,
                    romanization,
                    &mut ranks,
                    |s| s.replace(TONES, "").replace(' ', ""),
                );
            }
        }
        Romanization::Yale => {
            fn to_yale_no_tones(s: &str) -> String {
                parse_jyutpings(s)
                    .unwrap()
                    .into_iter()
                    .map(|mut jyutping| {
                        jyutping.tone = None;
                        jyutping.to_yale_no_diacritics()
                    })
                    .join(" ")
            }

            let has_tone = remove_yale_diacritics(&query) != query;

            if has_tone && query.contains(' ') {
                lookup_index(
                    &query,
                    |query| pr_indices.search(true, query, romanization),
                    dict,
                    script,
                    romanization,
                    &mut ranks,
                    to_yale,
                );
            } else if has_tone {
                lookup_index(
                    &query,
                    |query| pr_indices.search(true, query, romanization),
                    dict,
                    script,
                    romanization,
                    &mut ranks,
                    |s| to_yale(s).replace(' ', ""),
                );
            } else if query.contains(' ') {
                lookup_index(
                    &query,
                    |query| pr_indices.search(true, query, romanization),
                    dict,
                    script,
                    romanization,
                    &mut ranks,
                    to_yale,
                );

                lookup_index(
                    &query,
                    |query| pr_indices.search(false, query, romanization),
                    dict,
                    script,
                    romanization,
                    &mut ranks,
                    to_yale_no_tones,
                );
            } else {
                lookup_index(
                    &query,
                    |query| pr_indices.search(true, query, romanization),
                    dict,
                    script,
                    romanization,
                    &mut ranks,
                    |s| to_yale(s).replace(' ', ""),
                );

                lookup_index(
                    &query,
                    |query| pr_indices.search(false, query, romanization),
                    dict,
                    script,
                    romanization,
                    &mut ranks,
                    |s| to_yale_no_tones(s).replace(' ', ""),
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
    variant: &RichVariant,
    query_normalized: &str,
    query_script: Script,
    script: Script,
) -> (Index, Score, MatchedInfix) {
    let variant_normalized = &unicode::normalize(&pick_variant(variant, query_script).word)[..];
    // The query has to be fully contained by the variant
    let occurrence_index = match variant_normalized.find(query_normalized) {
        Some(i) => i,
        None => return (usize::MAX, usize::MAX, MatchedInfix::default()),
    };
    let length_diff: usize = variant_normalized.chars().count() - query_normalized.chars().count();
    let variant_normalized_original = &unicode::normalize(&pick_variant(variant, script).word)[..];
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
        let entry = dict.get_entry(id);
        let frequency_count = get_max_frequency_count(&entry.variants);
        entry
            .variants
            .0
            .iter()
            .enumerate()
            .for_each(|(variant_index, variant)| {
                let (occurrence_index, length_diff, matched_variant) =
                    score_variant_query(variant, &query_normalized, query_script, script);
                if occurrence_index < usize::MAX && length_diff <= 2 {
                    ranks.push(VariantSearchRank {
                        id,
                        def_len: dict.get_entry(id).defs.len(),
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

pub enum CombinedSearchRank {
    Variant(BinaryHeap<VariantSearchRank>),
    Pr(BinaryHeap<PrSearchRank>),
    All(
        BinaryHeap<VariantSearchRank>,
        BinaryHeap<PrSearchRank>,
        BinaryHeap<EnglishSearchRank>,
    ),
}

// Auto recognize the type of the query
pub fn combined_search<D>(
    dict: &D,
    query: &str,
    script: Script,
    romanization: Romanization,
) -> CombinedSearchRank
where
    D: RichDictLike + VariantIndexLike + FstPrIndicesLike + EnglishIndexLike,
{
    // if the query has CJK characters, it can only be a variant
    if query.chars().any(unicode::is_cjk) {
        return CombinedSearchRank::Variant(variant_search(dict, dict, query, script));
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
    let variants_ranks = variant_search(dict, dict, query, query_script);
    let pr_ranks = pr_search(dict, dict, query, script, romanization);
    let english_results = english_search(dict, dict, query, script);

    CombinedSearchRank::All(variants_ranks, pr_ranks, english_results)
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

#[cfg(feature = "embedding-search")]
use fastembed::{EmbeddingBase, FlagEmbedding};
#[cfg(feature = "embedding-search")]
use finalfusion::prelude::*;
#[cfg(feature = "embedding-search")]
use ndarray::Array1;
#[cfg(feature = "embedding-search")]
pub fn english_embedding_search(
    english_embeddings: &Embeddings<VocabWrap, StorageWrap>,
    dict: &ArchivedRichDict,
    query: &str,
) -> Vec<EnglishSearchRank> {
    // let mut start_time = std::time::Instant::now();

    let model: FlagEmbedding = FlagEmbedding::try_new(Default::default()).unwrap();

    let query_embedding = model
        .embed(vec![format!("query: {query}")], None)
        .unwrap()
        .remove(0);
    let query_embedding: Array1<f32> = query_embedding.into();

    // println!(
    //     "calculating query's embedding took {:?}",
    //     start_time.elapsed()
    // );

    let query_embedding_norm: f32 = query_embedding.dot(&query_embedding).sqrt();
    let query_embedding_normalized = query_embedding.mapv(|x| x / query_embedding_norm);

    // start_time = std::time::Instant::now();

    let mut ranks: Vec<(String, f32)> = vec![];

    for (id, v) in english_embeddings {
        let cosine_similarity = v.dot(&query_embedding_normalized);
        if cosine_similarity > 0.5 {
            ranks.push((id.to_string(), cosine_similarity));
        }
    }

    ranks.sort_by(|(_, s1), (_, s2)| s2.total_cmp(s1));

    // println!("embedding search took {:?}", start_time.elapsed());

    ranks
        .into_iter()
        .take(50)
        .flat_map(|(ids, similarity)| {
            ids.split(';')
                .map(|id| {
                    let indices = id.split(",").collect_vec();
                    assert_eq!(indices.len(), 2);
                    let entry_id = indices[0].parse::<EntryId>().unwrap();
                    let def_index = indices[1].parse::<usize>().unwrap();

                    let entry = dict.get(&entry_id).unwrap();
                    let def = &entry.defs[def_index];
                    let eng: Clause = def.eng.as_ref().unwrap();

                    let eng = clause_to_string(&eng);
                    let matched_eng = vec![MatchedSegment {
                        segment: eng,
                        matched: true,
                    }];

                    EnglishSearchRank {
                        entry_id,
                        def_index,
                        score: (100.0 * similarity) as Score,
                        matched_eng,
                    }
                })
                .collect::<Vec<_>>()
        })
        .collect()
}

pub fn english_search(
    english_index: &dyn EnglishIndexLike,
    dict: &dyn RichDictLike,
    query: &str,
    script: Script,
) -> BinaryHeap<EnglishSearchRank> {
    let query = unicode::normalize_english_word_for_search_index(query);
    let results = english_index
        .get(query.as_str())
        .unwrap_or(fuzzy_english_search(english_index, &[query.clone()]));
    results
        .into_iter()
        .map(
            |EnglishIndexData {
                 entry_id,
                 def_index,
                 score,
             }| {
                let entry = dict.get_entry(entry_id);
                let def = &entry.defs[def_index as usize];
                let eng: Clause = def.eng.as_ref().unwrap().clone();
                let eng = clause_to_string(&eng);
                let matched_eng = match_eng_words(&eng, &query);
                let frequency_count = get_max_frequency_count(&dict.get_entry(entry_id).variants);
                EnglishSearchRank {
                    entry_id,
                    def_len: entry.defs.len(),
                    def_index,
                    variant: pick_variant(
                        dict.get_entry(entry_id).variants.0.first().unwrap(),
                        script,
                    )
                    .word,
                    score,
                    frequency_count,
                    matched_eng,
                }
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
