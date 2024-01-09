use crate::dict::{clause_to_string, Clause, EntryId};
use crate::jyutping::{parse_jyutpings, remove_yale_diacritics, LaxJyutPing};
use crate::lihkg_frequencies::LIHKG_FREQUENCIES;
use crate::pr_index::{FstPrIndices, PrLocation, MAX_DELETIONS};
use crate::rich_dict::{ArchivedRichDict, RichLine};

use super::charlist::CHARLIST;
use super::dict::{Variant, Variants};
use super::english_index::{ArchivedEnglishIndex, EnglishIndexData, EnglishSearchRank};
use super::iconic_simps::ICONIC_SIMPS;
use super::iconic_trads::ICONIC_TRADS;
use super::jyutping::{LaxJyutPings, Romanization};
use super::rich_dict::RichEntry;
use super::unicode;
use super::word_frequencies::WORD_FREQUENCIES;
use finalfusion::prelude::*;
use fst::automaton::Levenshtein;
use itertools::Itertools;
use rkyv::Deserialize;
use sif_embedding::{SentenceEmbedder, Sif};
use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet};
use std::collections::{BinaryHeap, HashSet};
use strsim::{generic_levenshtein, levenshtein, normalized_levenshtein};
use vtext::tokenize::{Tokenizer, VTextTokenizerParams};
use wordfreq::WordFreq;

/// Max score is 100
type Score = usize;

const MAX_SCORE: Score = 100;

type Index = usize;

#[derive(Copy, Clone, PartialEq, Eq)]
pub enum Script {
    Simplified,
    Traditional,
}

/// A Map from entry ID to variants and simplified variants
pub type VariantsMap = BTreeMap<EntryId, ComboVariants>;

#[derive(Debug, Clone, PartialEq)]
pub struct ComboVariant {
    pub word_trad: String,
    pub word_simp: String,
    pub prs: LaxJyutPings,
}

pub type ComboVariants = Vec<ComboVariant>;

pub fn rich_dict_to_variants_map(dict: &ArchivedRichDict) -> VariantsMap {
    dict.iter()
        .map(|(id, entry)| {
            (
                *id,
                create_combo_variants(
                    &entry.variants.deserialize(&mut rkyv::Infallible).unwrap(),
                    &entry
                        .variants_simp
                        .deserialize(&mut rkyv::Infallible)
                        .unwrap(),
                ),
            )
        })
        .collect()
}

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

#[derive(
    Clone,
    Eq,
    PartialEq,
    Debug,
    serde::Serialize,
    serde::Deserialize,
    rkyv::Archive,
    rkyv::Deserialize,
    rkyv::Serialize,
)]
pub struct MatchedSegment {
    pub segment: String,
    pub matched: bool,
}

#[derive(Clone, Eq, PartialEq, Debug)]
pub struct PrSearchRank {
    pub id: EntryId,
    pub variant_index: Index,
    pub pr_index: Index,
    pub jyutping: String,
    pub matched_pr: Vec<MatchedSegment>,
    pub num_matched_initial_chars: u32,
    pub num_matched_final_chars: u32,
    pub score: Score,
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
            .then(other.id.cmp(&self.id))
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

pub fn get_entry_id(variants_map: &VariantsMap, query: &str, script: Script) -> Option<EntryId> {
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

pub fn get_entry_group(
    variants_map: &VariantsMap,
    dict: &ArchivedRichDict,
    id: EntryId,
) -> Vec<RichEntry> {
    let query_word_set: HashSet<&str> = variants_map
        .get(&id)
        .unwrap()
        .iter()
        .map(|ComboVariant { word_trad, .. }| word_trad.as_str())
        .collect();
    sort_entry_group(
        dict.iter()
            .filter_map(|(id, entry)| {
                let current_word_set: HashSet<&str> = variants_map
                    .get(&id)
                    .unwrap()
                    .iter()
                    .map(|ComboVariant { word_trad, .. }| word_trad.as_str())
                    .collect();
                if query_word_set
                    .intersection(&current_word_set)
                    .next()
                    .is_some()
                {
                    Some(entry.deserialize(&mut rkyv::Infallible).unwrap())
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

fn get_entry_frequency(entry_id: EntryId) -> u8 {
    *WORD_FREQUENCIES.get(&entry_id).unwrap_or(&50)
}

pub fn pr_search(
    pr_indices: &FstPrIndices,
    dict: &ArchivedRichDict,
    query: &str,
    romanization: Romanization,
) -> BinaryHeap<PrSearchRank> {
    let mut ranks = BinaryHeap::new();
    let query = unicode::normalize(query);

    if query.is_empty() {
        return ranks;
    }

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
        dict: &ArchivedRichDict,
        romanization: Romanization,
        ranks: &mut BinaryHeap<PrSearchRank>,
        pr_variant_generator: fn(&str) -> String,
    ) {
        let max_deletions = (query.chars().count() - 1).min(MAX_DELETIONS);

        let lev = Levenshtein::new(query, max_deletions as u32).unwrap();

        for PrLocation {
            entry_id,
            variant_index,
            pr_index,
        } in search(lev)
        {
            let jyutping: LaxJyutPing = dict.get(&entry_id).unwrap().variants.0
                [variant_index as Index]
                .prs
                .0[pr_index as Index]
                .deserialize(&mut rkyv::Infallible)
                .unwrap();
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
                ranks.push(PrSearchRank {
                    id: entry_id,
                    variant_index: variant_index as Index,
                    pr_index: pr_index as Index,
                    jyutping,
                    matched_pr,
                    num_matched_initial_chars,
                    num_matched_final_chars,
                    score: MAX_SCORE - distance,
                });
            }
        }
    }

    match romanization {
        Romanization::Jyutping => {
            const TONES: [char; 6] = ['1', '2', '3', '4', '5', '6'];

            if query.contains(TONES) && query.contains(' ') {
                lookup_index(
                    &query,
                    pr_indices.tone_and_space(),
                    dict,
                    romanization,
                    &mut ranks,
                    |s| s.to_string(),
                );
            } else if query.contains(TONES) {
                lookup_index(
                    &query,
                    pr_indices.tone(),
                    dict,
                    romanization,
                    &mut ranks,
                    |s| s.replace(' ', ""),
                );
            } else if query.contains(' ') {
                lookup_index(
                    &query,
                    pr_indices.space(),
                    dict,
                    romanization,
                    &mut ranks,
                    |s| s.replace(TONES, ""),
                );
            } else {
                lookup_index(
                    &query,
                    pr_indices.none(),
                    dict,
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
                    pr_indices.tone_and_space(),
                    dict,
                    romanization,
                    &mut ranks,
                    to_yale,
                );
            } else if has_tone {
                lookup_index(
                    &query,
                    pr_indices.tone(),
                    dict,
                    romanization,
                    &mut ranks,
                    |s| to_yale(s).replace(' ', ""),
                );
            } else if query.contains(' ') {
                lookup_index(
                    &query,
                    pr_indices.tone_and_space(),
                    dict,
                    romanization,
                    &mut ranks,
                    to_yale,
                );

                lookup_index(
                    &query,
                    pr_indices.space(),
                    dict,
                    romanization,
                    &mut ranks,
                    to_yale_no_tones,
                );
            } else {
                lookup_index(
                    &query,
                    pr_indices.tone(),
                    dict,
                    romanization,
                    &mut ranks,
                    |s| to_yale(s).replace(' ', ""),
                );

                lookup_index(
                    &query,
                    pr_indices.none(),
                    dict,
                    romanization,
                    &mut ranks,
                    |s| to_yale_no_tones(s).replace(' ', ""),
                );
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

#[derive(Debug, Clone, Eq, PartialEq, Default)]
pub struct MatchedInfix {
    pub prefix: String,
    pub query: String,
    pub suffix: String,
}

#[derive(Clone, Eq, PartialEq)]
pub struct VariantSearchRank {
    pub id: EntryId,
    pub variant_index: Index,
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
            .then_with(|| other.length_diff.cmp(&self.length_diff))
            .then_with(|| self.frequency_count.cmp(&other.frequency_count))
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
    variant: &ComboVariant,
    query_normalized: &str,
    query_script: Script,
) -> (Index, Score, MatchedInfix) {
    let variant_normalized = &unicode::normalize(&pick_variant(variant, query_script).word)[..];
    // The query has to be fully contained by the variant
    let occurrence_index = match variant_normalized.find(query_normalized) {
        Some(i) => i,
        None => return (usize::MAX, usize::MAX, MatchedInfix::default()),
    };
    let length_diff = variant_normalized.chars().count() - query_normalized.chars().count();
    let matched_variant = MatchedInfix {
        prefix: variant_normalized[..occurrence_index].to_string(),
        query: query_normalized.to_string(),
        suffix: variant_normalized[occurrence_index + query_normalized.len()..].to_string(),
    };
    (occurrence_index, length_diff, matched_variant)
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
        let frequency_count = *variants
            .iter()
            .max_by(|variant1, variant2| {
                LIHKG_FREQUENCIES
                    .get(&variant1.word_trad)
                    .unwrap_or(&0)
                    .cmp(LIHKG_FREQUENCIES.get(&variant2.word_trad).unwrap_or(&0))
            })
            .map(|most_frequent_variant| {
                LIHKG_FREQUENCIES
                    .get(&most_frequent_variant.word_trad)
                    .unwrap_or(&0)
            })
            .unwrap_or(&0);
        variants
            .iter()
            .enumerate()
            .for_each(|(variant_index, variant)| {
                let (occurrence_index, length_diff, matched_variant) =
                    score_variant_query(variant, &query_normalized, query_script);
                if occurrence_index < usize::MAX && length_diff <= 2 {
                    ranks.push(VariantSearchRank {
                        id: *id,
                        variant_index,
                        occurrence_index,
                        length_diff,
                        matched_variant,
                        frequency_count,
                    });
                }
            });
    });
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
            .then(other.id.cmp(&self.id))
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
    variants_map: &VariantsMap,
    dict: &ArchivedRichDict,
    query: &str,
    max_first_index_in_eg: usize,
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

    use std::sync::Arc;
    use std::sync::Mutex;

    let ranks: Arc<Mutex<BinaryHeap<_>>> = Arc::new(Mutex::new(BinaryHeap::new()));

    use rayon::iter::IntoParallelRefIterator;
    use rayon::iter::ParallelIterator;

    variants_map.par_iter().for_each(|(entry_id, _)| {
        let entry = dict.get(entry_id).unwrap();
        for (def_index, def) in entry.defs.iter().enumerate() {
            for (eg_index, eg) in def.egs.iter().enumerate() {
                let get_line_in_script = |script| match script {
                    Script::Traditional => eg.yue.as_ref().map(|line| {
                        let line: RichLine = line.deserialize(&mut rkyv::Infallible).unwrap();
                        line.to_string()
                    }),
                    Script::Simplified => eg.yue_simp.deserialize(&mut rkyv::Infallible).unwrap(),
                };
                let line: Option<String> = get_line_in_script(query_script);
                if let Some(line) = line {
                    let line_len = line.chars().count();
                    if let Some(first_index) = line.find(&query_normalized) {
                        let char_index = line[..first_index].chars().count();
                        if char_index <= max_first_index_in_eg {
                            let line = if query_script == script {
                                line
                            } else {
                                get_line_in_script(script).unwrap()
                            };
                            let matched_eg = MatchedInfix {
                                prefix: line[..first_index].to_string(),
                                query: line[first_index..first_index + query_normalized.len()]
                                    .to_string(),
                                suffix: line[first_index + query_normalized.len()..].to_string(),
                            };
                            ranks.lock().unwrap().push(EgSearchRank {
                                id: *entry_id,
                                def_index,
                                eg_index,
                                eg_length: line_len,
                                matched_eg,
                            });
                        }
                    }
                }
            }
        }
    });

    Arc::try_unwrap(ranks).unwrap().into_inner().unwrap()
}

pub enum CombinedSearchRank {
    Variant(BinaryHeap<VariantSearchRank>),
    Pr(BinaryHeap<PrSearchRank>),
    All(
        BinaryHeap<VariantSearchRank>,
        BinaryHeap<PrSearchRank>,
        Vec<EnglishSearchRank>,
    ),
}

// Auto recognize the type of the query
pub fn combined_search(
    variants_map: &VariantsMap,
    pr_indices: Option<&FstPrIndices>,
    english_index: &ArchivedEnglishIndex,
    english_embeddings: &Embeddings<VocabWrap, StorageWrap>,
    sif_model: &Sif<VocabWrap, StorageWrap>,
    dict: &ArchivedRichDict,
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
    let pr_ranks = if let Some(pr_indices) = pr_indices {
        pr_search(pr_indices, dict, query, romanization)
    } else {
        BinaryHeap::new()
    };
    let english_results = english_search(english_index, dict, query);

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

pub fn english_embedding_search(
    english_embeddings: &Embeddings<VocabWrap, StorageWrap>,
    sif_model: &Sif<Embeddings<VocabWrap, StorageWrap>, WordFreq>,
    dict: &ArchivedRichDict,
    query: &str,
) -> Vec<EnglishSearchRank> {
    let tokenizer = VTextTokenizerParams::default().lang("en").build().unwrap();
    let separator = sif_embedding::DEFAULT_SEPARATOR.to_string();
    let query_tokenized = tokenizer
        .tokenize(&unicode::normalize_english_word_for_embedding(query))
        .collect::<Vec<_>>()
        .join(&separator)
        .to_lowercase();

    let query_embeddings = sif_model.embeddings([query_tokenized]).unwrap();
    let query_embedding = query_embeddings.row(0);

    let query_embedding_norm: f32 = query_embedding.dot(&query_embedding).sqrt();
    let query_embedding_normalized = query_embedding.mapv(|x| x / query_embedding_norm);

    let mut ranks: Vec<(String, f32)> = vec![];

    for (id, v) in english_embeddings {
        let cosine_similarity = v.dot(&query_embedding_normalized);
        if cosine_similarity > 0.5 {
            ranks.push((id.to_string(), cosine_similarity));
        }
    }

    ranks.sort_by(|(_, s1), (_, s2)| s2.total_cmp(s1));

    ranks
        .into_iter()
        .take(50)
        .map(|(id, similarity)| {
            let indices = id.split(",").collect_vec();
            assert_eq!(indices.len(), 3);
            let entry_id = indices[0].parse::<EntryId>().unwrap();
            let def_index = indices[1].parse::<usize>().unwrap();
            let phrase_index = indices[2].parse::<usize>().unwrap();

            let entry = dict.get(&entry_id).unwrap();
            let def = &entry.defs[def_index];
            let eng: Clause = def
                .eng
                .as_ref()
                .unwrap()
                .deserialize(&mut rkyv::Infallible)
                .unwrap();

            let eng = clause_to_string(&eng);
            let mut matched_eng = eng
                .split(';')
                .enumerate()
                .flat_map(|(current_phrase_index, current_phrase)| {
                    vec![
                        MatchedSegment {
                            segment: current_phrase.to_string(),
                            matched: current_phrase_index == phrase_index,
                        },
                        MatchedSegment {
                            segment: ";".to_string(),
                            matched: false,
                        },
                    ]
                })
                .collect_vec();
            // Remove extra ";" at the end
            matched_eng.pop();

            EnglishSearchRank {
                entry_id,
                def_index,
                score: (100.0 * similarity) as Score,
                matched_eng,
            }
        })
        .collect()
}

pub fn english_search(
    english_index: &ArchivedEnglishIndex,
    dict: &ArchivedRichDict,
    query: &str,
) -> Vec<EnglishSearchRank> {
    let query = unicode::normalize_english_word_for_search_index(query);
    let results = english_index
        .get(query.as_str())
        .map(|results| results.deserialize(&mut rkyv::Infallible).unwrap())
        .unwrap_or(fuzzy_english_search(english_index, &[query.clone()]));
    results
        .into_iter()
        .map(
            |EnglishIndexData {
                 entry_id,
                 def_index,
                 score,
             }| {
                let entry = dict.get(&entry_id).unwrap();
                let def = &entry.defs[def_index as usize];
                let eng: Clause = def
                    .eng
                    .as_ref()
                    .unwrap()
                    .deserialize(&mut rkyv::Infallible)
                    .unwrap();
                let eng = clause_to_string(&eng);
                let matched_eng = match_eng_words(&eng, &query);
                EnglishSearchRank {
                    entry_id,
                    def_index,
                    score,
                    matched_eng,
                }
            },
        )
        .collect()
}

fn fuzzy_english_search<'a>(
    english_index: &'a ArchivedEnglishIndex,
    queries: &[String],
) -> Vec<EnglishIndexData> {
    english_index
        .iter()
        .fold(
            (60, None), // must have a score of at least 60 out of 100
            |(max_score, max_entries), (phrase, entries)| {
                let (mut next_max_score, mut next_max_entries) = (max_score, max_entries);
                queries.iter().for_each(|query| {
                    let current_score = score_english_query(query, phrase);
                    if current_score > max_score {
                        (next_max_score, next_max_entries) = (current_score, Some(entries))
                    }
                });
                (next_max_score, next_max_entries)
            },
        )
        .1
        .map(|results| results.deserialize(&mut rkyv::Infallible).unwrap())
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
