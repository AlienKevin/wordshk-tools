use crate::{jyutping::Romanization, rich_dict::RichDict};
use kdam::tqdm;
use serde::{Deserialize, Serialize};
use std::{
    collections::{HashMap, HashSet},
    hash::{Hash, Hasher},
};
use xxhash_rust::xxh3::xxh3_64;

const MAX_DELETIONS: usize = 2;
const MAX_CANDIDATES: usize = 10;

#[derive(Serialize, Deserialize, Default)]
pub struct PrIndices {
    pub tone_and_space: Vec<PrIndex>,
    pub tone: Vec<PrIndex>,
    pub space: Vec<PrIndex>,
    pub none: Vec<PrIndex>,
}

impl PrIndices {
    pub fn default() -> Self {
        Self {
            tone_and_space: vec![HashMap::default(); MAX_DELETIONS + 1],
            tone: vec![HashMap::default(); MAX_DELETIONS + 1],
            space: vec![HashMap::default(); MAX_DELETIONS + 1],
            none: vec![HashMap::default(); MAX_DELETIONS + 1],
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct PrLocation {
    #[serde(rename = "e")]
    pub entry_id: usize,

    #[serde(rename = "v")]
    pub variant_index: u8,

    #[serde(rename = "p")]
    pub pr_index: u8,
}

impl PartialEq for PrLocation {
    fn eq(&self, other: &Self) -> bool {
        self.entry_id == other.entry_id
    }
}

impl Eq for PrLocation {}

impl Hash for PrLocation {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.entry_id.hash(state);
    }
}

// map from hash of permuted pr to entry ids
pub type PrIndex = HashMap<u64, HashSet<PrLocation>>;

/// Generate a deletion index for a string
/// ```
/// use wordshk_tools::pr_index::generate_deletion_neighborhood;
/// use std::collections::HashMap;
///
/// assert_eq!(generate_deletion_neighborhood("abc", 1), HashMap::from_iter([
///     ("abc".to_string(), 0),
///
///     ("bc".to_string(), 1),
///     ("ac".to_string(), 1),
///     ("ab".to_string(), 1),
/// ]));
///
/// assert_eq!(generate_deletion_neighborhood("abcdef", 2), HashMap::from_iter([
///     ("abcdef".to_string(), 0),
///
///     ("bcdef".to_string(), 1),
///     ("acdef".to_string(), 1),
///     ("abdef".to_string(), 1),
///     ("abcef".to_string(), 1),
///     ("abcdf".to_string(), 1),
///     ("abcde".to_string(), 1),
///
///     ("cdef".to_string(), 2),
///     ("bdef".to_string(), 2),
///     ("bcef".to_string(), 2),
///     ("bcdf".to_string(), 2),
///     ("bcde".to_string(), 2),
///     
///     ("cdef".to_string(), 2),
///     ("adef".to_string(), 2),
///     ("acef".to_string(), 2),
///     ("acdf".to_string(), 2),
///     ("acde".to_string(), 2),
///
///     ("bdef".to_string(), 2),
///     ("adef".to_string(), 2),
///     ("abef".to_string(), 2),
///     ("abdf".to_string(), 2),
///     ("abde".to_string(), 2),
///
///     ("bcef".to_string(), 2),
///     ("acef".to_string(), 2),
///     ("abef".to_string(), 2),
///     ("abcf".to_string(), 2),
///     ("abce".to_string(), 2),
///
///     ("bcdf".to_string(), 2),
///     ("acdf".to_string(), 2),
///     ("abdf".to_string(), 2),
///     ("abcf".to_string(), 2),
///     ("abcd".to_string(), 2),
///
///     ("bcde".to_string(), 2),
///     ("acde".to_string(), 2),
///     ("abde".to_string(), 2),
///     ("abce".to_string(), 2),
///     ("abcd".to_string(), 2),
/// ]));
/// ```
pub fn generate_deletion_neighborhood(s: &str, max_deletions: usize) -> HashMap<String, usize> {
    let mut results: HashMap<String, usize> = HashMap::new();

    fn recursive_generate(
        input: &str,
        deletions: usize,
        max_deletions: usize,
        results: &mut HashMap<String, usize>,
    ) {
        if deletions >= max_deletions {
            return;
        }

        for (index, _) in input.char_indices() {
            let mut sub_str = input.to_string();
            sub_str.remove(index);
            results
                .entry(sub_str.clone())
                .and_modify(|old_deletions| *old_deletions = (*old_deletions).min(deletions + 1))
                .or_insert(deletions + 1);
            recursive_generate(&sub_str, deletions + 1, max_deletions, results);
        }
    }

    results.insert(s.to_string(), 0);
    recursive_generate(s, 0, max_deletions, &mut results);
    results
}

fn generate_deletion_index(s: &str) -> HashMap<String, usize> {
    let max_deletions = ((s.chars().count() as f32 * 0.35) as usize).min(MAX_DELETIONS);
    generate_deletion_neighborhood(s, max_deletions)
}

fn generate_jyutping_variants(pr_location: PrLocation, pr: String, index: &mut PrIndices) {
    let tones = &['1', '2', '3', '4', '5', '6'];

    let insert = |index: &mut Vec<PrIndex>, variant: String, deletions: usize| {
        index[deletions]
            .entry(xxh3_64(variant.as_bytes()))
            .and_modify(|locations| {
                if locations.len() < MAX_CANDIDATES {
                    locations.insert(pr_location);
                }
            })
            .or_insert(HashSet::new());
    };

    if pr.contains(' ') {
        for (variant, deletions) in generate_deletion_index(&pr) {
            insert(&mut index.tone_and_space, variant, deletions);
        }

        for (variant, deletions) in generate_deletion_index(&pr.replace(' ', "")) {
            insert(&mut index.tone, variant, deletions);
        }

        for (variant, deletions) in generate_deletion_index(&pr.replace(tones, "")) {
            insert(&mut index.space, variant, deletions);
        }

        for (variant, deletions) in generate_deletion_index(&pr.replace(' ', "").replace(tones, ""))
        {
            insert(&mut index.none, variant, deletions);
        }
    } else {
        for (variant, deletions) in generate_deletion_index(&pr) {
            insert(&mut index.tone, variant, deletions);
        }

        for (variant, deletions) in generate_deletion_index(&pr.replace(tones, "")) {
            insert(&mut index.none, variant, deletions);
        }
    }
}

fn generate_yale_variants(pr_location: PrLocation, pr: String, index: &mut PrIndices) {
    todo!("add support for yale")
}

pub fn generate_pr_indices(dict: &RichDict, romanization: Romanization) -> PrIndices {
    let mut index = PrIndices::default();
    for (&entry_id, entry) in tqdm!(dict.iter(), desc = "Building pr index") {
        for (variant_index, variant) in entry.variants.0.iter().enumerate() {
            for (pr_index, pr) in variant.prs.0.iter().enumerate() {
                // only add standard jyutping to pr index
                if pr.is_standard_jyutping() {
                    match romanization {
                        Romanization::Jyutping => generate_jyutping_variants(
                            PrLocation {
                                entry_id,
                                variant_index: variant_index.try_into().unwrap(),
                                pr_index: pr_index.try_into().unwrap(),
                            },
                            pr.to_string(),
                            &mut index,
                        ),
                        Romanization::Yale => generate_yale_variants(
                            PrLocation {
                                entry_id,
                                variant_index: variant_index.try_into().unwrap(),
                                pr_index: pr_index.try_into().unwrap(),
                            },
                            pr.to_string(),
                            &mut index,
                        ),
                    }
                }
            }
        }
    }
    index
}
