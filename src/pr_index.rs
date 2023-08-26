use crate::{jyutping::Romanization, rich_dict::RichDict};
use kdam::tqdm;
use serde::{Deserialize, Serialize};
use std::{
    collections::{HashMap, HashSet},
    hash::{Hash, Hasher},
};

const MAX_DELETIONS: usize = 2;
const MAX_CANDIDATES: usize = 10;

#[derive(Serialize, Deserialize)]
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
    pub variant_index: usize,

    #[serde(rename = "p")]
    pub pr_index: usize,
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

// map from (deleted) pr to entry ids
pub type PrIndex = HashMap<String, HashSet<PrLocation>>;

fn generate_deletion_index(s: &str) -> HashMap<String, usize> {
    let mut results = HashMap::new();
    let max_deletions = ((s.len() as f32 * 0.3) as usize).min(MAX_DELETIONS);

    fn recursive_generate(
        input: &str,
        deletions: usize,
        max_deletions: usize,
        output: &mut HashMap<String, usize>,
    ) {
        if deletions >= max_deletions {
            return;
        }

        for (index, _) in input.char_indices() {
            let mut sub_str = input.to_string();
            sub_str.remove(index);
            if !sub_str.is_empty() && !output.contains_key(&sub_str) {
                output
                    .entry(sub_str.clone())
                    .and_modify(|old_deletions| {
                        *old_deletions = (*old_deletions).min(deletions + 1)
                    })
                    .or_insert(deletions + 1);
                recursive_generate(&sub_str, deletions + 1, max_deletions, output);
            }
        }
    }

    results.insert(s.to_string(), 0);
    recursive_generate(s, 0, max_deletions, &mut results);
    results
}

fn generate_pr_variants(pr_location: PrLocation, pr: String, index: &mut PrIndices) {
    let tones = &['1', '2', '3', '4', '5', '6'];

    if pr.contains(' ') {
        for (variant, deletions) in generate_deletion_index(&pr) {
            index.tone_and_space[deletions]
                .entry(variant)
                .and_modify(|locations| {
                    if locations.len() < MAX_CANDIDATES {
                        locations.insert(pr_location);
                    }
                })
                .or_insert(HashSet::new());
        }

        for (variant, deletions) in generate_deletion_index(&pr.replace(' ', "")) {
            index.tone[deletions]
                .entry(variant)
                .and_modify(|locations| {
                    if locations.len() < MAX_CANDIDATES {
                        locations.insert(pr_location);
                    }
                })
                .or_insert(HashSet::new());
        }

        for (variant, deletions) in generate_deletion_index(&pr.replace(' ', "").replace(tones, ""))
        {
            index.none[deletions]
                .entry(variant)
                .and_modify(|locations| {
                    if locations.len() < MAX_CANDIDATES {
                        locations.insert(pr_location);
                    }
                })
                .or_insert(HashSet::new());
        }
    } else {
        for (variant, deletions) in generate_deletion_index(&pr) {
            index.tone[deletions]
                .entry(variant)
                .and_modify(|locations| {
                    if locations.len() < MAX_CANDIDATES {
                        locations.insert(pr_location);
                    }
                })
                .or_insert(HashSet::new());
        }

        for (variant, deletions) in generate_deletion_index(&pr.replace(tones, "")) {
            index.none[deletions]
                .entry(variant)
                .and_modify(|locations| {
                    if locations.len() < MAX_CANDIDATES {
                        locations.insert(pr_location);
                    }
                })
                .or_insert(HashSet::new());
        }
    }
}

pub fn generate_pr_indices(dict: &RichDict, romanization: Romanization) -> PrIndices {
    let mut index = PrIndices::default();
    for (&entry_id, entry) in tqdm!(dict.iter(), desc = "Building pr index") {
        for (variant_index, variant) in entry.variants.0.iter().enumerate() {
            for (pr_index, pr) in variant.prs.0.iter().enumerate() {
                // only add standard jyutping to pr index
                if pr.is_standard_jyutping() {
                    generate_pr_variants(
                        PrLocation {
                            entry_id,
                            variant_index,
                            pr_index,
                        },
                        match romanization {
                            Romanization::Jyutping => pr.to_string(),
                            Romanization::Yale => pr.to_yale_no_diacritics(),
                        },
                        &mut index,
                    );
                }
            }
        }
    }
    index
}
