use crate::{
    dict::EntryId,
    jyutping::{remove_yale_tones, Romanization},
    search::RichDictLike,
    sqlite_db::SqliteDb,
};
use fst::{
    automaton::{Automaton, Levenshtein, Str},
    IntoStreamer, Map, MapBuilder,
};
use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeMap, BTreeSet, HashMap},
    hash::Hash,
};

pub const MAX_DELETIONS: usize = 2;

#[derive(Serialize, Deserialize, Default)]
pub struct PrIndices {
    pub tone: PrIndex,
    pub none: PrIndex,
}

impl PrIndices {
    pub fn default() -> Self {
        Self {
            tone: BTreeMap::default(),
            none: BTreeMap::default(),
        }
    }
}

pub struct FstPrIndices {
    pub tone: Map<Vec<u8>>,
    pub none: Map<Vec<u8>>,

    pub locations: HashMap<u64, BTreeSet<PrLocation>>,
}

pub enum FstSearchResult {
    Levenshtein(BTreeSet<PrLocation>),
    Prefix(BTreeSet<PrLocation>),
}

pub trait FstPrIndicesLike {
    fn search(
        &self,
        has_tone: bool,
        query_no_space: &str,
        romanization: Romanization,
    ) -> FstSearchResult;
}

impl FstPrIndicesLike for SqliteDb {
    fn search(&self, has_tone: bool, query: &str, romanization: Romanization) -> FstSearchResult {
        let conn = self.conn();
        let mut stmt = conn
            .prepare(&format!(
                "SELECT fst FROM pr_index_fsts_{romanization} WHERE name = ?"
            ))
            .unwrap();
        let fst_bytes: Vec<u8> = stmt
            .query_row([if has_tone { "tone" } else { "none" }], |row| row.get(0))
            .ok()
            .unwrap();
        let fst = Map::new(fst_bytes).unwrap();

        let mut stmt = conn
            .prepare(&format!(
                "SELECT locations FROM pr_index_locations_{romanization} WHERE id = ?"
            ))
            .unwrap();

        let query_no_space = query.replace(' ', "");

        let prefix_automaton = Str::new(&query_no_space).starts_with();

        let max_deletions = (query_no_space.chars().count() - 1).min(MAX_DELETIONS);
        let levenshtein_automaton =
            Levenshtein::new(&query_no_space, max_deletions as u32).unwrap();

        let prefix_results: BTreeSet<PrLocation> = fst
            .search(prefix_automaton)
            .into_stream()
            .into_values()
            .into_iter()
            .flat_map(|loc| -> BTreeSet<PrLocation> {
                let locations_text: String = stmt.query_row([loc], |row| row.get(0)).unwrap();
                serde_json::from_str(&locations_text).unwrap()
            })
            .collect();

        if prefix_results.is_empty() {
            FstSearchResult::Levenshtein(
                fst.search(levenshtein_automaton)
                    .into_stream()
                    .into_values()
                    .into_iter()
                    .flat_map(|loc| -> BTreeSet<PrLocation> {
                        let locations_text: String =
                            stmt.query_row([loc], |row| row.get(0)).unwrap();
                        serde_json::from_str(&locations_text).unwrap()
                    })
                    .collect(),
            )
        } else {
            FstSearchResult::Prefix(prefix_results)
        }
    }
}

impl FstPrIndicesLike for FstPrIndices {
    fn search(&self, has_tone: bool, query: &str, romanization: Romanization) -> FstSearchResult {
        let fst = if has_tone { &self.tone } else { &self.none };

        let query_no_space = query.replace(' ', "");

        let prefix_automaton = Str::new(&query_no_space).starts_with();

        let max_deletions = (query_no_space.chars().count() - 1).min(MAX_DELETIONS);
        let levenshtein_automaton =
            Levenshtein::new(&query_no_space, max_deletions as u32).unwrap();

        let prefix_results: BTreeSet<PrLocation> = fst
            .search(prefix_automaton)
            .into_stream()
            .into_values()
            .into_iter()
            .flat_map(|loc| &self.locations[&loc])
            .cloned()
            .collect();

        if prefix_results.is_empty() {
            FstSearchResult::Levenshtein(
                fst.search(levenshtein_automaton)
                    .into_stream()
                    .into_values()
                    .into_iter()
                    .flat_map(|loc| &self.locations[&loc])
                    .cloned()
                    .collect(),
            )
        } else {
            FstSearchResult::Prefix(prefix_results)
        }
    }
}

impl FstPrIndices {
    pub fn fst_size_in_bytes(&self) -> usize {
        self.tone.as_fst().as_bytes().len() + self.none.as_fst().as_bytes().len()
    }

    pub fn locations_size_in_bytes(&self) -> usize {
        self.locations.iter().fold(0, |acc, (k, v)| {
            acc + std::mem::size_of_val(k) + v.len() * std::mem::size_of::<PrLocation>()
        })
    }
}

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct PrLocation {
    #[serde(rename = "e")]
    pub entry_id: EntryId,

    #[serde(rename = "v")]
    pub variant_index: u8,

    #[serde(rename = "p")]
    pub pr_index: u8,
}

pub type PrIndex = BTreeMap<String, BTreeSet<PrLocation>>;

fn generate_pr_variants(
    pr_location: PrLocation,
    pr: String,
    index: &mut PrIndices,
    remove_tones: fn(&str) -> String,
) {
    let insert = |index: &mut PrIndex, variant: String| {
        index
            .entry(variant.clone())
            .and_modify(|locations| {
                locations.insert(pr_location);
            })
            .or_insert(BTreeSet::from_iter([pr_location]));
    };

    insert(&mut index.tone, pr.replace(' ', ""));
    insert(&mut index.none, remove_tones(&pr).replace(' ', ""))
}

pub fn generate_pr_indices(dict: &dyn RichDictLike, romanization: Romanization) -> PrIndices {
    let mut indices = PrIndices::default();
    for entry_id in dict.get_ids() {
        let entry = dict.get_entry(entry_id);
        for (variant_index, variant) in entry.variants.0.iter().enumerate() {
            for (pr_index, pr) in variant.prs.0.iter().enumerate() {
                // only add standard jyutping to pr index
                if pr.is_standard_jyutping() {
                    match romanization {
                        Romanization::Jyutping => generate_pr_variants(
                            PrLocation {
                                entry_id: entry_id,
                                variant_index: variant_index.try_into().unwrap(),
                                pr_index: pr_index.try_into().unwrap(),
                            },
                            pr.to_string(),
                            &mut indices,
                            |s| s.replace(&['1', '2', '3', '4', '5', '6'], ""),
                        ),
                        Romanization::Yale => generate_pr_variants(
                            PrLocation {
                                entry_id: entry_id,
                                variant_index: variant_index.try_into().unwrap(),
                                pr_index: pr_index.try_into().unwrap(),
                            },
                            pr.to_yale(),
                            &mut indices,
                            remove_yale_tones,
                        ),
                    }
                }
            }
        }
    }
    indices
}

pub fn pr_indices_into_fst(indices: PrIndices) -> FstPrIndices {
    let mut tone_builder = MapBuilder::memory();
    let mut none_builder = MapBuilder::memory();

    let mut locations = HashMap::new();
    let mut location_index = 0;

    for (pr, locs) in indices.tone {
        let loc_index = locations.entry(locs).or_insert({
            location_index += 1;
            location_index - 1
        });
        tone_builder.insert(pr, *loc_index).unwrap();
    }

    for (pr, locs) in indices.none {
        let loc_index = locations.entry(locs).or_insert({
            location_index += 1;
            location_index - 1
        });
        none_builder.insert(pr, *loc_index).unwrap();
    }

    let tone = tone_builder.into_map();
    let none = none_builder.into_map();

    FstPrIndices {
        tone,
        none,
        locations: locations.into_iter().map(|(k, v)| (v, k)).collect(),
    }
}
