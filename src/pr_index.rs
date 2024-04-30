use crate::{
    dict::EntryId,
    jyutping::{remove_yale_tones, Romanization},
    search::RichDictLike,
};
use fst::{automaton::Levenshtein, IntoStreamer, Map, MapBuilder};
use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeMap, BTreeSet, HashMap},
    hash::Hash,
};

pub const MAX_DELETIONS: usize = 1;

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
    tone: Map<Vec<u8>>,
    none: Map<Vec<u8>>,

    locations: HashMap<u64, BTreeSet<PrLocation>>,
}

impl FstPrIndices {
    pub fn tone<'a>(&'a self) -> impl FnOnce(Levenshtein) -> BTreeSet<PrLocation> + 'a {
        |query| self.search(&self.tone, query)
    }

    pub fn none<'a>(&'a self) -> impl FnOnce(Levenshtein) -> BTreeSet<PrLocation> + 'a {
        |query| self.search(&self.none, query)
    }

    fn search(&self, map: &Map<Vec<u8>>, query: Levenshtein) -> BTreeSet<PrLocation> {
        map.search(query)
            .into_stream()
            .into_values()
            .into_iter()
            .flat_map(|loc| &self.locations[&loc])
            .cloned()
            .collect()
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
