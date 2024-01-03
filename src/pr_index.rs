use crate::{
    dict::EntryId,
    jyutping::{remove_yale_tones, LaxJyutPing, Romanization},
    rich_dict::ArchivedRichDict,
};
use fst::{automaton::Levenshtein, IntoStreamer, Map, MapBuilder};
use rkyv::Deserialize as _;
use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeMap, BTreeSet, HashMap},
    hash::Hash,
};

pub const MAX_DELETIONS: usize = 1;

#[derive(Serialize, Deserialize, Default)]
pub struct PrIndices {
    pub tone_and_space: PrIndex,
    pub tone: PrIndex,
    pub space: PrIndex,
    pub none: PrIndex,
}

impl PrIndices {
    pub fn default() -> Self {
        Self {
            tone_and_space: BTreeMap::default(),
            tone: BTreeMap::default(),
            space: BTreeMap::default(),
            none: BTreeMap::default(),
        }
    }
}

pub struct FstPrIndices {
    tone_and_space: Map<Vec<u8>>,
    tone: Map<Vec<u8>>,
    space: Map<Vec<u8>>,
    none: Map<Vec<u8>>,

    locations: HashMap<u64, BTreeSet<PrLocation>>,
}

impl FstPrIndices {
    pub fn tone_and_space<'a>(&'a self) -> impl FnOnce(Levenshtein) -> BTreeSet<PrLocation> + 'a {
        |query| self.search(&self.tone_and_space, query)
    }

    pub fn tone<'a>(&'a self) -> impl FnOnce(Levenshtein) -> BTreeSet<PrLocation> + 'a {
        |query| self.search(&self.tone, query)
    }

    pub fn space<'a>(&'a self) -> impl FnOnce(Levenshtein) -> BTreeSet<PrLocation> + 'a {
        |query| self.search(&self.space, query)
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

    if pr.contains(' ') {
        insert(&mut index.tone, pr.replace(' ', ""));
        insert(&mut index.space, remove_tones(&pr));
        insert(&mut index.none, remove_tones(&pr).replace(' ', ""));
        insert(&mut index.tone_and_space, pr);
    } else {
        insert(&mut index.none, remove_tones(&pr));
        insert(&mut index.tone, pr);
    }
}

pub fn generate_pr_indices(dict: &ArchivedRichDict, romanization: Romanization) -> PrIndices {
    let mut indices = PrIndices::default();
    for (&entry_id, entry) in dict.iter() {
        for (variant_index, variant) in entry.variants.0.iter().enumerate() {
            for (pr_index, pr) in variant.prs.0.iter().enumerate() {
                let pr: LaxJyutPing = pr.deserialize(&mut rkyv::Infallible).unwrap();
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
    let mut tone_and_space_builder = MapBuilder::memory();
    let mut tone_builder = MapBuilder::memory();
    let mut space_builder = MapBuilder::memory();
    let mut none_builder = MapBuilder::memory();

    let mut locations = HashMap::new();
    let mut location_index = 0;

    for (pr, locs) in indices.tone_and_space {
        let loc_index = locations.entry(locs).or_insert({
            location_index += 1;
            location_index - 1
        });
        tone_and_space_builder.insert(pr, *loc_index).unwrap();
    }
    for (pr, locs) in indices.tone {
        let loc_index = locations.entry(locs).or_insert({
            location_index += 1;
            location_index - 1
        });
        tone_builder.insert(pr, *loc_index).unwrap();
    }
    for (pr, locs) in indices.space {
        let loc_index = locations.entry(locs).or_insert({
            location_index += 1;
            location_index - 1
        });
        space_builder.insert(pr, *loc_index).unwrap();
    }
    for (pr, locs) in indices.none {
        let loc_index = locations.entry(locs).or_insert({
            location_index += 1;
            location_index - 1
        });
        none_builder.insert(pr, *loc_index).unwrap();
    }

    let tone_and_space = tone_and_space_builder.into_map();
    let tone = tone_builder.into_map();
    let space = space_builder.into_map();
    let none = none_builder.into_map();

    FstPrIndices {
        tone_and_space,
        tone,
        space,
        none,
        locations: locations.into_iter().map(|(k, v)| (v, k)).collect(),
    }
}
