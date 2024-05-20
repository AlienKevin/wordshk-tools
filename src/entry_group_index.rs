use std::collections::HashSet;

use rayon::iter::{IntoParallelRefIterator, ParallelIterator};

use crate::{
    dict::EntryId,
    rich_dict::{RichDict, RichEntry},
    search::{get_entry_frequency, RichDictLike},
};

pub trait EntryGroupIndex {
    fn get_entry_group(&self, id: EntryId) -> Vec<EntryId>;
}

pub fn get_entry_group<D>(dict: &D, id: EntryId) -> Vec<RichEntry>
where
    D: RichDictLike + EntryGroupIndex,
{
    dict.get_entry_group(id)
        .iter()
        .map(|id| dict.get_entry(*id))
        .collect()
}

pub fn get_entry_group_ids(dict: &RichDict, id: EntryId) -> Vec<EntryId> {
    let entry = dict.get_entry(id);
    let query_word_set: HashSet<&str> = entry.variants.to_words_set();
    sort_entry_group_ids(
        dict,
        dict.par_iter()
            .filter_map(|(id, entry)| {
                let current_word_set: HashSet<&str> = entry.variants.to_words_set();
                if query_word_set
                    .intersection(&current_word_set)
                    .next()
                    .is_some()
                {
                    Some(*id)
                } else {
                    None
                }
            })
            .collect(),
    )
}

fn sort_entry_group_ids(dict: &dyn RichDictLike, mut entry_group: Vec<EntryId>) -> Vec<EntryId> {
    entry_group.sort_by(|a, b| {
        get_entry_frequency(*a)
            .cmp(&get_entry_frequency(*b))
            .reverse()
            .then(
                dict.get_entry(*a)
                    .defs
                    .len()
                    .cmp(&dict.get_entry(*b).defs.len())
                    .reverse(),
            )
            .then(a.cmp(b))
    });
    entry_group
}
