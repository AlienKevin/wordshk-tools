use std::collections::HashSet;

use rayon::iter::{IntoParallelRefIterator, ParallelIterator};

use crate::{
    dict::EntryId,
    rich_dict::{RichDict, RichEntry},
    search::{get_entry_frequency, RichDictLike},
    sqlite_db::SqliteDb,
};

pub trait EntryGroupIndex {
    fn get_entry_group(&self, id: EntryId) -> Vec<EntryId>;
}

impl EntryGroupIndex for SqliteDb {
    fn get_entry_group(&self, id: EntryId) -> Vec<EntryId> {
        let conn = self.conn();
        let mut stmt = conn
            .prepare("SELECT group_ids FROM entry_group_index WHERE entry_id = ?")
            .unwrap();
        let group_ids: String = stmt.query_row(&[&id], |row| row.get(0)).unwrap();
        serde_json::from_str(&group_ids).unwrap()
    }
}

pub fn get_entry_group<D>(dict: &D, id: EntryId) -> Vec<RichEntry>
where
    D: RichDictLike + EntryGroupIndex,
{
    dict.get_entry_group(id)
        .iter()
        .filter_map(|id| dict.get_entry(*id))
        .collect()
}

pub fn get_entry_group_ids(dict: &RichDict, id: EntryId) -> Option<Vec<EntryId>> {
    let entry = dict.get_entry(id)?;
    let query_word_set: HashSet<&str> = entry.variants.to_words_set();
    Some(sort_entry_group_ids(
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
    ))
}

// Assumes all EntryIds in entry_group are valid
fn sort_entry_group_ids(dict: &dyn RichDictLike, mut entry_group: Vec<EntryId>) -> Vec<EntryId> {
    entry_group.sort_by(|a, b| {
        get_entry_frequency(*a)
            .cmp(&get_entry_frequency(*b))
            .reverse()
            .then(
                dict.get_entry(*a)
                    .unwrap()
                    .defs
                    .len()
                    .cmp(&dict.get_entry(*b).unwrap().defs.len())
                    .reverse(),
            )
            .then(a.cmp(b))
    });
    entry_group
}
