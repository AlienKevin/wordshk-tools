use crate::{
    dict::EntryId,
    rich_dict::{RichDict, RichEntry},
    search::Script,
    sqlite_db::SqliteDb,
};
use std::collections::{BTreeSet, HashMap};

pub type VariantIndex = HashMap<char, BTreeSet<EntryId>>;

pub trait VariantIndexLike: Sync + Send {
    fn get(&self, c: char) -> Option<BTreeSet<EntryId>>;
}

impl VariantIndexLike for SqliteDb {
    fn get(&self, c: char) -> Option<BTreeSet<EntryId>> {
        let conn = self.conn();
        let mut stmt = conn
            .prepare(&format!(
                "SELECT entry_ids FROM variant_index WHERE char = ?"
            ))
            .unwrap();
        let index_data_string: Option<String> =
            stmt.query_row([c.to_string()], |row| row.get(0)).ok();
        index_data_string.map(|string| serde_json::from_str(&string).unwrap())
    }
}

pub fn generate_variant_index(dict: &RichDict) -> VariantIndex {
    let mut index = HashMap::new();

    dict.iter().for_each(|(_, entry)| {
        index_entry(entry, &mut index);
    });

    index
}

fn index_entry(entry: &RichEntry, index: &mut VariantIndex) {
    entry.variants.0.iter().for_each(|variant| {
        let word_trad = crate::unicode::normalize(variant.word.as_str());
        let word_simp = crate::unicode::normalize(variant.word_simp.as_str());
        for c in word_trad.chars().chain(word_simp.chars()) {
            index
                .entry(c)
                .or_insert_with(BTreeSet::new)
                .insert(entry.id);
        }
    });
}
