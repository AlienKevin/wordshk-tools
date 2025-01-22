use crate::{
    dict::EntryId,
    rich_dict::{RichDict, RichEntry},
    sqlite_db::SqliteDb,
};
use std::collections::{BTreeSet, HashMap};

pub type MandarinVariantIndex = HashMap<char, BTreeSet<EntryId>>;

pub trait MandarinVariantIndexLike: Sync + Send {
    fn get(&self, c: char) -> Option<BTreeSet<EntryId>>;
}

impl MandarinVariantIndexLike for SqliteDb {
    fn get(&self, c: char) -> Option<BTreeSet<EntryId>> {
        let conn = self.conn();
        let mut stmt = conn
            .prepare(&format!(
                "SELECT entry_ids FROM mandarin_variant_index WHERE char = ?"
            ))
            .unwrap();
        let english_index_data_string: Option<String> =
            stmt.query_row([c.to_string()], |row| row.get(0)).ok();
        english_index_data_string.map(|string| serde_json::from_str(&string).unwrap())
    }
}

pub fn generate_variant_index(dict: &RichDict) -> MandarinVariantIndex {
    let mut index_simp = HashMap::new();

    dict.iter().for_each(|(_, entry)| {
        index_entry(entry, &mut index_simp);
    });

    index_simp
}

fn index_entry(entry: &RichEntry, index_simp: &mut MandarinVariantIndex) {
    entry.mandarin_variants.0.iter().for_each(|variant| {
        let word_simp = crate::unicode::normalize(variant.word_simp.as_str());
        for c in word_simp.chars() {
            index_simp
                .entry(c)
                .or_insert_with(BTreeSet::new)
                .insert(entry.id);
        }
    });
}
