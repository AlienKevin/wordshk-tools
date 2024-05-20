use crate::{
    dict::EntryId,
    rich_dict::{RichDict, RichEntry},
    search::Script,
    sqlite_db::SqliteDb,
};
use std::collections::{BTreeSet, HashMap};

pub type VariantIndex = HashMap<char, BTreeSet<EntryId>>;

pub trait VariantIndexLike: Sync + Send {
    fn get(&self, c: char, script: Script) -> Option<BTreeSet<EntryId>>;
}

impl VariantIndexLike for SqliteDb {
    fn get(&self, c: char, script: Script) -> Option<BTreeSet<EntryId>> {
        let conn = self.conn();
        let mut stmt = conn
            .prepare(&format!(
                "SELECT entry_ids FROM variant_index_{script} WHERE char = ?"
            ))
            .unwrap();
        let english_index_data_bytes: Option<Vec<u8>> =
            stmt.query_row([c.to_string()], |row| row.get(0)).ok();
        english_index_data_bytes.map(|bytes| serde_json::from_slice(&bytes).unwrap())
    }
}

pub fn generate_variant_index(dict: &RichDict) -> (VariantIndex, VariantIndex) {
    let mut index_trad = HashMap::new();
    let mut index_simp = HashMap::new();

    dict.iter().for_each(|(_, entry)| {
        index_entry(entry, &mut index_trad, &mut index_simp);
    });

    (index_trad, index_simp)
}

fn index_entry(entry: &RichEntry, index_trad: &mut VariantIndex, index_simp: &mut VariantIndex) {
    entry.variants.0.iter().for_each(|variant| {
        let word_trad = crate::unicode::normalize(variant.word.as_str());
        let word_simp = crate::unicode::normalize(variant.word_simp.as_str());
        for c in word_trad.chars() {
            index_trad
                .entry(c)
                .or_insert_with(BTreeSet::new)
                .insert(entry.id);
        }
        for c in word_simp.chars() {
            index_simp
                .entry(c)
                .or_insert_with(BTreeSet::new)
                .insert(entry.id);
        }
    });
}
