use crate::{
    dict::EntryId,
    rich_dict::{ArchivedRichDict, ArchivedRichEntry},
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
        use rkyv::Deserialize;

        let conn = self.conn();
        let mut stmt = conn
            .prepare(match script {
                Script::Traditional => {
                    "SELECT entry_ids_rkyv FROM variant_index_trad WHERE char = ?"
                }
                Script::Simplified => {
                    "SELECT entry_ids_rkyv FROM variant_index_simp WHERE char = ?"
                }
            })
            .unwrap();
        let english_index_data_rkyv_bytes: Option<Vec<u8>> =
            stmt.query_row([c.to_string()], |row| row.get(0)).ok();
        english_index_data_rkyv_bytes.map(|bytes| {
            unsafe { rkyv::archived_root::<BTreeSet<EntryId>>(&bytes[..]) }
                .deserialize(&mut rkyv::Infallible)
                .unwrap()
        })
    }
}

pub fn generate_variant_index(dict: &ArchivedRichDict) -> (VariantIndex, VariantIndex) {
    let mut index_trad = HashMap::new();
    let mut index_simp = HashMap::new();

    dict.iter().for_each(|(_, entry)| {
        index_entry(entry, &mut index_trad, &mut index_simp);
    });

    (index_trad, index_simp)
}

fn index_entry(
    entry: &ArchivedRichEntry,
    index_trad: &mut VariantIndex,
    index_simp: &mut VariantIndex,
) {
    entry
        .variants
        .0
        .iter()
        .enumerate()
        .for_each(|(i, variant)| {
            let word_trad = crate::unicode::normalize(variant.word.as_str());
            let word_simp = crate::unicode::normalize(entry.variants_simp[i].as_str());
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
