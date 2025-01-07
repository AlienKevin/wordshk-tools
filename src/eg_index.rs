use std::collections::{HashMap, HashSet};

use crate::{search::RichDictLike, sqlite_db::SqliteDb};

pub type EgIndex = HashMap<u32, HashSet<u32>>;
pub type EgLocation = (u32, usize, usize);
pub type CharacterIdMapping = HashMap<char, u32>;
pub type EgIdMapping = HashMap<u32, (EgLocation, String, String)>;

pub trait EgIndexLike {
    /// Retrieves the `(entry_id, def_index, eg_index)` tuples for a given character.
    fn get_egs_by_character(&self, character: char) -> HashSet<(EgLocation, String, String)>;
}

pub fn generate_eg_index(dict: &dyn RichDictLike) -> (EgIndex, CharacterIdMapping, EgIdMapping) {
    let mut egs_index: EgIndex = HashMap::new();
    let mut character_id_mapping: CharacterIdMapping = HashMap::new();
    let mut eg_id_mapping: EgIdMapping = HashMap::new();
    let mut current_eg_id: u32 = 0;

    for entry_id in dict.get_ids() {
        if let Some(entry) = dict.get_entry(entry_id) {
            for (def_index, def) in entry.defs.iter().enumerate() {
                for (eg_index, eg) in def.egs.iter().enumerate() {
                    // Combine both traditional and simplified versions of the eg if available
                    let eg_yue = eg.yue.as_ref().map(|s| s.to_string());
                    let eg_yue_simp = eg.yue_simp.as_ref().map(|s| s.clone());
                    let (eg_yue, eg_yue_simp) = match (eg_yue, eg_yue_simp) {
                        (Some(eg_yue), Some(eg_yue_simp)) => (eg_yue, eg_yue_simp),
                        _ => continue,
                    };

                    for c in eg_yue.chars().chain(eg_yue_simp.chars()) {
                        let new_char_id = character_id_mapping.len() as u32;
                        let char_id = character_id_mapping.entry(c).or_insert(new_char_id);
                        egs_index
                            .entry(*char_id)
                            .or_insert_with(HashSet::new)
                            .insert(current_eg_id);
                    }

                    // Map eg_id to (entry_id, def_index, eg_index)
                    eg_id_mapping.insert(
                        current_eg_id,
                        ((entry_id, def_index, eg_index), eg_yue, eg_yue_simp),
                    );
                    current_eg_id += 1;
                }
            }
        }
    }

    (egs_index, character_id_mapping, eg_id_mapping)
}

pub fn get_egs_by_character(
    eg_index: &EgIndex,
    character_id_mapping: &CharacterIdMapping,
    eg_id_mapping: &EgIdMapping,
    character: char,
) -> HashSet<(EgLocation, String, String)> {
    character_id_mapping
        .get(&character)
        .and_then(|character_id| {
            eg_index.get(&character_id).map(|eg_ids| {
                eg_ids
                    .iter()
                    .filter_map(|eg_id| eg_id_mapping.get(eg_id))
                    .cloned()
                    .collect()
            })
        })
        .unwrap_or(HashSet::new())
}

impl EgIndexLike for SqliteDb {
    fn get_egs_by_character(&self, character: char) -> HashSet<(EgLocation, String, String)> {
        let conn = self.conn();
        let mut stmt = conn
            .prepare(
                "SELECT e.entry_id, e.def_index, e.eg_index, e.eg_yue, e.eg_yue_simp
                 FROM character_eg_ids c
                 JOIN eg e ON c.eg_id = e.eg_id
                 JOIN character_ids ch ON c.character_id = ch.character_id
                 WHERE ch.character = ?",
            )
            .unwrap();

        let mut rows = stmt
            .query(rusqlite::params![character.to_string()])
            .unwrap();
        let mut results = HashSet::new();

        while let Some(row) = rows.next().unwrap() {
            let entry_id: u32 = row.get(0).unwrap();
            let def_index: usize = row.get(1).unwrap();
            let eg_index: usize = row.get(2).unwrap();
            let eg_yue: String = row.get(3).unwrap();
            let eg_yue_simp: String = row.get(4).unwrap();
            results.insert(((entry_id, def_index, eg_index), eg_yue, eg_yue_simp));
        }

        results
    }
}
