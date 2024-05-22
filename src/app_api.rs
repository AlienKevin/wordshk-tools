use crate::dict::EntryId;
use crate::entry_group_index::get_entry_group_ids;
use crate::jyutping::Romanization;
use crate::pr_index::{generate_pr_indices, pr_indices_into_fst};
use crate::rich_dict::RichEntry;
use crate::search::Script;
use crate::variant_index::generate_variant_index;

use super::english_index::generate_english_index;
use super::parse::parse_dict;
use super::rich_dict::{enrich_dict, EnrichDictOptions, RichDict};
use std::collections::BTreeSet;

pub struct Api {
    dict: RichDict,
}

impl Api {
    pub fn new(csv: &str) -> Self {
        let api = Api::get_new_dict(csv);
        api
    }

    fn insert_rich_entry(conn: &rusqlite::Connection, entry: &RichEntry) -> rusqlite::Result<()> {
        conn.execute(
            "INSERT INTO rich_dict (id, entry) VALUES (?, ?)",
            rusqlite::params![entry.id, serde_json::to_string(entry).unwrap()],
        )?;
        Ok(())
    }

    fn insert_group_ids(
        conn: &rusqlite::Connection,
        dict: &RichDict,
        entry_id: EntryId,
    ) -> rusqlite::Result<()> {
        conn.execute(
            "INSERT INTO entry_group_index (entry_id, group_ids) VALUES (?, ?)",
            rusqlite::params![
                entry_id,
                serde_json::to_string(&get_entry_group_ids(dict, entry_id)).unwrap()
            ],
        )?;
        Ok(())
    }

    fn insert_variant(
        conn: &rusqlite::Connection,
        entry: &RichEntry,
        script: Script,
    ) -> rusqlite::Result<()> {
        for variant in entry.variants.0.iter() {
            conn.execute(
                &format!(
                    "INSERT INTO variant_map_{script} (variant, entry_id) VALUES (?, ?) ON CONFLICT(variant) DO NOTHING"
                ),
                rusqlite::params![match script {
                    Script::Traditional => variant.word.as_str(),
                    Script::Simplified => variant.word_simp.as_str(),
                }, entry.id,],
            )?;
        }
        Ok(())
    }

    fn insert_english_index_data(
        conn: &rusqlite::Connection,
        phrase: &str,
        english_index_data: &Vec<crate::english_index::EnglishIndexData>,
    ) -> rusqlite::Result<()> {
        conn.execute(
            "INSERT INTO english_index (phrase, english_index_data) VALUES (?, ?)",
            rusqlite::params![phrase, serde_json::to_string(english_index_data).unwrap()],
        )?;
        Ok(())
    }

    fn insert_variant_index_data(
        conn: &rusqlite::Connection,
        c: char,
        entry_ids: &BTreeSet<EntryId>,
        script: Script,
    ) -> rusqlite::Result<()> {
        conn.execute(
            match script {
                Script::Traditional => {
                    "INSERT INTO variant_index_trad (char, entry_ids) VALUES (?, ?)"
                }
                Script::Simplified => {
                    "INSERT INTO variant_index_simp (char, entry_ids) VALUES (?, ?)"
                }
            },
            rusqlite::params![c.to_string(), serde_json::to_string(entry_ids).unwrap()],
        )?;
        Ok(())
    }

    fn insert_pr_index(
        conn: &rusqlite::Connection,
        dict: &RichDict,
        romanization: Romanization,
    ) -> rusqlite::Result<()> {
        let fst = pr_indices_into_fst(generate_pr_indices(dict, romanization));
        for (id, locations) in fst.locations {
            conn.execute(
                &format!(
                    "INSERT INTO pr_index_locations_{romanization} (id, locations) VALUES (?, ?)"
                ),
                rusqlite::params![id, serde_json::to_string(&locations).unwrap()],
            )?;
        }
        conn.execute(
            &format!("INSERT INTO pr_index_fsts_{romanization} (name, fst) VALUES (?, ?)"),
            rusqlite::params!["none", fst.none.as_fst().as_bytes(),],
        )?;
        conn.execute(
            &format!("INSERT INTO pr_index_fsts_{romanization} (name, fst) VALUES (?, ?)"),
            rusqlite::params!["tone", fst.tone.as_fst().as_bytes(),],
        )?;
        Ok(())
    }

    pub fn export_dict_as_sqlite_db<P>(&self, db_path: P, version: &str) -> rusqlite::Result<()>
    where
        P: AsRef<std::path::Path>,
    {
        let mut conn = rusqlite::Connection::open(db_path)?;
        let tx = conn.transaction()?;

        tx.execute(
            "CREATE TABLE rich_dict (
                id INTEGER PRIMARY KEY,
                entry TEXT NOT NULL
            )",
            [],
        )?;

        tx.execute(
            "CREATE TABLE english_index (
                phrase TEXT PRIMARY KEY,
                english_index_data TEXT NOT NULL
            )",
            [],
        )?;

        tx.execute(
            "CREATE TABLE variant_index_trad (
                char TEXT PRIMARY KEY,
                entry_ids TEXT NOT NULL
            )",
            [],
        )?;
        tx.execute(
            "CREATE TABLE variant_index_simp (
                char TEXT PRIMARY KEY,
                entry_ids TEXT NOT NULL
            )",
            [],
        )?;

        tx.execute(
            "CREATE TABLE pr_index_locations_jyutping (
                id INTEGER PRIMARY KEY,
                locations TEXT NOT NULL
            )",
            [],
        )?;
        tx.execute(
            "CREATE TABLE pr_index_fsts_jyutping (
                name TEXT PRIMARY KEY,
                fst BLOB NOT NULL
            )",
            [],
        )?;

        tx.execute(
            "CREATE TABLE pr_index_locations_yale (
                id INTEGER PRIMARY KEY,
                locations TEXT NOT NULL
            )",
            [],
        )?;
        tx.execute(
            "CREATE TABLE pr_index_fsts_yale (
                name TEXT PRIMARY KEY,
                fst BLOB NOT NULL
            )",
            [],
        )?;

        tx.execute(
            "CREATE TABLE entry_group_index (
                entry_id INTEGER PRIMARY KEY,
                group_ids TEXT NOT NULL
            )",
            [],
        )?;

        tx.execute(
            "CREATE TABLE variant_map_trad (
                variant TEXT PRIMARY KEY,
                entry_id INTEGER NOT NULL
            )",
            [],
        )?;
        tx.execute(
            "CREATE TABLE variant_map_simp (
                variant TEXT PRIMARY KEY,
                entry_id INTEGER NOT NULL
            )",
            [],
        )?;

        // Keep track of dict version in a separate metadata table
        tx.execute(
            "CREATE TABLE rich_dict_metadata (
            key TEXT NOT NULL UNIQUE,
            value TEXT
        )",
            [],
        )?;
        tx.execute(
            "INSERT INTO rich_dict_metadata (key, value) VALUES ('version', ?)",
            rusqlite::params![version],
        )?;

        for entry in self.dict.values() {
            Self::insert_rich_entry(&tx, entry)?;
            Self::insert_variant(&tx, entry, Script::Simplified)?;
            Self::insert_variant(&tx, entry, Script::Traditional)?;
            Self::insert_group_ids(&tx, &self.dict, entry.id)?;
        }

        for (phrase, english_index_data) in generate_english_index(&self.dict) {
            Self::insert_english_index_data(&tx, &phrase, &english_index_data)?;
        }

        Self::insert_pr_index(&tx, &self.dict, Romanization::Jyutping)?;
        Self::insert_pr_index(&tx, &self.dict, Romanization::Yale)?;

        let (index_trad, index_simp) = generate_variant_index(&self.dict);
        for (c, entry_ids) in index_trad {
            Self::insert_variant_index_data(&tx, c, &entry_ids, Script::Traditional)?;
        }
        for (c, entry_ids) in index_simp {
            Self::insert_variant_index_data(&tx, c, &entry_ids, Script::Simplified)?;
        }

        tx.commit()?;

        Ok(())
    }

    fn get_new_dict(csv: &str) -> Self {
        let dict = parse_dict(csv.as_bytes()).unwrap();
        let dict = crate::dict::filter_unfinished_entries(dict);
        let rich_dict = enrich_dict(
            &dict,
            &EnrichDictOptions {
                remove_dead_links: true,
            },
        );
        Api { dict: rich_dict }
    }
}
