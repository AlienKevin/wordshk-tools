use rkyv::{AlignedVec, Deserialize};

use crate::jyutping::Romanization;
use crate::pr_index::{generate_pr_indices, pr_indices_into_fst, FstPrIndices};
use crate::rich_dict::{ArchivedRichDict, RichDictWrapper};

use super::english_index::generate_english_index;
use super::parse::parse_dict;
use super::rich_dict::{enrich_dict, EnrichDictOptions, RichDict};
use std::fs;
use std::path::Path;

pub struct Api {
    dict_data: AlignedVec,
    pub fst_pr_indices: FstPrIndices,
}

fn serialize_dict<P: AsRef<Path>>(output_path: &P, dict: &RichDict) {
    fs::write(output_path, rkyv::to_bytes::<_, 1024>(dict).unwrap())
        .expect("Unable to output serialized RichDict");
}

impl Api {
    pub unsafe fn new(app_dir: &str, csv: &str, romanization: Romanization) -> Self {
        let api = Api::get_new_dict(app_dir, csv, romanization);
        Api::generate_english_index(app_dir, &api.dict());
        #[cfg(feature = "embedding-search")]
        Api::generate_english_embeddings(app_dir, &api.dict());
        api
    }

    pub unsafe fn dict(&self) -> &ArchivedRichDict {
        unsafe { rkyv::archived_root::<RichDict>(&self.dict_data) }
    }

    fn insert_rich_entry(
        conn: &rusqlite::Connection,
        entry: &crate::rich_dict::RichEntry,
    ) -> rusqlite::Result<()> {
        conn.execute(
            "INSERT INTO rich_dict (id, entry_rkyv) VALUES (?, ?)",
            rusqlite::params![
                entry.id,
                rkyv::to_bytes::<_, 1024>(entry).unwrap().as_slice()
            ],
        )?;
        Ok(())
    }

    fn insert_english_index_data(
        conn: &rusqlite::Connection,
        phrase: &str,
        english_index_data: &Vec<crate::english_index::EnglishIndexData>,
    ) -> rusqlite::Result<()> {
        conn.execute(
            "INSERT INTO english_index (phrase, english_index_data_rkyv) VALUES (?, ?)",
            rusqlite::params![
                phrase,
                rkyv::to_bytes::<_, 1024>(english_index_data)
                    .unwrap()
                    .as_slice()
            ],
        )?;
        Ok(())
    }

    pub fn export_dict_as_sqlite_db(&self, db_path: &Path, version: &str) -> rusqlite::Result<()> {
        use rkyv::Deserialize;

        let conn = rusqlite::Connection::open(db_path)?;
        conn.execute(
            "CREATE TABLE rich_dict (
                id INTEGER PRIMARY KEY,
                entry_rkyv BLOB NOT NULL
            )",
            [],
        )?;

        conn.execute(
            "CREATE TABLE english_index (
                phrase TEXT PRIMARY KEY,
                english_index_data_rkyv BLOB NOT NULL
            )",
            [],
        )?;

        // Keep track of dict version in a separate metadata table
        conn.execute(
            "CREATE TABLE rich_dict_metadata (
            key TEXT NOT NULL UNIQUE,
            value TEXT
        )",
            [],
        )?;
        conn.execute(
            "INSERT INTO rich_dict_metadata (key, value) VALUES ('version', ?)",
            rusqlite::params![version],
        )?;

        for entry in unsafe { self.dict().values() } {
            Self::insert_rich_entry(&conn, &entry.deserialize(&mut rkyv::Infallible).unwrap())?;
        }

        for (phrase, english_index_data) in generate_english_index(unsafe { self.dict() }) {
            Self::insert_english_index_data(&conn, &phrase, &english_index_data)?;
        }

        Ok(())
    }

    pub unsafe fn load(app_dir: &str, romanization: Romanization) -> Self {
        let dict_path = Path::new(app_dir).join("dict.rkyv");
        // Read the data from the file
        let dict_data = fs::read(dict_path).expect("Unable to read serialized data");
        let mut aligned_dict_data = AlignedVec::with_capacity(dict_data.len());
        aligned_dict_data.extend_from_slice(&dict_data);

        let dict = unsafe { rkyv::archived_root::<RichDict>(&dict_data) };

        let dict = RichDictWrapper::new(dict.deserialize(&mut rkyv::Infallible).unwrap());
        let pr_indices = generate_pr_indices(&dict, romanization);

        Api {
            dict_data: aligned_dict_data,
            fst_pr_indices: pr_indices_into_fst(pr_indices),
        }
    }

    unsafe fn get_new_dict(app_dir: &str, csv: &str, romanization: Romanization) -> Self {
        let dict = parse_dict(csv.as_bytes()).unwrap();
        let dict = crate::dict::filter_unfinished_entries(dict);
        let rich_dict = enrich_dict(
            &dict,
            &EnrichDictOptions {
                remove_dead_links: true,
            },
        );
        let api_path = Path::new(app_dir).join("dict.rkyv");
        serialize_dict(&api_path, &rich_dict);
        Self::load(app_dir, romanization)
    }

    fn generate_english_index(app_dir: &str, dict: &ArchivedRichDict) {
        let index_path = Path::new(app_dir).join("english_index.rkyv");
        let english_index = generate_english_index(dict);
        fs::write(
            index_path,
            rkyv::to_bytes::<_, 1024>(&english_index).unwrap(),
        )
        .expect("Unable to output serialized english index");
    }

    #[cfg(feature = "embedding-search")]
    fn generate_english_embeddings(app_dir: &str, dict: &ArchivedRichDict) {
        let embeddings_path = Path::new(app_dir).join("english_embeddings.fifu");
        let embeddings_bytes = super::english_embedding::generate_english_embeddings(dict).unwrap();
        fs::write(embeddings_path, embeddings_bytes)
            .expect("Unable to output serialized english embeddings");
    }
}
