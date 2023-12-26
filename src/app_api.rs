use crate::jyutping::Romanization;
use crate::pr_index::generate_pr_indices;
use crate::rich_dict::ArchivedRichDict;

use super::english_index::generate_english_index;
use super::parse::parse_dict;
use super::rich_dict::{enrich_dict, EnrichDictOptions, RichDict};
use std::fs;
use std::path::Path;

pub struct Api {
    dict_data: Vec<u8>,
}

fn serialize_dict<P: AsRef<Path>>(output_path: &P, dict: &RichDict) {
    fs::write(output_path, rkyv::to_bytes::<_, 1024>(dict).unwrap())
        .expect("Unable to output serialized RichDict");
}

impl Api {
    pub unsafe fn new(app_dir: &str, csv: &str, romanization: Romanization) -> Self {
        let api = Api::get_new_dict(app_dir, csv);
        Api::generate_index(app_dir, &api.dict(), romanization);
        api
    }

    pub unsafe fn dict(&self) -> &ArchivedRichDict {
        unsafe { rkyv::archived_root::<RichDict>(&self.dict_data) }
    }

    pub unsafe fn load(app_dir: &str) -> Self {
        let dict_path = Path::new(app_dir).join("dict.rkyv");
        // Read the data from the file
        let dict_data = fs::read(dict_path).expect("Unable to read serialized data");
        Api { dict_data }
    }

    unsafe fn get_new_dict(app_dir: &str, csv: &str) -> Self {
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
        Self::load(app_dir)
    }

    fn generate_index(app_dir: &str, dict: &ArchivedRichDict, romanization: Romanization) {
        let index_path = Path::new(app_dir).join("english_index.rkyv");
        let english_index = generate_english_index(dict);
        fs::write(
            index_path,
            rkyv::to_bytes::<_, 1024>(&english_index).unwrap(),
        )
        .expect("Unable to output serailized english index");

        let index_path = Path::new(app_dir).join("pr_indices.rkyv");
        let pr_indices = generate_pr_indices(dict, romanization);
        fs::write(index_path, rkyv::to_bytes::<_, 1024>(&pr_indices).unwrap())
            .expect("Unable to output serialized pr index");
    }
}
