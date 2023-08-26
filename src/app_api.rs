use crate::jyutping::Romanization;
use crate::pr_index::generate_pr_indices;

use super::english_index::generate_english_index;
use super::parse::parse_dict;
use super::rich_dict::{enrich_dict, EnrichDictOptions, RichDict};
use flate2::write::GzEncoder;
use flate2::Compression;
use serde::{Deserialize, Serialize};
use std::fs;
use std::io::prelude::*;
use std::path::Path;

#[derive(Serialize, Deserialize)]
pub struct Api {
    pub dict: RichDict,
}

fn serialize_api<P: AsRef<Path>>(output_path: &P, api: &Api) {
    let mut e = GzEncoder::new(Vec::new(), Compression::default());
    e.write_all(serde_json::to_string(&api).unwrap().as_bytes())
        .unwrap();
    fs::write(output_path, e.finish().unwrap()).expect("Unable to output serialized RichDict");
}

impl Api {
    pub fn new(app_dir: &str, csv: &str, romanization: Romanization) -> Self {
        let api_path = Path::new(app_dir).join("api.json");
        let api = Api::get_new_dict(&api_path, csv);
        Api::generate_index(app_dir, &api.dict, romanization);
        api
    }

    fn get_new_dict<P: AsRef<Path>>(api_path: &P, csv: &str) -> Api {
        let dict = parse_dict(csv.as_bytes()).unwrap();
        let dict = crate::dict::filter_unfinished_entries(dict);
        let new_api = Api {
            dict: enrich_dict(
                &dict,
                &EnrichDictOptions {
                    remove_dead_links: true,
                },
            ),
        };
        serialize_api(api_path, &new_api);
        new_api
    }

    fn generate_index(app_dir: &str, dict: &RichDict, romanization: Romanization) {
        let index_path = Path::new(app_dir).join("english_index.json");
        let english_index = generate_english_index(dict);
        let mut e = GzEncoder::new(Vec::new(), Compression::default());
        e.write_all(serde_json::to_string(&english_index).unwrap().as_bytes())
            .unwrap();
        fs::write(index_path, e.finish().unwrap())
            .expect("Unable to output serailized english index");

        let index_path = Path::new(app_dir).join("pr_indices.json");
        let pr_indices = generate_pr_indices(dict, romanization);
        let mut e = GzEncoder::new(Vec::new(), Compression::default());
        e.write_all(serde_json::to_string(&pr_indices).unwrap().as_bytes())
            .unwrap();
        fs::write(index_path, e.finish().unwrap()).expect("Unable to output serailized pr index");
    }
}
