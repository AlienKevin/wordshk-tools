use super::english_index::generate_english_index;
use super::lean_rich_dict::{to_lean_rich_entry, LeanRichEntry};
use super::parse::parse_dict;
use super::rich_dict::{enrich_dict, RichDict};
use flate2::read::GzDecoder;
use reqwest;
use serde::{Deserialize, Serialize};
use std::fs;
use std::io::prelude::*;
use std::path::Path;

#[derive(Serialize, Deserialize)]
pub struct Api {
    pub dict: RichDict,
}

fn serialize_api<P: AsRef<Path>>(output_path: &P, api: &Api) {
    fs::write(output_path, serde_json::to_string(&api).unwrap())
        .expect("Unable to output serialized RichDict");
}

impl Api {
    pub fn new(app_dir: &str) -> Self {
        let api_path = Path::new(app_dir).join("api.json");
        let api = Api::get_new_dict(&api_path);
        Api::generate_index(app_dir, &api.dict);
        api
    }

    fn get_new_dict<P: AsRef<Path>>(api_path: &P) -> Api {
        let csv_url = "https://words.hk/static/all.csv.gz";
        let csv_gz_data = reqwest::blocking::get(csv_url).unwrap().bytes().unwrap();
        let mut gz = GzDecoder::new(&csv_gz_data[..]);
        let mut csv_data = String::new();
        gz.read_to_string(&mut csv_data).unwrap();
        let csv_data_remove_first_line = csv_data.get(csv_data.find('\n').unwrap() + 1..).unwrap();
        let csv_data_remove_two_lines = csv_data_remove_first_line
            .get(csv_data_remove_first_line.find('\n').unwrap() + 1..)
            .unwrap();
        let dict = parse_dict(csv_data_remove_two_lines.as_bytes()).unwrap();
        let new_api = Api {
            dict: enrich_dict(&dict),
        };
        serialize_api(api_path, &new_api);
        new_api
    }

    fn generate_index(app_dir: &str, dict: &RichDict) {
        let index_path = Path::new(app_dir).join("english_index.json");
        let english_index = generate_english_index(&dict);
        fs::write(index_path, serde_json::to_string(&english_index).unwrap())
            .expect("Unable to output serailized Index");
    }
}
