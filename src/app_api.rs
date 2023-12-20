use crate::jyutping::Romanization;
use crate::pr_index::generate_pr_indices;

use super::english_index::generate_english_index;
use super::parse::parse_dict;
use super::rich_dict::{enrich_dict, EnrichDictOptions, RichDict};
use flate2::read::GzDecoder;
use flate2::write::GzEncoder;
use flate2::Compression;
use rmp_serde::Serializer;
use serde::{Deserialize, Serialize};
use std::fs;
use std::io::Read;
use std::io::Write;
use std::path::Path;

#[derive(Serialize, Deserialize)]
pub struct Api {
    pub dict: RichDict,
}

fn serialize_dict<P: AsRef<Path>>(output_path: &P, dict: &RichDict) {
    let mut e = GzEncoder::new(Vec::new(), Compression::default());
    e.write_all(serde_json::to_string(dict).unwrap().as_bytes())
        .unwrap();
    fs::write(output_path, e.finish().unwrap()).expect("Unable to output serialized RichDict");
}

impl Api {
    pub fn new(app_dir: &str, csv: &str, romanization: Romanization) -> Self {
        let api_path = Path::new(app_dir).join("dict.json");
        let api = Api::get_new_dict(&api_path, csv);
        Api::generate_index(app_dir, &api.dict, romanization);
        api
    }

    pub fn load(app_dir: &str) -> Self {
        let dict_path = Path::new(app_dir).join("dict.json");
        // Read the compressed data from the file
        let compressed_data = fs::read(dict_path).expect("Unable to read serialized data");
        // Create a GzDecoder from the compressed data
        let mut d = GzDecoder::new(&compressed_data[..]);
        // Decompress and read the decoded data into a String
        let mut decoded_data = String::new();
        d.read_to_string(&mut decoded_data).unwrap();
        // Deserialize the data back into the Api type
        Api { dict: serde_json::from_str(&decoded_data).expect("Unable to deserialize Api data") }
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
        serialize_dict(api_path, &new_api.dict);
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

        let index_path = Path::new(app_dir).join("pr_indices.msgpack");
        let pr_indices = generate_pr_indices(dict, romanization);
        let mut buf = Vec::new();
        pr_indices
            .serialize(&mut Serializer::new(&mut buf))
            .unwrap();
        let mut e = GzEncoder::new(Vec::new(), Compression::default());
        e.write_all(&buf).unwrap();
        fs::write(index_path, e.finish().unwrap()).expect("Unable to output serialized pr index");
    }
}
