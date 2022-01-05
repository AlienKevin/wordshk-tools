use super::dict::LaxJyutPing;
use super::emit_html::rich_entry_to_xml;
use super::parse::{parse_dict, parse_pr};
use super::rich_dict::{enrich_dict, RichDict};
use super::search;
use chrono::{DateTime, Utc};
use flate2::read::GzDecoder;
use reqwest;
use serde::{Deserialize, Serialize};
use std::fs;
use std::io::prelude::*;
use std::path::Path;

#[derive(Serialize, Deserialize)]
pub struct Api {
    pub dict: RichDict,
    pub release_time: DateTime<Utc>,
}

fn serialize_api<P: AsRef<Path>>(output_path: &P, api: &Api) {
    fs::write(output_path, serde_json::to_string(&api).unwrap())
        .expect("Unable to output serailized RichDict");
}

fn deserialize_api<P: AsRef<Path>>(input_path: &P) -> Option<Api> {
    match fs::read_to_string(input_path) {
        Ok(json) => match serde_json::from_str(&json) {
            Ok(api) => Some(api),
            Err(_) => None,
        },
        Err(_) => None,
    }
}

impl Api {
    pub fn new(app_dir: &str) -> Self {
        let api_path = Path::new(app_dir).join("api.json");
        let current_time = Utc::now();
        match deserialize_api(&api_path) {
            Some(api) => {
                if current_time
                    .signed_duration_since(api.release_time)
                    .num_weeks()
                    > 1
                {
                    Api::get_new_dict(&api_path)
                } else {
                    api
                }
            }
            None => Api::get_new_dict(&api_path),
        }
    }

    pub fn pr_search(&self, capacity: usize, query: &str) -> Vec<PrSearchResult> {
        pr_search_helper(capacity, &self.dict, &parse_pr(query))
    }

    pub fn variant_search(&self, capacity: usize, query: &str) -> Vec<VariantSearchResult> {
        variant_search_helper(capacity, &self.dict, query)
    }

    pub fn get_entry_html(&self, id: usize) -> String {
        rich_entry_to_xml(self.dict.get(&id).unwrap())
    }

    fn get_new_dict<P: AsRef<Path>>(api_path: &P) -> Api {
        let new_release_time = Utc::now();
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
            release_time: new_release_time,
        };
        serialize_api(api_path, &new_api);
        new_api
    }
}

pub struct PrSearchResult {
    pub id: usize,
    pub variant: String,
    pub pr: String,
}

fn pr_search_helper(capacity: usize, dict: &RichDict, query: &LaxJyutPing) -> Vec<PrSearchResult> {
    let mut ranks = search::pr_search(dict, query);
    let mut results = vec![];
    let mut i = 0;
    while ranks.len() > 0 && i < capacity {
        let search::PrSearchRank {
            id,
            variant_index,
            pr_index,
            ..
        } = ranks.pop().unwrap();
        let entry = dict.get(&id).unwrap();
        let variant = &entry.variants.0[variant_index];
        results.push(PrSearchResult {
            id,
            variant: variant.word.clone(),
            pr: variant.prs.0[pr_index].to_string(),
        });
        i += 1;
    }
    results
}

pub struct VariantSearchResult {
    pub id: usize,
    pub variant: String,
}

fn variant_search_helper(
    capacity: usize,
    dict: &RichDict,
    query: &str,
) -> Vec<VariantSearchResult> {
    let mut ranks = search::variant_search(dict, query);
    let mut results = vec![];
    let mut i = 0;
    while ranks.len() > 0 && i < capacity {
        let search::VariantSearchRank {
            id, variant_index, ..
        } = ranks.pop().unwrap();
        let entry = dict.get(&id).unwrap();
        let variant = &entry.variants.0[variant_index];
        results.push(VariantSearchResult {
            id,
            variant: variant.word.clone(),
        });
        i += 1;
    }
    results
}
