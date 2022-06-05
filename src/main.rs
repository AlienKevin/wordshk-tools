use std::io;
use std::process;
use wordshk_tools::app_api::Api;
use wordshk_tools::emit_apple_dict::rich_dict_to_xml;
use wordshk_tools::english_index::generate_english_index;
use wordshk_tools::parse::parse_dict;
use wordshk_tools::rich_dict::{enrich_dict, EnrichDictOptions};

const APP_TMP_DIR: &str = "./app_tmp";

fn main() {
    generate_api_json();
}

fn generate_api_json() {
    let _ = Api::new(APP_TMP_DIR);
}

fn generate_apple_dict() {
    match parse_dict(io::stdin()) {
        Err(err) => {
            println!("error reading csv file: {}", err);
            process::exit(1);
        }
        Ok(dict) => {
            print!("{}", rich_dict_to_xml(enrich_dict(&dict, &EnrichDictOptions { remove_dead_links: true })));
        }
    }
}
