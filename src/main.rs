use wordshk_tools::{parse_dict, dict_to_xml};
use std::process;

fn main() {
    match parse_dict() {
        Err(err) => {
            println!("error reading csv file: {}", err);
            process::exit(1);
        }
        Ok(dict) => {
            print!("{}", dict_to_xml(dict));
        }
    }
}
