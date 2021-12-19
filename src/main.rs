use std::process;
use wordshk_tools::{dict_to_xml, parse_dict};

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
