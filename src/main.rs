use wordshk_tools::{parse_dict, to_apple_dict};
use std::process;

fn main() {
    match parse_dict() {
        Err(err) => {
            println!("error reading csv file: {}", err);
            process::exit(1);
        }
        Ok(dict) => {
            print!("{}", to_apple_dict(dict));
        }
    }
}
