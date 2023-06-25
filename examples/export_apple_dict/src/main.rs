use std::io::Write;
use wordshk_tools::{
    emit_apple_dict::rich_dict_to_xml,
    parse::parse_dict,
    rich_dict::{enrich_dict, EnrichDictOptions},
};

fn main() {
    let mut xml_file = std::fs::File::create("apple_dict/wordshk.xml").expect("Failed to create the target XML file");

    static DATA_FILE: &'static str = include_str!("../../wordshk.csv");
    let dict = parse_dict(DATA_FILE.as_bytes()).expect("Failed to parse dict");
    let rich_dict = enrich_dict(
        &dict,
        &EnrichDictOptions {
            remove_dead_links: true,
        },
    );
    
    xml_file
        .write_all(rich_dict_to_xml(rich_dict).as_bytes())
        .expect("Failed to write to target XML file");
}
