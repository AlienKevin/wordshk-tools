use wordshk_tools::parse::parse_dict;

fn main() {
    static DATA_FILE: &'static str = include_str!("../all.csv");
    let dict = parse_dict(DATA_FILE.as_bytes());
    println!("{:?}", dict);
}
