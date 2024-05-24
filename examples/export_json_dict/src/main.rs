const APP_TMP_DIR: &str = "./app_tmp";

fn main() {
    export_json_dict();
}

fn export_json_dict() {
    std::fs::create_dir(APP_TMP_DIR).ok();
    let dict = wordshk_tools::parse::parse_dict(include_str!("../../wordshk.csv").as_bytes()).unwrap();
    let dict = wordshk_tools::dict::filter_unfinished_entries(dict);
    let json_path = std::path::Path::new(APP_TMP_DIR).join("dict.json");
    std::fs::write(json_path, serde_json::to_string(&dict).unwrap()).unwrap();
}
