use wordshk_tools::app_api::Api;

const APP_TMP_DIR: &str = "./app_tmp";

fn main() {
    std::fs::create_dir(APP_TMP_DIR).ok();
    generate_api_json();
}

fn generate_api_json() {
    let _ = Api::new(APP_TMP_DIR, include_str!("../../wordshk.csv"));
}
