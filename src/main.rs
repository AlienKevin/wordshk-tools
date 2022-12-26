use wordshk_tools::app_api::Api;

const APP_TMP_DIR: &str = "./app_tmp";

fn main() {
    std::fs::create_dir(APP_TMP_DIR).ok();
    generate_api_json();
}

fn generate_api_json() {
    // Request your own URL from https://words.hk/faiman/request_data/
    let csv_url = "https://words.hk/static/generated/all-1671998701.csv.gz";
    let _ = Api::new(APP_TMP_DIR, csv_url);
}
