use wordshk_tools::{app_api::Api, jyutping::Romanization};

const APP_TMP_DIR: &str = "./app_tmp";

fn main() {
    let api = unsafe { Api::load(APP_TMP_DIR, Romanization::Jyutping) };
    api.export_dict_as_sqlite_db("dict.db").unwrap();
}
