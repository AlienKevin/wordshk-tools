use std::io;
use std::process;
use wordshk_tools::app_api::Api;
use wordshk_tools::dict::{
    JyutPing, JyutPingCoda, JyutPingInitial, JyutPingNucleus, JyutPingTone, LaxJyutPing,
    LaxJyutPingSegment,
};
use wordshk_tools::emit_apple_dict::rich_dict_to_xml;
use wordshk_tools::parse::parse_dict;
use wordshk_tools::rich_dict::enrich_dict;

const APP_TMP_DIR: &str = "./app_tmp";
const ENTRY_HTML_TEST_PATH: &str = "./app_tmp/test_entry.html";

fn main() {
    generate_html();
}

fn do_variant_search() {
    let api = Api::new(APP_TMP_DIR);
    // 說 is a variant that does not appear in dictionary variant data
    // Two words contains 說明: 說明 and 説明書
    let queries = vec!["說明"];

    let max_num_of_results = 10;
    queries.iter().for_each(|query| {
        println!("{}\n", query.to_string());
        api.variant_search(max_num_of_results, query)
            .iter()
            .for_each(|result| println!("{}", result.variant,));
        println!("\n----\n");
    });
}

fn do_pr_search() {
    let api = Api::new(APP_TMP_DIR);
    let queries = vec![("麪包", "ming baau"), ("學生", "hok6 saang")];
    let max_num_of_results = 10;
    queries.iter().for_each(|(intended, query)| {
        println!("{}\t{}\n", intended, query.to_string());
        api.pr_search(max_num_of_results, query)
            .iter()
            .for_each(|result| println!("{}\t{}", result.variant, result.pr,));
        println!("\n----\n");
    });
}

fn generated_apple_dict() {
    match parse_dict(io::stdin()) {
        Err(err) => {
            println!("error reading csv file: {}", err);
            process::exit(1);
        }
        Ok(dict) => {
            print!("{}", rich_dict_to_xml(enrich_dict(&dict)));
        }
    }
}

fn generate_html() {
    use std::fs;
    let api = Api::new(APP_TMP_DIR);
    let test_id = 80623; // test entry id for 麪包
    let html = format!(
        "<html>\n<head>\n<meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">
<link rel=\"stylesheet\" href=\"test_entry.css\"></head>\n<body>\n{}\n</body>\n</html>\n",
        api.get_entry_html(test_id)
    );
    fs::write(ENTRY_HTML_TEST_PATH, html).expect("Unable to write test entry HTML");
}
