use std::process;
use wordshk_tools::emit::prs_to_string;
use wordshk_tools::emit_apple_dict::dict_to_xml;
use wordshk_tools::parse::{
    parse_dict, JyutPing, JyutPingCoda, JyutPingInitial, JyutPingNucleus, LaxJyutPingSegment,
};
use wordshk_tools::search::{pr_search, variant_search, PrSearchResult, VariantSearchResult};

fn main() {
    do_variant_search();
}

fn do_variant_search() {
    match parse_dict() {
        Err(err) => {
            println!("error reading csv file: {}", err);
            process::exit(1);
        }
        Ok(dict) => {
            // 說 is a variant that does not appear in dictionary variant data
            // Two words contains 說明: 說明 and 説明書
            let mut results = variant_search(&dict, "說明");
            let mut i = 0;
            while results.len() > 0 && i < 10 {
                let VariantSearchResult {
                    id, variant_index, ..
                } = results.pop().unwrap();
                let entry = dict.get(&id).unwrap();
                let variant = &entry.variants[variant_index];
                println!("{} {}", variant.word, prs_to_string(&variant.prs));
                i += 1;
            }
        }
    }
}

fn do_pr_search() {
    match parse_dict() {
        Err(err) => {
            println!("error reading csv file: {}", err);
            process::exit(1);
        }
        Ok(dict) => {
            // test 麪包 jyutping search with slight typos
            let mut results = pr_search(
                &dict,
                &vec![
                    LaxJyutPingSegment::Standard(JyutPing {
                        initial: Some(JyutPingInitial::M),
                        nucleus: JyutPingNucleus::I,
                        coda: Some(JyutPingCoda::Ng),
                        tone: None,
                    }),
                    LaxJyutPingSegment::Standard(JyutPing {
                        initial: Some(JyutPingInitial::B),
                        nucleus: JyutPingNucleus::Aa,
                        coda: Some(JyutPingCoda::U),
                        tone: None,
                    }),
                ],
            );
            let mut i = 0;
            while results.len() > 0 && i < 10 {
                let PrSearchResult {
                    id, variant_index, ..
                } = results.pop().unwrap();
                let entry = dict.get(&id).unwrap();
                let variant = &entry.variants[variant_index];
                println!("{} {}", variant.word, prs_to_string(&variant.prs));
                i += 1;
            }
        }
    }
}

fn generated_apple_dict() {
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
