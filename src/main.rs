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

fn main() {
    do_pr_search();
}

// fn do_variant_search() {
//     let api = Api::new(APP_TMP_DIR);
//     // 說 is a variant that does not appear in dictionary variant data
//     // Two words contains 說明: 說明 and 説明書
//     let mut results = api::variant_search("說明");
//     let mut i = 0;
//     while results.len() > 0 && i < 10 {
//         let VariantSearchResult {
//             id, variant_index, ..
//         } = results.pop().unwrap();
//         let entry = dict.get(&id).unwrap();
//         let variant = &entry.variants.0[variant_index];
//         println!("{} {}", variant.word, &variant.prs.to_string());
//         i += 1;
//     }
// }

fn do_pr_search() {
    let api = Api::new(APP_TMP_DIR);
    let queries = vec![
        (
            "麪包",
            LaxJyutPing(vec![
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
            ]),
        ),
        (
            "學生",
            LaxJyutPing(vec![
                LaxJyutPingSegment::Standard(JyutPing {
                    initial: Some(JyutPingInitial::H),
                    nucleus: JyutPingNucleus::O,
                    coda: Some(JyutPingCoda::K),
                    tone: Some(JyutPingTone::T6),
                }),
                LaxJyutPingSegment::Standard(JyutPing {
                    initial: Some(JyutPingInitial::S),
                    nucleus: JyutPingNucleus::Aa,
                    coda: Some(JyutPingCoda::Ng),
                    tone: None,
                }),
            ]),
        ),
    ];
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
