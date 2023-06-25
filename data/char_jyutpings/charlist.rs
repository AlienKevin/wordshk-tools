use indexmap::set::IndexSet;
use lazy_static::lazy_static;
use std::collections::HashMap;

pub type CharList = HashMap<char, IndexSet<String>>;

lazy_static! {
    pub static ref CHARLIST: CharList = {
        let list: HashMap<char, HashMap<String, usize>> =
            serde_json::from_str(include_str!("charlist_processed.json")).unwrap();
        list.iter()
            .map(|(c, pr_frequencies)| {
                let mut freqs = pr_frequencies.iter().collect::<Vec<(&String, &usize)>>();
                freqs.sort_by(|a, b| b.1.cmp(a.1));
                (*c, freqs.iter().map(|(pr, _)| pr.to_string()).collect())
            })
            .collect()
    };
}
