use lazy_static::lazy_static;
use std::collections::HashMap;

lazy_static! {
    pub static ref LIHKG_FREQUENCIES: HashMap<String, usize> = {
        let m: HashMap<String, usize> = HashMap::new();
        include_str!("lihkg_frequencies.tsv")
            .lines()
            .map(|line| {
                let mut iter = line.split('\t');
                let pr = iter.next().unwrap();
                let freq = iter.next().unwrap().parse::<usize>().unwrap();
                (pr.to_string(), freq)
            })
            .collect()
    };
}
