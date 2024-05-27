use itertools::Itertools;
use lazy_static::lazy_static;
use std::collections::HashMap;

use crate::{
    dict::EntryId,
    rich_dict::{MandarinVariant, MandarinVariants},
};

lazy_static! {
    pub static ref MANDARIN_VARIANTS: HashMap<EntryId, MandarinVariants> = {
        let mut variants_map: HashMap<EntryId, MandarinVariants> = HashMap::new();
        include_str!("mandarin_variants.tsv")
            .lines()
            .skip(1) // skip the header line
            .for_each(|line| {
                let mut iter = line.split('\t');
                let entry_id = iter.next().unwrap().parse::<EntryId>().unwrap();
                let def_index = iter.next().unwrap().parse::<usize>().unwrap();
                let _variants = iter.next().unwrap();
                let mandarin_variants = iter
                    .next()
                    .unwrap()
                    .split('/')
                    .map(|variant| variant.to_string())
                    .unique()
                    .collect::<Vec<String>>();
                if let Some(variants) = variants_map.get_mut(&entry_id) {
                    for variant in &mut variants.0 {
                        for new_variant in &mandarin_variants {
                            if new_variant == &variant.word_simp {
                                variant.def_indices.push(def_index);
                            }
                        }
                    }
                } else {
                    let mut variants = vec![];
                    for variant in mandarin_variants {
                        variants.push(MandarinVariant {
                            word_simp: variant,
                            def_indices: vec![def_index],
                        });
                    }
                    variants_map.insert(entry_id, MandarinVariants(variants));
                }
            });
        variants_map
    };
}
