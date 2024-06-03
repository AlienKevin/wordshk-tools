use lazy_static::lazy_static;
use std::collections::HashMap;

lazy_static! {
    pub static ref HONG_KONG_VARIANT_MAP_SAFE: HashMap<char, char> = {
        include_str!("hk_variant_map_safe.tsv")
            .lines()
            .map(|line| {
                let mut iter = line.split('\t');
                let other_variant = iter.next().unwrap();
                assert_eq!(other_variant.chars().count(), 1);
                let safe_variant = iter.next().unwrap();
                assert_eq!(safe_variant.chars().count(), 1);
                (
                    other_variant.chars().next().unwrap(),
                    safe_variant.chars().next().unwrap(),
                )
            })
            .collect()
    };
}
