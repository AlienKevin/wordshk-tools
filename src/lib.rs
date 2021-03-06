#[macro_use]
mod tests;
pub mod app_api;
pub mod dict;
pub mod emit_apple_dict;
pub mod emit_html;
pub mod english_index;
mod hk_variant_map_safe;
#[path = "../data/simp_trad_conversions/iconic_simps.rs"]
mod iconic_simps;
#[path = "../data/simp_trad_conversions/iconic_trads.rs"]
mod iconic_trads;
pub mod jyutping;
pub mod lean_rich_dict;
pub mod parse;
pub mod rich_dict;
pub mod search;
pub mod unicode;
mod variant_to_us_english;
mod word_frequencies;
