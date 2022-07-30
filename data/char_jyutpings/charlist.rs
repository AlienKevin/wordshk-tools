use lazy_static::lazy_static;
use std::collections::HashMap;

pub type CharList = HashMap<char, HashMap<String, usize>>;

lazy_static! {
	pub static ref CHARLIST: CharList =
		serde_json::from_str(include_str!("charlist_processed.json")).unwrap();
}
