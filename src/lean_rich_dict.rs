use super::rich_dict::{RichDef, RichEntry};
use serde::Deserialize;
use serde::Serialize;

#[derive(Debug, PartialEq, Clone, Serialize, Deserialize)]
pub struct LeanRichEntry {
	pub id: usize,
	pub variants: Vec<LeanVariant>,
	pub variants_simp: Vec<String>,
	pub poses: Vec<String>,
	pub labels: Vec<String>,
	pub sims: Vec<String>,
	pub ants: Vec<String>,
	pub defs: Vec<RichDef>,
	pub published: bool,
}

/// A variant of a \[word\] with \[prs\] (pronounciations)
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LeanVariant {
	pub word: String,
	pub prs: String,
}

pub fn to_lean_rich_entry(entry: &RichEntry) -> LeanRichEntry {
	LeanRichEntry {
		id: entry.id,
		variants: entry
			.variants
			.0
			.iter()
			.map(|variant| LeanVariant {
				word: variant.word.clone(),
				prs: variant.prs.to_string(),
			})
			.collect(),
		variants_simp: entry.variants_simp.clone(),
		poses: entry.poses.clone(),
		labels: entry.labels.clone(),
		sims: entry.sims.clone(),
		ants: entry.ants.clone(),
		defs: entry.defs.clone(),
		published: entry.published,
	}
}
