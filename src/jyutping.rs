use super::unicode;
use itertools::Itertools;
use lazy_static::lazy_static;
use regex::Regex;
use serde::Deserialize;
use serde::Serialize;
use std::fmt;
use std::ops::Range;
use std::str::FromStr;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LaxJyutPings(pub Vec<LaxJyutPing>);
impl fmt::Display for LaxJyutPings {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		write!(f, "{}", self.0.iter().map(|pr| pr.to_string()).join(", "))
	}
}

/// JyutPing encoding with initial, nucleus, coda, and tone
///
/// Phonetics info based on: <https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.148.6501&rep=rep1&type=pdf>
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct JyutPing {
	pub initial: Option<JyutPingInitial>,
	pub nucleus: Option<JyutPingNucleus>,
	pub coda: Option<JyutPingCoda>,
	pub tone: Option<JyutPingTone>,
}

impl fmt::Display for JyutPing {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		write!(f, "{}", {
			self.initial
				.as_ref()
				.map(|i| i.to_string())
				.unwrap_or("".to_string())
				+ &self
					.nucleus
					.as_ref()
					.map(|i| i.to_string())
					.unwrap_or("".to_string())
				+ &self
					.coda
					.as_ref()
					.map(|i| i.to_string())
					.unwrap_or("".to_string())
				+ &self
					.tone
					.as_ref()
					.map(|i| i.to_string())
					.unwrap_or("".to_string())
		})
	}
}

impl JyutPing {
	pub fn is_empty(&self) -> bool {
		self.initial.is_none()
			&& self.nucleus.is_none()
			&& self.coda.is_none()
			&& self.tone.is_none()
	}
	pub fn to_string_without_tone(&self) -> String {
		self.initial
			.as_ref()
			.map(|i| i.to_string())
			.unwrap_or("".to_string())
			+ &self
				.nucleus
				.as_ref()
				.map(|i| i.to_string())
				.unwrap_or("".to_string())
			+ &self
				.coda
				.as_ref()
				.map(|i| i.to_string())
				.unwrap_or("".to_string())
	}
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LaxJyutPing(pub Vec<LaxJyutPingSegment>);

impl fmt::Display for LaxJyutPing {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		write!(f, "{}", self.0.iter().map(|seg| seg.to_string()).join(" "))
	}
}

impl LaxJyutPing {
	pub fn to_string_without_tone(&self) -> String {
		self.0
			.iter()
			.map(|seg| seg.to_string_without_tone())
			.collect::<Vec<String>>()
			.join(" ")
	}
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum LaxJyutPingSegment {
	Standard(JyutPing),
	Nonstandard(String),
}

impl fmt::Display for LaxJyutPingSegment {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		write!(
			f,
			"{}",
			match self {
				LaxJyutPingSegment::Standard(pr) => pr.to_string(),
				LaxJyutPingSegment::Nonstandard(pr_str) => pr_str.clone(),
			}
		)
	}
}

impl LaxJyutPingSegment {
	pub fn to_string_without_tone(&self) -> String {
		match self {
			LaxJyutPingSegment::Standard(pr) => pr.to_string_without_tone(),
			LaxJyutPingSegment::Nonstandard(pr_str) => pr_str.clone(),
		}
	}
}

/// Initial segment of a JyutPing, optional
///
/// Eg: 's' in "sap6"
///
#[derive(strum::EnumString, strum::Display, Debug, Clone, PartialEq, Serialize, Deserialize)]
#[strum(ascii_case_insensitive)]
#[strum(serialize_all = "lowercase")]
pub enum JyutPingInitial {
	B,
	P,
	M,
	F,
	D,
	T,
	N,
	L,
	G,
	K,
	Ng,
	H,
	Gw,
	Kw,
	W,
	Z,
	C,
	S,
	J,
}

/// Nucleus segment of a Jyutping, not required in case of /ng/ and /m/
///
/// Eg: 'a' in "sap6"
///
#[derive(strum::EnumString, strum::Display, Debug, Clone, PartialEq, Serialize, Deserialize)]
#[strum(ascii_case_insensitive)]
#[strum(serialize_all = "lowercase")]
pub enum JyutPingNucleus {
	Aa,
	I,
	U,
	E,
	O,
	Yu,
	Oe,
	A,
	Eo,
}

/// Coda segment of a Jyutping, optional
///
/// Eg: 'p' in "sap6"
///
#[derive(strum::EnumString, strum::Display, Debug, Clone, PartialEq, Serialize, Deserialize)]
#[strum(ascii_case_insensitive)]
#[strum(serialize_all = "lowercase")]
pub enum JyutPingCoda {
	P,
	T,
	K, // stop
	M,
	N,
	Ng, // nasal
	I,
	U, // vowel
}

/// Tone segment of a Jyutping, optional.
/// Six tones from 1 to 6.
///
/// Eg: '6' in "sap6"
///
#[derive(strum::EnumString, strum::Display, Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum JyutPingTone {
	#[strum(serialize = "1")]
	T1,
	#[strum(serialize = "2")]
	T2,
	#[strum(serialize = "3")]
	T3,
	#[strum(serialize = "4")]
	T4,
	#[strum(serialize = "5")]
	T5,
	#[strum(serialize = "6")]
	T6,
}

/// Parse [LaxJyutPing] pronunciation
pub fn parse_pr(str: &str) -> LaxJyutPing {
	LaxJyutPing(
		str.split_whitespace()
			.map(|pr_seg| match parse_jyutping(pr_seg) {
				Some(pr) => {
					if pr.is_empty() {
						LaxJyutPingSegment::Nonstandard(pr_seg.to_string())
					} else {
						LaxJyutPingSegment::Standard(pr)
					}
				}
				None => LaxJyutPingSegment::Nonstandard(pr_seg.to_string()),
			})
			.collect(),
	)
}

/// Parse [JyutPing] pronunciation
pub fn parse_jyutping(str: &str) -> Option<JyutPing> {
	let mut start = 0;

	let initial: Option<JyutPingInitial> = parse_jyutping_initial(str).map(|(_initial, _start)| {
		start = _start;
		_initial
	});

	let nucleus: Option<JyutPingNucleus> =
		parse_jyutping_nucleus(start, str).map(|(_nucleus, _start)| {
			start = _start;
			_nucleus
		});

	let coda: Option<JyutPingCoda> = parse_jyutping_coda(start, str).map(|(_coda, _start)| {
		start = _start;
		_coda
	});
	let tone: Option<JyutPingTone> = parse_jyutping_tone(start, str);

	Some(JyutPing {
		initial,
		nucleus,
		coda,
		tone,
	})
}

fn parse_jyutping_component<T: FromStr>(start: usize, str: &str) -> Option<(T, usize)> {
	get_slice(str, start..start + 2)
		.and_then(|first_two| match T::from_str(first_two) {
			Ok(component) => Some((component, start + 2)),
			Err(_) => get_slice(str, start..start + 1).and_then(|first_one| {
				match T::from_str(first_one) {
					Ok(component) => Some((component, start + 1)),
					Err(_) => None,
				}
			}),
		})
		.or(
			get_slice(str, start..start + 1).and_then(|first_one| match T::from_str(first_one) {
				Ok(component) => Some((component, start + 1)),
				Err(_) => None,
			}),
		)
}

fn parse_jyutping_initial(str: &str) -> Option<(JyutPingInitial, usize)> {
	parse_jyutping_component::<JyutPingInitial>(0, str)
}

fn parse_jyutping_nucleus(start: usize, str: &str) -> Option<(JyutPingNucleus, usize)> {
	parse_jyutping_component::<JyutPingNucleus>(start, str)
}

fn parse_jyutping_coda(start: usize, str: &str) -> Option<(JyutPingCoda, usize)> {
	parse_jyutping_component::<JyutPingCoda>(start, str)
}

fn parse_jyutping_tone(start: usize, str: &str) -> Option<JyutPingTone> {
	// println!("{} {} {}", str, start, str.len());
	get_slice(str, start..str.len()).and_then(|substr| match JyutPingTone::from_str(substr) {
		Ok(tone) => Some(tone),
		Err(_) => None,
	})
}

fn get_slice(s: &str, range: Range<usize>) -> Option<&str> {
	if s.len() > range.start && s.len() >= range.end {
		Some(&s[range])
	} else {
		None
	}
}

// Source: lib/cantonese.py:is_valid_jyutping_form
fn is_standard_jyutping(s: &str) -> bool {
	lazy_static! {
		static ref RE: Regex = Regex::new(r"^(b|p|m|f|d|t|n|l|g|k|ng|h|gw|kw|w|z|c|s|j)?(i|ip|it|ik|im|in|ing|iu|yu|yut|yun|u|up|ut|uk|um|un|ung|ui|e|ep|et|ek|em|en|eng|ei|eu|eot|eon|eoi|oe|oet|oek|oeng|o|ot|ok|on|ong|oi|ou|op|om|a|ap|at|ak|am|an|ang|ai|au|aa|aap|aat|aak|aam|aan|aang|aai|aau|m|ng)[1-6]$").unwrap();
	}
	RE.is_match(s)
}

// Source: zidin/definition.py:looks_like_jyutping
pub fn looks_like_pr(s: &str) -> bool {
	let segs = s.split_whitespace();
	let similarity_score: u32 = segs
		.clone()
		.map(|seg| {
			let mut cleaned_seg = seg.to_string();
			cleaned_seg.retain(|c| !unicode::is_english_punc(c));
			if is_standard_jyutping(&cleaned_seg) {
				1
			} else {
				0
			}
		})
		.sum();
	(similarity_score as f64 / segs.count() as f64) > 0.7
}
