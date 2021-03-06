use super::unicode;
use itertools::Itertools;
use lazy_static::lazy_static;
use regex::Regex;
use serde::Deserialize;
use serde::Serialize;
use std::fmt;
use std::ops::Range;
use std::str::FromStr;
use std::collections::HashMap;
use csv;

#[derive(Copy, Clone, Debug)]
pub enum Romanization {
    Jyutping,
    YaleNumbers,
    YaleDiacritics,
    CantonesePinyin,
    Guangdong,
    SidneyLau,
    Ipa,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LaxJyutPings(pub Vec<LaxJyutPing>);
impl fmt::Display for LaxJyutPings {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		write!(f, "{}", self.0.iter().map(|pr| pr.to_string()).join(", "))
	}
}

pub type JyutPings = Vec<JyutPing>;

/// JyutPing encoding with initial, nucleus, coda, and tone
///
/// Phonetics info based on: <https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.148.6501&rep=rep1&type=pdf>
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct JyutPing {
	#[serde(rename = "i")]
	pub initial: Option<JyutPingInitial>,

	#[serde(rename = "n")]
	pub nucleus: Option<JyutPingNucleus>,

	#[serde(rename = "c")]
	pub coda: Option<JyutPingCoda>,

	#[serde(rename = "t")]
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

	pub fn to_jyutpings(&self) -> Option<JyutPings> {
		let mut jyutpings = vec![];
		for seg in &self.0 {
			match seg {
				LaxJyutPingSegment::Standard(jyutping) => {
					jyutpings.push(jyutping.clone());
				},
				LaxJyutPingSegment::Nonstandard(_) => {
					return None;
				}
			}
		}
		Some(jyutpings)
	}
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum LaxJyutPingSegment {
	#[serde(rename = "S")]
	Standard(JyutPing),

	#[serde(rename = "N")]
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
				Some(pr) => LaxJyutPingSegment::Standard(pr),
				None => LaxJyutPingSegment::Nonstandard(pr_seg.to_string()),
			})
			.collect(),
	)
}

pub fn parse_jyutpings(str: &str) -> Option<JyutPings> {
	let mut jyutpings = vec![];
	for pr_seg in str.split_whitespace() {
		match parse_jyutping(pr_seg) {
			Some(jyutping) => {
				jyutpings.push(jyutping);
			},
			None => {
				return None;
			},
		}
	}
	Some(jyutpings)
}

pub fn parse_continuous_prs(str: &str, romanization: Romanization) -> Option<Vec<JyutPings>> {
	if let Some(last_char) = str.chars().next_back() {
		if last_char.is_digit(10) {
			let jyutping = lookup_romanization_with_tone(str, romanization);

			match jyutping {
				Some(jyutping) => {
					return Some(vec![vec![jyutping]]);
				},
				None => {
					return None;
				}
			}
		}
	}
	let results = parse_continuous_prs_helper(str, romanization);
	if results.len() == 0 {
		None
	} else {
		Some(results)
	}
}

fn parse_continuous_prs_helper(str: &str, romanization: Romanization) -> Vec<JyutPings> {
	use Romanization::*;

	let standard_check = match romanization {
		Jyutping => is_standard_jyutping_optional_tone,
		YaleNumbers => is_standard_yale_with_numbers_optional_tone,
		CantonesePinyin => is_standard_cantonese_pinyin_optional_tone,
		SidneyLau => is_standard_sidney_lau_optional_tone,
		_ => panic!("Unsupported romanization {:?} in parse_continuous_prs_helper()", romanization),
	};
	let mut jyutpings = vec![];
	let mut i = 1;
	while i <= str.len() {
		let head_word = &str[0..i];
		if standard_check(head_word) {
			if let Some(head_jyutping) = convert_to_jyutping(head_word, romanization) {
				if i == str.len() {
					jyutpings.push(vec![head_jyutping]);
				} else {
					let sub_jyutpings = parse_continuous_prs_helper(&str[i..], romanization);
					for sub_jyutping in sub_jyutpings {
						let mut sub_jyutping: JyutPings = sub_jyutping.clone();
						sub_jyutping.insert(0, head_jyutping.clone());
						jyutpings.push(sub_jyutping);
					}
				}
			}
		}
		i += 1;
	}
	jyutpings
}

fn lookup_romanization_with_tone(s: &str, romanization: Romanization) -> Option<JyutPing> {
	use Romanization::*;

	let jyutping_index = match romanization {
		YaleNumbers => ROMANIZATION_MAPS.yale_numbers_to_jyutping.get(s),
		CantonesePinyin => ROMANIZATION_MAPS.cantonese_pinyin_to_jyutping.get(s),
		SidneyLau => ROMANIZATION_MAPS.sidney_lau_to_jyutping.get(s),
		Jyutping => {
			return parse_jyutping(s);
		},
		_ => panic!("Unsupported romanization {:?} in convert_to_jyutping()", romanization)
	};

	match jyutping_index {
		Some(jyutping_index) => {
			let jyutping = ROMANIZATION_MAPS.jyutpings[*jyutping_index].clone();
			Some(jyutping)
		},
		None => {
			None
		}
	}
}

pub fn convert_to_jyutping(s: &str, romanization: Romanization) -> Option<JyutPing> {
	let jyutping = lookup_romanization_with_tone(s, romanization);

	match jyutping {
		Some(_) => {
			jyutping
		},
		None => {
			use Romanization::*;
			let jyutping_index = match romanization {
				YaleNumbers => ROMANIZATION_MAPS.yale_numbers_without_tone_to_jyutping.get(s),
				CantonesePinyin => ROMANIZATION_MAPS.cantonese_pinyin_without_tone_to_jyutping.get(s),
				SidneyLau => ROMANIZATION_MAPS.sidney_lau_without_tone_to_jyutping.get(s),
				Jyutping => panic!("Should never reach this line. Jyutping without tone should be handled in parse_jyutping() already"),
				_ => panic!("Unsupported romanization {:?} in convert_to_jyutping()", romanization)
			};
			match jyutping_index {
				Some(jyutping_index) => {
					let mut jyutping = ROMANIZATION_MAPS.jyutpings[*jyutping_index].clone();
					jyutping.tone = None;
					Some(jyutping)
				},
				None => {
					None
				}
			}
		}
    }
}

pub fn convert_to_jyutpings(s: &str, romanization: Romanization) -> Option<Vec<JyutPings>> {
	let jyutping_segments_re = Regex::new(r"^\s*([a-z]+[0-9]?\s*)+$").unwrap();
	if !jyutping_segments_re.is_match(s) {
		return None;
	}
	let jyutping_segment_re = Regex::new(r"[a-z]+[0-9]?").unwrap();
    let mut jyutping_possibilities: Vec<JyutPings> = vec![];
	for seg in jyutping_segment_re.find_iter(s) {
		match parse_continuous_prs(seg.as_str(), romanization) {
			Some(jyutpings) => {
				if jyutping_possibilities.len() == 0 {
					jyutping_possibilities = jyutpings;
				} else {
					jyutping_possibilities = jyutping_possibilities.iter().flat_map(|possibility| {
						jyutpings.iter().map(move |jyutping| {
							let mut result = possibility.clone();
							result.extend(jyutping.clone());
							result
						})
					}).collect();
				}
				continue;
			},
			None => {
				return None;
			}
		}
	}
	Some(jyutping_possibilities)
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
	let tone: Option<JyutPingTone> = parse_jyutping_tone(start, str).map(|(_tone, _start)| {
		start = _start;
		_tone
	});

	// part of the str is not matched
	if start < str.len() {
		None
	} else {
		Some(JyutPing {
			initial,
			nucleus,
			coda,
			tone,
		})
	}
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

fn parse_jyutping_tone(start: usize, str: &str) -> Option<(JyutPingTone, usize)> {
	// println!("{} {} {}", str, start, str.len());
	get_slice(str, start..str.len()).and_then(|substr| match JyutPingTone::from_str(substr) {
		Ok(tone) => Some((tone, start + 1)),
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

lazy_static! {
	static ref JYUTPING_WITHOUT_TONE_REGEX: &'static str = r"^(b|p|m|f|d|t|n|l|g|k|ng|h|gw|kw|w|z|c|s|j)?(i|ip|it|ik|im|in|ing|iu|yu|yut|yun|u|ut|uk|um|un|ung|ui|e|ep|et|ek|em|en|eng|ei|eu|eot|eon|eoi|oe|oet|oek|oeng|o|ot|ok|on|ong|oi|ou|op|om|a|ap|at|ak|am|an|ang|ai|au|aa|aap|aat|aak|aam|aan|aang|aai|aau|m|ng)";
	static ref YALE_WITHOUT_TONE_REGEX: &'static str = r"^(b|p|m|f|d|t|n|l|g|k|ng|h|gw|kw|w|j|ch|s|y)?(i|ip|it|ik|im|in|ing|iu|yu|yut|yun|u|ut|uk|um|un|ung|ui|e|ep|et|ek|em|en|eng|ei|eeu|eui|eu|eut|euk|eun|eung|o|ot|ok|on|ong|oi|ou|op|om|a|ap|at|ak|am|an|ang|ai|au|a|aap|aat|aak|aam|aan|aang|aai|aau|m|ng)";
	static ref CANTONESE_PINYIN_WITHOUT_TONE_REGEX: &'static str = r"^(b|p|m|f|d|t|n|l|g|k|ng|h|gw|kw|w|dz|ts|s|j)?(i|ip|it|ik|im|in|ing|iu|y|yt|yn|u|ut|uk|um|un|ung|ui|e|ep|et|ek|em|en|eng|ei|eu|eot|eon|eoy|oe|oet|oek|oeng|o|ot|ok|on|ong|oi|ou|op|om|a|ap|at|ak|am|an|ang|ai|au|aa|aap|aat|aak|aam|aan|aang|aai|aau|m|ng)";
	static ref SIDNEY_LAU_WITHOUT_TONE_REGEX: &'static str = r"^(b|p|m|f|d|t|n|l|g|k|ng|h|gw|kw|w|j|ch|s|y)?(i|ip|it|ik|im|in|ing|iu|ue|uet|uen|oo|oot|uk|um|oon|ung|ooi|e|ep|et|ek|em|en|eng|ei|euh|ui|eu|ut|euk|un|eung|o|ot|ok|on|ong|oh|oi|ou|op|om|a|ap|at|ak|am|an|ang|ai|au|a|aap|aat|aak|aam|aan|aang|aai|aau|m|ng)";
}

// Source: lib/cantonese.py:is_valid_jyutping_form
fn is_standard_jyutping(s: &str) -> bool {
	lazy_static! {
		static ref RE: Regex = Regex::new(&(JYUTPING_WITHOUT_TONE_REGEX.to_owned() + r"[1-6]$")).unwrap();
	}
	RE.is_match(s)
}

fn is_standard_jyutping_optional_tone(s: &str) -> bool {
	lazy_static! {
		static ref RE: Regex = Regex::new(&(JYUTPING_WITHOUT_TONE_REGEX.to_owned() + r"[1-6]?$")).unwrap();
	}
	RE.is_match(s)
}

// source: https://jyutping.org/blog/table/
fn is_standard_yale_with_numbers(s: &str) -> bool {
	lazy_static! {
		static ref RE: Regex = Regex::new(&(YALE_WITHOUT_TONE_REGEX.to_owned() + r"[1-6]$")).unwrap();
	}
	RE.is_match(s)
}

fn is_standard_yale_with_numbers_optional_tone(s: &str) -> bool {
	lazy_static! {
		static ref RE: Regex = Regex::new(&(YALE_WITHOUT_TONE_REGEX.to_owned() + r"[1-6]?$")).unwrap();
	}
	RE.is_match(s)
}

// source: https://jyutping.org/blog/table/
// We are ignoring diacritics here
// fn is_standard_yale_with_diacritics(s: &str) -> bool {
// 	lazy_static! {
// 		static ref RE: Regex = Regex::new(r"^(b|p|m|f|d|t|n|l|g|k|ng|h|gw|kw|w|j|ch|s|y)?(i|ih|ip|ihp|it|iht|ik|ihk|im|ihm|in|ihn|ing|ihng|iu|iuhng|yu|yuh|yut|yuht|yun|yuhn|u|uh|ut|uht|uk|uhk|um|uhm|un|uhn|ung|uhng|ui|uih|e|eh|ep|ehp|et|eht|ek|ehk|em|ehm|en|ehn|eng|ehng|ei|eih|eeu|eeuh|eui|euih|eu|euh|eut|euht|euk|euhk|eun|euhn|eung|euhng|o|oh|ot|oht|ok|ohk|on|ohn|ong|ohng|oi|oih|ou|ouh|op|ohp|om|ohm|a|ah|ap|ahp|at|aht|ak|ahk|am|ahm|an|ahn|ang|ahng|ai|aih|au|auh|aap|aahp|aat|aaht|aak|aahk|aam|aahm|aan|aahn|aang|aahng|aai|aaih|aau|aauh|m|mh|ng|ngh)$").unwrap();
// 	}

// 	RE.is_match(&remove_diacritics(&unicode::normalize(&s), find_yale_diacritics))
// }

// fn remove_diacritics(s: &str, find_diacritics: fn(char) -> char) -> String {
// 	let chars = s.chars();
//     chars.fold("".to_string(), |acc, c| acc + &find_diacritics(c).to_string())
// }

// fn find_yale_diacritics(c: char) -> char {
// 	match c {
// 		'??'|'??'|'??' => 'a',
// 		'??'|'??'|'??' => 'e',
// 		'??'|'??'|'??' => 'i',
// 		'??'|'??'|'??' => 'o',
// 		'??'|'??'|'??' => 'u',
// 		_ => c
// 	}
// }

// Source: https://jyutping.org/blog/table/
fn is_standard_cantonese_pinyin(s: &str) -> bool {
	lazy_static! {
		static ref RE: Regex = Regex::new(&(CANTONESE_PINYIN_WITHOUT_TONE_REGEX.to_owned() + r"[1-9]$")).unwrap();
	}
	RE.is_match(s)
}

fn is_standard_cantonese_pinyin_optional_tone(s: &str) -> bool {
	lazy_static! {
		static ref RE: Regex = Regex::new(&(CANTONESE_PINYIN_WITHOUT_TONE_REGEX.to_owned() + r"[1-9]?$")).unwrap();
	}
	RE.is_match(s)
}

// Source: https://jyutping.org/blog/table/
fn is_standard_sidney_lau(s: &str) -> bool {
	lazy_static! {
		static ref RE: Regex = Regex::new(&(SIDNEY_LAU_WITHOUT_TONE_REGEX.to_owned() + r"[1-6]$")).unwrap();
	}
	RE.is_match(s)
}

fn is_standard_sidney_lau_optional_tone(s: &str) -> bool {
	lazy_static! {
		static ref RE: Regex = Regex::new(&(SIDNEY_LAU_WITHOUT_TONE_REGEX.to_owned() + r"[1-6]?$")).unwrap();
	}
	RE.is_match(s)
}

// Source: zidin/definition.py:looks_like_jyutping
pub fn looks_like_pr(s: &str, romanization: Romanization) -> bool {
	use Romanization::*;
	let segs = s.split_whitespace();
	let standard_check = match romanization {
		Jyutping => is_standard_jyutping,
		YaleNumbers => is_standard_yale_with_numbers,
		CantonesePinyin => is_standard_cantonese_pinyin,
		SidneyLau => is_standard_sidney_lau,
		_ => panic!("Unsupported romanization {:?} in looks_like_pr()", romanization),
	};
	let similarity_score: u32 = segs
		.clone()
		.map(|seg| {
			let mut cleaned_seg = seg.to_string();
			cleaned_seg.retain(|c| !unicode::is_english_punc(c));
			if standard_check(&cleaned_seg) {
				1
			} else {
				0
			}
		})
		.sum();
	(similarity_score as f64 / segs.count() as f64) > 0.7
}

#[derive(Default)]
struct RomanizationMaps {
    pub yale_numbers_to_jyutping: RomanizationMap,
    pub yale_numbers_without_tone_to_jyutping: RomanizationMap,
    pub cantonese_pinyin_to_jyutping: RomanizationMap,
    pub cantonese_pinyin_without_tone_to_jyutping: RomanizationMap,
    pub sidney_lau_to_jyutping: RomanizationMap,
    pub sidney_lau_without_tone_to_jyutping: RomanizationMap,

    pub jyutpings: Vec<JyutPing>
}

type RomanizationMap = HashMap<String, usize>;

lazy_static! {
    static ref ROMANIZATION_MAPS: RomanizationMaps = {
        let tsv = include_str!("cantonese_romanizations.tsv");
        let mut rdr = csv::ReaderBuilder::new().delimiter(b'\t').from_reader(tsv.as_bytes());
        let mut maps: RomanizationMaps = RomanizationMaps::default();
        let mut i = 0;
        for result in rdr.records() {
            let entry = result.unwrap();
            maps.jyutpings.push(parse_jyutping(&entry[0]).unwrap());
            maps.yale_numbers_to_jyutping.insert(entry[1].to_string(), i);
            maps.yale_numbers_without_tone_to_jyutping.insert(unicode::remove_last_char(&entry[1]), i);
            maps.cantonese_pinyin_to_jyutping.insert(entry[2].to_string(), i);
            maps.cantonese_pinyin_without_tone_to_jyutping.insert(unicode::remove_last_char(&entry[2]), i);
            maps.sidney_lau_to_jyutping.insert(entry[3].to_string(), i);
            maps.sidney_lau_without_tone_to_jyutping.insert(unicode::remove_last_char(&entry[3]), i);

            i += 1;
        }
        maps
    };
}
