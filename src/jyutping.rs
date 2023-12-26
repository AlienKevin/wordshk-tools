use itertools::Itertools;
use lazy_static::lazy_static;
use regex::Regex;
use serde::Deserialize;
use serde::Serialize;
use std::fmt;
use std::ops::Range;
use std::str::FromStr;

use crate::unicode;

#[derive(Copy, Clone, Debug)]
pub enum Romanization {
    Jyutping,
    Yale,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, rkyv::Archive, rkyv::Deserialize, rkyv::Serialize)]
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
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, rkyv::Archive, rkyv::Deserialize, rkyv::Serialize)]
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

    /// Convert to Yale romanization without diacritics
    /// ```
    /// use wordshk_tools::jyutping::{parse_jyutping, JyutPing};
    /// assert_eq!(parse_jyutping("jyut1").unwrap().to_yale_no_diacritics(), "yut");
    /// assert_eq!(parse_jyutping("jyut2").unwrap().to_yale_no_diacritics(), "yut");
    /// assert_eq!(parse_jyutping("jyut3").unwrap().to_yale_no_diacritics(), "yut");
    /// assert_eq!(parse_jyutping("jyut4").unwrap().to_yale_no_diacritics(), "yuht");
    /// assert_eq!(parse_jyutping("jyut5").unwrap().to_yale_no_diacritics(), "yuht");
    /// assert_eq!(parse_jyutping("jyut6").unwrap().to_yale_no_diacritics(), "yuht");
    ///
    /// assert_eq!(parse_jyutping("m4").unwrap().to_yale_no_diacritics(), "m");
    /// assert_eq!(parse_jyutping("ng4").unwrap().to_yale_no_diacritics(), "ng");
    ///
    /// assert_eq!(parse_jyutping("hoe1").unwrap().to_yale_no_diacritics(), "heu");
    /// assert_eq!(parse_jyutping("hoe2").unwrap().to_yale_no_diacritics(), "heu");
    /// assert_eq!(parse_jyutping("hoe3").unwrap().to_yale_no_diacritics(), "heu");
    /// assert_eq!(parse_jyutping("hoe4").unwrap().to_yale_no_diacritics(), "heuh");
    /// assert_eq!(parse_jyutping("hoe5").unwrap().to_yale_no_diacritics(), "heuh");
    /// assert_eq!(parse_jyutping("hoe6").unwrap().to_yale_no_diacritics(), "heuh");
    ///
    /// assert_eq!(parse_jyutping("leot1").unwrap().to_yale_no_diacritics(), "leut");
    /// assert_eq!(parse_jyutping("leot2").unwrap().to_yale_no_diacritics(), "leut");
    /// assert_eq!(parse_jyutping("leot3").unwrap().to_yale_no_diacritics(), "leut");
    /// assert_eq!(parse_jyutping("leot4").unwrap().to_yale_no_diacritics(), "leuht");
    /// assert_eq!(parse_jyutping("leot5").unwrap().to_yale_no_diacritics(), "leuht");
    /// assert_eq!(parse_jyutping("leot6").unwrap().to_yale_no_diacritics(), "leuht");
    ///
    /// assert_eq!(parse_jyutping("hoi1").unwrap().to_yale_no_diacritics(), "hoi");
    /// assert_eq!(parse_jyutping("mui5").unwrap().to_yale_no_diacritics(), "muih");
    /// assert_eq!(parse_jyutping("miu1").unwrap().to_yale_no_diacritics(), "miu");
    /// assert_eq!(parse_jyutping("deu6").unwrap().to_yale_no_diacritics(), "deuh");
    ///
    /// assert_eq!(parse_jyutping("deu").unwrap().to_yale_no_diacritics(), "deu");
    /// ```
    pub fn to_yale_no_diacritics(&self) -> String {
        let result = self
            .initial
            .as_ref()
            .map(|i| i.to_yale())
            .unwrap_or("".to_string())
            + &to_yale_rime(self.nucleus, self.coda, self.tone, false);
        result.replace("yy", "y")
    }

    /// Convert to Yale romanization with diacritics
    /// ```
    /// use wordshk_tools::jyutping::{parse_jyutping, JyutPing, JyutPingInitial, JyutPingNucleus, JyutPingCoda, JyutPingTone};
    /// assert_eq!(parse_jyutping("jyut1").unwrap().to_yale(), "yūt");
    /// assert_eq!(parse_jyutping("jyut2").unwrap().to_yale(), "yút");
    /// assert_eq!(parse_jyutping("jyut3").unwrap().to_yale(), "yut");
    /// assert_eq!(parse_jyutping("jyut4").unwrap().to_yale(), "yùht");
    /// assert_eq!(parse_jyutping("jyut5").unwrap().to_yale(), "yúht");
    /// assert_eq!(parse_jyutping("jyut6").unwrap().to_yale(), "yuht");
    ///
    /// assert_eq!(parse_jyutping("m4").unwrap().to_yale(), "m̀");
    /// assert_eq!(parse_jyutping("ng4").unwrap().to_yale(), "ǹg");
    ///
    /// assert_eq!(parse_jyutping("hoe1").unwrap().to_yale(), "hēu");
    /// assert_eq!(parse_jyutping("hoe2").unwrap().to_yale(), "héu");
    /// assert_eq!(parse_jyutping("hoe3").unwrap().to_yale(), "heu");
    /// assert_eq!(parse_jyutping("hoe4").unwrap().to_yale(), "hèuh");
    /// assert_eq!(parse_jyutping("hoe5").unwrap().to_yale(), "héuh");
    /// assert_eq!(parse_jyutping("hoe6").unwrap().to_yale(), "heuh");
    ///
    /// assert_eq!(parse_jyutping("leot1").unwrap().to_yale(), "lēut");
    /// assert_eq!(parse_jyutping("leot2").unwrap().to_yale(), "léut");
    /// assert_eq!(parse_jyutping("leot3").unwrap().to_yale(), "leut");
    /// assert_eq!(parse_jyutping("leot4").unwrap().to_yale(), "lèuht");
    /// assert_eq!(parse_jyutping("leot5").unwrap().to_yale(), "léuht");
    /// assert_eq!(parse_jyutping("leot6").unwrap().to_yale(), "leuht");
    ///
    /// assert_eq!(parse_jyutping("hoi1").unwrap().to_yale(), "hōi");
    /// assert_eq!(parse_jyutping("mui5").unwrap().to_yale(), "múih");
    /// assert_eq!(parse_jyutping("miu1").unwrap().to_yale(), "mīu");
    /// assert_eq!(parse_jyutping("deu6").unwrap().to_yale(), "deuh");
    /// ```
    pub fn to_yale(&self) -> String {
        let result = self
            .initial
            .as_ref()
            .map(|i| i.to_yale())
            .unwrap_or("".to_string())
            + &to_yale_rime(self.nucleus, self.coda, self.tone, true);
        let result = result.replace("yy", "y");
        match (result.as_str(), self.tone) {
            ("m", Some(JyutPingTone::T1)) => "m̄".to_string(),
            ("m", Some(JyutPingTone::T2 | JyutPingTone::T5)) => "ḿ".to_string(),
            ("m", Some(JyutPingTone::T4)) => "m̀".to_string(),

            ("ng", Some(JyutPingTone::T1)) => "n̄g".to_string(),
            ("ng", Some(JyutPingTone::T2 | JyutPingTone::T5)) => "ńg".to_string(),
            ("ng", Some(JyutPingTone::T4)) => "ǹg".to_string(),

            _ => result,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, rkyv::Archive, rkyv::Deserialize, rkyv::Serialize)]
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
                }
                LaxJyutPingSegment::Nonstandard(_) => {
                    return None;
                }
            }
        }
        Some(jyutpings)
    }

    pub fn is_standard_jyutping(&self) -> bool {
        self.0.iter().all(|seg| match seg {
            LaxJyutPingSegment::Standard(_) => true,
            LaxJyutPingSegment::Nonstandard(_) => false,
        })
    }

    pub fn to_yale_no_diacritics(&self) -> String {
        self.0
            .iter()
            .map(|seg| match seg {
                LaxJyutPingSegment::Standard(jyutping) => jyutping.to_yale_no_diacritics(),
                LaxJyutPingSegment::Nonstandard(s) => s.to_string(),
            })
            .collect::<Vec<String>>()
            .join(" ")
    }

    pub fn to_yale(&self) -> String {
        self.0
            .iter()
            .map(|seg| match seg {
                LaxJyutPingSegment::Standard(jyutping) => jyutping.to_yale(),
                LaxJyutPingSegment::Nonstandard(s) => s.to_string(),
            })
            .collect::<Vec<String>>()
            .join(" ")
    }
}

lazy_static::lazy_static! {
    pub static ref YALE_TONE_MARK_REGEX_NO_DIACRITICS: Regex = Regex::new(r"([aeiou])h").unwrap();
}

pub fn remove_yale_diacritics(s: &str) -> String {
    let chars = s.chars();
    chars
        .fold("".to_string(), |acc, c| {
            acc + &find_yale_diacritics(c).to_string()
        })
        // combining accute accent mark ◌́
        .replace('\u{0301}', "")
        // combining grave accent mark ◌̀
        .replace('\u{0300}', "")
        // combining macron ◌̄
        .replace('\u{0304}', "")
}

fn find_yale_diacritics(c: char) -> char {
    match c {
        'ā' | 'á' | 'à' => 'a',
        'ē' | 'é' | 'è' => 'e',
        'ī' | 'í' | 'ì' => 'i',
        'ō' | 'ó' | 'ò' => 'o',
        'ū' | 'ú' | 'ù' => 'u',
        _ => c,
    }
}

pub fn remove_yale_tones(s: &str) -> String {
    YALE_TONE_MARK_REGEX_NO_DIACRITICS
        .replace_all(&remove_yale_diacritics(s), "$1")
        .to_string()
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, rkyv::Archive, rkyv::Deserialize, rkyv::Serialize)]
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
#[derive(
    strum::EnumString, strum::Display, Debug, Clone, Copy, PartialEq, Serialize, Deserialize, rkyv::Archive, rkyv::Deserialize, rkyv::Serialize
)]
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

impl JyutPingInitial {
    pub fn to_yale(&self) -> String {
        match self {
            Self::Z => "j".to_string(),
            Self::C => "ch".to_string(),
            Self::J => "y".to_string(),
            _ => self.to_string(),
        }
    }
}

/// Nucleus segment of a Jyutping, not required in case of /ng/ and /m/
///
/// Eg: 'a' in "sap6"
///
#[derive(
    strum::EnumString, strum::Display, Debug, Clone, Copy, PartialEq, Serialize, Deserialize, rkyv::Archive, rkyv::Deserialize, rkyv::Serialize
)]
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

fn to_yale_rime(
    nucleus: Option<JyutPingNucleus>,
    coda: Option<JyutPingCoda>,
    tone: Option<JyutPingTone>,
    show_diacritics: bool,
) -> String {
    fn is_lower_tone(tone: &JyutPingTone) -> bool {
        match tone {
            JyutPingTone::T4 => true,
            JyutPingTone::T5 => true,
            JyutPingTone::T6 => true,
            _ => false,
        }
    }
    let tone_mark = if let Some(tone) = tone {
        if is_lower_tone(&tone) && nucleus.is_some() {
            "h"
        } else {
            ""
        }
    } else {
        ""
    };

    enum ReplaceResult {
        NoChange(String),
        Replaced(String),
    }

    impl ReplaceResult {
        fn replace_first(self, pat: char, to: &str) -> ReplaceResult {
            match self {
                ReplaceResult::Replaced(_) => self,
                ReplaceResult::NoChange(s) => {
                    let result = s.replacen(pat, to, 1);
                    if s == result {
                        ReplaceResult::NoChange(s)
                    } else {
                        ReplaceResult::Replaced(result)
                    }
                }
            }
        }

        fn unwrap(self) -> String {
            match self {
                ReplaceResult::Replaced(s) => s,
                ReplaceResult::NoChange(s) => s,
            }
        }
    }

    let add_diacritics = |nucleus: String| {
        if show_diacritics {
            match tone {
                Some(JyutPingTone::T1) => ReplaceResult::NoChange(nucleus)
                    .replace_first('a', "ā")
                    .replace_first('e', "ē")
                    .replace_first('o', "ō")
                    .replace_first('i', "ī")
                    .replace_first('u', "ū")
                    .unwrap(),
                Some(JyutPingTone::T2 | JyutPingTone::T5) => ReplaceResult::NoChange(nucleus)
                    .replace_first('a', "á")
                    .replace_first('e', "é")
                    .replace_first('o', "ó")
                    .replace_first('i', "í")
                    .replace_first('u', "ú")
                    .unwrap(),
                Some(JyutPingTone::T4) => ReplaceResult::NoChange(nucleus)
                    .replace_first('a', "à")
                    .replace_first('e', "è")
                    .replace_first('o', "ò")
                    .replace_first('i', "ì")
                    .replace_first('u', "ù")
                    .unwrap(),
                _ => nucleus,
            }
        } else {
            nucleus
        }
    };

    let result = match (nucleus.as_ref(), coda.as_ref()) {
        (Some(JyutPingNucleus::Aa), None) => add_diacritics("a".to_string()) + tone_mark,
        (Some(JyutPingNucleus::Oe | JyutPingNucleus::Eo), coda) => match coda {
            Some(JyutPingCoda::I | JyutPingCoda::U) => {
                add_diacritics("eu".to_string())
                    + &coda.map(|c| c.to_string()).unwrap_or("".to_string())
                    + tone_mark
            }
            _ => {
                add_diacritics("eu".to_string())
                    + tone_mark
                    + &coda.map(|c| c.to_string()).unwrap_or("".to_string())
            }
        },
        _ => match coda {
            Some(JyutPingCoda::I | JyutPingCoda::U) => {
                add_diacritics(nucleus.map(|c| c.to_string()).unwrap_or("".to_string()))
                    + &coda.map(|c| c.to_string()).unwrap_or("".to_string())
                    + tone_mark
            }
            _ => {
                add_diacritics(nucleus.map(|c| c.to_string()).unwrap_or("".to_string()))
                    + tone_mark
                    + &coda.map(|c| c.to_string()).unwrap_or("".to_string())
            }
        },
    };

    result
}

/// Coda segment of a Jyutping, optional
///
/// Eg: 'p' in "sap6"
///
#[derive(
    strum::EnumString, strum::Display, Debug, Clone, Copy, PartialEq, Serialize, Deserialize, rkyv::Archive, rkyv::Deserialize, rkyv::Serialize
)]
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
#[derive(
    strum::EnumString, strum::Display, Debug, Clone, Copy, PartialEq, Serialize, Deserialize, rkyv::Archive, rkyv::Deserialize, rkyv::Serialize
)]
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
            }
            None => {
                return None;
            }
        }
    }
    Some(jyutpings)
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
    static ref JYUTPING_WITH_TONE_REGEX: Regex =
        Regex::new(&(JYUTPING_WITHOUT_TONE_REGEX.to_owned() + r"[1-6]$")).unwrap();
}

// Source: lib/cantonese.py:is_valid_jyutping_form
pub fn is_standard_jyutping(s: &str) -> bool {
    JYUTPING_WITH_TONE_REGEX.is_match(s)
}

// Source: zidin/definition.py:looks_like_jyutping
pub fn looks_like_jyutping(s: &str) -> bool {
    let s_lower = s.to_ascii_lowercase();
    let segs = s_lower.split_whitespace();
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

/// ```
/// use wordshk_tools::jyutping::jyutping_to_yale;
/// assert_eq!(jyutping_to_yale("hoi1 coi2".to_string()), "hōi chói".to_string());
/// assert_eq!(jyutping_to_yale("m4 hou2 hai2 dou6".to_string()), "m̀ hóu hái douh".to_string());
/// assert_eq!(jyutping_to_yale("tung4 joeng6".to_string()), "tùhng yeuhng".to_string());
/// ```
pub fn jyutping_to_yale(jyutping: &str) -> String {
    jyutping
        .split(" ")
        .map(|syllable| parse_jyutping(&syllable).unwrap().to_yale())
        .join(" ")
}
