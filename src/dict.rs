use itertools::Itertools;
use serde::Deserialize;
use serde::Serialize;
use std::collections::{HashMap, HashSet};
use std::fmt;

/// A dictionary is a list of entries
pub type Dict = HashMap<usize, Entry>;

/// An entry contains some information about a word.
///
/// \[id\] the word's unique identifier used by words.hk: 116878
///
/// \[variants\] variants of the word: 㗎:gaa3,咖:gaa3,𡃉:gaa3
///
/// \[pos\] grammaticall positions of the word: 動詞, 名詞, 形容詞
///
/// \[labels\] labels on the word: 術語, 俚語, 專名
///
/// \[sims\] synonyms of the word: 武士 is a synonym of 騎士
///
/// \[ants\] antonyms of the word: 放電 is an antonym of 充電
///
/// \[refs\] urls to references for this entry: <http://dictionary.reference.com/browse/tart?s=t>
///
/// \[imgs\] urls to images for this entry: <https://upload.wikimedia.org/wikipedia/commons/7/79/Naihuangbao.jpg>
///
/// \[defs\] a list of definitions for this word
///
#[derive(Debug, PartialEq)]
pub struct Entry {
    pub id: usize,
    pub variants: Variants,
    pub poses: Vec<String>,
    pub labels: Vec<String>,
    pub sims: Vec<String>,
    pub ants: Vec<String>,
    pub refs: Vec<String>,
    pub imgs: Vec<String>,
    pub defs: Vec<Def>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Variants(pub Vec<Variant>);

impl Variants {
    pub fn to_words(&self) -> Vec<&str> {
        self.0.iter().map(|variant| &variant.word[..]).collect()
    }
    pub fn to_words_set(&self) -> HashSet<&str> {
        self.0
            .iter()
            .map(|variant| &variant.word[..])
            .into_iter()
            .collect()
    }
}

/// A variant of a \[word\] with \[prs\] (pronounciations)
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Variant {
    pub word: String,
    pub prs: LaxJyutPings,
}

/// Two types of segments: text or link. See [Segment]
///
/// \[Text\] normal text
///
/// \[Link\] a link to another entry
///
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum SegmentType {
    Text,
    Link,
}

/// A segment can be a text or a link
///
/// Text: 非常鬆軟。（量詞：件／籠）
///
/// Link: A link to the entry 雞蛋 would be #雞蛋
///
pub type Segment = (SegmentType, String);

/// A line consists of one or more [Segment]s
///
/// Empty line: `vec![(Text, "")]`
///
/// Simple line: `vec![(Text, "用嚟圍喺BB牀邊嘅布（量詞：塊）")]`
///
/// Mixed line: `vec![(Text, "一種加入"), (Link, "蝦籽"), (Text, "整嘅廣東麪")]`
///
pub type Line = Vec<Segment>;

/// A clause consists of one or more [Line]s. Appears in explanations and example sentences
///
/// Single-line clause: `vec![vec![(Text, "一行白鷺上青天")]]`
///
/// Multi-line clause: `vec![vec![(Text, "一行白鷺上青天")], vec![(Text, "兩個黃鸝鳴翠柳")]]`
///
pub type Clause = Vec<Line>; // can be multiline

/// A definition of a word
///
/// Here's an example of the definition of the word 年畫
///
/// \[yue\] Cantonese explanation of the word's meaning: 東亞民間慶祝#新春 嘅畫種（量詞：幅）
///
/// \[eng\] English explanation of the word's meaning: new year picture in East Asia
///
/// \[alts\] Word with similar meaning in other languages: jpn:年画；ねんが, kor:세화, vie:Tranh tết
///
/// \[egs\] Example sentences usually with Jyutping pronunciations and English translations
///
#[derive(Debug, PartialEq)]
pub struct Def {
    pub yue: Clause,
    pub eng: Option<Clause>,
    pub alts: Vec<AltClause>,
    pub egs: Vec<Eg>,
}

/// A clause in an alternative language other than Cantonese and English
///
/// \[[AltLang]\] language tag
///
/// \[[Clause]\] A sequence of texts and links
///
pub type AltClause = (AltLang, Clause);

/// Language tags for alternative languages other than Cantonese and English
///
/// From my observation, the tags seem to be alpha-3 codes in [ISO 639-2]
///
/// [ISO 639-2]: https://www.loc.gov/standards/iso639-2/php/code_list.php
///
#[derive(Debug, PartialEq, Clone, Copy, Serialize, Deserialize)]
pub enum AltLang {
    Jpn, // Japanese
    Kor, // Korean
    Por, // Portuguese
    Vie, // Vietnamese
    Lat, // Latin
    Fra, // French
}

impl AltLang {
    /// Convert [AltLang] to a language name in Cantonese
    pub fn to_yue_name(&self) -> String {
        match self {
            AltLang::Jpn => "日文",
            AltLang::Kor => "韓文",
            AltLang::Por => "葡萄牙文",
            AltLang::Vie => "越南文",
            AltLang::Lat => "拉丁文",
            AltLang::Fra => "法文",
        }
        .to_string()
    }
}

/// An example sentence in Mandarin, Cantonese, and/or English
///
/// \[zho\] Mandarin example with optional Jyutping pronunciation: 可否見面？ (ho2 fau2 gin3 min6?)
///
/// \[yue\] Cantonese example with optional Jyutping pronunciation: 可唔可以見面？ (ho2 m4 ho2 ji5 gin3 min6?)
///
/// \[eng\] English example: Can we meet up?
///
#[derive(Debug, Clone, PartialEq)]
pub struct Eg {
    pub zho: Option<PrLine>,
    pub yue: Option<PrLine>,
    pub eng: Option<Line>,
}

/// An example sentence with optional Jyutping pronunciation
///
/// Eg: 可唔可以見面？ (ho2 m4 ho2 ji5 gin3 min6?)
///
pub type PrLine = (Line, Option<String>);

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
