use super::jyutping::LaxJyutPings;
use itertools::Itertools;
use serde::Deserialize;
use serde::Serialize;
use std::collections::{BTreeMap, HashSet};

/// A dictionary is a list of entries
pub type Dict = BTreeMap<EntryId, Entry>;

pub type EntryId = u32;

/// An entry contains some information about a word.
///
/// \[id\] the word's unique identifier used by words.hk: 116878
///
/// \[variants\] variants of the word: 㗎:gaa3,咖:gaa3,𡃉:gaa3
///
/// \[pos\] grammatical positions of the word: 動詞, 名詞, 形容詞
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
#[derive(Clone, Debug, PartialEq)]
pub struct Entry {
    pub id: EntryId,
    pub variants: Variants,
    pub poses: Vec<String>,
    pub labels: Vec<String>,
    pub sims: Vec<String>,
    pub ants: Vec<String>,
    pub refs: Vec<String>,
    pub imgs: Vec<String>,
    pub defs: Vec<Def>,
    pub published: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Variants(pub Vec<Variant>);

impl Variants {
    pub fn to_words(&self) -> Vec<&str> {
        self.0.iter().map(|variant| &variant.word[..]).collect()
    }
    pub fn to_words_set(&self) -> HashSet<&str> {
        self.0.iter().map(|variant| &variant.word[..]).collect()
    }
}

/// A variant of a \[word\] with \[prs\] (pronounciations)
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Variant {
    #[serde(rename = "w")]
    pub word: String,

    #[serde(rename = "p")]
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
    #[serde(rename = "T")]
    Text,

    #[serde(rename = "L")]
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

pub fn line_to_strings(line: &Line) -> Vec<String> {
    line.iter().map(|seg| seg.1.to_string()).collect()
}

pub fn line_to_string(line: &Line) -> String {
    line_to_strings(line).join("")
}

/// A clause consists of one or more [Line]s. Appears in explanations and example sentences
///
/// Single-line clause: `vec![vec![(Text, "一行白鷺上青天")]]`
///
/// Multi-line clause: `vec![vec![(Text, "一行白鷺上青天")], vec![(Text, "兩個黃鸝鳴翠柳")]]`
///
pub type Clause = Vec<Line>; // can be multiline

pub fn clause_to_string(clause: &Clause) -> String {
    clause.iter().map(line_to_string).join("\n")
}

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
#[derive(Clone, Debug, PartialEq)]
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

fn is_unfinished_line(line: &Line) -> bool {
    line.iter()
        .any(|seg| seg.1.contains("XX") || seg.1.contains("xxx") || seg.1.contains("[ChatGPT]"))
        || *line == vec![(SegmentType::Text, "X".to_string())]
        || *line == vec![(SegmentType::Text, "x".to_string())]
}

pub fn filter_unfinished_entries(dict: Dict) -> Dict {
    dict.into_iter()
        .filter(|(_, entry)| {
            // no definition should contain unfinished lines
            !entry.defs.iter().any(|def| {
                def.yue.iter().any(is_unfinished_line)
                    || def
                        .eng
                        .as_ref()
                        .map_or(false, |clause| clause.iter().any(is_unfinished_line))
                    || def.egs.iter().any(|eg| {
                        eg.zho
                            .as_ref()
                            .map_or(false, |(line, _)| is_unfinished_line(line))
                            || eg
                                .yue
                                .as_ref()
                                .map_or(false, |(line, _)| is_unfinished_line(line))
                            || eg
                                .eng
                                .as_ref()
                                .map_or(false, |line| is_unfinished_line(line))
                    })
            })
        })
        .collect()
}
