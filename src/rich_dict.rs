use crate::jyutping::LaxJyutPings;
use crate::mandarin_variants::MANDARIN_VARIANTS;
use crate::search::RichDictLike;

use super::charlist::CHARLIST;
use super::dict::{
    line_to_string, line_to_strings, AltClause, Clause, Def, Dict, Eg, EntryId, Line, PrLine,
    Segment, SegmentType, Variants,
};
use super::unicode;
use itertools::Itertools;
use serde::Deserialize;
use serde::Serialize;
use std::cmp;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::fmt;

pub type RichDict = BTreeMap<EntryId, RichEntry>;

impl RichDictLike for RichDict {
    fn get_entry(&self, id: EntryId) -> Option<RichEntry> {
        self.get(&id).cloned()
    }

    fn get_entries(&self, ids: &[EntryId]) -> HashMap<EntryId, RichEntry> {
        ids.iter()
            .filter_map(|id| self.get(id).cloned().map(|entry| (*id, entry)))
            .collect()
    }

    fn get_ids(&self) -> Vec<EntryId> {
        self.keys().cloned().collect_vec()
    }
}

#[derive(Debug, PartialEq, Clone, Serialize, Deserialize)]
pub struct RichEntry {
    #[serde(rename = "i")]
    pub id: EntryId,

    #[serde(rename = "v")]
    pub variants: RichVariants,

    #[serde(rename = "p")]
    pub poses: Vec<String>,

    #[serde(rename = "l")]
    pub labels: Vec<String>,

    #[serde(rename = "s")]
    pub sims: Vec<Segment>,

    #[serde(rename = "ss")]
    pub sims_simp: Vec<String>,

    #[serde(rename = "a")]
    pub ants: Vec<Segment>,

    #[serde(rename = "as")]
    pub ants_simp: Vec<String>,

    #[serde(rename = "m")]
    pub mandarin_variants: MandarinVariants,

    #[serde(skip)]
    pub refs: Vec<String>,

    #[serde(skip)]
    pub imgs: Vec<String>,

    #[serde(rename = "d")]
    pub defs: Vec<RichDef>,

    #[serde(rename = "pb")]
    pub published: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RichVariants(pub Vec<RichVariant>);

impl RichVariants {
    pub fn to_words(&self) -> Vec<&str> {
        self.0.iter().map(|variant| &variant.word[..]).collect()
    }
    pub fn to_words_set(&self) -> HashSet<&str> {
        self.0.iter().map(|variant| &variant.word[..]).collect()
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MandarinVariant {
    #[serde(rename = "ws")]
    pub word_simp: String,

    #[serde(rename = "d")]
    pub def_indices: Vec<usize>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MandarinVariants(pub Vec<MandarinVariant>);

impl Default for MandarinVariants {
    fn default() -> Self {
        MandarinVariants(vec![])
    }
}

/// A variant of a \[word\] with \[prs\] (pronounciations)
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RichVariant {
    #[serde(rename = "w")]
    pub word: String,

    #[serde(rename = "ws")]
    pub word_simp: String,

    #[serde(rename = "p")]
    pub prs: LaxJyutPings,
}

#[derive(Debug, PartialEq, Clone, Serialize, Deserialize)]
pub struct RichDef {
    #[serde(rename = "y")]
    pub yue: Clause,

    #[serde(rename = "ys")]
    pub yue_simp: Clause,

    #[serde(rename = "e")]
    pub eng: Option<Clause>,

    #[serde(skip)]
    pub alts: Vec<AltClause>,

    #[serde(rename = "eg")]
    pub egs: Vec<RichEg>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RichEg {
    #[serde(rename = "z")]
    pub zho: Option<RichLine>,

    #[serde(rename = "zs")]
    pub zho_simp: Option<String>,

    #[serde(rename = "y")]
    pub yue: Option<RichLine>,

    #[serde(rename = "ys")]
    pub yue_simp: Option<String>,

    #[serde(rename = "e")]
    pub eng: Option<Line>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum RichLine {
    #[serde(rename = "R")]
    Ruby(RubyLine),

    #[serde(rename = "T")]
    Text(WordLine),
}

impl fmt::Display for RichLine {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Ruby(line) => write!(
                f,
                "{}",
                line.iter().map(|segment| segment.to_string()).join("")
            ),
            Self::Text(line) => write!(
                f,
                "{}",
                line.iter().map(|(_, word)| word.to_string()).join("")
            ),
        }
    }
}

/// A styled text segment
///
/// Normal: `(TextStyle::Normal, "好")`
///
/// Bold: `(TextStyle::Bold, "good")`
///
pub type Text = (TextStyle, String);

/// Text styles, can be bold or normal
#[derive(Debug, PartialEq, Clone, Serialize, Deserialize)]
pub enum TextStyle {
    #[serde(rename = "B")]
    Bold,
    #[serde(rename = "N")]
    Normal,
}

/// A segment containing a [Word]
///
/// `(SegmentType::Text, vec![(TextStyle::Bold, "兩"), (TextStyle::Normal, "周")])`
///
pub type WordSegment = (SegmentType, Word);

/// A consecutive series of [Text]s
///
/// `vec![(TextStyle::Bold, "兩"), (TextStyle::Normal, "周")]`
///
#[derive(Debug, PartialEq, Clone, Serialize, Deserialize)]
pub struct Word(pub Vec<Text>);

impl fmt::Display for Word {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}",
            self.0
                .iter()
                .map(|(_, seg)| seg.clone())
                .collect::<Vec<String>>()
                .join("")
        )
    }
}

/// A segment marked with pronunciation (called "ruby" in HTML)
///
/// Segment can be one of
/// * a single punctuation
/// * or a [Word] with its pronunciations
/// * or a linked segment with a series of [Word]s
/// and their pronunciations
#[derive(Debug, PartialEq, Clone, Serialize, Deserialize)]
pub enum RubySegment {
    #[serde(rename = "P")]
    Punc(String),

    #[serde(rename = "W")]
    Word(Word, Vec<String>),

    #[serde(rename = "L")]
    LinkedWord(Vec<(Word, Vec<String>)>),
}

impl fmt::Display for RubySegment {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Punc(punc) => write!(f, "{}", punc),
            Self::Word(word, _) => write!(f, "{}", word),
            Self::LinkedWord(words) => {
                write!(
                    f,
                    "{}",
                    words.iter().map(|(word, _)| word.to_string()).join("")
                )
            }
        }
    }
}

/// A line consists of one or more [RubySegment]s
pub type RubyLine = Vec<RubySegment>;

/// A line consists of one or more [WordSegment]s
pub type WordLine = Vec<WordSegment>;

pub struct RichDictWrapper {
    dict: RichDict,
}

impl RichDictWrapper {
    pub fn new(dict: RichDict) -> Self {
        Self { dict }
    }
}

impl RichDictLike for RichDictWrapper {
    fn get_entry(&self, id: EntryId) -> Option<RichEntry> {
        self.dict.get_entry(id)
    }

    fn get_entries(&self, ids: &[EntryId]) -> HashMap<EntryId, RichEntry> {
        self.dict.get_entries(ids)
    }

    fn get_ids(&self) -> Vec<EntryId> {
        self.dict.get_ids()
    }
}

// source: https://stackoverflow.com/a/35907071/6798201
// Important: strings are normalized before comparison
// This ensures that "Hello" in an <eg> can be identified as the variant "hello"
fn find_variants(haystack: &[&str], needle: &[&str]) -> Vec<(usize, usize)> {
    haystack
        .windows(needle.len())
        .enumerate()
        .filter_map(|(i, window)| {
            if unicode::normalize(&window.join("")) == unicode::normalize(&needle.join("")) {
                Some((i, i + needle.len() - 1))
            } else {
                None
            }
        })
        .collect()
}

pub fn tokenize(variants: &[&str], text: &str) -> Vec<Word> {
    let mut i = 0;
    let mut words: Vec<Word> = vec![];
    let gs = unicode::to_graphemes(text);
    let mut start_end_pairs: Vec<(usize, usize)> = vec![];
    variants.iter().for_each(|variant| {
        let variant = unicode::to_graphemes(variant);
        find_variants(&gs, &variant)
            .iter()
            .for_each(|(start_index, end_index)| {
                // filter out short variants
                start_end_pairs
                    .retain(|(start, end)| !(*start_index <= *start && *end <= *end_index));
                // add variant indices if a longer variant does not already exist
                if !start_end_pairs
                    .iter()
                    .any(|(start, end)| *start <= *start_index && *end_index <= *end)
                {
                    start_end_pairs.push((*start_index, *end_index));
                }
            });
    });
    // if variants.contains(&"camp camp 哋".to_string()) {
    //     println!("{:?}", start_end_pairs);
    // }
    let mut is_bolded = false;
    while i < gs.len() {
        let g = gs[i];
        if unicode::test_g(unicode::is_cjk, g) {
            if start_end_pairs.iter().any(|(start, _)| start == &i) {
                is_bolded = true;
            }
            words.push(Word(vec![(
                if is_bolded {
                    TextStyle::Bold
                } else {
                    TextStyle::Normal
                },
                g.to_string(),
            )]));
            if start_end_pairs.iter().any(|(_, end)| end == &i) {
                is_bolded = false;
            }
            i += 1;
        } else if unicode::test_g(unicode::is_alphanumeric, g) {
            let mut j = i;
            let mut prev_bold_end_index: Option<usize> = None;
            let mut bold_start_index = None;
            let mut bold_end_index = None;
            let mut word = vec![];
            // a segment can be an alphanumeric char followed by any number of alphanumeric chars,
            // whitespace, and decimal points.
            while j < gs.len() && !unicode::test_g(unicode::is_cjk, gs[j]) {
                if start_end_pairs.iter().any(|(start, _)| start == &j) {
                    bold_start_index = Some(j);
                    is_bolded = true;
                }
                if start_end_pairs.iter().any(|(_, end)| end == &j) {
                    bold_end_index = Some(j);
                    is_bolded = false;
                }
                if let (Some(start), Some(end)) = (bold_start_index, bold_end_index) {
                    let prev_end = prev_bold_end_index.map(|x| x + 1).unwrap_or(i);
                    if prev_end < start {
                        word.push((TextStyle::Normal, gs[prev_end..start].join("")));
                    }
                    word.push((TextStyle::Bold, gs[start..end + 1].join("")));
                    prev_bold_end_index = bold_end_index;
                    bold_start_index = None;
                    bold_end_index = None;
                }
                j += 1;
            }
            let mut word_end = j;
            while word_end >= 1 {
                if unicode::test_g(unicode::is_alphanumeric, gs[word_end - 1]) {
                    break;
                } else {
                    word_end -= 1;
                }
            }
            if let Some(bold_end) = prev_bold_end_index {
                if bold_end + 1 < word_end {
                    let rest = gs[bold_end + 1..word_end].join("").trim_end().to_string();
                    if !rest.is_empty() {
                        word.push((TextStyle::Normal, rest));
                    }
                }
            } else if let Some(bold_start) = bold_start_index {
                word.push((
                    TextStyle::Bold,
                    gs[bold_start..word_end].join("").trim_end().into(),
                ));
            } else {
                word.push((
                    TextStyle::Normal,
                    gs[i..word_end].join("").trim_end().into(),
                ));
            }
            words.push(Word(word));
            // push the punctuations after the word into words
            while word_end < j {
                if !unicode::test_g(char::is_whitespace, gs[word_end]) {
                    words.push(Word(vec![(TextStyle::Normal, gs[word_end].to_string())]));
                }
                word_end += 1;
            }
            i = j;
        } else {
            // a punctuation or space
            if !unicode::test_g(char::is_whitespace, g) {
                words.push(Word(vec![(
                    if is_bolded {
                        TextStyle::Bold
                    } else {
                        TextStyle::Normal
                    },
                    g.to_string(),
                )]));
            }
            i += 1;
        }
    }
    words
}

/// Flatten a [Line] by breaking each [Segment] into tokens
///
/// A token can be:
/// * A Chinese character
/// * A series of consecutive alphanumeric words (includes spaces in between)
/// * A punctuation mark
///
/// Tokens are units of pronunciation matching (see [match_ruby])
///
pub fn flatten_line(variants: &[&str], line: &Line) -> WordLine {
    let mut bit_line: WordLine = vec![];
    line.iter().for_each(|(seg_type, seg): &Segment| {
        bit_line.extend::<WordLine>(
            tokenize(variants, seg)
                .iter()
                .map(|text| (seg_type.clone(), text.clone()))
                .collect(),
        );
    });
    bit_line
}

fn unflatten_word_line(line: &WordLine) -> WordLine {
    let mut i = 0;
    let mut unflattened_line: WordLine = vec![];
    while i < line.len() {
        let mut link_word = vec![];
        while let (SegmentType::Link, seg) = &line[i] {
            link_word.extend(seg.0.clone());
            i += 1;
            if i >= line.len() {
                break;
            }
        }
        if !link_word.is_empty() {
            unflattened_line.push((SegmentType::Link, Word(link_word)));
        } else {
            unflattened_line.push(line[i].clone());
            i += 1;
        }
    }
    unflattened_line
}

fn create_ruby_segment(seg_type: &SegmentType, word: &Word, prs: &[&str]) -> RubySegment {
    let prs = prs.iter().map(|x| x.to_string()).collect();
    if *seg_type == SegmentType::Link {
        RubySegment::LinkedWord(vec![(word.clone(), prs)])
    } else {
        RubySegment::Word(word.clone(), prs)
    }
}

/// Match a [Line] to its pronunciations and bold the variants
pub fn match_ruby(variants: &[&str], line: &Line, prs: &Vec<&str>) -> RubyLine {
    let line = flatten_line(variants, line);
    let pr_scores = match_ruby_construct_table(&line, prs);
    let pr_map = match_ruby_backtrack(&line, prs, &pr_scores);
    // if variants[0] == "大家噉話" {
    //     println!("{:?}", pr_map);
    // }
    // index into consecutive cjk characters with inaccurate jyutping ruby annotation
    let mut consecutive_cjk_index = 0;
    let flattened_ruby_line = line
        .iter()
        .enumerate()
        .map(|(i, (seg_type, word))| match pr_map.get(&i) {
            Some(j) => {
                consecutive_cjk_index = 0;
                create_ruby_segment(seg_type, word, &prs[*j..j + 1])
            }
            None => {
                let word_str = word.to_string();
                if unicode::test_g(unicode::is_punc, &word_str) {
                    consecutive_cjk_index = 0;
                    RubySegment::Punc(word_str)
                } else {
                    let start = {
                        let mut j = i;
                        while j >= 1 && pr_map.get(&j).is_none() {
                            j -= 1;
                        }
                        match pr_map.get(&j) {
                            Some(start) => *start + 1,
                            None => 0,
                        }
                    };
                    let end = {
                        let mut j = i + 1;
                        while j < line.len() && pr_map.get(&j).is_none() {
                            j += 1;
                        }
                        match pr_map.get(&j) {
                            Some(end) => *end,
                            None => prs.len(),
                        }
                    };
                    // Check for consecutive Chinese characters with inaccurate jyutpings.
                    // Handles edge cases like 卅 saa1 aa6 and 卌 sei3 aa6
                    if end - start >= 2 &&
                    // current char is a Chinese character
                    unicode::test_g(unicode::is_cjk, &word_str)
                    {
                        let next_char_is_inaccurate_cjk = line
                                .get(i + 1)
                                .map(|(_segment_type, word)| {
                                    unicode::test_g(unicode::is_cjk, &word.to_string())
                                })
                                .unwrap_or(false) // next char is also a Chinese character
                                // next char also has inaccurate jyutping
                            && pr_map.get(&(i + 1)).is_none();
                        let prev_char_is_inaccurate_cjk = i >= 1 && line
                                .get(i - 1)
                                .map(|(_segment_type, word)| {
                                    unicode::test_g(unicode::is_cjk, &word.to_string())
                                })
                                .unwrap_or(false) // prev char is also a Chinese character
                                // prev char also has inaccurate jyutping
                            && pr_map.get(&(i - 1)).is_none();
                        if prev_char_is_inaccurate_cjk || next_char_is_inaccurate_cjk {
                            consecutive_cjk_index += 1;
                            create_ruby_segment(
                                seg_type,
                                word,
                                &prs[start + consecutive_cjk_index - 1
                                    ..start + consecutive_cjk_index],
                            )
                        } else {
                            consecutive_cjk_index = 0;
                            create_ruby_segment(seg_type, word, &prs[start..end])
                        }
                    } else {
                        consecutive_cjk_index = 0;
                        create_ruby_segment(seg_type, word, &prs[start..end])
                    }
                }
            }
        })
        .collect::<RubyLine>();
    unflatten_ruby_line(&flattened_ruby_line)
}

fn unflatten_ruby_line(line: &RubyLine) -> RubyLine {
    let mut i = 0;
    let mut unflattened_line = vec![];
    while i < line.len() {
        let mut link_pairs = vec![];
        while let RubySegment::LinkedWord(pairs) = &line[i] {
            link_pairs.extend(pairs.clone());
            i += 1;
            if i >= line.len() {
                break;
            }
        }
        if !link_pairs.is_empty() {
            unflattened_line.push(RubySegment::LinkedWord(link_pairs));
        } else {
            unflattened_line.push(line[i].clone());
            i += 1;
        }
    }
    unflattened_line
}

enum PrMatch {
    Full,
    Half,
    Zero,
}

fn pr_match_to_score(m: PrMatch) -> usize {
    match m {
        PrMatch::Full => 4,
        PrMatch::Half => 2,
        PrMatch::Zero => 0,
    }
}

fn match_pr(seg: &str, pr: &str) -> PrMatch {
    if seg.chars().count() != 1 {
        return PrMatch::Zero;
    }
    let c = seg.chars().next().unwrap();
    if unicode::is_chinese_punc(c) {
        PrMatch::Zero
    } else {
        match CHARLIST.get(&c) {
            Some(c_prs) => {
                match c_prs.get(pr) {
                    Some(_) => PrMatch::Full,
                    None => {
                        // try half pr (without tones), to accommodate for tone changes
                        let half_c_prs = c_prs
                            .iter()
                            .map(|pr| {
                                if let Some(tail) = pr.chars().last() {
                                    if tail.is_ascii_digit() {
                                        &pr[0..pr.len() - 1]
                                    } else {
                                        pr
                                    }
                                } else {
                                    pr
                                }
                            })
                            .collect::<Vec<&str>>();
                        // found the half pr
                        if half_c_prs.contains(&pr) {
                            PrMatch::Half
                        } else {
                            PrMatch::Zero
                        }
                    }
                }
            }
            None => PrMatch::Zero,
        }
    }
}

fn match_ruby_construct_table(line: &WordLine, prs: &Vec<&str>) -> Vec<Vec<usize>> {
    let m = line.len() + 1;
    let n = prs.len() + 1;
    let mut pr_scores = vec![vec![0; n]; m];
    // println!("m: {}, n: {}", m, n);
    for i in 1..m {
        for j in 1..n {
            // println!("i: {}, j: {}", i, j);
            let (_, word) = &line[i - 1];
            let cell_pr_match = match_pr(&word.to_string(), prs[j - 1]);
            match cell_pr_match {
                PrMatch::Full | PrMatch::Half => {
                    pr_scores[i][j] = pr_scores[i - 1][j - 1] + pr_match_to_score(cell_pr_match);
                }
                PrMatch::Zero => {
                    let top_pr_score = pr_scores[i - 1][j];
                    let left_pr_score = pr_scores[i][j - 1];
                    pr_scores[i][j] =
                        cmp::max(top_pr_score, left_pr_score) + pr_match_to_score(cell_pr_match);
                }
            }
        }
    }
    pr_scores
}

fn match_ruby_backtrack(
    line: &WordLine,
    prs: &[&str],
    pr_scores: &Vec<Vec<usize>>,
) -> HashMap<usize, usize> {
    let mut pr_map = HashMap::new();
    let mut i = pr_scores.len() - 1;
    let mut j = pr_scores[0].len() - 1;

    while i > 0 && j > 0 {
        // println!("i: {}, j: {}", i, j);
        let (_, word) = &line[i - 1];
        match match_pr(&word.to_string(), prs[j - 1]) {
            PrMatch::Full | PrMatch::Half => {
                pr_map.insert(i - 1, j - 1);
                // backtrack to the top left
                i -= 1;
                j -= 1;
            }
            PrMatch::Zero => {
                let left_score = pr_scores[i - 1][j];
                let right_score = pr_scores[i][j - 1];
                if left_score > right_score {
                    // backtrack to left
                    i -= 1;
                } else if left_score < right_score {
                    // backtrack to top
                    j -= 1;
                } else {
                    // a tie, default to move left
                    i -= 1;
                }
            }
        }
    }
    pr_map
}

pub struct EnrichDictOptions {
    pub remove_dead_links: bool,
}

pub fn enrich_dict(dict: &Dict, options: &EnrichDictOptions) -> RichDict {
    use rayon::prelude::*;
    dict.par_iter()
        .map(|(id, entry)| {
            let variants = &entry.variants.to_words();
            let rich_defs = entry
                .defs
                .iter()
                .map(|def| {
                    let yue = enrich_clause(&def.yue, dict, options);
                    let yue_simp = clause_to_simplified(&yue);
                    let eng = def
                        .eng
                        .as_ref()
                        .map(|eng| enrich_clause(eng, dict, options));
                    let alts: Vec<AltClause> = def
                        .alts
                        .iter()
                        .map(|(alt_lang, alt)| (*alt_lang, enrich_clause(alt, dict, options)))
                        .collect();
                    RichDef {
                        yue,
                        yue_simp,
                        eng,
                        alts,
                        egs: def
                            .egs
                            .iter()
                            .map(|eg| enrich_eg(variants, eg, dict, options))
                            .collect(),
                    }
                })
                .collect();
            let sims = enrich_sims_or_ants(&entry.sims, dict, options);
            let sims_simp = get_simplified_sims_or_ants(&line_to_strings(&sims));
            let ants = enrich_sims_or_ants(&entry.ants, dict, options);
            let ants_simp = get_simplified_sims_or_ants(&line_to_strings(&ants));
            let mandarin_variants = MANDARIN_VARIANTS
                .get(&id)
                .map(|mandarin_variants| mandarin_variants.clone())
                .unwrap_or(MandarinVariants(vec![]));
            (
                *id,
                RichEntry {
                    id: *id,
                    variants: enrich_variant(&entry.variants, &entry.defs),
                    poses: entry.poses.clone(),
                    labels: entry.labels.clone(),
                    sims,
                    sims_simp,
                    ants,
                    ants_simp,
                    mandarin_variants,
                    refs: entry.refs.clone(),
                    imgs: entry.imgs.clone(),
                    defs: rich_defs,
                    published: entry.published,
                },
            )
        })
        .collect::<RichDict>()
}

pub fn enrich_clause(clause: &Clause, dict: &Dict, options: &EnrichDictOptions) -> Clause {
    if options.remove_dead_links {
        clause
            .iter()
            .map(|line| enrich_line(line, dict, options))
            .collect()
    } else {
        clause.clone()
    }
}

pub fn enrich_line(line: &Line, dict: &Dict, options: &EnrichDictOptions) -> Line {
    if options.remove_dead_links {
        line.iter()
            .map(|(seg_type, seg)| {
                if seg_type == &SegmentType::Link {
                    let new_seg_type = if is_live_link(seg, dict) {
                        SegmentType::Link
                    } else {
                        SegmentType::Text
                    };
                    (new_seg_type, seg.clone())
                } else {
                    (seg_type.clone(), seg.clone())
                }
            })
            .collect()
    } else {
        line.clone()
    }
}

pub fn enrich_sims_or_ants(
    sims_or_ants: &[String],
    dict: &Dict,
    options: &EnrichDictOptions,
) -> Vec<Segment> {
    if options.remove_dead_links {
        sims_or_ants
            .iter()
            .map(|sim_or_ant| {
                if is_live_link(sim_or_ant, dict) {
                    (SegmentType::Link, sim_or_ant.clone())
                } else {
                    (SegmentType::Text, sim_or_ant.clone())
                }
            })
            .collect()
    } else {
        sims_or_ants
            .iter()
            .map(|sim_or_ant| (SegmentType::Link, sim_or_ant.clone()))
            .collect()
    }
}

fn is_live_link(link: &str, dict: &Dict) -> bool {
    dict.iter()
        .any(|(_id, entry)| entry.variants.to_words_set().contains(link))
}

pub fn enrich_pr_line(
    variants: &[&str],
    pr_line: &PrLine,
    dict: &Dict,
    options: &EnrichDictOptions,
) -> RichLine {
    let line = enrich_line(&pr_line.0, dict, options);
    match &pr_line.1 {
        Some(pr) => RichLine::Ruby(match_ruby(variants, &line, &unicode::to_words(pr))),
        None => RichLine::Text(unflatten_word_line(&flatten_line(variants, &line))),
    }
}

pub fn enrich_eg(variants: &[&str], eg: &Eg, dict: &Dict, options: &EnrichDictOptions) -> RichEg {
    let eng = eg.eng.as_ref().map(|eng| enrich_line(eng, dict, options));
    let zho = eg
        .zho
        .as_ref()
        .map(|zho| enrich_pr_line(variants, zho, dict, options));
    let zho_simp = zho
        .as_ref()
        .map(|zho| unicode::to_simplified(&zho.to_string()));
    let yue = eg
        .yue
        .as_ref()
        .map(|yue| enrich_pr_line(variants, yue, dict, options));
    let yue_simp = yue
        .as_ref()
        .map(|yue| unicode::to_simplified(&yue.to_string()));
    RichEg {
        zho,
        zho_simp,
        yue,
        yue_simp,
        eng,
    }
}

pub fn get_simplified_rich_line(simp_line: &str, trad_line: &RichLine) -> RichLine {
    let mut simp_line_chars = simp_line.chars().peekable();
    match trad_line {
        RichLine::Ruby(ruby_line) => RichLine::Ruby(
            ruby_line
                .iter()
                .map(|seg| match seg {
                    RubySegment::Punc(_) => {
                        while simp_line_chars.peek().unwrap().is_whitespace() {
                            // get rid of all whitespace before punctuation
                            simp_line_chars.next();
                        }
                        // skip over punctuation
                        simp_line_chars.next();
                        seg.clone()
                    }
                    RubySegment::Word(word, prs) => RubySegment::Word(
                        replace_contents_in_word(word, &mut simp_line_chars),
                        prs.to_vec(),
                    ),
                    RubySegment::LinkedWord(segs) => RubySegment::LinkedWord(
                        segs.iter()
                            .map(|(word, prs)| {
                                (
                                    replace_contents_in_word(word, &mut simp_line_chars),
                                    prs.to_vec(),
                                )
                            })
                            .collect(),
                    ),
                })
                .collect(),
        ),
        RichLine::Text(word_line) => RichLine::Text(
            word_line
                .iter()
                .map(|(seg_type, word)| {
                    (
                        seg_type.clone(),
                        replace_contents_in_word(word, &mut simp_line_chars),
                    )
                })
                .collect(),
        ),
    }
}

pub fn replace_contents_in_word(
    target_word: &Word,
    content: &mut std::iter::Peekable<std::str::Chars<'_>>,
) -> Word {
    Word(
        target_word
            .0
            .iter()
            .map(|(seg_type, seg)| {
                if !seg.chars().next().unwrap().is_whitespace() {
                    while content.peek().unwrap().is_whitespace() {
                        // get rid of all whitespace before word
                        content.next();
                    }
                }
                (
                    seg_type.clone(),
                    content.take(seg.chars().count()).collect(),
                )
            })
            .collect(),
    )
}

// Needs the defs context to handle one traditional to multiple simplified variants
// Like 乾 -> 乾隆/干净
fn enrich_variant(trad_variants: &Variants, defs: &[Def]) -> RichVariants {
    let mut lines: Vec<Line> = defs
        .iter()
        .flat_map(|def| {
            def.egs
                .iter()
                .filter_map(|eg| eg.zho.as_ref().map(|zho| zho.0.clone()))
        })
        .collect();
    let yue_lines = defs.iter().flat_map(|def| {
        def.egs
            .iter()
            .filter_map(|eg| eg.yue.as_ref().map(|yue| yue.0.clone()))
    });
    lines.extend(yue_lines);

    let mut simp_variants = HashMap::new();

    lines.iter().for_each(|line| {
        let line_str = line_to_string(line);
        let line_str_simp = unicode::to_simplified(&line_str);
        trad_variants
            .0
            .iter()
            .enumerate()
            .for_each(|(variant_index, trad_variant)| {
                let trad_variant = &trad_variant.word;
                if let Some(variant_start_index) = line_str.find(trad_variant) {
                    let variant_end_index = variant_start_index + trad_variant.len();
                    simp_variants.insert(
                        variant_index,
                        (line_str_simp[variant_start_index..variant_end_index]).to_string(),
                    );
                }
            });
    });
    RichVariants(
        trad_variants
            .0
            .iter()
            .enumerate()
            .map(|(variant_index, trad_variant)| RichVariant {
                word: trad_variant.word.clone(),
                word_simp: if let Some(simp_variant) = simp_variants.get(&variant_index) {
                    simp_variant.clone()
                } else {
                    unicode::to_simplified(&trad_variant.word)
                },
                prs: trad_variant.prs.clone(),
            })
            .collect(),
    )
}

fn get_simplified_sims_or_ants(sims_or_ants: &[String]) -> Vec<String> {
    sims_or_ants
        .iter()
        .map(|sim_or_ant| unicode::to_simplified(sim_or_ant))
        .collect()
}

fn clause_to_simplified(clause: &Clause) -> Clause {
    clause.iter().map(line_to_simplified).collect()
}

fn line_to_simplified(line: &Line) -> Line {
    line.iter()
        .map(|(seg_type, seg)| (seg_type.clone(), unicode::to_simplified(seg)))
        .collect()
}
