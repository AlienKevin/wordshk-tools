use super::dict::{
    line_to_string, AltClause, Clause, Def, Dict, Eg, Line, PrLine, Segment, SegmentType, Variants,
};
use super::unicode;
use lazy_static::lazy_static;
use serde::Deserialize;
use serde::Serialize;
use std::cmp;
use std::collections::HashMap;
use std::fmt;

pub type RichDict = HashMap<usize, RichEntry>;

#[derive(Debug, PartialEq, Clone, Serialize, Deserialize)]
pub struct RichEntry {
    pub id: usize,
    pub variants: Variants,
    pub variants_simp: Vec<String>,
    pub poses: Vec<String>,
    pub labels: Vec<String>,
    pub sims: Vec<String>,
    pub ants: Vec<String>,
    pub refs: Vec<String>,
    pub imgs: Vec<String>,
    pub defs: Vec<RichDef>,
    pub published: bool,
}

#[derive(Debug, PartialEq, Clone, Serialize, Deserialize)]
pub struct RichDef {
    pub yue: Clause,
    pub yue_simp: Clause,
    pub eng: Option<Clause>,
    pub alts: Vec<AltClause>,
    pub egs: Vec<RichEg>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RichEg {
    pub zho: Option<RichLine>,
    pub zho_simp: Option<String>,
    pub yue: Option<RichLine>,
    pub yue_simp: Option<String>,
    pub eng: Option<Line>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum RichLine {
    Ruby(RubyLine),
    Text(WordLine),
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
    Bold,
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
    Punc(String),
    Word(Word, Vec<String>),
    LinkedWord(Vec<(Word, Vec<String>)>),
}

/// A line consists of one or more [RubySegment]s
pub type RubyLine = Vec<RubySegment>;

/// A line consists of one or more [WordSegment]s
pub type WordLine = Vec<WordSegment>;

type CharList = HashMap<char, HashMap<String, usize>>;

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

pub fn tokenize(variants: &Vec<&str>, text: &str) -> Vec<Word> {
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
                start_end_pairs = start_end_pairs
                    .iter()
                    .filter(|(start, end)| !(*start_index <= *start && *end <= *end_index))
                    .cloned()
                    .collect();
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
                match (bold_start_index, bold_end_index) {
                    (Some(start), Some(end)) => {
                        let prev_end = prev_bold_end_index.map(|x| x + 1).unwrap_or(i);
                        if prev_end < start {
                            word.push((TextStyle::Normal, gs[prev_end..start].join("").into()));
                        }
                        word.push((TextStyle::Bold, gs[start..end + 1].join("").into()));
                        prev_bold_end_index = bold_end_index;
                        bold_start_index = None;
                        bold_end_index = None;
                    }
                    (_, _) => {}
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
                    if rest.len() > 0 {
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
pub fn flatten_line(variants: &Vec<&str>, line: &Line) -> WordLine {
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
        if link_word.len() > 0 {
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
pub fn match_ruby(variants: &Vec<&str>, line: &Line, prs: &Vec<&str>) -> RubyLine {
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
                        while j >= 1 && pr_map.get(&j) == None {
                            j -= 1;
                        }
                        match pr_map.get(&j) {
                            Some(start) => *start + 1,
                            None => 0,
                        }
                    };
                    let end = {
                        let mut j = i + 1;
                        while j < line.len() && pr_map.get(&j) == None {
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
                                .iter()
                                .nth(i + 1)
                                .map(|(_segment_type, word)| {
                                    unicode::test_g(unicode::is_cjk, &word.to_string())
                                })
                                .unwrap_or(false) // next char is also a Chinese character
                                // next char also has inaccurate jyutping
                            && pr_map.get(&(i + 1)).is_none();
                        let prev_char_is_inaccurate_cjk = i >= 1 && line
                                .iter()
                                .nth(i - 1)
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
        if link_pairs.len() > 0 {
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
        return PrMatch::Zero;
    } else {
        match CHARLIST.get(&c) {
            Some(c_prs) => {
                match c_prs.get(pr) {
                    Some(_) => PrMatch::Full,
                    None => {
                        // try half pr (without tones), to accommodate for tone changes
                        let half_c_prs = c_prs
                            .keys()
                            .map(|pr| {
                                if let Some(tail) = pr.chars().last() {
                                    if tail.is_digit(10) {
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
                        if half_c_prs.contains(&&pr[..]) {
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
    prs: &Vec<&str>,
    pr_scores: &Vec<Vec<usize>>,
) -> HashMap<usize, usize> {
    let mut pr_map = HashMap::new();
    let mut i = pr_scores.len() - 1;
    let mut j = pr_scores[0].len() - 1;

    while i > 0 && j > 0 {
        // println!("i: {}, j: {}", i, j);
        let (_, word) = &line[i - 1];
        match match_pr(&word.to_string(), &prs[j - 1]) {
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

lazy_static! {
    static ref CHARLIST: CharList = {
        let charlist = include_bytes!("charlist.json");
        serde_json::from_slice(charlist).unwrap()
    };
}

pub fn enrich_dict(dict: &Dict) -> RichDict {
    dict.iter()
        .map(|(id, entry)| {
            let variants = &entry.variants.to_words();
            let rich_defs = entry
                .defs
                .iter()
                .map(|def| RichDef {
                    yue: def.yue.clone(),
                    yue_simp: clause_to_simplified(&def.yue),
                    eng: def.eng.clone(),
                    alts: def.alts.clone(),
                    egs: def.egs.iter().map(|eg| enrich_eg(variants, eg)).collect(),
                })
                .collect();
            (
                *id,
                RichEntry {
                    id: *id,
                    variants: entry.variants.clone(),
                    variants_simp: get_simplified_variants(&entry.variants, &entry.defs),
                    poses: entry.poses.clone(),
                    labels: entry.labels.clone(),
                    sims: entry.sims.clone(),
                    ants: entry.ants.clone(),
                    refs: entry.refs.clone(),
                    imgs: entry.imgs.clone(),
                    defs: rich_defs,
                    published: entry.published,
                },
            )
        })
        .collect::<RichDict>()
}

pub fn enrich_pr_line(variants: &Vec<&str>, pr_line: &PrLine) -> RichLine {
    match pr_line {
        (line, Some(pr)) => RichLine::Ruby(match_ruby(variants, line, &unicode::to_words(pr))),
        (line, None) => RichLine::Text(unflatten_word_line(&flatten_line(variants, line))),
    }
}

pub fn enrich_eg(variants: &Vec<&str>, eg: &Eg) -> RichEg {
    RichEg {
        zho: eg.zho.as_ref().map(|zho| enrich_pr_line(variants, &zho)),
        zho_simp: eg
            .zho
            .as_ref()
            .map(|zho| line_to_string(&line_to_simplified(&zho.0))),
        yue: eg.yue.as_ref().map(|yue| enrich_pr_line(variants, &yue)),
        yue_simp: eg
            .yue
            .as_ref()
            .map(|yue| line_to_string(&line_to_simplified(&yue.0))),
        eng: eg.eng.clone(),
    }
}

fn get_simplified_variants(trad_variants: &Variants, defs: &Vec<Def>) -> Vec<String> {
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
                        (&line_str_simp[variant_start_index..variant_end_index]).to_string(),
                    );
                }
            });
    });
    trad_variants
        .0
        .iter()
        .enumerate()
        .map(|(variant_index, trad_variant)| {
            if let Some(simp_variant) = simp_variants.get(&variant_index) {
                simp_variant.clone()
            } else {
                unicode::to_simplified(&trad_variant.word)
            }
        })
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
