//! A parser for [`words.hk`] (粵典)
//!
//! Parses all entries marked with OK and store the results as a list of entries.
//! This parser does not use any regular expressions, backtracking or other inefficient
//! parsing techniques. It is powered by a library called [`lip`] that provides
//! flexible parser combinators and supports friendly error messages.
//!
//! A note on doc format: we generally put examples after a colon ':'.
//!
//! [`words.hk`]: https://words.hk
//! [`lip`]: https://github.com/AlienKevin/lip
//!

#[macro_use]
mod tests;

use indoc::indoc;
use lazy_static::lazy_static;
use lip::ParseResult;
use lip::*;
use std::cmp;
use std::collections::HashMap;
use std::collections::HashSet;
use std::convert::identity;
use std::error::Error;
use std::fs;
use std::io;
use std::ops::Range;
use std::str::FromStr;
use strum::EnumString;
use unicode_names2;
use unicode_segmentation::UnicodeSegmentation;

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
    pub variants: Vec<Variant>,
    pub poses: Vec<String>,
    pub labels: Vec<String>,
    pub sims: Vec<String>,
    pub ants: Vec<String>,
    pub refs: Vec<String>,
    pub imgs: Vec<String>,
    pub defs: Vec<Def>,
}

/// A variant of a \[word\] with \[prs\] (pronounciations)
#[derive(Debug, Clone, PartialEq)]
pub struct Variant {
    pub word: String,
    pub prs: Vec<String>,
}

/// Two types of segments: text or link. See [Segment]
///
/// \[Text\] normal text
///
/// \[Link\] a link to another entry
///
#[derive(Debug, Clone, PartialEq)]
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

/// A segment containing a [Word]
///
/// `(SegmentType::Text, vec![(TextStyle::Bold, "兩"), (TextStyle::Normal, "周")])`
///
pub type WordSegment = (SegmentType, Word);

/// A consecutive series of [Text]s
///
/// `vec![(TextStyle::Bold, "兩"), (TextStyle::Normal, "周")]`
///
pub type Word = Vec<Text>;

/// A styled text segment
///
/// Normal: `(TextStyle::Normal, "好")`
///
/// Bold: `(TextStyle::Bold, "good")`
///
pub type Text = (TextStyle, String);

/// Text styles, can be bold or normal
#[derive(Debug, PartialEq, Clone)]
pub enum TextStyle {
    Bold,
    Normal,
}

/// A segment marked with pronunciation (called "ruby" in HTML)
///
/// Segment can be one of
/// * a single punctuation
/// * or a [Word] with its pronunciations
/// * or a linked segment with a series of [Word]s
/// and their pronunciations
#[derive(Debug, PartialEq, Clone)]
pub enum RubySegment {
    Punc(String),
    Word(Word, Vec<String>),
    LinkedWord(Vec<(Word, Vec<String>)>),
}

/// A line consists of one or more [Segment]s
///
/// Empty line: `vec![(Text, "")]`
///
/// Simple line: `vec![(Text, "用嚟圍喺BB牀邊嘅布（量詞：塊）")]`
///
/// Mixed line: `vec![(Text, "一種加入"), (Link, "蝦籽"), (Text, "整嘅廣東麪")]`
///
pub type Line = Vec<Segment>;

/// A line consists of one or more [RubySegment]s
pub type RubyLine = Vec<RubySegment>;

/// A line consists of one or more [WordSegment]s
pub type WordLine = Vec<WordSegment>;

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
#[derive(Debug, PartialEq, Clone, Copy)]
pub enum AltLang {
    Jpn, // Japanese
    Kor, // Korean
    Por, // Portuguese
    Vie, // Vietnamese
    Lat, // Latin
    Fra, // French
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

/// Parse the whole words.hk CSV database into a [Dict]
pub fn parse_dict() -> Result<Dict, Box<dyn Error>> {
    // Build the CSV reader and iterate over each record.
    let mut rdr = csv::Reader::from_reader(io::stdin());
    let mut dict: Dict = HashMap::new();
    for result in rdr.records() {
        let entry = result?;
        if &entry[4] == "OK" {
            let id: usize = entry[0].parse().unwrap();
            let head = &entry[1];
            let content = &entry[2];
            // entry[3] is always an empty string
            let head_parse_result = sequence(
                "",
                succeed!(|word, prs| Variant { word, prs })
                    .keep(take_chomped(chomp_while1c(&(|c: &char| c != &':'), "word")))
                    .keep(sequence(
                        ":",
                        BoxedParser::new(take_chomped(chomp_while1c(
                            &(|c: &char| c != &':' && c != &','),
                            "jyutping",
                        ))),
                        ":",
                        space0(),
                        "",
                        Trailing::Forbidden,
                    )),
                ",",
                space0(),
                "",
                Trailing::Forbidden,
            )
            .run(head, ());
            let entry: Option<Entry> = match head_parse_result {
                ParseResult::Ok {
                    output: head_result,
                    ..
                } => match parse_content(id, head_result).run(content, ()) {
                    ParseResult::Ok {
                        output: content_result,
                        ..
                    } => content_result,
                    ParseResult::Err { message, .. } => {
                        // println!("Error in #{}: {:?}", id, message);
                        None
                    }
                },
                ParseResult::Err { message, .. } => {
                    // println!("Error in #{}: {:?}", id, message);
                    None
                }
            };
            match entry {
                Some(e) => {
                    // println!("{:?}", e);
                    dict.insert(id, e);
                }
                None => {}
            };
        }
    }
    Ok(dict)
}

/// Parse tags on a word like pos, label, and sim
///
/// For example, here's the label tags of the word 佛系:
///
/// ```
/// # use wordshk_tools::*;
/// # let source = indoc::indoc! {"
/// (label:外來語)(label:潮語)
/// # "};
///
/// // which parses to:
///
/// # lip::assert_succeed(parse_tags("label"), source,
/// vec!["外來語".into(), "潮語".into()]
/// # );
/// ```
///
pub fn parse_tags<'a>(name: &'static str) -> lip::BoxedParser<'a, Vec<String>, ()> {
    return zero_or_more(
        succeed!(|tag| tag)
            .skip(token("("))
            .skip(token(name))
            .skip(token(":"))
            .keep(take_chomped(chomp_while1c(&(|c: &char| c != &')'), name)))
            .skip(token(")")),
    );
}

/// Parse a newline character
///
/// Supports both Windows "\r\n" and Unix "\n"
fn parse_br<'a>() -> lip::BoxedParser<'a, (), ()> {
    chomp_if(|c| c == "\r\n" || c == "\n", "a newline")
}

/// Parse a [Line]
///
/// For example, here's an English line:
///
/// ```
/// # use wordshk_tools::*;
/// # use wordshk_tools::SegmentType::*;
/// # let source = indoc::indoc! {"
/// My headphone cord was knotted.
/// # "};
///
/// // which parses to:
///
/// # lip::assert_succeed(parse_line("eng"), source,
/// vec![(Text, "My headphone cord was knotted.".into())]
/// # );
/// ```
pub fn parse_line<'a>(name: &'static str) -> lip::BoxedParser<'a, Line, ()> {
    succeed!(identity).keep(one_or_more(succeed!(|seg| seg).keep(one_of!(
            succeed!(|string| (SegmentType::Link, string))
                .skip(token("#"))
                .keep(take_chomped(chomp_while1c(
                    &(|c: &char| !is_punctuation(*c) && !c.is_whitespace()),
                    name
                )))
                .skip(optional("", token(" "))),
            succeed!(|string| (SegmentType::Text, string)).keep(take_chomped(chomp_while1c(
                &(|c: &char| *c != '#' && *c != '\n' && *c != '\r'),
                name
            )))
        ))))
}

/// Parse a [Line] tagged in front by its name/category
///
/// For example, here's an English line:
///
/// ```
/// # use wordshk_tools::*;
/// # use wordshk_tools::SegmentType::*;
/// # let source = indoc::indoc! {"
/// eng:My headphone cord was knotted.
/// # "};
///
/// // which parses to:
///
/// # lip::assert_succeed(parse_named_line("eng"), source,
/// vec![(Text, "My headphone cord was knotted.".into())]
/// # );
/// ```
///
pub fn parse_named_line<'a>(name: &'static str) -> lip::BoxedParser<'a, Line, ()> {
    succeed!(|clause| clause)
        .skip(token(name))
        .skip(token(":"))
        .keep(parse_line(name))
}

/// Parse a partial pronunciation [Line], until the opening paren of Jyutping pronunciations
///
/// For the following pronunciation line:
///
/// 可唔可以見面？ (ho2 m4 ho2 ji5 gin3 min6?)
///
/// This function will parse everything up until the '(':
///
/// ```
/// # use wordshk_tools::*;
/// # use wordshk_tools::SegmentType::*;
/// # let source = indoc::indoc! {"
/// 可唔可以見面？
/// # "};
///
/// // which parses to:
///
/// # lip::assert_succeed(parse_partial_pr_line("yue"), source,
/// vec![(Text, "可唔可以見面？".into())]
/// # );
/// ```
///
pub fn parse_partial_pr_line<'a>(name: &'static str) -> lip::BoxedParser<'a, Line, ()> {
    succeed!(identity).keep(one_or_more(succeed!(|seg| seg).keep(one_of!(
            succeed!(|string: String| (SegmentType::Link, string.trim_end().to_string()))
                .skip(token("#"))
                .keep(take_chomped(chomp_while1c(
                    &(|c: &char| !is_punctuation(*c) && !c.is_whitespace() && *c != '('),
                    name
                )))
                .skip(optional("", token(" "))),
            succeed!(|string: String| (SegmentType::Text, string.trim_end().to_string())).keep(
                take_chomped(chomp_while1c(
                    &(|c: &char| *c != '#' && *c != '\n' && *c != '\r' && *c != '('),
                    name
                ))
            )
        ))))
}

/// Parse a partial *named* pronunciation [Line], until the opening paren of Jyutping pronunciations
///
/// For the following *named* pronunciation line:
///
/// yue:可唔可以見面？ (ho2 m4 ho2 ji5 gin3 min6?)
///
/// This function will parse everything up until the '(':
///
/// ```
/// # use wordshk_tools::*;
/// # use wordshk_tools::SegmentType::*;
/// # let source = indoc::indoc! {"
/// yue:可唔可以見面？
/// # "};
///
/// // which parses to:
///
/// # lip::assert_succeed(parse_partial_pr_named_line("yue"), source,
/// vec![(Text, "可唔可以見面？".into())]
/// # );
/// ```
///
pub fn parse_partial_pr_named_line<'a>(name: &'static str) -> lip::BoxedParser<'a, Line, ()> {
    succeed!(identity)
        .skip(token(name))
        .skip(token(":"))
        .keep(parse_partial_pr_line(name))
}

/// Parse a [Clause] (can be single or multiline)
///
/// For example, here's a Cantonese clause:
///
/// ```
/// # use wordshk_tools::*;
/// # use wordshk_tools::SegmentType::*;
/// # let source = indoc::indoc! {"
/// 一行白鷺上青天
///
/// 兩個黃鸝鳴翠柳
/// # "};
///
/// // which parses to:
///
/// # lip::assert_succeed(parse_clause("yue"), source,
/// vec![vec![(Text, "一行白鷺上青天".into())], vec![(Text, "".into())], vec![(Text, "兩個黃鸝鳴翠柳".into())]]
/// # );
/// ```
///
pub fn parse_clause<'a>(expecting: &'static str) -> lip::BoxedParser<'a, Clause, ()> {
    succeed!(|first_line: Line, lines: Clause| {
        let mut all_lines = vec![first_line];
        all_lines.extend(lines);
        all_lines
    })
    .keep(parse_line(expecting))
    .keep(zero_or_more_until(
        succeed!(identity).skip(parse_br()).keep(one_of!(
            // non-empty line
            succeed!(identity).keep(parse_line(expecting)),
            // empty line
            succeed!(|_| vec!((SegmentType::Text, "".to_string()))).keep(token(""))
        )),
        succeed!(|_| ()).skip(parse_br()).keep(one_of!(
            succeed!(|_| ()).keep(token("<eg>")),
            succeed!(|_| ()).keep(token("----")),
            succeed!(|_| ())
                .keep(chomp_ifc(|c| *c != '\r' && *c != '\n', "any char"))
                .skip(chomp_ifc(|c| *c != '\r' && *c != '\n', "any char"))
                .skip(chomp_ifc(|c| *c != '\r' && *c != '\n', "any char"))
                .skip(chomp_ifc(|c| *c == ':', "colon `:`"))
        )),
    ))
    .skip(optional((), parse_br()))
}

/// Parse a named [Clause] (can be single or multiline)
///
/// For example, here's a named Cantonese clause:
///
/// ```
/// # use wordshk_tools::*;
/// # use wordshk_tools::SegmentType::*;
/// # let source = indoc::indoc! {"
/// yue:一行白鷺上青天
///
/// 兩個黃鸝鳴翠柳
/// # "};
///
/// // which parses to:
///
/// # lip::assert_succeed(parse_named_clause("yue"), source,
/// vec![vec![(Text, "一行白鷺上青天".into())], vec![(Text, "".into())], vec![(Text, "兩個黃鸝鳴翠柳".into())]]
/// # );
/// ```
///
pub fn parse_named_clause<'a>(name: &'static str) -> lip::BoxedParser<'a, Clause, ()> {
    succeed!(identity)
        .skip(token(name))
        .skip(token(":"))
        .keep(parse_clause(name))
}

/// Parse a clause in an alternative language
///
/// For example, here's a Japanese clause:
///
/// ```
/// # use wordshk_tools::*;
/// # use wordshk_tools::SegmentType::*;
/// # use wordshk_tools::AltLang;
/// # let source = indoc::indoc! {"
/// jpn:年画；ねんが
/// # "};
///
/// // which parses to:
///
/// # lip::assert_succeed(parse_alt_clause(), source,
/// (AltLang::Jpn, vec![vec![(Text, "年画；ねんが".into())]])
/// # );
/// ```
///
pub fn parse_alt_clause<'a>() -> lip::BoxedParser<'a, AltClause, ()> {
    (succeed!(|alt_lang: Located<String>, clause: Clause| (alt_lang, clause))
        .keep(located(take_chomped(chomp_while1c(
            |c: &char| *c != ':',
            "alternative languages",
        ))))
        .skip(token(":"))
        .keep(parse_clause("alternative language clause")))
    .and_then(|(alt_lang, clause)| match &alt_lang.value[..] {
        "jpn" => succeed!(|_| (AltLang::Jpn, clause)).keep(token("")),
        "kor" => succeed!(|_| (AltLang::Kor, clause)).keep(token("")),
        "por" => succeed!(|_| (AltLang::Por, clause)).keep(token("")),
        "vie" => succeed!(|_| (AltLang::Vie, clause)).keep(token("")),
        "lat" => succeed!(|_| (AltLang::Lat, clause)).keep(token("")),
        "fra" => succeed!(|_| (AltLang::Fra, clause)).keep(token("")),
        _ => {
            let from = alt_lang.from;
            let to = alt_lang.to;
            problem(
                format!("Invalid alternative language: {}", alt_lang.value),
                move |_| from,
                move |_| to,
            )
        }
    })
}

/// Parse a Jyutping pronunciation line, for Cantonese (yue) and Mandarin (zho)
///
/// For example, here's a Cantonese pronunciation line:
///
/// ```
/// # use wordshk_tools::*;
/// # use wordshk_tools::SegmentType::*;
/// # let source = indoc::indoc! {"
/// yue:我個耳筒繑埋咗一嚿。 (ngo5 go3 ji5 tung2 kiu5 maai4 zo2 jat1 gau6.)
/// # "};
///
/// // which parses to:
///
/// # lip::assert_succeed(parse_pr_line("yue"), source,
/// (vec![(Text, "我個耳筒繑埋咗一嚿。".into())], Some("ngo5 go3 ji5 tung2 kiu5 maai4 zo2 jat1 gau6.".into()))
/// # );
/// ```
///
pub fn parse_pr_line<'a>(name: &'static str) -> lip::BoxedParser<'a, PrLine, ()> {
    succeed!(|line, pr| (line, pr))
        .keep(parse_partial_pr_named_line(name))
        .keep(optional(
            None,
            succeed!(Some)
                .skip(token("("))
                .keep(take_chomped(chomp_while1c(
                    &|c: &char| *c != ')',
                    "jyutping",
                )))
                .skip(token(")")),
        ))
}

/// Parse an example for a word
///
/// For example, here's an example for the word 便:
///
/// ```
/// # use wordshk_tools::*;
/// # use wordshk_tools::SegmentType::*;
/// # let source = indoc::indoc! {"
/// <eg>
/// zho:後邊 (hau6 bin6)
/// yue:#後便 (hau6 bin6)
/// eng:back side
/// # "};
///
/// // which parses to:
///
/// # lip::assert_succeed(parse_eg(), source,
/// Eg {
///     zho: Some((vec![(Text, "後邊".into())], Some("hau6 bin6".into()))),
///     yue: Some((vec![(Link, "後便".into())], Some("hau6 bin6".into()))),
///     eng: Some(vec![(Text, "back side".into())]),
/// }
/// # );
/// ```
///
pub fn parse_eg<'a>() -> lip::BoxedParser<'a, Eg, ()> {
    succeed!(|zho, yue, eng| Eg { zho, yue, eng })
        .skip(token("<eg>"))
        .skip(parse_br())
        .keep(optional(
            None,
            succeed!(Some)
                .keep(parse_pr_line("zho"))
                .skip(optional((), parse_br())),
        ))
        .keep(optional(
            None,
            succeed!(Some)
                .keep(parse_pr_line("yue"))
                .skip(optional((), parse_br())),
        ))
        // only a single line is accepted in eg
        .keep(optional(None, succeed!(Some).keep(parse_named_line("eng"))))
        .skip(optional((), parse_br()))
}

/// Parse a rich definition
///
/// Rich definitions start with an <explanation> tag and
/// contains one or more <eg> tags.
///
/// For example, here's part of the rich definition for the word 便:
///
/// ```
/// # use wordshk_tools::*;
/// # use wordshk_tools::SegmentType::*;
/// # let source = indoc::indoc! {"
/// <explanation>
/// yue:用於方位詞之後。書寫時，亦會用#邊 代替本字
/// eng:suffix for directional/positional noun
/// <eg>
/// yue:#開便 (hoi1 bin6)
/// eng:outside
/// <eg>
/// yue:#呢便 (nei1 bin6)
/// eng:this side
/// # "};
///
/// // which parses to:
///
/// # lip::assert_succeed(parse_rich_def(), source,
/// Def {
///     yue: vec![vec![(Text, "用於方位詞之後。書寫時，亦會用".into()), (Link, "邊".into()), (Text, "代替本字".into())]],
///     eng: Some(vec![vec![(Text, "suffix for directional/positional noun".into())]]),
///     alts: vec![],
///     egs: vec![ Eg {
///             zho: None,
///             yue: Some((vec![(Link, "開便".into())], Some("hoi1 bin6".into()))),
///             eng: Some(vec![(Text, "outside".into())]),
///         },
///         Eg {
///             zho: None,
///             yue: Some((vec![(Link, "呢便".into())], Some("nei1 bin6".into()))),
///             eng: Some(vec![(Text, "this side".into())]),
///         },
///     ],
/// }
/// # );
/// ```
///
pub fn parse_rich_def<'a>() -> lip::BoxedParser<'a, Def, ()> {
    succeed!(|yue, eng, alts, egs| Def {
        yue,
        eng,
        alts,
        egs
    })
    .skip(token("<explanation>"))
    .skip(parse_br())
    .keep(parse_named_clause("yue"))
    .keep(optional(
        None,
        succeed!(Some).keep(parse_named_clause("eng")),
    ))
    .keep(zero_or_more(
        succeed!(|clause| clause)
            .keep(parse_alt_clause())
            .skip(optional((), parse_br())),
    ))
    .keep(one_or_more(parse_eg()))
}

/// Parse a simple definition
///
/// For example, here's a simple definition for the word 奸爸爹
///
/// ```
/// # use wordshk_tools::*;
/// # use wordshk_tools::SegmentType::*;
/// # let source = indoc::indoc! {"
/// yue:#加油
/// eng:cheer up
/// jpn:頑張って（がんばって）
/// # "};
///
/// // which parses to:
///
/// # lip::assert_succeed(parse_simple_def(), source,
/// Def {
///     yue: vec![vec![(Link, "加油".into())]],
///     eng: Some(vec![vec![(Text, "cheer up".into())]]),
///     alts: vec![(AltLang::Jpn, vec![vec![(Text, "頑張って（がんばって）".into())]])],
///     egs: vec![],
/// }
/// # );
/// ```
///
pub fn parse_simple_def<'a>() -> lip::BoxedParser<'a, Def, ()> {
    succeed!(|yue, eng, alts| Def {
        yue,
        eng,
        egs: Vec::new(),
        alts
    })
    .keep(parse_named_clause("yue"))
    .keep(optional(
        None,
        succeed!(Some).keep(parse_named_clause("eng")),
    ))
    .keep(zero_or_more(
        succeed!(|clause| clause)
            .keep(parse_alt_clause())
            .skip(optional((), parse_br())),
    ))
}

/// Parse a series of definitions for a word, separated by "\-\-\-\-"
///
/// For example, here's a series of definitions for the word 兄
///
/// ```
/// # use wordshk_tools::*;
/// # use wordshk_tools::SegmentType::*;
/// # let source = indoc::indoc! {"
/// <explanation>
/// yue:同父母或者同監護人，年紀比你大嘅男性
/// eng:elder brother
/// <eg>
/// yue:#兄弟 (hing1 dai6)
/// eng:brothers
/// ----
/// yue:對男性朋友嘅尊稱
/// eng:politely addressing a male friend
/// # "};
///
/// // which parses to:
///
/// # lip::assert_succeed(parse_defs(), source,
/// vec![
///     Def {
///         yue: vec![vec![(Text, "同父母或者同監護人，年紀比你大嘅男性".into())]],
///         eng: Some(vec![vec![(Text, "elder brother".into())]]),
///         alts: vec![],
///         egs: vec![
///             Eg {
///                 zho: None,
///                 yue: Some((vec![(Link, "兄弟".into())], Some("hing1 dai6".into()))),
///                 eng: Some(vec![(Text, "brothers".into())]),
///             }
///         ],
///     },
///     Def {
///         yue: vec![vec![(Text, "對男性朋友嘅尊稱".into())]],
///         eng: Some(vec![vec![(Text, "politely addressing a male friend".into())]]),
///         alts: vec![],
///         egs: vec![],
///     },
/// ]
/// # );
/// ```
///
pub fn parse_defs<'a>() -> lip::BoxedParser<'a, Vec<Def>, ()> {
    succeed!(|defs| defs).keep(one_or_more(
        succeed!(|def| def)
            .keep(one_of!(parse_simple_def(), parse_rich_def()))
            .skip(optional(
                (),
                succeed!(|_| ()).keep(token("----")).skip(parse_br()),
            )),
    ))
}

/// Parse the content of an [Entry]
///
/// id and variants are parsed by [parse_dict] and passed in to this function.
///
/// For example, here's the content of the Entry for 奸爸爹
///
/// ```
/// # use wordshk_tools::*;
/// # use wordshk_tools::SegmentType::*;
/// # use wordshk_tools::AltLang;
/// # let source = indoc::indoc! {"
/// (pos:語句)(label:外來語)(label:潮語)(label:香港)
/// yue:#加油
/// eng:cheer up
/// jpn:頑張って（がんばって）
/// # "};
///
/// let id = 98634;
/// let variants = vec![(Variant {word: "奸爸爹".into(), prs: vec!["gaan1 baa1 de1".into()]})];
///
/// // which parses to:
///
/// # lip::assert_succeed(parse_content(id, variants.clone()), source,
/// Some(Entry {
/// id: id,
/// variants: variants,
/// poses: vec!["語句".into()],
/// labels: vec!["外來語".into(), "潮語".into(), "香港".into()],
/// sims: vec![],
/// ants: vec![],
/// refs: vec![],
/// imgs: vec![],
/// defs: vec![Def {
///     yue: vec![vec![(Link, "加油".into())]],
///     eng: Some(vec![vec![(Text, "cheer up".into())]]),
///     alts: vec![(AltLang::Jpn, vec![vec![(Text, "頑張って（がんばって）".into())]])],
///     egs: vec![],
///     }]
/// })
/// # );
/// ```
///
pub fn parse_content<'a>(
    id: usize,
    variants: Vec<Variant>,
) -> lip::BoxedParser<'a, Option<Entry>, ()> {
    one_of!(
        succeed!(|poses, labels, sims, ants, refs, imgs, defs| Some(Entry {
            id,
            variants,
            poses,
            labels,
            sims,
            ants,
            refs,
            imgs,
            defs,
        }))
        .keep(parse_tags("pos"))
        .keep(parse_tags("label"))
        .keep(parse_tags("sim"))
        .keep(parse_tags("ant"))
        .keep(parse_tags("ref"))
        .keep(parse_tags("img"))
        .skip(parse_br())
        .keep(parse_defs()),
        succeed!(|_| None).keep(token("未有內容 NO DATA"))
    )
}

type CharList = HashMap<char, HashMap<String, usize>>;

// type WordList = HashMap<String, Vec<String>>;

// source: https://stackoverflow.com/a/35907071/6798201
fn find_subsequences<T>(haystack: &[T], needle: &[T]) -> Vec<(usize, usize)>
where
    for<'a> &'a [T]: PartialEq,
{
    haystack
        .windows(needle.len())
        .enumerate()
        .filter_map(|(i, window)| {
            if window == needle {
                Some((i, i + needle.len() - 1))
            } else {
                None
            }
        })
        .collect()
}

fn tokenize(variants: &Vec<String>, text: &str) -> Vec<Word> {
    let mut i = 0;
    let mut words: Vec<Word> = vec![];
    let gs = UnicodeSegmentation::graphemes(&text[..], true).collect::<Vec<&str>>();
    let mut start_end_pairs: Vec<(usize, usize)> = vec![];
    variants.iter().for_each(|variant| {
        let variant = UnicodeSegmentation::graphemes(&variant[..], true).collect::<Vec<&str>>();
        find_subsequences(&gs, &variant)
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
        if test_g(is_cjk, g) {
            if start_end_pairs.iter().any(|(start, _)| start == &i) {
                is_bolded = true;
            }
            words.push(vec![(
                if is_bolded {
                    TextStyle::Bold
                } else {
                    TextStyle::Normal
                },
                g.to_string(),
            )]);
            if start_end_pairs.iter().any(|(_, end)| end == &i) {
                is_bolded = false;
            }
            i += 1;
        } else if test_g(is_alphanumeric, g) {
            let mut j = i;
            let mut prev_bold_end_index: Option<usize> = None;
            let mut bold_start_index = None;
            let mut bold_end_index = None;
            let mut word: Word = vec![];
            while j < gs.len()
                && (test_g(is_alphanumeric, gs[j]) || (test_g(char::is_whitespace, gs[j])))
            {
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
            if let Some(bold_end) = prev_bold_end_index {
                if bold_end + 1 < j {
                    let rest = gs[bold_end + 1..j].join("").trim_end().to_string();
                    if rest.len() > 0 {
                        word.push((TextStyle::Normal, rest));
                    }
                }
            } else if let Some(bold_start) = bold_start_index {
                word.push((
                    TextStyle::Bold,
                    gs[bold_start..j].join("").trim_end().into(),
                ));
            } else {
                word.push((TextStyle::Normal, gs[i..j].join("").trim_end().into()));
            }
            words.push(word);
            i = j;
        } else {
            // a punctuation or space
            if !test_g(char::is_whitespace, g) {
                words.push(vec![(
                    if is_bolded {
                        TextStyle::Bold
                    } else {
                        TextStyle::Normal
                    },
                    g.to_string(),
                )]);
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
pub fn flatten_line(variants: &Vec<String>, line: &Line) -> WordLine {
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

fn word_to_string(word: &Word) -> String {
    word.iter()
        .map(|(_, seg)| seg.clone())
        .collect::<Vec<String>>()
        .join("")
}

fn get_xml_start_tag(style: &TextStyle) -> &'static str {
    match style {
        TextStyle::Normal => "",
        TextStyle::Bold => "<b>",
    }
}

fn get_xml_end_tag(style: &TextStyle) -> &'static str {
    match style {
        TextStyle::Normal => "",
        TextStyle::Bold => "</b>",
    }
}

fn word_to_xml(word: &Word) -> String {
    word.iter()
        .map(|(style, seg)| {
            format!(
                "{start_tag}{content}{end_tag}",
                start_tag = get_xml_start_tag(style),
                content = xml_escape(seg),
                end_tag = get_xml_end_tag(style),
            )
        })
        .collect::<Vec<String>>()
        .join("")
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
pub fn match_ruby(variants: &Vec<String>, line: &Line, prs: &Vec<&str>) -> RubyLine {
    let line = flatten_line(variants, line);
    let pr_scores = match_ruby_construct_table(&line, prs);
    let pr_map = match_ruby_backtrack(&line, prs, &pr_scores);
    // println!("{:?}", pr_map);
    let flattened_ruby_line = line
        .iter()
        .enumerate()
        .map(|(i, (seg_type, word))| match pr_map.get(&i) {
            Some(j) => create_ruby_segment(seg_type, word, &prs[*j..j + 1]),
            None => {
                let word_str = word_to_string(word);
                if test_g(is_punctuation, &word_str) {
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
                    create_ruby_segment(seg_type, word, &prs[start..end])
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
        PrMatch::Full => 2,
        PrMatch::Half => 1,
        PrMatch::Zero => 0,
    }
}

fn match_pr(seg: &String, pr: &str) -> PrMatch {
    if seg.chars().count() != 1 {
        return PrMatch::Zero;
    }
    let c = seg.chars().next().unwrap();
    match CHARLIST.get(&c) {
        Some(c_prs) => {
            match c_prs.get(pr) {
                Some(_) => PrMatch::Full,
                None => {
                    // try half pr (without tones), to accomodate for tone changes
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

fn match_ruby_construct_table(line: &WordLine, prs: &Vec<&str>) -> Vec<Vec<usize>> {
    let m = line.len() + 1;
    let n = prs.len() + 1;
    let mut pr_scores = vec![vec![0; n]; m];
    // println!("m: {}, n: {}", m, n);
    for i in 1..m {
        for j in 1..n {
            // println!("i: {}, j: {}", i, j);
            let (_, word) = &line[i - 1];
            let cell_pr_match = match_pr(&word_to_string(&word), prs[j - 1]);
            match cell_pr_match {
                PrMatch::Full | PrMatch::Half => {
                    pr_scores[i][j] = pr_scores[i - 1][j - 1] + pr_match_to_score(cell_pr_match);
                }
                PrMatch::Zero => {
                    let top_pr_score = pr_scores[i - 1][j];
                    let left_pr_score = pr_scores[i][j - 1];
                    pr_scores[i][j] = cmp::max(top_pr_score, left_pr_score);
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
        match match_pr(&word_to_string(&word), &prs[j - 1]) {
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
    static ref PUNCTUATIONS: HashSet<char> = {
        HashSet::from([
            // Shared punctuations
            '@', '#', '$', '%', '^', '&', '*',
            // English punctuations
            '~', '`', '!',  '(', ')', '-', '_', '{', '}', '[', ']', '|', '\\', ':', ';',
            '"', '\'', '<', '>', ',', '.', '?', '/',
            // Chinese punctuations
            '～', '·', '！', '：', '；', '“', '”', '‘', '’', '【', '】', '（', '）',
            '「', '」', '《', '》', '？', '，', '。', '、', '／', '＋'
        ])
    };

    static ref CHARLIST: CharList = {
        let charlist_file = fs::File::open("charlist.json").unwrap();
        let charlist_reader = io::BufReader::new(charlist_file);
        serde_json::from_reader(charlist_reader).unwrap()
    };
}

/// Test whether a character is a Chinese/English punctuation
fn is_punctuation(c: char) -> bool {
    PUNCTUATIONS.contains(&c)
}

/// Test if a character is latin small or capital letter
fn is_latin(c: char) -> bool {
    if let Some(name) = unicode_names2::name(c) {
        let name = format!("{}", name);
        name.starts_with("LATIN SMALL LETTER") || name.starts_with("LATIN CAPITAL LETTER")
    } else {
        false
    }
}

fn is_alphanumeric(c: char) -> bool {
    let cp = c as i32;
    (0x30 <= cp && cp < 0x40) || (0xFF10 <= cp && cp < 0xFF20) || is_latin(c)
}

fn is_cjk(c: char) -> bool {
    let cp = c as i32;
    (0x3400 <= cp && cp <= 0x4DBF)
        || (0x4E00 <= cp && cp <= 0x9FFF)
        || (0xF900 <= cp && cp <= 0xFAFF)
        || (0x20000 <= cp && cp <= 0x2FFFF)
}

fn test_g(f: fn(char) -> bool, g: &str) -> bool {
    if let Some(c) = g.chars().next() {
        g.chars().count() == 1 && f(c)
    } else {
        false
    }
}

/// Escape '<' and '&' in an XML string
fn xml_escape(s: &String) -> String {
    s.replace("<", "&lt;").replace("&", "&amp;")
}

fn segment_to_xml((seg_type, seg): &Segment) -> String {
    match seg_type {
        SegmentType::Text => xml_escape(seg),
        SegmentType::Link => link_to_xml(&xml_escape(&seg), &xml_escape(&seg)),
    }
}

fn word_segment_to_xml((seg_type, word): &WordSegment) -> String {
    match seg_type {
        SegmentType::Text => word_to_xml(word),
        SegmentType::Link => link_to_xml(&word_to_string(&word), &word_to_xml(&word)),
    }
}

fn link_to_xml(link: &String, word: &String) -> String {
    format!(
        r#"<a href="x-dictionary:d:{}:{dict_id}">{}</a>"#,
        link,
        word,
        dict_id = "wordshk"
    )
}

fn clause_to_xml_with_class_name(class_name: &str, clause: &Clause) -> String {
    format!(
        "<div class=\"{}\">{}</div>",
        class_name,
        clause
            .iter()
            .map(|line| {
                line.iter()
                    .map(segment_to_xml)
                    .collect::<Vec<String>>()
                    .join("")
            })
            .collect::<Vec<String>>()
            .join("\n")
    )
}

// Convert simple PrClause (without Pr) to XML
fn simple_pr_clause_to_xml(variants: &Vec<String>, clause: &Clause) -> String {
    format!(
        "<div class=\"{}\">{}</div>",
        "pr-clause",
        clause
            .iter()
            .map(|line| {
                flatten_line(variants, line)
                    .iter()
                    .map(word_segment_to_xml)
                    .collect::<Vec<String>>()
                    .join("")
            })
            .collect::<Vec<String>>()
            .join("\n")
    )
}

/// Convert a [Clause] to an Apple Dictionary XML string
fn clause_to_xml(clause: &Clause) -> String {
    clause_to_xml_with_class_name("clause", clause)
}

/// Convert a [PrLine] to an Apple Dictionary XML string
fn pr_line_to_xml(variants: &Vec<String>, (line, pr): &PrLine) -> String {
    match pr {
        Some(pr) => {
            let prs = pr.unicode_words().collect::<Vec<&str>>();
            let ruby_line = match_ruby(variants, line, &prs);
            let mut output = "<ruby class=\"pr-clause\">".to_string();
            ruby_line.iter().for_each(|seg| match seg {
                RubySegment::LinkedWord(pairs) => {
                    let mut ruby = "<ruby>".to_string();
                    let mut word_str = String::new();
                    pairs.iter().for_each(|(word, prs)| {
                        ruby += &format!(
                            "\n<rb>{}</rb>\n<rt>{}</rt>",
                            word_to_xml(word),
                            prs.join(" ")
                        );
                        word_str += &word_to_string(word);
                    });
                    ruby += "\n</ruby>";
                    output += &format!("<rb>{}</rb><rt></rt>", &link_to_xml(&word_str, &ruby));
                }
                RubySegment::Word(word, prs) => {
                    output += &format!(
                        "\n<rb>{}</rb>\n<rt>{}</rt>",
                        word_to_xml(word),
                        prs.join(" ")
                    );
                }
                RubySegment::Punc(punc) => {
                    output += &format!("\n<rb>{}</rb>\n<rt></rt>", punc);
                }
            });
            output += "\n</ruby>";
            output
        }
        None => simple_pr_clause_to_xml(variants, &vec![line.clone()]),
    }
}

/// Convert [AltLang] to a language name in Cantonese
fn to_yue_lang_name(lang: AltLang) -> String {
    match lang {
        AltLang::Jpn => "日文",
        AltLang::Kor => "韓文",
        AltLang::Por => "葡萄牙文",
        AltLang::Vie => "越南文",
        AltLang::Lat => "拉丁文",
        AltLang::Fra => "法文",
    }
    .to_string()
}

fn to_xml_badge_helper(is_emphasized: bool, tag: &String) -> String {
    format!(
        "<span class=\"badge{}\">{}</span>",
        if is_emphasized { "-em" } else { "-weak" },
        tag
    )
}

fn to_xml_badge_em(tag: &String) -> String {
    to_xml_badge_helper(true, tag)
}

fn to_xml_badge(tag: &String) -> String {
    to_xml_badge_helper(false, tag)
}

/// Convert a [Dict] to Apple Dictionary XML format
pub fn dict_to_xml(dict: Dict) -> String {
    let front_back_matter_filename = "apple_dict/front_back_matter.html";
    let front_back_matter = fs::read_to_string(front_back_matter_filename).expect(&format!(
        "Something went wrong when I tried to read {}",
        front_back_matter_filename
    ));

    let header = format!(
        r#"<?xml version="1.0" encoding="UTF-8"?>
<d:dictionary xmlns="http://www.w3.org/1999/xhtml" xmlns:d="http://www.apple.com/DTDs/DictionaryService-1.0.rng">
<d:entry id="front_back_matter" d:title="Front/Back Matter">
{}
</d:entry>
"#,
        front_back_matter
    );

    let entries = dict
        .iter()
        .map(|(_id, entry)| {
            let variant_words = entry.variants.iter().map(|variant| variant.word.clone()).collect::<Vec<String>>();
            let entry_str = format!(
                indoc! {r#"
                <d:entry id="{id}" d:title="{variant_0_word}">
                <div class="entry">
                {variants_index}
                <div class ="entry-head">
                {variants_word_pr}
                {tags}
                </div>
                <div>
                {defs}
                </div>
                </div>
                </d:entry>"#},
                id = entry.id,
                variant_0_word = entry.variants[0].word,
                variants_index = entry
                    .variants
                    .iter()
                    .map(|variant| {
                        let prs = variant.prs.join(", ");
                        format!(
                            indoc!{r#"<d:index d:value="{word}" d:pr="{prs}"/>
                            {pr_indices}"#},
                            word = variant.word,
                            prs = prs,
                            pr_indices = variant.prs.iter().map(|pr| {
                                let word_and_pr = variant.word.clone() + " " + &pr;
                                format!(r#"<d:index d:value="{pr}" d:title="{word_and_pr}" d:priority="2"/>{simple_pr}"#,
                                    pr = pr,
                                    word_and_pr = word_and_pr,
                                    simple_pr = {
                                        let simple_pr = pr.split_whitespace().map(|seg|
                                            if seg.chars().last().unwrap().is_digit(10) {
                                                let mut chars = seg.chars();
                                                chars.next_back();
                                                chars.as_str()
                                            } else {
                                                seg
                                            }).collect::<Vec<&str>>().join(" ");
                                        if simple_pr == *pr {
                                            "".to_string()
                                        } else {
                                            format!(r#"<d:index d:value="{simple_pr}" d:title="{word_and_pr}" d:priority="2"/>"#,
                                                simple_pr = simple_pr,
                                                word_and_pr = word_and_pr
                                            )
                                        }
                                    })
                                }
                            ).collect::<Vec<String>>().join("\n")
                        )
                    })
                    .collect::<Vec<String>>()
                    .join("\n"),
                variants_word_pr = entry
                    .variants
                    .iter()
                    .map(|variant| {
                        format!(
                            indoc! {r#"<div>
                            <span d:priority="2"><h1>{}</h1></span>
                            <span class="prs"><span d:pr="JYUTPING">{}</span></span>
                            </div>"#},
                            variant.word,
                            variant.prs.join(", ")
                        )
                    })
                    .collect::<Vec<String>>()
                    .join("\n"),
                tags = "<div class=\"tags\">\n".to_string()
            + &(if entry.poses.len() > 0 { format!("<span>詞性：{}</span>\n", entry.poses.iter().map(to_xml_badge_em).collect::<Vec<String>>().join("，")) } else { "".to_string() })
            + &(if entry.labels.len() > 0 { format!("<span> ｜ 標籤：{}</span>\n", entry.labels.iter().map(to_xml_badge).collect::<Vec<String>>().join("，")) } else { "".to_string() })
            + &(if entry.sims.len() > 0 { format!("<span> ｜ 近義：{}</span>\n", entry.sims.join("，")) } else { "".to_string() })
            + &(if entry.ants.len() > 0 { format!("<span> ｜ 反義：{}</span>\n", entry.ants.join("，")) } else { "".to_string() })
            // TODO: add refs 
            // TODO: add imgs
            + "</div>",
                defs = "<ol>\n".to_string()
                    + &entry
                        .defs
                        .iter()
                        .map(|def| {
                            "<li>\n".to_string()
                                + "<div class=\"def-head\">\n"
                                + &format!("<div class=\"def-yue\"> <div>【粵】</div> {} </div>\n", clause_to_xml(&def.yue))
                                + &def.eng.clone().map_or("".to_string(), |eng| {
                                    format!("<div class=\"def-eng\"> <div>【英】</div> {} </div>\n", clause_to_xml(&eng))
                                })
                                + &def
                                    .alts
                                    .iter()
                                    .map(|(lang, clause)| {
                                        format!(
                                            "<div class=\"def-alt\"> <div>【{lang_name}】</div> {clause} </div>\n",
                                            lang_name = to_yue_lang_name(*lang),
                                            clause = clause_to_xml(clause)
                                        )
                                    })
                                    .collect::<Vec<String>>()
                                    .join("")
                                + "</div>\n"
                                + &def
                                    .egs
                                    .iter()
                                    .map(|eg| {
                                        "<div class=\"eg\">\n".to_string()
                                        + &eg.zho.clone().map_or("".to_string(), |zho| {
                                            let clause = pr_line_to_xml(&variant_words, &zho);
                                            format!(
                                                "<div class=\"eg-clause\"> <div class=\"lang-tag-ch\">（中）</div> {} </div>\n",
                                                clause
                                            )
                                        }) + &eg.yue.clone().map_or("".to_string(), |yue| {
                                            let clause = pr_line_to_xml(&variant_words, &yue);
                                            format!(
                                                "<div class=\"eg-clause\"> <div class=\"lang-tag-ch\">（粵）</div> {} </div>\n",
                                                clause
                                            )
                                        }) + &eg.eng.clone().map_or("".to_string(), |eng| {
                                            format!("<div class=\"eg-clause\"> <div>（英）</div> <div class=\"eng-eg\">{}</div> </div>\n", clause_to_xml(&vec![eng]))
                                        })
                                        + "</div>"
                                    })
                                    .collect::<Vec<String>>()
                                    .join("\n")
                                + "</li>\n"
                        })
                        .collect::<Vec<String>>()
                        .join("\n")
                    + "</ol>"
            );

            entry_str
        })
        .collect::<Vec<String>>()
        .join("\n\n");
    header.to_string() + &entries + "\n</d:dictionary>\n"
}

// fn tokenize(text: &str) -> Vec<&str> {
//     let mut i = 0;
//     let mut tokens = vec![];
//     let gs = UnicodeSegmentation::graphemes(&text[..], true).collect::<Vec<&str>>();
//     while i < gs.len() {
//         let g = gs[i];
//         let mut token = String::new();
//         while i < gs.len() && test_g(is_cjk, g) {
//             token += g;
//             i += 1;
//         }
//         if token.len() > 0 {
//             tokens.push(token.as_str());
//             token = String::new();
//         }
//         while i < gs.len() && test_g(is_alphanumeric, g) {
//             let mut j = i + 1;
//             while j < gs.len() && (test_g(is_alphanumeric, gs[j]) || (test_g(char::is_whitespace, gs[j]))) {
//                 j+=1;
//             }
//             tokens.push(gs[i..j].join("").trim_end());
//             i = j;
//         }
//         if token.len() > 0 {
//             tokens.push(token.as_str());
//         }
//         while i < gs.len() && !test_g(is_cjk, g) && !test_g(is_alphanumeric, g) { // a punctuation or space
//             if !test_g(char::is_whitespace, g) {
//                 tokens.push(g);
//             }
//             i += 1;
//         }
//     }
//     tokens
// }

/// JyutPing encoding with initial, nucleus (required), coda, and tone
///
/// Phonetics info based on: <https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.148.6501&rep=rep1&type=pdf>
#[derive(Debug, PartialEq)]
pub struct JyutPing {
    pub initial: Option<JyutPingInitial>,
    pub nucleus: JyutPingNucleus,
    pub coda: Option<JyutPingCoda>,
    pub tone: Option<JyutPingTone>,
}

/// Initial segment of a JyutPing, optional
///
/// Eg: 's' in "sap6"
///
#[derive(EnumString, Debug, PartialEq)]
#[strum(ascii_case_insensitive)]
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

/// Nucleus segment of a Jyutping, always required
///
/// Eg: 'a' in "sap6"
///
#[derive(EnumString, Debug, PartialEq)]
#[strum(ascii_case_insensitive)]
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
#[derive(EnumString, Debug, PartialEq)]
#[strum(ascii_case_insensitive)]
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
#[derive(EnumString, Debug, PartialEq)]
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

/// Parse [JyutPing] pronunciation
pub fn parse_jyutping(str: &String) -> Option<JyutPing> {
    let mut start = 0;

    let initial: Option<JyutPingInitial> = parse_jyutping_initial(str).map(|(_initial, _start)| {
        start = _start;
        _initial
    });

    let nucleus: JyutPingNucleus;
    match parse_jyutping_nucleus(start, str) {
        Some((_nucleus, _start)) => {
            nucleus = _nucleus;
            start = _start;
        }
        None => {
            return None;
        }
    };

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

fn parse_jyutping_component<T: FromStr>(start: usize, str: &String) -> Option<(T, usize)> {
    get_slice(str, start..start + 2).and_then(|first_two| match T::from_str(first_two) {
        Ok(component) => Some((component, start + 2)),
        Err(_) => {
            get_slice(str, start..start + 1).and_then(|first_one| match T::from_str(first_one) {
                Ok(component) => Some((component, start + 1)),
                Err(_) => None,
            })
        }
    })
}

fn parse_jyutping_initial(str: &String) -> Option<(JyutPingInitial, usize)> {
    parse_jyutping_component::<JyutPingInitial>(0, str)
}

fn parse_jyutping_nucleus(start: usize, str: &String) -> Option<(JyutPingNucleus, usize)> {
    parse_jyutping_component::<JyutPingNucleus>(start, str)
}

fn parse_jyutping_coda(start: usize, str: &String) -> Option<(JyutPingCoda, usize)> {
    parse_jyutping_component::<JyutPingCoda>(start, str)
}

fn parse_jyutping_tone(start: usize, str: &String) -> Option<JyutPingTone> {
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
