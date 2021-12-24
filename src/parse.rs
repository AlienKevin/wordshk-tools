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

use super::unicode;

use lip::ParseResult;
use lip::*;
use std::collections::HashMap;
use std::convert::identity;
use std::error::Error;
use std::io;
use std::ops::Range;
use std::str::FromStr;

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
    pub prs: Vec<LaxJyutPing>,
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
                        BoxedParser::new(
                            take_chomped(chomp_while1c(
                                &(|c: &char| c != &':' && c != &','),
                                "jyutping",
                            ))
                            .map(|pr_str| {
                                pr_str
                                    .split_whitespace()
                                    .map(|pr_seg| match parse_jyutping(&pr_seg.to_string()) {
                                        Some(pr) => LaxJyutPingSegment::Standard(pr),
                                        None => LaxJyutPingSegment::Nonstandard(pr_seg.to_string()),
                                    })
                                    .collect::<LaxJyutPing>()
                            }),
                        ),
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
/// # use wordshk_tools::parse::{parse_tags};
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
/// # use wordshk_tools::parse::{parse_line, SegmentType::*};
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
                    &(|c: &char| !unicode::is_punctuation(*c) && !c.is_whitespace()),
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
/// # use wordshk_tools::parse::{parse_named_line, SegmentType::*};
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
/// # use wordshk_tools::parse::{parse_partial_pr_line, SegmentType::*};
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
                    &(|c: &char| !unicode::is_punctuation(*c) && !c.is_whitespace() && *c != '('),
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
/// # use wordshk_tools::parse::{parse_partial_pr_named_line, SegmentType::*};
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
/// # use wordshk_tools::parse::{parse_clause, SegmentType::*};
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
/// # use wordshk_tools::parse::{parse_named_clause, SegmentType::*};
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
/// # use wordshk_tools::parse::{parse_alt_clause, AltLang, SegmentType::*};
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
/// # use wordshk_tools::parse::{parse_pr_line, SegmentType::*};
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
/// # use wordshk_tools::parse::{parse_eg, Eg, SegmentType::*};
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
/// # use wordshk_tools::parse::{parse_rich_def, Def, Eg, SegmentType::*};
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
/// # use wordshk_tools::parse::{parse_simple_def, Def, AltLang, SegmentType::*};
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
/// # use wordshk_tools::parse::{parse_defs, Def, Eg, SegmentType::*};
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
/// # use wordshk_tools::parse::{parse_content, Def, Entry, Variant, AltLang, SegmentType::*};
/// # let source = indoc::indoc! {"
/// (pos:語句)(label:外來語)(label:潮語)(label:香港)
/// yue:#加油
/// eng:cheer up
/// jpn:頑張って（がんばって）
/// # "};
///
/// let id = 98634;
/// let variants = vec![(Variant {word: "奸爸爹".into(), prs: vec![]})]; // prs omitted for brevity
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

/// JyutPing encoding with initial, nucleus (required), coda, and tone
///
/// Phonetics info based on: <https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.148.6501&rep=rep1&type=pdf>
#[derive(Debug, Clone, PartialEq)]
pub struct JyutPing {
    pub initial: Option<JyutPingInitial>,
    pub nucleus: JyutPingNucleus,
    pub coda: Option<JyutPingCoda>,
    pub tone: Option<JyutPingTone>,
}

pub fn jyutping_to_string(pr: &JyutPing) -> String {
    let pr = pr.clone();
    pr.initial.map(|i| i.to_string()).unwrap_or("".to_string())
        + &pr.nucleus.to_string()
        + &pr.coda.map(|i| i.to_string()).unwrap_or("".to_string())
        + &pr.tone.map(|i| i.to_string()).unwrap_or("".to_string())
}

pub fn jyutping_to_string_without_tone(pr: &JyutPing) -> String {
    let pr = pr.clone();
    pr.initial.map(|i| i.to_string()).unwrap_or("".to_string())
        + &pr.nucleus.to_string()
        + &pr.coda.map(|i| i.to_string()).unwrap_or("".to_string())
}

pub type LaxJyutPing = Vec<LaxJyutPingSegment>;

#[derive(Debug, Clone, PartialEq)]
pub enum LaxJyutPingSegment {
    Standard(JyutPing),
    Nonstandard(String),
}

/// Initial segment of a JyutPing, optional
///
/// Eg: 's' in "sap6"
///
#[derive(strum::EnumString, strum::Display, Debug, Clone, PartialEq)]
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

/// Nucleus segment of a Jyutping, always required
///
/// Eg: 'a' in "sap6"
///
#[derive(strum::EnumString, strum::Display, Debug, Clone, PartialEq)]
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
#[derive(strum::EnumString, strum::Display, Debug, Clone, PartialEq)]
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
#[derive(strum::EnumString, strum::Display, Debug, Clone, PartialEq)]
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
