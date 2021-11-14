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

use lazy_static::lazy_static;
use lip::ParseResult;
use lip::*;
use std::collections::HashSet;
use std::error::Error;
use std::io;
use std::process;

/// A dictionary is a list of entries
type Dict = Vec<Entry>;

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
struct Entry {
    id: usize,
    variants: Vec<Variant>,
    poses: Vec<String>,
    labels: Vec<String>,
    sims: Vec<String>,
    ants: Vec<String>,
    refs: Vec<String>,
    imgs: Vec<String>,
    defs: Vec<Def>,
}

/// A variant of a \[word\] with \[prs\] (pronounciations)
#[derive(Debug, Clone, PartialEq)]
struct Variant {
    word: String,
    prs: Vec<String>,
}

/// Two types of segments: text or link. See [Segment]
///
/// \[Text\] normal text
///
/// \[Link\] a link to another entry
///
#[derive(Debug, Clone, PartialEq)]
enum SegmentType {
    Text,
    Link,
}

/// A segment can be a text or a link
///
/// Text: 非常鬆軟。（量詞：件／籠）
///
/// Link: A link to the entry 雞蛋 would be #雞蛋
///
type Segment = (SegmentType, String);

/// A line consists of one or more [Segment]s
///
/// Empty line: `vec![(Text, "")]`
///
/// Simple line: `vec![(Text, "用嚟圍喺BB牀邊嘅布（量詞：塊）")]`
///
/// Mixed line: `vec![(Text, "一種加入"), (Link, "蝦籽"), (Text, "整嘅廣東麪")]`
///
type Line = Vec<Segment>;

/// A clause consists of one or more [Line]s. Appears in explanations and example sentences
///
/// Single-line clause: `vec![vec![(Text, "一行白鷺上青天")]]`
///
/// Multi-line clause: `vec![vec![(Text, "一行白鷺上青天")], vec![(Text, "兩個黃鸝鳴翠柳")]]`
///
type Clause = Vec<Line>; // can be multiline

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
struct Def {
    yue: Clause,
    eng: Option<Clause>,
    alts: Vec<AltClause>,
    egs: Vec<Eg>,
}

/// A clause in an alternative language other than Cantonese and English
///
/// \[[AltLang]\] language tag
///
/// \[[Clause]\] A sequence of texts and links
///
type AltClause = (AltLang, Clause);

/// Language tags for alternative languages other than Cantonese and English
///
/// From my observation, the tags seem to be alpha-3 codes in [ISO 639-2]
///
/// [ISO 639-2]: https://www.loc.gov/standards/iso639-2/php/code_list.php
///
#[derive(Debug, PartialEq)]
enum AltLang {
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
struct Eg {
    zho: Option<PrClause>,
    yue: Option<PrClause>,
    eng: Option<Clause>,
}

/// An example sentence with optional Jyutping pronunciation
///
/// Eg: 可唔可以見面？ (ho2 m4 ho2 ji5 gin3 min6?)
///
type PrClause = (Clause, Option<String>);

/// Parse the whole words.hk CSV database into a [Dict]
fn parse_dict() -> Result<Dict, Box<dyn Error>> {
    // Build the CSV reader and iterate over each record.
    let mut rdr = csv::Reader::from_reader(io::stdin());
    let mut dict: Dict = Vec::new();
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
                        println!("Error in #{}: {:?}", id, message);
                        None
                    }
                },
                ParseResult::Err { message, .. } => {
                    println!("Error in #{}: {:?}", id, message);
                    None
                }
            };
            match entry {
                Some(e) => {
                    println!("{:?}", e);
                    dict.push(e);
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
/// (label:外來語)(label:潮語)
///
/// which parses to:
///
/// `vec!["外來語", "潮語"]`
///
fn parse_tags<'a>(name: &'static str) -> lip::BoxedParser<'a, Vec<String>, ()> {
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

/// Parse a [Clause]
///
/// For example, here's an English clause:
///
/// My headphone cord was knotted.
///
/// which parses to:
///
/// `vec![vec![(Text, "My headphone cord was knotted.")]]`
fn parse_clause<'a>(name: &'static str) -> lip::BoxedParser<'a, Clause, ()> {
    succeed!(|clause| vec!(clause)).keep(one_or_more(succeed!(|seg| seg).keep(one_of!(
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

/// Parse a [Clause] tagged in front by its name/category
///
/// For example, here's an English clause:
///
/// eng:My headphone cord was knotted.
///
/// which parses to:
///
/// `vec![vec![(Text, "My headphone cord was knotted.")]]`
///
fn parse_named_clause<'a>(name: &'static str) -> lip::BoxedParser<'a, Clause, ()> {
    succeed!(|clause| clause)
        .skip(token(name))
        .skip(token(":"))
        .keep(parse_clause(name))
}

/// Parse a partial pronunciation [Clause], until the opening paren of Jyutping pronunciations
///
/// For the following pronunciation clause:
///
/// 可唔可以見面？ (ho2 m4 ho2 ji5 gin3 min6?)
///
/// This function will parse everything up until the '(':
///
/// 可唔可以見面？
///
/// which parses to:
///
/// `vec![vec![(Text, "可唔可以見面？")]]`
///
fn parse_partial_pr_clause<'a>(name: &'static str) -> lip::BoxedParser<'a, Clause, ()> {
    succeed!(|clause| vec!(clause)).keep(one_or_more(succeed!(|seg| seg).keep(one_of!(
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

/// Parse a partial *named* pronunciation [Clause], until the opening paren of Jyutping pronunciations
///
/// For the following *named* pronunciation clause:
///
/// yue:可唔可以見面？ (ho2 m4 ho2 ji5 gin3 min6?)
///
/// This function will parse everything up until the '(':
///
/// yue:可唔可以見面？
///
/// which parses to:
///
/// `vec![vec![(Text, "可唔可以見面？")]]`
///
fn parse_partial_pr_named_clause<'a>(name: &'static str) -> lip::BoxedParser<'a, Clause, ()> {
    succeed!(|clause| clause)
        .skip(token(name))
        .skip(token(":"))
        .keep(parse_partial_pr_clause(name))
}

/// Parse an English [Clause]
///
/// For example, here's an English clause:
///
/// eng:My headphone cord was knotted.
///
/// which parses to:
///
/// `vec![vec![(Text, "My headphone cord was knotted.")]]`
///
fn parse_eng_clause<'a>() -> lip::BoxedParser<'a, Clause, ()> {
    parse_named_clause("eng")
}

/// Parse a multiline clause
///
/// For example, here's a multiline clause:
///
/// 一行白鷺上青天
///
/// 兩個黃鸝鳴翠柳
///
/// which parses to:
///
/// `vec![vec![(Text, "一行白鷺上青天")], vec![(Text, "兩個黃鸝鳴翠柳")]]`
///
fn parse_multiline_clause<'a>(name: &'static str) -> lip::BoxedParser<'a, Clause, ()> {
    succeed!(|first_line: Clause, lines: Clause| {
        let mut all_lines = first_line;
        all_lines.extend(lines);
        all_lines
    })
    .skip(token(name))
    .skip(token(":"))
    .keep(succeed!(|line: Clause| line).keep(
        succeed!(|single_line_clause: Clause| single_line_clause).keep(parse_clause("a line")),
    ))
    .keep(zero_or_more_until(
        succeed!(|line| line).skip(parse_br()).keep(one_of!(
            // non-empty line
            succeed!(|line: Clause| line[0].clone()).keep(parse_clause("a nonempty line")),
            // empty line
            succeed!(|_| vec!((SegmentType::Text, "".to_string()))).keep(token(""))
        )),
        succeed!(|_| ()).skip(parse_br()).keep(one_of!(
            succeed!(|_| ()).keep(token("<eg>")),
            succeed!(|_| ())
                .keep(chomp_ifc(|c| *c != '\r' && *c != '\n', "any char"))
                .skip(chomp_ifc(|c| *c != '\r' && *c != '\n', "any char"))
                .skip(chomp_ifc(|c| *c != '\r' && *c != '\n', "any char"))
                .skip(chomp_ifc(|c| *c == ':', "colon `:`"))
        )),
    ))
    .skip(optional((), parse_br()))
}

/// Parse a clause in an alternative language
///
/// For example, here's a Japanese clause:
///
/// jpn:年画；ねんが
///
/// which parses to:
///
/// `(AltLang::Jpn, vec![vec![(Text, "年画；ねんが")]])`
///
fn parse_alt_clause<'a>() -> lip::BoxedParser<'a, AltClause, ()> {
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

/// Parse a Jyutping pronunciation clause, for Cantonese (yue) and Mandarin (zho)
///
/// For example, here's a Cantonese pronunciation clause:
///
/// yue:我個耳筒繑埋咗一嚿。 (ngo5 go3 ji5 tung2 kiu5 maai4 zo2 jat1 gau6.)
///
/// which parses to:
///
/// `(vec![vec![(Text, 我個耳筒繑埋咗一嚿。)]], Some("ngo5 go3 ji5 tung2 kiu5 maai4 zo2 jat1 gau6."))`
///
fn parse_pr_clause<'a>(name: &'static str) -> lip::BoxedParser<'a, PrClause, ()> {
    succeed!(|clause, pr| (clause, pr))
        .keep(parse_partial_pr_named_clause(name))
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
/// zho:後邊 (hau6 bin6)
/// yue:#後便 (hau6 bin6)
/// eng:back side
///
/// which parses to:
///
/// ```
/// Eg {
///     zho: (vec![vec![(Text, "後邊")]], Some("hau6 bin6")),
///     yue: (vec![vec![(Link, "後便")]], Some("hau6 bin6")),
///     end: vec![vec![(Text, "back side")]],
/// }
/// ```
///
fn parse_eg<'a>() -> lip::BoxedParser<'a, Eg, ()> {
    succeed!(|zho, yue, eng| Eg { zho, yue, eng })
        .skip(token("<eg>"))
        .skip(parse_br())
        .keep(optional(
            None,
            succeed!(Some)
                .keep(parse_pr_clause("zho"))
                .skip(optional((), parse_br())),
        ))
        .keep(optional(
            None,
            succeed!(Some)
                .keep(parse_pr_clause("yue"))
                .skip(optional((), parse_br())),
        ))
        .keep(optional(None, succeed!(Some).keep(parse_eng_clause())))
        .skip(optional((), parse_br()))
}

/// Parse a rich definition
///
/// Rich definitions start with an <explanation> tag and
/// contains one or more <eg> tags.
///
/// For example, here's part of the rich definition for the word 便:
///
/// <explanation>
/// yue:用於方位詞之後。書寫時，亦會用#邊 代替本字
/// eng:suffix for directional/positional noun
/// <eg>
/// yue:#開便 (hoi1 bin6)
/// eng:outside
/// <eg>
/// yue:#呢便 (nei1 bin6)
/// eng:this side
///
/// which parses to:
///
/// ```
/// Def {
///     yue: vec![vec![(Text, "用於方位詞之後。書寫時，亦會用"), (Link, "邊"), (Text, "代替本字")]],
///     eng: Some(vec![vec![(Text, "suffix for directional/positional noun")]]),
///     alts: vec![],
///     egs: vec![ Eg {
///          zho: None,
///          yue: Some((vec![vec![(Link, "開便")]], Some("hoi1 bin6"))),
///          eng: Some(vec![vec!["outside"]]),
///     },
///     Eg {
///          zho: None,
///          yue: Some((vec![vec![(Link, "呢便")]], Some("nei1 bin6"))),
///          eng: Some(vec![vec!["this side"]]),
///     },
///     ],
/// }
/// ```
///
fn parse_rich_def<'a>() -> lip::BoxedParser<'a, Def, ()> {
    succeed!(|yue, eng, alts, egs| Def {
        yue,
        eng,
        alts,
        egs
    })
    .skip(token("<explanation>"))
    .skip(parse_br())
    .keep(parse_multiline_clause("yue"))
    .keep(optional(
        None,
        succeed!(Some).keep(parse_multiline_clause("eng")),
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
/// yue:#加油
/// eng:cheer up
/// jpn:頑張って（がんばって）
///
/// which parses to:
///
/// ```
/// Def {
///     yue: vec![vec![(Link, "加油")]],
///     eng: Some(vec![vec![(Text, "cheer up")]]),
///     alts: vec![(AltLang::Jpn, vec![vec![(Text, "頑張って（がんばって）")]])],
///     egs: vec![],
/// }
/// ```
///
fn parse_simple_def<'a>() -> lip::BoxedParser<'a, Def, ()> {
    succeed!(|yue, eng, alts| Def {
        yue,
        eng,
        egs: Vec::new(),
        alts
    })
    .keep(parse_multiline_clause("yue"))
    .keep(optional(
        None,
        succeed!(Some).keep(parse_multiline_clause("eng")),
    ))
    .keep(zero_or_more(
        succeed!(|clause| clause)
            .keep(parse_alt_clause())
            .skip(optional((), parse_br())),
    ))
}

/// Parse a series of definitions for a word, separated by "----"
///
/// For example, here's a series of definitions for the word 兄
///
/// <explanation>
/// yue:同父母或者同監護人，年紀比你大嘅男性
/// eng:elder brother
/// <eg>
/// yue:#兄弟 (hing1 dai6)
/// eng:brothers
/// ----
/// <explanation>
/// yue:對男性朋友嘅尊稱
/// eng:politely addressing a male friend
///
/// which parses to:
///
/// ```
/// vec![
///     Def {
///         yue: vec![vec![(Text, "同父母或者同監護人，年紀比你大嘅男性")]],
///         eng: Some(vec![vec![(Text, "elder brother")]]),
///         alts: vec![],
///         egs: vec![
///             zho: None,
///             yue: Some((vec![vec![(Link, "兄弟")]], Some("hing1 dai6"))),
///             eng: Some(vec![vec![(Text, "brothers")]]),
///         ],
///     },
///     Def {
///         yue: vec![vec![(Text, "對男性朋友嘅尊稱")]],
///         eng: Some(vec![vec![(Text, "politely addressing a male friend")]]),
///         alts: vec![],
///         egs: vec![],
///     },
/// ]
/// ```
///
fn parse_defs<'a>() -> lip::BoxedParser<'a, Vec<Def>, ()> {
    succeed!(|defs| defs).keep(one_or_more(
        succeed!(|def| def)
            .keep(one_of!(parse_simple_def(), parse_rich_def()))
            .skip(optional(
                (),
                succeed!(|_| ())
                    .skip(parse_br())
                    .keep(token("----"))
                    .skip(parse_br()),
            )),
    ))
}

/// Parse the content of an [Entry]
///
/// id and variants are parsed by [parse_dict] and passed in to this function.
///
/// For example, here's the content of the Entry for 奸爸爹
///
/// (pos:語句)(label:外來語)(label:潮語)(label:香港)
/// yue:#加油
/// eng:cheer up
/// jpn:頑張って（がんばって）
///
/// ```
/// Some(Entry {
/// id: 98634,
/// variants: vec![("奸爸爹", vec!["gaan1 baa1 de1"])],
/// poses: vec!["語句"],
/// labels: vec!["外來語", "潮語", "香港"],
/// sims: vec![],
/// ants: vec![],
/// refs: vec![],
/// imgs: vec![],
/// defs: vec![Def {
///     yue: vec![vec![(Link, "加油")]],
///     eng: Some(vec![vec![(Text, "cheer up")]]),
///     alts: vec![(AltLang::Jpn, vec![vec![(Text, "頑張って（がんばって）")]])],
///     egs: vec![],
/// }]
/// })
/// ```
///
fn parse_content<'a>(id: usize, variants: Vec<Variant>) -> lip::BoxedParser<'a, Option<Entry>, ()> {
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
            '「', '」', '《', '》', '？', '，', '。'
        ])
    };
}

fn is_punctuation(c: char) -> bool {
    PUNCTUATIONS.contains(&c)
}

fn main() {
    if let Err(err) = parse_dict() {
        println!("error reading csv file: {}", err);
        process::exit(1);
    }
}

fn text(string: &'static str) -> Segment {
    (SegmentType::Text, string.to_string())
}

fn link(string: &'static str) -> Segment {
    (SegmentType::Link, string.to_string())
}

fn simple_clause(string: &'static str) -> Clause {
    vec![vec![text(string)]]
}

#[cfg(test)]
#[test]
fn test_parse_pr() {
    assert_succeed(
        parse_pr_clause("yue"),
        "yue:《飛狐外傳》 (fei1 wu4 ngoi6 zyun2)",
        (
            simple_clause("《飛狐外傳》"),
            Some("fei1 wu4 ngoi6 zyun2".to_string()),
        ),
    );
    assert_succeed(parse_pr_clause("yue"), "yue:《哈利波特》出外傳喎，你會唔會睇啊？ (\"haa1 lei6 bo1 dak6\" ceot1 ngoi6 zyun2 wo3, nei5 wui5 m4 wui5 tai2 aa3?)",
    (simple_clause("《哈利波特》出外傳喎，你會唔會睇啊？"), Some("\"haa1 lei6 bo1 dak6\" ceot1 ngoi6 zyun2 wo3, nei5 wui5 m4 wui5 tai2 aa3?".to_string())));
    assert_succeed(
        parse_pr_clause("yue"),
        "yue:佢唔係真喊架，扮嘢㗎咋。 (keoi5 m4 hai6 zan1 haam3 gaa3, baan6 je5 gaa3 zaa3.)",
        (
            simple_clause("佢唔係真喊架，扮嘢㗎咋。"),
            Some("keoi5 m4 hai6 zan1 haam3 gaa3, baan6 je5 gaa3 zaa3.".to_string()),
        ),
    );
    assert_succeed(parse_pr_clause("yue"), "yue:條八婆好扮嘢㗎，連嗌個叉飯都要講英文。 (tiu4 baat3 po4 hou2 baan6 je5 gaa3, lin4 aai3 go3 caa1 faan6 dou1 jiu3 gong2 jing1 man2.)", (simple_clause("條八婆好扮嘢㗎，連嗌個叉飯都要講英文。"), Some("tiu4 baat3 po4 hou2 baan6 je5 gaa3, lin4 aai3 go3 caa1 faan6 dou1 jiu3 gong2 jing1 man2.".to_string())));
}

#[test]
fn test_parse_eg() {
    assert_succeed(
        parse_eg(),
        "<eg>
yue:你可唔可以唔好成日zip呀，吓！ (nei5 ho2 m4 ho2 ji5 m4 hou2 sing4 jat6 zip4 aa3, haa2!)
eng:Stop tsking!",
        Eg {
            zho: None,
            yue: Some((
                simple_clause("你可唔可以唔好成日zip呀，吓！"),
                Some("nei5 ho2 m4 ho2 ji5 m4 hou2 sing4 jat6 zip4 aa3, haa2!".to_string()),
            )),
            eng: Some(simple_clause("Stop tsking!")),
        },
    );
    assert_succeed(
        parse_eg(),
        "<eg>
yue:佢今日心神恍惚，時時做錯嘢，好似有心事喎。 (keoi5 gam1 jat6 sam1 san4 fong2 fat1, si4 si4 zou6 co3 je5, hou2 ci5 jau5 sam1 si6 wo3.)",
        Eg {
            zho: None,
            yue: Some((simple_clause("佢今日心神恍惚，時時做錯嘢，好似有心事喎。"), Some("keoi5 gam1 jat6 sam1 san4 fong2 fat1, si4 si4 zou6 co3 je5, hou2 ci5 jau5 sam1 si6 wo3.".to_string()))),
            eng: None,
        },
    );
    assert_succeed(
        parse_eg(),
        "<eg>
yue:#難民營 (naan6 man4 jing4)
eng:refugee camp",
        Eg {
            zho: None,
            yue: Some((
                vec![vec![link("難民營")]],
                Some("naan6 man4 jing4".to_string()),
            )),
            eng: Some(simple_clause("refugee camp")),
        },
    )
}

#[test]
fn test_parse_simple_def() {
    assert_succeed(
        parse_simple_def(),
        "yue:嘗試去爬上一個更高的位置，泛指响職業上
eng:try to archive a higher position in career
fra:briguer une promotion",
        Def {
            yue: simple_clause("嘗試去爬上一個更高的位置，泛指响職業上"),
            eng: Some(simple_clause("try to archive a higher position in career")),
            alts: vec![(AltLang::Fra, simple_clause("briguer une promotion"))],
            egs: vec![],
        },
    );

    assert_succeed(
        parse_simple_def(),
        "yue:地方名
eng:Macau; Macao
por:Macau",
        Def {
            yue: simple_clause("地方名"),
            eng: Some(simple_clause("Macau; Macao")),
            alts: vec![(AltLang::Por, simple_clause("Macau"))],
            egs: vec![],
        },
    );

    assert_succeed(
        parse_simple_def(),
        "yue:東亞民間慶祝#新春 嘅畫種（量詞：幅）
eng:new year picture in East Asia
jpn:年画；ねんが
kor:세화
vie:Tranh tết",
        Def {
            yue: vec![vec![
                text("東亞民間慶祝"),
                link("新春"),
                text("嘅畫種（量詞：幅）"),
            ]],
            eng: Some(simple_clause("new year picture in East Asia")),
            alts: vec![
                (AltLang::Jpn, simple_clause("年画；ねんが")),
                (AltLang::Kor, simple_clause("세화")),
                (AltLang::Vie, simple_clause("Tranh tết")),
            ],
            egs: vec![],
        },
    );

    // string like "ccc:", where c stands for any character, appearing in clause
    assert_succeed(
        parse_simple_def(),
        "yue:abc

def",
        Def {
            yue: vec![vec![text("abc")], vec![text("")], vec![text("def")]],
            eng: None,
            alts: vec![],
            egs: vec![],
        },
    );

    // a concrete example of "ccc:" used in the dataset
    assert_succeed(
        parse_simple_def(),
        "yue:第二人稱代詞；用嚟稱呼身處喺自己面前又或者同自己講緊嘢嘅眾數對象
eng:you: second person plural",
        Def {
            yue: simple_clause("第二人稱代詞；用嚟稱呼身處喺自己面前又或者同自己講緊嘢嘅眾數對象"),
            eng: Some(simple_clause("you: second person plural")),
            alts: vec![],
            egs: vec![],
        },
    );
    // more complicated "ccc:" style clause with multiple consecutive empty lines
    assert_succeed(
        parse_simple_def(),
        "yue:abc:def:ghi


eng:opq:rst:

uvw",
        Def {
            yue: vec![vec![text("abc:def:ghi")], vec![text("")], vec![text("")]],
            eng: Some(vec![
                vec![text("opq:rst:")],
                vec![text("")],
                vec![text("uvw")],
            ]),
            alts: vec![],
            egs: vec![],
        },
    );
}

#[test]
fn test_parse_rich_def() {
    assert_succeed(
        parse_rich_def(),
        "<explanation>
yue:表現不屑而發出嘅聲音
eng:tsk
<eg>
yue:你可唔可以唔好成日zip呀，吓！ (nei5 ho2 m4 ho2 ji5 m4 hou2 sing4 jat6 zip4 aa3, haa2!)
eng:Stop tsking!",
        Def {
            yue: simple_clause("表現不屑而發出嘅聲音"),
            eng: Some(simple_clause("tsk")),
            alts: vec![],
            egs: vec![Eg {
                zho: None,
                yue: Some((
                    simple_clause("你可唔可以唔好成日zip呀，吓！"),
                    Some("nei5 ho2 m4 ho2 ji5 m4 hou2 sing4 jat6 zip4 aa3, haa2!".to_string()),
                )),
                eng: Some(simple_clause("Stop tsking!")),
            }],
        },
    );

    assert_succeed(
        parse_rich_def(),
        "<explanation>
yue:字面義指一個人做出啲好古怪嘅表情或者動作，或者大聲講嘢，令其他人感到困擾。引申出表達意見，尤其是過度表達意見嘅行為。
eng:to clench one's teeth or show weird gesture, which implies speaking improperly or with exaggerating manner.
<eg>
yue:觀棋不語真君子，旁觀者不得𪘲牙聳䚗（依牙鬆鋼）。 (gun1 kei4 bat1 jyu5 zan1 gwan1 zi2，pong4 gun1 ze2 bat1 dak1 ji1 ngaa4 sung1 gong3.)
eng:Gentlemen observe chess match with respectful silence. Spectators are not allowed to disturb the competitors.",
        Def {
            yue: simple_clause("字面義指一個人做出啲好古怪嘅表情或者動作，或者大聲講嘢，令其他人感到困擾。引申出表達意見，尤其是過度表達意見嘅行為。"),
            eng: Some(simple_clause("to clench one's teeth or show weird gesture, which implies speaking improperly or with exaggerating manner.")),
            alts: vec![],
            egs: vec![Eg {
                zho: None,
                yue: Some((simple_clause("觀棋不語真君子，旁觀者不得𪘲牙聳䚗（依牙鬆鋼）。"), Some("gun1 kei4 bat1 jyu5 zan1 gwan1 zi2，pong4 gun1 ze2 bat1 dak1 ji1 ngaa4 sung1 gong3.".to_string()))),
eng: Some(simple_clause("Gentlemen observe chess match with respectful silence. Spectators are not allowed to disturb the competitors.")),
            }],
        }
        )
}

#[test]
fn test_parse_defs() {
    assert_succeed(
        parse_defs(),
        "<explanation>
yue:表現不屑而發出嘅聲音
eng:tsk
<eg>
yue:你可唔可以唔好成日zip呀，吓！ (nei5 ho2 m4 ho2 ji5 m4 hou2 sing4 jat6 zip4 aa3, haa2!)
eng:Stop tsking!",
        vec![Def {
            yue: simple_clause("表現不屑而發出嘅聲音"),
            eng: Some(simple_clause("tsk")),
            alts: vec![],
            egs: vec![Eg {
                zho: None,
                yue: Some((
                    simple_clause("你可唔可以唔好成日zip呀，吓！"),
                    Some("nei5 ho2 m4 ho2 ji5 m4 hou2 sing4 jat6 zip4 aa3, haa2!".to_string()),
                )),
                eng: Some(simple_clause("Stop tsking!")),
            }],
        }],
    );
    assert_succeed(
        parse_defs(),
        "yue:「#仆街」嘅代名詞",
        vec![Def {
            yue: vec![vec![text("「"), link("仆街"), text("」嘅代名詞")]],
            eng: None,
            alts: vec![],
            egs: vec![],
        }],
    );

    assert_succeed(
        parse_defs(),
        r#"<explanation>
yue:#天干

#地支
eng:Heavenly Stems

Earthly Branches
<eg>
yue:乙等 / 乙級 (jyut6 dang2 / jyut6 kap1)
eng:B grade"#,
        vec![Def {
            yue: vec![vec![link("天干")], vec![text("")], vec![link("地支")]],
            eng: Some(vec![
                vec![text("Heavenly Stems")],
                vec![text("")],
                vec![text("Earthly Branches")],
            ]),
            alts: vec![],
            egs: vec![Eg {
                zho: None,
                yue: Some((
                    simple_clause("乙等 / 乙級"),
                    Some("jyut6 dang2 / jyut6 kap1".to_string()),
                )),
                eng: Some(simple_clause("B grade")),
            }],
        }],
    );

    assert_succeed(parse_defs(), r#"<explanation>
yue:「#天干」同「#地支」嘅合稱。十天干分別係「#甲#乙#丙#丁#戊#己#庚#辛#壬#癸」。 十二地支係：「#子#丑#寅#卯#辰#巳#午#未#申#酉#戌#亥」。 天干同地支組合就成為以「#甲子」為首嘅六十干支循環。

干支循環通常用嚟計年份。天干亦可以獨立用嚟順序將物件命名，第一個叫「甲」、第二個叫「乙」，如此類推。用法類似西方嘅「A, B, C」 或 「α, β, γ」。中國傳統紀時間嘅方法係將一日分成十二個時辰，每一個時辰由一個地支表示，「子時」係半夜 (11pm 至 1am)，如此類推。
eng:Literally ""Heavenly Stems and Earthly Branches"". It is a traditional Chinese system of counting. Heavenly Stems and Earthly Branches are collectively known as ""Stem-Branch"".

The 10 Heavenly Stems are 甲(gaap3) 乙(jyut6) 丙(bing2) 丁(ding1) 戊(mou6) 己(gei2) 庚(gang1) 辛(san1) 壬(jam4) 癸(gwai3).

The 12 Earthly Branches are 子(zi2) 丑(cau2) 寅(jan4) 卯(maau5) 辰(san4) 巳(zi6) 午(ng5) 未(mei6) 申(san1) 酉(jau5) 戌(seot1) 亥(hoi6). Each Heavenly Stem is paired with an Earthly Branch to form the ""stem-branch"" sexagenary (i.e. 60 element) cycle that starts with 甲子 (gaap3 zi2)

The sexagenary cycle is often used for counting years in the Chinese calendar. Heavenly Stems are also used independently to name things in a particular order -- the first is labeled ""gaap3"", the second ""jyut6"", the third ""bing2"", and so on. It is similar to how ""A, B, C"" and ""α, β, γ"" are used in western cultures. Earthly Branches are also traditionally used to denote time. One day is divided into twelve slots called Chinese-hours (#時辰), starting from 子時 (zi2 si4), which is 11pm to 1am.
<eg>
yue:乙等 / 乙級 (jyut6 dang2 / jyut6 kap1)
eng:B grade"#
, vec![Def {
    yue: vec!(vec!(text("「"), link("天干"), text("」同「"), link("地支"), text("」嘅合稱。十天干分別係「"), link("甲"), link("乙"), link("丙"), link("丁"), link("戊"), link("己"), link("庚"), link("辛"), link("壬"), link("癸"), text("」。 十二地支係：「"), link("子"), link("丑"), link("寅"), link("卯"), link("辰"), link("巳"), link("午"), link("未"), link("申"), link("酉"), link("戌"), link("亥"), text("」。 天干同地支組合就成為以「"), link("甲子"), text("」為首嘅六十干支循環。")),
vec!(text("")),
vec!(text("干支循環通常用嚟計年份。天干亦可以獨立用嚟順序將物件命名，第一個叫「甲」、第二個叫「乙」，如此類推。用法類似西方嘅「A, B, C」 或 「α, β, γ」。中國傳統紀時間嘅方法係將一日分成十二個時辰，每一個時辰由一個地支表示，「子時」係半夜 (11pm 至 1am)，如此類推。"))),
eng: Some(vec!(vec!(text(r#"Literally ""Heavenly Stems and Earthly Branches"". It is a traditional Chinese system of counting. Heavenly Stems and Earthly Branches are collectively known as ""Stem-Branch""."#)),
vec!(text("")),
vec!(text(r#"The 10 Heavenly Stems are 甲(gaap3) 乙(jyut6) 丙(bing2) 丁(ding1) 戊(mou6) 己(gei2) 庚(gang1) 辛(san1) 壬(jam4) 癸(gwai3)."#)),
vec!(text("")),
vec!(text(r#"The 12 Earthly Branches are 子(zi2) 丑(cau2) 寅(jan4) 卯(maau5) 辰(san4) 巳(zi6) 午(ng5) 未(mei6) 申(san1) 酉(jau5) 戌(seot1) 亥(hoi6). Each Heavenly Stem is paired with an Earthly Branch to form the ""stem-branch"" sexagenary (i.e. 60 element) cycle that starts with 甲子 (gaap3 zi2)"#)),
vec!(text("")),
vec!(text(r#"The sexagenary cycle is often used for counting years in the Chinese calendar. Heavenly Stems are also used independently to name things in a particular order -- the first is labeled ""gaap3"", the second ""jyut6"", the third ""bing2"", and so on. It is similar to how ""A, B, C"" and ""α, β, γ"" are used in western cultures. Earthly Branches are also traditionally used to denote time. One day is divided into twelve slots called Chinese-hours ("#), link("時辰"), text(r#"), starting from 子時 (zi2 si4), which is 11pm to 1am."#))),
),
    alts: vec![],
    egs: vec![Eg {
        zho: None,
        yue: Some((
            simple_clause("乙等 / 乙級"),
            Some("jyut6 dang2 / jyut6 kap1".to_string()),
        )),
        eng: Some(simple_clause("B grade")),
    }],
}]);
}

#[test]
fn test_parse_content() {
    {
        let id = 103022;
        let variants = vec![
            Variant {
                word: "zip".to_string(),
                prs: vec!["zip4".to_string()],
            },
            Variant {
                word: "jip".to_string(),
                prs: vec!["zip4".to_string()],
            },
        ];
        assert_succeed(
            parse_content(id, variants.clone()),
            "(pos:動詞)(pos:擬聲詞)
<explanation>
yue:表現不屑而發出嘅聲音
eng:tsk
<eg>
yue:你可唔可以唔好成日zip呀，吓！ (nei5 ho2 m4 ho2 ji5 m4 hou2 sing4 jat6 zip4 aa3, haa2!)
eng:Stop tsking!",
            Some(Entry {
                id,
                variants,
                poses: vec!["動詞".to_string(), "擬聲詞".to_string()],
                labels: vec![],
                sims: vec![],
                ants: vec![],
                refs: vec![],
                imgs: vec![],
                defs: vec![Def {
                    yue: simple_clause("表現不屑而發出嘅聲音"),
                    eng: Some(simple_clause("tsk")),
                    alts: vec![],
                    egs: vec![Eg {
                        zho: None,
                        yue: Some((
                            simple_clause("你可唔可以唔好成日zip呀，吓！"),
                            Some(
                                "nei5 ho2 m4 ho2 ji5 m4 hou2 sing4 jat6 zip4 aa3, haa2!"
                                    .to_string(),
                            ),
                        )),
                        eng: Some(simple_clause("Stop tsking!")),
                    }],
                }],
            }),
        );
    }
    {
        let id = 20;
        let variants = vec![Variant {
            word: "hihi".to_string(),
            prs: vec!["haai1 haai1".to_string()],
        }];
        assert_succeed(
                parse_content(id, variants.clone()),
                "(pos:動詞)(label:潮語)(label:粗俗)(ref:http://evchk.wikia.com/wiki/%E9%AB%98%E7%99%BB%E7%B2%97%E5%8F%A3Filter)
yue:「#仆街」嘅代名詞",
                Some(Entry {
                    id,
                    variants,
                    poses: vec!["動詞".to_string()],
                    labels: vec!["潮語".to_string(), "粗俗".to_string()],
                    sims: vec![],
                    ants: vec![],
                    refs: vec!["http://evchk.wikia.com/wiki/%E9%AB%98%E7%99%BB%E7%B2%97%E5%8F%A3Filter".to_string()],
                    imgs: vec![],
                    defs: vec![Def {
                        yue: vec!(vec!(text("「"), link("仆街"), text("」嘅代名詞"))),
                        eng: None,
                        alts: vec![],
                        egs: vec![],
                    }]
                })
            );
    }
}
