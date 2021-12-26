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

use super::dict::*;
use super::unicode;

use lip::ParseResult;
use lip::*;
use std::collections::HashMap;
use std::convert::identity;
use std::error::Error;
use std::io;
use std::ops::Range;
use std::str::FromStr;

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
                    .keep(
                        sequence(
                            ":",
                            BoxedParser::new(
                                take_chomped(chomp_while1c(
                                    &(|c: &char| c != &':' && c != &','),
                                    "jyutping",
                                ))
                                .map(|pr_str| {
                                    LaxJyutPing(
                                        pr_str
                                            .split_whitespace()
                                            .map(|pr_seg| {
                                                match parse_jyutping(&pr_seg.to_string()) {
                                                    Some(pr) => LaxJyutPingSegment::Standard(pr),
                                                    None => LaxJyutPingSegment::Nonstandard(
                                                        pr_seg.to_string(),
                                                    ),
                                                }
                                            })
                                            .collect(),
                                    )
                                }),
                            ),
                            ":",
                            space0(),
                            "",
                            Trailing::Forbidden,
                        )
                        .map(|prs| LaxJyutPings(prs)),
                    ),
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
                } => match parse_content(id, Variants(head_result)).run(content, ()) {
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
/// # use wordshk_tools::dict::{SegmentType::*};
/// # use wordshk_tools::parse::{parse_line};
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
/// # use wordshk_tools::dict::{SegmentType::*};
/// # use wordshk_tools::parse::{parse_named_line};
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
/// # use wordshk_tools::dict::{SegmentType::*};
/// # use wordshk_tools::parse::{parse_partial_pr_line};
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
/// # use wordshk_tools::dict::{SegmentType::*};
/// # use wordshk_tools::parse::{parse_partial_pr_named_line};
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
/// # use wordshk_tools::dict::{SegmentType::*};
/// # use wordshk_tools::parse::{parse_clause};
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
/// # use wordshk_tools::dict::{SegmentType::*};
/// # use wordshk_tools::parse::{parse_named_clause};
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
/// # use wordshk_tools::dict::{AltLang, SegmentType::*};
/// # use wordshk_tools::parse::{parse_alt_clause};
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
/// # use wordshk_tools::dict::{SegmentType::*};
/// # use wordshk_tools::parse::{parse_pr_line};
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
/// # use wordshk_tools::dict::{Eg, SegmentType::*};
/// # use wordshk_tools::parse::{parse_eg};
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
/// # use wordshk_tools::dict::{Def, Eg, SegmentType::*};
/// # use wordshk_tools::parse::{parse_rich_def};
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
/// # use wordshk_tools::dict::{Def, AltLang, SegmentType::*};
/// # use wordshk_tools::parse::{parse_simple_def};
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
/// # use wordshk_tools::dict::{Def, Eg, SegmentType::*};
/// # use wordshk_tools::parse::{parse_defs};
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
/// # use wordshk_tools::dict::{Def, Entry, Variants, Variant, LaxJyutPings, AltLang, SegmentType::*};
/// # use wordshk_tools::parse::{parse_content};
/// # let source = indoc::indoc! {"
/// (pos:語句)(label:外來語)(label:潮語)(label:香港)
/// yue:#加油
/// eng:cheer up
/// jpn:頑張って（がんばって）
/// # "};
///
/// let id = 98634;
/// // prs omitted below for brevity
/// let variants = Variants(vec![(Variant {word: "奸爸爹".into(), prs: LaxJyutPings(vec![])})]);
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
pub fn parse_content<'a>(id: usize, variants: Variants) -> lip::BoxedParser<'a, Option<Entry>, ()> {
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

/// Parse [JyutPing] pronunciation
pub fn parse_jyutping(str: &str) -> Option<JyutPing> {
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

fn parse_jyutping_component<T: FromStr>(start: usize, str: &str) -> Option<(T, usize)> {
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