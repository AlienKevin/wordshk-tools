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
use super::jyutping::*;
use super::unicode;

use lip::ParseResult;
use lip::*;
use std::collections::BTreeMap;
use std::convert::identity;
use std::error::Error;
use std::io;

/// Parse the whole words.hk CSV database into a [Dict]
pub fn parse_dict<R: io::Read>(input: R) -> Result<Dict, Box<dyn Error>> {
    // Build the CSV reader and iterate over each record.
    let mut rdr = csv::ReaderBuilder::new()
        .has_headers(false)
        .from_reader(input);
    let mut dict: Dict = BTreeMap::new();
    for result in rdr.records() {
        let entry = result?;
        // if &entry[4] == "OK" {
        let id: EntryId = entry[0].parse().unwrap();
        let head = &entry[1];
        let content = &entry[2];
        let published = &entry[5] == "已公開";
        // entry[3] is always an empty string
        let head_parse_result = sequence(
            succeed!(),
            || {
                succeed!(|word, prs| Variant { word, prs })
                    .keep(take_chomped(chomp_while1c(&(|c: char| c != ':'), "word")))
                    .keep(
                        sequence(
                            token(":"),
                            || {
                                take_chomped(chomp_while1c(
                                    &(|c: char| c != ':' && c != ','),
                                    "jyutping",
                                ))
                                .map(|pr_str| parse_pr(&pr_str))
                            },
                            || token(":"),
                            space0,
                            succeed!(),
                            Trailing::Forbidden,
                        )
                        .map(LaxJyutPings),
                    )
            },
            || token(","),
            space0,
            succeed!(),
            Trailing::Forbidden,
        )
        .run(head, ());
        let entry: Option<Entry> = match head_parse_result {
            ParseResult::Ok {
                output: head_result,
                ..
            } => match parse_content(id, Variants(head_result), published).run(content, ()) {
                ParseResult::Ok {
                    output: content_result,
                    ..
                } => content_result,
                ParseResult::Err {
                    message, from, to, ..
                } => {
                    println!("Content error in #{}: {:?}", id, message);
                    println!("{}\n\n", display_error(content, message, from, to));
                    None
                }
            },
            ParseResult::Err {
                message, from, to, ..
            } => {
                println!("Header error in #{}: {:?}", id, message);
                println!("{}\n\n", display_error(content, message, from, to));
                None
            }
        };
        if let Some(e) = entry {
            // println!("{:?}", e);
            dict.insert(id, e);
        };
        // }
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
pub fn parse_tags(name: &str) -> impl lip::Parser<Output = Vec<String>, State = ()> {
    return zero_or_more(
        succeed!(|tag| tag)
            .skip(token("("))
            .backtrackable()
            .skip(token(name))
            .skip(token(":"))
            .keep(take_chomped(chomp_while1c(&(|c: char| c != ')'), name)))
            .skip(token(")")),
    );
}

/// Parse a newline character
///
/// Supports both Windows "\r\n" and Unix "\n"
fn parse_br<'a>() -> impl lip::Parser<'a, Output = (), State = ()> {
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
pub fn parse_line(name: &str) -> impl lip::Parser<Output = Line, State = ()> {
    succeed!(remove_extra_spaces_around_link).keep(one_or_more(succeed!(|seg| seg).keep(one_of!(
            succeed!(|string| (SegmentType::Link, string))
                .skip(token("#"))
                .keep(take_chomped(chomp_while1c(
                    &(|c: char| !unicode::is_punc(c) && !c.is_whitespace()),
                    name
                ))),
            succeed!(|string| (SegmentType::Text, string)).keep(take_chomped(chomp_while1c(
                &(|c: char| c != '#' && c != '\n' && c != '\r'),
                name
            )))
        ))))
}

fn remove_extra_spaces_around_link(segs: Vec<Segment>) -> Vec<Segment> {
    let mut i = 0;
    let mut output_segs = segs.clone();
    output_segs.iter_mut().for_each(|seg| {
        // Remove the whitespace before a link if the whitespace
        // is preceeded by a CJK character or another whitespace (double whitespace)
        let mut seg_chars = seg.1.chars();
        if i + 1 < segs.len()
            && segs[i + 1].0 == SegmentType::Link
            && seg_chars
                .next_back()
                .map(char::is_whitespace)
                .unwrap_or(false)
            && seg_chars
                .next_back()
                .map(|c| unicode::is_cjk(c) || char::is_whitespace(c))
                .unwrap_or(false)
        {
            seg.1 = unicode::remove_last_char(&seg.1);
        }

        // Remove the whitespace after a link if the whitespace
        // is followed by a CJK character or another whitespace (double whitespace)
        seg_chars = seg.1.chars();
        if i >= 1
            && segs[i - 1].0 == SegmentType::Link
            && seg_chars.next().map(char::is_whitespace).unwrap_or(false)
            && seg_chars
                .next()
                .map(|c| unicode::is_cjk(c) || char::is_whitespace(c))
                .unwrap_or(false)
        {
            seg.1 = unicode::remove_first_char(&seg.1);
        }

        // Remove trailing whitespace from the last seg
        if i == segs.len() - 1
            && seg
                .1
                .chars()
                .next_back()
                .map(char::is_whitespace)
                .unwrap_or(false)
        {
            seg.1 = unicode::remove_last_char(&seg.1);
        }

        i += 1;
    });
    // Filter out empty segs resulted from whitespace trimming above
    output_segs
        .into_iter()
        .filter(|seg| !seg.1.is_empty())
        .collect()
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
pub fn parse_named_line(name: &str) -> impl lip::Parser<Output = Line, State = ()> {
    succeed!(|clause| clause)
        .skip(token(name))
        .skip(token(":"))
        .keep(parse_line(name))
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
pub fn parse_clause(expecting: &str) -> impl lip::Parser<Output = Clause, State = ()> {
    succeed!(|first_line: Line, lines: Clause| {
        let mut all_lines = vec![first_line];
        all_lines.extend(lines);
        let mut i = 1;
        while i <= all_lines.len()
            && all_lines[all_lines.len() - i].len() == 1
            && all_lines[all_lines.len() - i][0].1.is_empty()
        {
            i += 1;
        }
        for _ in 0..i - 1 {
            all_lines.pop();
        }
        all_lines
    })
    .keep(parse_line(expecting))
    .keep(zero_or_more_until(
        "line",
        succeed!(identity).skip(parse_br()).keep(one_of!(
            // non-empty line
            succeed!(identity).keep(parse_line(expecting)),
            // empty line
            succeed!(|_| vec![(SegmentType::Text, "".to_string())]).keep(succeed!())
        )),
        "end of clause",
        succeed!(|_| ()).keep(one_of!(
            token("\n<eg>"),
            token("\n----"),
            token("\nzho:"),
            token("\nyue:"),
            token("\neng:"),
            token("\njpn:"),
            token("\nkor:"),
            token("\npor:"),
            token("\nvie:"),
            token("\nlat:"),
            token("\nfra:")
        )),
    ))
    .skip(optional(parse_br()))
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
pub fn parse_named_clause(name: &str) -> impl lip::Parser<Output = Clause, State = ()> {
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
pub fn parse_alt_clause<'a>() -> impl lip::Parser<'a, Output = AltClause, State = ()> {
    succeed!(|alt_lang: AltLang, clause: Clause| (alt_lang, clause))
        .keep(one_of!(
            succeed!(|_| AltLang::Jpn).keep(token("jpn:")),
            succeed!(|_| AltLang::Kor).keep(token("kor:")),
            succeed!(|_| AltLang::Por).keep(token("por:")),
            succeed!(|_| AltLang::Vie).keep(token("vie:")),
            succeed!(|_| AltLang::Lat).keep(token("lat:")),
            succeed!(|_| AltLang::Fra).keep(token("fra:"))
        ))
        .keep(parse_clause("alternative language clause"))
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
pub fn parse_pr_line(name: &str) -> impl lip::Parser<Output = PrLine, State = ()> {
    (succeed!(|line: String| {
        let open_paren_index = line.rfind('(');
        // println!("open_paren_index: {:?}", open_paren_index);
        // println!("line.chars().next_back(): {:?}", line.chars().next_back());
        if open_paren_index.is_some() && line.ends_with(')') {
            let open_paren = open_paren_index.unwrap();
            let paren_segment = &line[open_paren + 1..line.len() - 1];
            // println!("paren_segment: {:?}", paren_segment);
            if looks_like_jyutping(paren_segment) {
                // println!("Found pr line with pr: {paren_segment}");
                return (
                    line[0..open_paren].to_string(),
                    Some(paren_segment.to_string()),
                );
            }
        }
        (line, None)
    })
    .skip(token(name))
    .skip(token(":"))
    .keep(take_chomped(chomp_while1c(
        &(|c: char| c != '\n' && c != '\r'),
        "line",
    ))))
    .map(move |(line, pr)| match parse_line(name).run(&line, ()) {
        ParseResult::Ok { output, .. } => (output, pr),
        ParseResult::Err {
            message, from, to, ..
        } => {
            println!("Pr line error: {:?}", message);
            println!("{}\n\n", display_error(&line, message, from, to));
            (vec![], pr)
        }
    })
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
///     lzh: None,
/// }
/// # );
/// ```
///
pub fn parse_eg<'a>() -> impl lip::Parser<'a, Output = Eg, State = ()> {
    succeed!(|zho, yue, eng, lzh| Eg { zho, yue, eng, lzh })
        .skip(token("<eg>"))
        .skip(parse_br())
        .keep(optional(parse_pr_line("zho").skip(optional(parse_br()))))
        .keep(optional(parse_pr_line("yue").skip(optional(parse_br()))))
        // only a single line is accepted in eg
        .keep(optional(parse_named_line("eng")))
        .skip(optional(parse_br()))
        // Very rare Classical Chinese line
        .keep(optional(parse_pr_line("lzh")))
        .skip(optional(parse_br()))
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
///             lzh: None,
///         },
///         Eg {
///             zho: None,
///             yue: Some((vec![(Link, "呢便".into())], Some("nei1 bin6".into()))),
///             eng: Some(vec![(Text, "this side".into())]),
///             lzh: None,
///         },
///     ],
/// }
/// # );
/// ```
///
pub fn parse_rich_def<'a>() -> impl lip::Parser<'a, Output = Def, State = ()> {
    succeed!(|yue, eng, alts, egs| Def {
        yue,
        eng,
        alts,
        egs
    })
    .skip(token("<explanation>"))
    .skip(parse_br())
    .keep(parse_named_clause("yue"))
    .keep(optional(parse_named_clause("eng")))
    .keep(zero_or_more(
        succeed!(|clause| clause)
            .keep(parse_alt_clause())
            .skip(optional(parse_br())),
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
pub fn parse_simple_def<'a>() -> impl lip::Parser<'a, Output = Def, State = ()> {
    succeed!(|yue, eng, alts| Def {
        yue,
        eng,
        egs: Vec::new(),
        alts
    })
    .keep(parse_named_clause("yue"))
    .keep(optional(parse_named_clause("eng")))
    .keep(zero_or_more(
        succeed!(|clause| clause)
            .keep(parse_alt_clause())
            .skip(optional(parse_br())),
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
///                 lzh: None,
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
pub fn parse_defs<'a>() -> impl lip::Parser<'a, Output = Vec<Def>, State = ()> {
    succeed!(|defs| defs).keep(one_or_more(
        succeed!(|def| def)
            .keep(one_of!(parse_simple_def(), parse_rich_def()))
            .skip(optional(token("----").skip(parse_br()))),
    ))
}

/// Parse the content of an [Entry]
///
/// id and variants are parsed by [parse_dict] and passed in to this function.
///
/// For example, here's the content of the Entry for 奸爸爹
///
/// ```
/// # use wordshk_tools::dict::{Def, Entry, Variants, Variant, AltLang, SegmentType::*};
/// # use wordshk_tools::jyutping::LaxJyutPings;
/// # use wordshk_tools::parse::{parse_content};
/// # let source = indoc::indoc! {"
/// (pos:語句)(label:外來語)(label:潮語)(label:香港)
/// yue:#加油
/// eng:cheer up
/// jpn:頑張って（がんばって）
/// # "};
///
/// let id = 98634;
/// let published = true;
/// // prs omitted below for brevity
/// let variants = Variants(vec![(Variant {word: "奸爸爹".into(), prs: LaxJyutPings(vec![])})]);
///
/// // which parses to:
///
/// # lip::assert_succeed(parse_content(id, variants.clone(), published), source,
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
///     }],
/// published: published,
/// })
/// # );
/// ```
///
pub fn parse_content<'a>(
    id: EntryId,
    variants: Variants,
    published: bool,
) -> impl lip::Parser<'a, Output = Option<Entry>, State = ()> {
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
            published,
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
