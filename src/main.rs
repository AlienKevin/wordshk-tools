use lip::ParseResult;
use lip::*;
use std::error::Error;
use std::io;
use std::process;

type Dict = Vec<Entry>;

#[derive(Debug, PartialEq)]
struct Entry {
    variants: Vec<Variant>,
    poses: Vec<String>,
    labels: Vec<String>,
    sims: Vec<String>,
    ants: Vec<String>,
    refs: Vec<String>,
    imgs: Vec<String>,
    defs: Vec<Def>,
}

#[derive(Debug, Clone, PartialEq)]
struct Variant {
    word: String,
    prs: Vec<String>,
}

#[derive(Debug, PartialEq)]
struct Def {
    yue: String,
    eng: Option<String>,
    alts: Vec<AltClause>,
    egs: Vec<Eg>,
}

type AltClause = (AltLang, String);

#[derive(Debug, PartialEq)]
enum AltLang {
    Jpn,
    Kor,
    Vie,
    Lat,
}

#[derive(Debug, Clone, PartialEq)]
struct Eg {
    zho: Option<PrClause>,
    yue: Option<PrClause>,
    eng: Option<String>,
}
type PrClause = (String, Option<String>);

fn to_apple_dict() -> Result<Dict, Box<dyn Error>> {
    // Build the CSV reader and iterate over each record.
    let mut rdr = csv::Reader::from_reader(io::stdin());
    let mut dict: Dict = Vec::new();
    for result in rdr.records() {
        let entry = result?;
        if &entry[4] == "OK" {
            let id = &entry[0];
            let head = &entry[1];
            let content = &entry[2];
            // entry[3] is always an empty string
            let head_parse_result = sequence(
                "",
                succeed!(|word, prs| Variant { word, prs })
                    .keep(take_chomped(chomp_while1(&(|c: &char| c != &':'), "word")))
                    .keep(sequence(
                        ":",
                        BoxedParser::new(take_chomped(chomp_while1(
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
                } => {
                    println!("{:?}", head_result);
                    match parse_content(head_result).run(content, ()) {
                        ParseResult::Ok {
                            output: content_result,
                            ..
                        } => content_result,
                        ParseResult::Err { message, .. } => {
                            println!("{:?}", message);
                            None
                        }
                    }
                }
                ParseResult::Err { message, .. } => {
                    println!("{:?}", message);
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

fn parse_tags<'a>(name: &'static str) -> lip::BoxedParser<'a, Vec<String>, ()> {
    return zero_or_more(
        succeed!(|tag| tag)
            .skip(token("("))
            .skip(token(name))
            .skip(token(":"))
            .keep(take_chomped(chomp_while1(&(|c: &char| c != &')'), name)))
            .skip(token(")")),
    );
}

fn parse_br<'a>() -> lip::BoxedParser<'a, (), ()> {
    newline1(indent(0))
}

fn parse_clause<'a, F: 'a>(name: &'static str, cont_parse: F) -> lip::BoxedParser<'a, String, ()>
where
    F: Fn(&char) -> bool,
{
    succeed!(|clause| clause)
        .skip(token(name))
        .skip(token(":"))
        .keep(take_chomped(chomp_while1(cont_parse, name)))
}

fn parse_yue_clause<'a>() -> lip::BoxedParser<'a, String, ()> {
    parse_clause("yue", |c: &char| *c != '\n' && *c != '\r')
}

fn parse_eng_clause<'a>() -> lip::BoxedParser<'a, String, ()> {
    parse_clause("eng", |c: &char| *c != '\n' && *c != '\r')
}

fn parse_alt_clause<'a>() -> lip::BoxedParser<'a, AltClause, ()> {
    (succeed!(|alt_lang: Located<String>, clause: String| (alt_lang, clause))
        .keep(located(take_chomped(chomp_while1(
            |c: &char| *c != ':',
            "alternative languages",
        ))))
        .skip(token(":"))
        .keep(take_chomped(chomp_while1(
            |c: &char| *c != '\n' && *c != '\r',
            "alternative language clause",
        ))))
    .and_then(|(alt_lang, clause)| match &alt_lang.value[..] {
        "jpn" => succeed!(|_| (AltLang::Jpn, clause)).keep(token("")),
        "kor" => succeed!(|_| (AltLang::Kor, clause)).keep(token("")),
        "vie" => succeed!(|_| (AltLang::Vie, clause)).keep(token("")),
        "lat" => succeed!(|_| (AltLang::Lat, clause)).keep(token("")),
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

fn parse_pr_clause<'a>(name: &'static str) -> lip::BoxedParser<'a, (String, Option<String>), ()> {
    succeed!(|clause, pr| (clause, pr))
        .keep(
            parse_clause(name, |c: &char| *c != '(' && *c != '\n' && *c != '\r')
                .map(|clause| clause.trim_end().to_string()),
        )
        .keep(optional(
            None,
            succeed!(Some)
                .skip(token("("))
                .keep(take_chomped(chomp_while1(
                    &|c: &char| *c != ')',
                    "jyutping",
                )))
                .skip(token(")")),
        ))
}

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

fn parse_explaination<'a>() -> lip::BoxedParser<'a, Def, ()> {
    succeed!(|yue, eng, alts, egs| Def {
        yue,
        eng,
        alts,
        egs
    })
    .skip(token("<explanation>"))
    .skip(parse_br())
    .keep(parse_yue_clause())
    .skip(parse_br())
    .keep(optional(
        None,
        succeed!(Some).keep(parse_eng_clause()).skip(parse_br()),
    ))
    .keep(zero_or_more(
        succeed!(|clause| clause)
            .keep(parse_alt_clause())
            .skip(optional((), parse_br())),
    ))
    .keep(one_or_more(parse_eg()))
}

fn parse_defs<'a>() -> lip::BoxedParser<'a, Vec<Def>, ()> {
    succeed!(|defs| defs).keep(one_or_more(
        succeed!(|def| def)
            .keep(one_of!(
                succeed!(|yue, eng, alts| Def {
                    yue,
                    eng,
                    egs: Vec::new(),
                    alts
                })
                .keep(parse_yue_clause())
                .skip(optional((), parse_br()))
                .keep(optional(None, succeed!(Some).keep(parse_eng_clause())))
                .skip(optional((), parse_br()))
                .keep(zero_or_more(
                    succeed!(|clause| clause)
                        .keep(parse_alt_clause())
                        .skip(optional((), parse_br()))
                )),
                parse_explaination()
            ))
            .skip(optional(
                (),
                succeed!(|_| ())
                    .skip(parse_br())
                    .keep(token("----"))
                    .skip(parse_br()),
            )),
    ))
}

fn parse_content<'a>(variants: Vec<Variant>) -> lip::BoxedParser<'a, Option<Entry>, ()> {
    one_of!(
        succeed!(|poses, labels, sims, ants, refs, imgs, defs| Some(Entry {
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

fn main() {
    if let Err(err) = to_apple_dict() {
        println!("error reading csv file: {}", err);
        process::exit(1);
    }
}

#[cfg(test)]
#[test]
fn test_parse_pr() {
    assert_succeed(
        parse_pr_clause("yue"),
        "yue:《飛狐外傳》 (fei1 wu4 ngoi6 zyun2)",
        (
            "《飛狐外傳》".to_string(),
            Some("fei1 wu4 ngoi6 zyun2".to_string()),
        ),
    );
    assert_succeed(parse_pr_clause("yue"), "yue:《哈利波特》出外傳喎，你會唔會睇啊？ (\"haa1 lei6 bo1 dak6\" ceot1 ngoi6 zyun2 wo3, nei5 wui5 m4 wui5 tai2 aa3?)",
    ("《哈利波特》出外傳喎，你會唔會睇啊？".to_string(), Some("\"haa1 lei6 bo1 dak6\" ceot1 ngoi6 zyun2 wo3, nei5 wui5 m4 wui5 tai2 aa3?".to_string())));
    assert_succeed(
        parse_pr_clause("yue"),
        "yue:佢唔係真喊架，扮嘢㗎咋。 (keoi5 m4 hai6 zan1 haam3 gaa3, baan6 je5 gaa3 zaa3.)",
        (
            "佢唔係真喊架，扮嘢㗎咋。".to_string(),
            Some("keoi5 m4 hai6 zan1 haam3 gaa3, baan6 je5 gaa3 zaa3.".to_string()),
        ),
    );
    assert_succeed(parse_pr_clause("yue"), "yue:條八婆好扮嘢㗎，連嗌個叉飯都要講英文。 (tiu4 baat3 po4 hou2 baan6 je5 gaa3, lin4 aai3 go3 caa1 faan6 dou1 jiu3 gong2 jing1 man2.)", ("條八婆好扮嘢㗎，連嗌個叉飯都要講英文。".to_string(), Some("tiu4 baat3 po4 hou2 baan6 je5 gaa3, lin4 aai3 go3 caa1 faan6 dou1 jiu3 gong2 jing1 man2.".to_string())));
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
                "你可唔可以唔好成日zip呀，吓！".to_string(),
                Some("nei5 ho2 m4 ho2 ji5 m4 hou2 sing4 jat6 zip4 aa3, haa2!".to_string()),
            )),
            eng: Some("Stop tsking!".to_string()),
        },
    );
    assert_succeed(
        parse_eg(),
        "<eg>
yue:佢今日心神恍惚，時時做錯嘢，好似有心事喎。 (keoi5 gam1 jat6 sam1 san4 fong2 fat1, si4 si4 zou6 co3 je5, hou2 ci5 jau5 sam1 si6 wo3.)",
        Eg {
            zho: None,
            yue: Some(("佢今日心神恍惚，時時做錯嘢，好似有心事喎。".to_string(), Some("keoi5 gam1 jat6 sam1 san4 fong2 fat1, si4 si4 zou6 co3 je5, hou2 ci5 jau5 sam1 si6 wo3.".to_string()))),
            eng: None,
        },
    );
}

#[test]
fn test_parse_explaination() {
    assert_succeed(
        parse_explaination(),
        "<explanation>
yue:表現不屑而發出嘅聲音
eng:tsk
<eg>
yue:你可唔可以唔好成日zip呀，吓！ (nei5 ho2 m4 ho2 ji5 m4 hou2 sing4 jat6 zip4 aa3, haa2!)
eng:Stop tsking!",
        Def {
            yue: "表現不屑而發出嘅聲音".to_string(),
            eng: Some("tsk".to_string()),
            alts: vec![],
            egs: vec![Eg {
                zho: None,
                yue: Some((
                    "你可唔可以唔好成日zip呀，吓！".to_string(),
                    Some("nei5 ho2 m4 ho2 ji5 m4 hou2 sing4 jat6 zip4 aa3, haa2!".to_string()),
                )),
                eng: Some("Stop tsking!".to_string()),
            }],
        },
    );
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
            yue: "表現不屑而發出嘅聲音".to_string(),
            eng: Some("tsk".to_string()),
            alts: vec![],
            egs: vec![Eg {
                zho: None,
                yue: Some((
                    "你可唔可以唔好成日zip呀，吓！".to_string(),
                    Some("nei5 ho2 m4 ho2 ji5 m4 hou2 sing4 jat6 zip4 aa3, haa2!".to_string()),
                )),
                eng: Some("Stop tsking!".to_string()),
            }],
        }],
    );
}

#[test]
fn test_parse_content() {
    {
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
            parse_content(variants.clone()),
            "(pos:動詞)(pos:擬聲詞)
<explanation>
yue:表現不屑而發出嘅聲音
eng:tsk
<eg>
yue:你可唔可以唔好成日zip呀，吓！ (nei5 ho2 m4 ho2 ji5 m4 hou2 sing4 jat6 zip4 aa3, haa2!)
eng:Stop tsking!",
            Some(Entry {
                variants,
                poses: vec!["動詞".to_string(), "擬聲詞".to_string()],
                labels: vec![],
                sims: vec![],
                ants: vec![],
                refs: vec![],
                imgs: vec![],
                defs: vec![Def {
                    yue: "表現不屑而發出嘅聲音".to_string(),
                    eng: Some("tsk".to_string()),
                    alts: vec![],
                    egs: vec![Eg {
                        zho: None,
                        yue: Some((
                            "你可唔可以唔好成日zip呀，吓！".to_string(),
                            Some(
                                "nei5 ho2 m4 ho2 ji5 m4 hou2 sing4 jat6 zip4 aa3, haa2!"
                                    .to_string(),
                            ),
                        )),
                        eng: Some("Stop tsking!".to_string()),
                    }],
                }],
            }),
        );
    }
    {
        let variants = vec![Variant {
            word: "hihi".to_string(),
            prs: vec!["haai1 haai1".to_string()],
        }];
        assert_succeed(
            parse_content(variants.clone()),
            "(pos:動詞)(label:潮語)(label:粗俗)(ref:http://evchk.wikia.com/wiki/%E9%AB%98%E7%99%BB%E7%B2%97%E5%8F%A3Filter)
yue:「#仆街」嘅代名詞",
            Some(Entry {
                variants,
                poses: vec!["動詞".to_string()],
                labels: vec!["潮語".to_string(), "粗俗".to_string()],
                sims: vec![],
                ants: vec![],
                refs: vec!["http://evchk.wikia.com/wiki/%E9%AB%98%E7%99%BB%E7%B2%97%E5%8F%A3Filter".to_string()],
                imgs: vec![],
                defs: vec![Def {
                    yue: "「#仆街」嘅代名詞".to_string(),
                    eng: None,
                    alts: vec![],
                    egs: vec![],
                }]
            })
        );
    }
}
