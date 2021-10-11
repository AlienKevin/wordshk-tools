use lip::ParseResult;
use lip::*;
use std::error::Error;
use std::io;
use std::process;

type Dict = Vec<Entry>;

#[derive(Debug)]
struct Entry {
    variants: Vec<Variant>,
    pos: String,
    labels: Vec<String>,
    sims: Vec<String>,
    ants: Vec<String>,
    refs: Vec<String>,
    imgs: Vec<String>,
    defs: Vec<Def>,
}

#[derive(Debug, Clone)]
struct Variant {
    word: String,
    prs: Vec<String>,
}

#[derive(Debug)]
struct Def {
    yue: String,
    eng: String,
    alts: Vec<AltClause>,
    egs: Vec<Eg>,
}

type AltClause = (AltLang, String);

#[derive(Debug)]
enum AltLang {
    Jpn,
    Kor,
    Vie,
    Lat,
}

#[derive(Debug, Clone)]
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
            succeed!(Some).keep(parse_pr_clause("zho")).skip(parse_br()),
        ))
        .keep(optional(
            None,
            succeed!(Some).keep(parse_pr_clause("yue")).skip(parse_br()),
        ))
        .keep(optional(None, succeed!(Some).keep(parse_eng_clause())))
        .skip(optional((), parse_br()))
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
                .skip(parse_br())
                .keep(parse_eng_clause())
                .skip(optional((), parse_br()))
                .keep(zero_or_more(
                    succeed!(|clause| clause)
                        .keep(parse_alt_clause())
                        .skip(optional((), parse_br()))
                )),
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
                .keep(parse_eng_clause())
                .skip(parse_br())
                .keep(zero_or_more(
                    succeed!(|clause| clause)
                        .keep(parse_alt_clause())
                        .skip(optional((), parse_br()))
                ))
                .keep(one_or_more(parse_eg()))
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
        succeed!(|pos, labels, sims, ants, refs, imgs, defs| Some(Entry {
            variants,
            pos,
            labels,
            sims,
            ants,
            refs,
            imgs,
            defs,
        }))
        .skip(token("(pos:"))
        .keep(take_chomped(chomp_while1(&(|c: &char| c != &')'), "pos")))
        .skip(token(")"))
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
