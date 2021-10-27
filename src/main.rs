use lip::ParseResult;
use lip::*;
use std::error::Error;
use std::io;
use std::process;

type Dict = Vec<Entry>;

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
    Fra,
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
                } => {
                    match parse_content(id, head_result).run(content, ()) {
                        ParseResult::Ok {
                            output: content_result,
                            ..
                        } => content_result,
                        ParseResult::Err { message, .. } => {
                            println!("Error in #{}: {:?}", id , message);
                            None
                        }
                    }
                }
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

fn parse_br<'a>() -> lip::BoxedParser<'a, (), ()> {
    chomp_if(|c| c == "\r\n" || c == "\n", "a newline")
}

fn parse_clause<'a, F: 'a>(name: &'static str, cont_parse: F) -> lip::BoxedParser<'a, String, ()>
where
    F: Fn(&char) -> bool,
{
    succeed!(|clause| clause)
        .skip(token(name))
        .skip(token(":"))
        .keep(take_chomped(chomp_while1c(cont_parse, name)))
}

fn parse_eng_clause<'a>() -> lip::BoxedParser<'a, String, ()> {
    parse_clause("eng", |c: &char| *c != '\n' && *c != '\r')
}

fn parse_multiline_clause<'a>(name: &'static str) -> lip::BoxedParser<'a, String, ()> {
    succeed!(|lines: Vec<String>| lines.join("\n"))
        .skip(token(name))
        .skip(token(":"))
        .keep(one_or_more_until(
            succeed!(|line| line).keep(one_of!(
                // empty line
                succeed!(|_| "".to_string()).keep(parse_br()),
                // non-empty line
                succeed!(|line| line)
                    .keep(take_chomped(chomp_while1c(
                        |c| *c != '\n' && *c != '\r',
                        "a line",
                    )))
                    .skip(optional((), parse_br()))
            )),
            succeed!(|_| ()).keep(one_of!(
                succeed!(|_| ()).keep(token("<eg>")),
                succeed!(|_| ())
                    .keep(chomp_ifc(|_| true, "any char"))
                    .skip(chomp_ifc(|_| true, "any char"))
                    .skip(chomp_ifc(|_| true, "any char"))
                    .skip(chomp_ifc(|c| *c == ':', "colon `:`"))
            )),
        ))
}

fn parse_alt_clause<'a>() -> lip::BoxedParser<'a, AltClause, ()> {
    (succeed!(|alt_lang: Located<String>, clause: String| (alt_lang, clause))
        .keep(located(take_chomped(chomp_while1c(
            |c: &char| *c != ':',
            "alternative languages",
        ))))
        .skip(token(":"))
        .keep(take_chomped(chomp_while1c(
            |c: &char| *c != '\n' && *c != '\r',
            "alternative language clause",
        ))))
    .and_then(|(alt_lang, clause)| match &alt_lang.value[..] {
        "jpn" => succeed!(|_| (AltLang::Jpn, clause)).keep(token("")),
        "kor" => succeed!(|_| (AltLang::Kor, clause)).keep(token("")),
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
                .keep(take_chomped(chomp_while1c(
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
        succeed!(Some).keep(parse_multiline_clause("eng"))
    ))
    .keep(zero_or_more(
        succeed!(|clause| clause)
            .keep(parse_alt_clause())
            .skip(optional((), parse_br()))
    ))
}

fn parse_defs<'a>() -> lip::BoxedParser<'a, Vec<Def>, ()> {
    succeed!(|defs| defs).keep(one_or_more(
        succeed!(|def| def)
            .keep(one_of!(
                parse_simple_def(),
                parse_rich_def()
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

    assert_succeed(
        parse_rich_def(),
        "<explanation>
yue:字面義指一個人做出啲好古怪嘅表情或者動作，或者大聲講嘢，令其他人感到困擾。引申出表達意見，尤其是過度表達意見嘅行為。
eng:to clench one's teeth or show weird gesture, which implies speaking improperly or with exaggerating manner.
<eg>
yue:觀棋不語真君子，旁觀者不得𪘲牙聳䚗（依牙鬆鋼）。 (gun1 kei4 bat1 jyu5 zan1 gwan1 zi2，pong4 gun1 ze2 bat1 dak1 ji1 ngaa4 sung1 gong3.)
eng:Gentlemen observe chess match with respectful silence. Spectators are not allowed to disturb the competitors.",
        Def {
            yue: "字面義指一個人做出啲好古怪嘅表情或者動作，或者大聲講嘢，令其他人感到困擾。引申出表達意見，尤其是過度表達意見嘅行為。".to_string(),
            eng: Some("to clench one's teeth or show weird gesture, which implies speaking improperly or with exaggerating manner.".to_string()),
            alts: vec![],
            egs: vec![Eg {
                zho: None,
                yue: Some(("觀棋不語真君子，旁觀者不得𪘲牙聳䚗（依牙鬆鋼）。".to_string(), Some("gun1 kei4 bat1 jyu5 zan1 gwan1 zi2，pong4 gun1 ze2 bat1 dak1 ji1 ngaa4 sung1 gong3.".to_string()))),
eng: Some("Gentlemen observe chess match with respectful silence. Spectators are not allowed to disturb the competitors.".to_string()),
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
    assert_succeed(
        parse_defs(),
        "yue:「#仆街」嘅代名詞",
        vec![Def {
            yue: "「#仆街」嘅代名詞".to_string(),
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
            yue: "#天干\n\n#地支".to_string(),
            eng: Some("Heavenly Stems\n\nEarthly Branches".to_string()),
            alts: vec![],
            egs: vec![Eg {
                zho: None,
                yue: Some((
                    "乙等 / 乙級".to_string(),
                    Some("jyut6 dang2 / jyut6 kap1".to_string()),
                )),
                eng: Some("B grade".to_string()),
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
    yue: r#"「#天干」同「#地支」嘅合稱。十天干分別係「#甲#乙#丙#丁#戊#己#庚#辛#壬#癸」。 十二地支係：「#子#丑#寅#卯#辰#巳#午#未#申#酉#戌#亥」。 天干同地支組合就成為以「#甲子」為首嘅六十干支循環。

干支循環通常用嚟計年份。天干亦可以獨立用嚟順序將物件命名，第一個叫「甲」、第二個叫「乙」，如此類推。用法類似西方嘅「A, B, C」 或 「α, β, γ」。中國傳統紀時間嘅方法係將一日分成十二個時辰，每一個時辰由一個地支表示，「子時」係半夜 (11pm 至 1am)，如此類推。"#.to_string(),
    eng: Some(r#"Literally ""Heavenly Stems and Earthly Branches"". It is a traditional Chinese system of counting. Heavenly Stems and Earthly Branches are collectively known as ""Stem-Branch"".

The 10 Heavenly Stems are 甲(gaap3) 乙(jyut6) 丙(bing2) 丁(ding1) 戊(mou6) 己(gei2) 庚(gang1) 辛(san1) 壬(jam4) 癸(gwai3).

The 12 Earthly Branches are 子(zi2) 丑(cau2) 寅(jan4) 卯(maau5) 辰(san4) 巳(zi6) 午(ng5) 未(mei6) 申(san1) 酉(jau5) 戌(seot1) 亥(hoi6). Each Heavenly Stem is paired with an Earthly Branch to form the ""stem-branch"" sexagenary (i.e. 60 element) cycle that starts with 甲子 (gaap3 zi2)

The sexagenary cycle is often used for counting years in the Chinese calendar. Heavenly Stems are also used independently to name things in a particular order -- the first is labeled ""gaap3"", the second ""jyut6"", the third ""bing2"", and so on. It is similar to how ""A, B, C"" and ""α, β, γ"" are used in western cultures. Earthly Branches are also traditionally used to denote time. One day is divided into twelve slots called Chinese-hours (#時辰), starting from 子時 (zi2 si4), which is 11pm to 1am."#.to_string()),
    alts: vec![],
    egs: vec![Eg {
        zho: None,
        yue: Some((
            "乙等 / 乙級".to_string(),
            Some("jyut6 dang2 / jyut6 kap1".to_string()),
        )),
        eng: Some("B grade".to_string()),
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
                        yue: "「#仆街」嘅代名詞".to_string(),
                        eng: None,
                        alts: vec![],
                        egs: vec![],
                    }]
                })
            );
    }
}
