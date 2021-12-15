use super::*;
use super::SegmentType::*;
use super::RubySegment::*;

fn text(string: &'static str) -> Segment {
    (Text, string.to_string())
}

fn link(string: &'static str) -> Segment {
    (Link, string.to_string())
}

fn simple_line(string: &'static str) -> Line {
    vec![text(string)]
}

fn simple_clause(string: &'static str) -> Clause {
    vec![simple_line(string)]
}

#[cfg(test)]
#[test]
fn test_parse_clause() {
    assert_succeed(
        parse_clause("yue"),
        "《#ABC》入面有#一個、#兩個 同埋#三個：字母",
        vec![vec![text("《"), link("ABC"), text("》入面有"), link("一個"), text("、"), link("兩個"), text("同埋"), link("三個"), text("：字母")]]
    );
}

#[test]
fn test_parse_pr() {
    assert_succeed(
        parse_pr_line("yue"),
        "yue:《飛狐外傳》 (fei1 wu4 ngoi6 zyun2)",
        (
            simple_line("《飛狐外傳》"),
            Some("fei1 wu4 ngoi6 zyun2".to_string()),
        ),
    );
    assert_succeed(parse_pr_line("yue"), "yue:《哈利波特》出外傳喎，你會唔會睇啊？ (\"haa1 lei6 bo1 dak6\" ceot1 ngoi6 zyun2 wo3, nei5 wui5 m4 wui5 tai2 aa3?)",
    (simple_line("《哈利波特》出外傳喎，你會唔會睇啊？"), Some("\"haa1 lei6 bo1 dak6\" ceot1 ngoi6 zyun2 wo3, nei5 wui5 m4 wui5 tai2 aa3?".to_string())));
    assert_succeed(
        parse_pr_line("yue"),
        "yue:佢唔係真喊架，扮嘢㗎咋。 (keoi5 m4 hai6 zan1 haam3 gaa3, baan6 je5 gaa3 zaa3.)",
        (
            simple_line("佢唔係真喊架，扮嘢㗎咋。"),
            Some("keoi5 m4 hai6 zan1 haam3 gaa3, baan6 je5 gaa3 zaa3.".to_string()),
        ),
    );
    assert_succeed(parse_pr_line("yue"), "yue:條八婆好扮嘢㗎，連嗌個叉飯都要講英文。 (tiu4 baat3 po4 hou2 baan6 je5 gaa3, lin4 aai3 go3 caa1 faan6 dou1 jiu3 gong2 jing1 man2.)", (simple_line("條八婆好扮嘢㗎，連嗌個叉飯都要講英文。"), Some("tiu4 baat3 po4 hou2 baan6 je5 gaa3, lin4 aai3 go3 caa1 faan6 dou1 jiu3 gong2 jing1 man2.".to_string())));
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
                simple_line("你可唔可以唔好成日zip呀，吓！"),
                Some("nei5 ho2 m4 ho2 ji5 m4 hou2 sing4 jat6 zip4 aa3, haa2!".to_string()),
            )),
            eng: Some(simple_line("Stop tsking!")),
        },
    );
    assert_succeed(
        parse_eg(),
        "<eg>
yue:佢今日心神恍惚，時時做錯嘢，好似有心事喎。 (keoi5 gam1 jat6 sam1 san4 fong2 fat1, si4 si4 zou6 co3 je5, hou2 ci5 jau5 sam1 si6 wo3.)",
        Eg {
            zho: None,
            yue: Some((simple_line("佢今日心神恍惚，時時做錯嘢，好似有心事喎。"), Some("keoi5 gam1 jat6 sam1 san4 fong2 fat1, si4 si4 zou6 co3 je5, hou2 ci5 jau5 sam1 si6 wo3.".to_string()))),
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
                vec![link("難民營")],
                Some("naan6 man4 jing4".to_string()),
            )),
            eng: Some(simple_line("refugee camp")),
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
                    simple_line("你可唔可以唔好成日zip呀，吓！"),
                    Some("nei5 ho2 m4 ho2 ji5 m4 hou2 sing4 jat6 zip4 aa3, haa2!".to_string()),
                )),
                eng: Some(simple_line("Stop tsking!")),
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
                yue: Some((simple_line("觀棋不語真君子，旁觀者不得𪘲牙聳䚗（依牙鬆鋼）。"), Some("gun1 kei4 bat1 jyu5 zan1 gwan1 zi2，pong4 gun1 ze2 bat1 dak1 ji1 ngaa4 sung1 gong3.".to_string()))),
eng: Some(simple_line("Gentlemen observe chess match with respectful silence. Spectators are not allowed to disturb the competitors.")),
            }],
        }
        )
}

#[test]
fn test_parse_defs() {
    assert_succeed(
        parse_defs(),
        "yue:a
eng:a
----
yue:b
eng:b",
        vec![Def {
            yue: simple_clause("a"),
            eng: Some(simple_clause("a")),
            alts: vec![],
            egs: vec![],
        },
        Def {
            yue: simple_clause("b"),
            eng: Some(simple_clause("b")),
            alts: vec![],
            egs: vec![],
        }]
    );

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
                    simple_line("你可唔可以唔好成日zip呀，吓！"),
                    Some("nei5 ho2 m4 ho2 ji5 m4 hou2 sing4 jat6 zip4 aa3, haa2!".to_string()),
                )),
                eng: Some(simple_line("Stop tsking!")),
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
                    simple_line("乙等 / 乙級"),
                    Some("jyut6 dang2 / jyut6 kap1".to_string()),
                )),
                eng: Some(simple_line("B grade")),
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
            simple_line("乙等 / 乙級"),
            Some("jyut6 dang2 / jyut6 kap1".to_string()),
        )),
        eng: Some(simple_line("B grade")),
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
                            simple_line("你可唔可以唔好成日zip呀，吓！"),
                            Some(
                                "nei5 ho2 m4 ho2 ji5 m4 hou2 sing4 jat6 zip4 aa3, haa2!"
                                    .to_string(),
                            ),
                        )),
                        eng: Some(simple_line("Stop tsking!")),
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

#[test]
fn test_is_latin() {
    assert!(is_latin('a'));
    assert!(is_latin('A'));
    assert!(is_latin('Ä'));
    assert!(is_latin('æ'));
    assert!(is_latin('â'));
    assert!(is_latin('b'));
    assert!(is_latin('Ņ'));
    assert!(is_latin('Ō'));
    assert!(is_latin('Ƽ'));

    assert!(!is_latin('ĳ')); // no ligatures
    assert!(!is_latin('Ω')); // no greek
    assert!(!is_latin('2')); // no digits
    assert!(!is_latin('我')); // no CJK
}

#[test]
fn test_text_to_bits() {
    assert_eq!(text_to_bits("我 upgrade 咗做 Win 10 之後"), vec!["我", "upgrade", "咗", "做", "Win 10", "之", "後"]);
    assert_eq!(text_to_bits("唔該幫我13蚊沽200股。"), vec!["唔", "該", "幫", "我", "13", "蚊", "沽", "200", "股", "。"]);
    assert_eq!(text_to_bits("「你好耐。」「55」"), vec!["「", "你", "好", "耐", "。", "」", "「", "55", "」"]);
    assert_eq!(text_to_bits("I mean，廣東話。"), vec!["I mean", "，", "廣", "東", "話", "。"]);
}

#[test]
fn test_flatten_line() {
    assert_eq!(flatten_line(&vec![text("我 "), link("upgrade"), text(" 咗做 Win 10 之後")]),
        vec![text("我"), link("upgrade"), text("咗"), text("做"), text("Win 10"), text("之"), text("後")]
    );
    assert_eq!(flatten_line(&vec![text("I mean，廣東話。")]), vec!["I mean", "，", "廣", "東", "話", "。"].iter().map(|x| text(x)).collect::<Vec<Segment>>());
    assert_eq!(flatten_line(&vec![text("I mean，"), link("廣東話"), text("。")]), vec![text("I mean"), text("，"), link("廣"), link("東"), link("話"), text("。")]);
}

#[test]
fn test_match_ruby() {
    assert_eq!(match_ruby(&vec![text("I mean，廣東話。")], &vec!["aai6", "min1", "gwong2", "dung1", "waa2"]),
    vec![Word("I mean".into(), vec!["aai6".into(), "min1".into()]),
    Punc("，".into()),
    Word("廣".into(), vec!["gwong2".into()]),
    Word("東".into(), vec!["dung1".into()]),
    Word("話".into(), vec!["waa2".into()]),
    Punc("。".into()),
    ]);

    assert_eq!(match_ruby(&vec![text("唔該，幫我13蚊沽200股。")],
        &"m4 goi1 bong1 ngo5 sap6 saam1 man1 gu1 ji6 baak3 gu2".split_whitespace().collect::<Vec<&str>>()),
    vec![Word("唔".into(), vec!["m4".into()]),
    Word("該".into(), vec!["goi1".into()]),
    Punc("，".into()),
    Word("幫".into(), vec!["bong1".into()]),
    Word("我".into(), vec!["ngo5".into()]),
    Word("13".into(), vec!["sap6".into(), "saam1".into()]),
    Word("蚊".into(), vec!["man1".into()]),
    Word("沽".into(), vec!["gu1".into()]),
    Word("200".into(), vec!["ji6".into(), "baak3".into()]),
    Word("股".into(), vec!["gu2".into()]),
    Punc("。".into()),
    ]);

    assert_eq!(match_ruby(&vec![text("我 "), link("upgrade"), text(" 咗做 Win 10 之後。")],
        &"ngo5 ap1 gwei1 zo2 zou6 win1 sap6 zi1 hau6".split_whitespace().collect::<Vec<&str>>()),
    vec![Word("我".into(), vec!["ngo5".into()]),
    LinkedWord(vec![("upgrade".into(), vec!["ap1".into(), "gwei1".into()])]),
    Word("咗".into(), vec!["zo2".into()]),
    Word("做".into(), vec!["zou6".into()]),
    Word("Win 10".into(), vec!["win1".into(), "sap6".into()]),
    Word("之".into(), vec!["zi1".into()]),
    Word("後".into(), vec!["hau6".into()]),
    Punc("。".into()),
    ]);

    // two full matches
    assert_eq!(match_ruby(&vec![link("經理")], &vec!["ging1".into(), "lei5".into()]),
    vec![LinkedWord(vec![("經".into(), vec!["ging1".into()]), ("理".into(), vec!["lei5".into()])])]);

    // one half match
    assert_eq!(match_ruby(&vec![link("經理")], &vec!["ging1".into(), "lei".into()]),
    vec![LinkedWord(vec![("經".into(), vec!["ging1".into()]), ("理".into(), vec!["lei".into()])])]);

    // two half matches
    assert_eq!(match_ruby(&vec![link("經理")], &vec!["ging".into(), "lei".into()]),
    vec![LinkedWord(vec![("經".into(), vec!["ging".into()]), ("理".into(), vec!["lei".into()])])]);
}

#[test]
fn test_parse_jyutping() {
    assert_eq!(
        parse_jyutping(&"ging1".to_string()),
        Some(JyutPing { initial: Some(JyutPingInitial::G), nucleus: JyutPingNucleus::I, coda: Some(JyutPingCoda::Ng), tone: Some(JyutPingTone::T1) })
    );

    assert_eq!(
        parse_jyutping(&"gwok3".to_string()),
        Some(JyutPing { initial: Some(JyutPingInitial::Gw), nucleus: JyutPingNucleus::O, coda: Some(JyutPingCoda::K), tone: Some(JyutPingTone::T3) })
    );

    assert_eq!(
        parse_jyutping(&"aa".to_string()),
        Some(JyutPing { initial: None, nucleus: JyutPingNucleus::Aa, coda: None, tone: None })
    );

    assert_eq!(
        parse_jyutping(&"aa".to_string()),
        Some(JyutPing { initial: None, nucleus: JyutPingNucleus::Aa, coda: None, tone: None })
    );

    assert_eq!(
        parse_jyutping(&"a2".to_string()),
        Some(JyutPing { initial: None, nucleus: JyutPingNucleus::A, coda: None, tone: Some(JyutPingTone::T2) })
    );

    assert_eq!(
        parse_jyutping(&"seoi5".to_string()),
        Some(JyutPing { initial: Some(JyutPingInitial::S), nucleus: JyutPingNucleus::Eo, coda: Some(JyutPingCoda::I), tone: Some(JyutPingTone::T5) })
    );
}
