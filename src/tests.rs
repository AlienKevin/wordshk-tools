use super::dict::*;
use super::parse::*;
use super::rich_dict::{Text, TextStyle, Word, WordSegment};
use lip::assert_succeed;

fn text(string: &'static str) -> Segment {
    (SegmentType::Text, string.to_string())
}

fn link(string: &'static str) -> Segment {
    (SegmentType::Link, string.to_string())
}

fn text_word(word: Word) -> WordSegment {
    (SegmentType::Text, word)
}

fn link_word(word: Word) -> WordSegment {
    (SegmentType::Link, word)
}

fn simple_line(string: &'static str) -> Line {
    vec![text(string)]
}

fn simple_clause(string: &'static str) -> Clause {
    vec![simple_line(string)]
}

fn bold(string: &'static str) -> Text {
    (TextStyle::Bold, string.to_string())
}

fn normal(string: &'static str) -> Text {
    (TextStyle::Normal, string.to_string())
}

fn bold_word(string: &'static str) -> Word {
    Word(vec![bold(string)])
}

fn normal_word(string: &'static str) -> Word {
    Word(vec![normal(string)])
}

#[cfg(test)]
#[test]
fn test_parse_clause() {
    assert_succeed(
        parse_clause("yue"),
        "《#ABC》入面有#一個、#兩個 同埋#三個：字母",
        vec![vec![
            text("《"),
            link("ABC"),
            text("》入面有"),
            link("一個"),
            text("、"),
            link("兩個"),
            text("同埋"),
            link("三個"),
            text("：字母"),
        ]],
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
            yue: Some((vec![link("難民營")], Some("naan6 man4 jing4".to_string()))),
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
        vec![
            Def {
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
            },
        ],
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
        let variants = Variants(vec![
            Variant {
                word: "zip".to_string(),
                prs: LaxJyutPings(vec![LaxJyutPing(vec![LaxJyutPingSegment::Standard(
                    JyutPing {
                        initial: Some(JyutPingInitial::Z),
                        nucleus: JyutPingNucleus::I,
                        coda: Some(JyutPingCoda::P),
                        tone: Some(JyutPingTone::T4),
                    },
                )])]),
            },
            Variant {
                word: "jip".to_string(),
                prs: LaxJyutPings(vec![LaxJyutPing(vec![LaxJyutPingSegment::Standard(
                    JyutPing {
                        initial: Some(JyutPingInitial::Z),
                        nucleus: JyutPingNucleus::I,
                        coda: Some(JyutPingCoda::P),
                        tone: Some(JyutPingTone::T4),
                    },
                )])]),
            },
        ]);
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
        let variants = Variants(vec![Variant {
            word: "hihi".to_string(),
            prs: LaxJyutPings(vec![LaxJyutPing(vec![
                LaxJyutPingSegment::Standard(JyutPing {
                    initial: Some(JyutPingInitial::H),
                    nucleus: JyutPingNucleus::Aa,
                    coda: Some(JyutPingCoda::I),
                    tone: Some(JyutPingTone::T1),
                }),
                LaxJyutPingSegment::Standard(JyutPing {
                    initial: Some(JyutPingInitial::H),
                    nucleus: JyutPingNucleus::Aa,
                    coda: Some(JyutPingCoda::I),
                    tone: Some(JyutPingTone::T1),
                }),
            ])]),
        }]);
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
    use super::unicode::is_latin;

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
fn test_tokenize() {
    use super::rich_dict::tokenize;

    assert_eq!(
        tokenize(&vec!["upgrade".into()], "我 upgrade 咗做 Win 10 之後"),
        vec![
            normal_word("我"),
            bold_word("upgrade"),
            normal_word("咗"),
            normal_word("做"),
            normal_word("Win 10"),
            normal_word("之"),
            normal_word("後")
        ]
    );
    assert_eq!(
        tokenize(&vec!["沽".into()], "唔該幫我13蚊沽200股。"),
        vec![
            normal_word("唔"),
            normal_word("該"),
            normal_word("幫"),
            normal_word("我"),
            normal_word("13"),
            normal_word("蚊"),
            bold_word("沽"),
            normal_word("200"),
            normal_word("股"),
            normal_word("。")
        ]
    );
    assert_eq!(
        tokenize(&vec!["好耐".into()], "「你好耐。」「55」"),
        vec![
            normal_word("「"),
            normal_word("你"),
            bold_word("好"),
            bold_word("耐"),
            normal_word("。"),
            normal_word("」"),
            normal_word("「"),
            normal_word("55"),
            normal_word("」")
        ]
    );
    assert_eq!(
        tokenize(&vec!["I mean".into()], "I mean，廣東話。"),
        vec![
            bold_word("I mean"),
            normal_word("，"),
            normal_word("廣"),
            normal_word("東"),
            normal_word("話"),
            normal_word("。")
        ]
    );
}

#[test]
fn test_flatten_line() {
    use super::rich_dict::flatten_line;

    assert_eq!(
        flatten_line(
            &vec!["upgrade".into()],
            &vec![text("我 "), link("upgrade"), text(" 咗做 Win 10 之後")]
        ),
        vec![
            text_word(normal_word("我")),
            link_word(bold_word("upgrade")),
            text_word(normal_word("咗")),
            text_word(normal_word("做")),
            text_word(normal_word("Win 10")),
            text_word(normal_word("之")),
            text_word(normal_word("後"))
        ]
    );
    assert_eq!(
        flatten_line(&vec!["I mean".into()], &vec![text("I mean，廣東話。")]),
        vec![
            text_word(bold_word("I mean")),
            text_word(normal_word("，")),
            text_word(normal_word("廣")),
            text_word(normal_word("東")),
            text_word(normal_word("話")),
            text_word(normal_word("。"))
        ]
    );
    assert_eq!(
        flatten_line(
            &vec!["I mean".into()],
            &vec![text("I mean，"), link("廣東話"), text("。")]
        ),
        vec![
            text_word(bold_word("I mean")),
            text_word(normal_word("，")),
            link_word(normal_word("廣")),
            link_word(normal_word("東")),
            link_word(normal_word("話")),
            text_word(normal_word("。"))
        ]
    );
}

#[test]
fn test_match_ruby() {
    use super::rich_dict::{match_ruby, RubySegment::*};

    assert_eq!(
        match_ruby(
            &vec!["I mean".into()],
            &vec![text("I mean，廣東話。")],
            &vec!["aai6", "min1", "gwong2", "dung1", "waa2"]
        ),
        vec![
            Word(bold_word("I mean"), vec!["aai6".into(), "min1".into()]),
            Punc("，".into()),
            Word(normal_word("廣"), vec!["gwong2".into()]),
            Word(normal_word("東"), vec!["dung1".into()]),
            Word(normal_word("話"), vec!["waa2".into()]),
            Punc("。".into()),
        ]
    );

    assert_eq!(
        match_ruby(
            &vec!["沽".into()],
            &vec![text("唔該，幫我13蚊沽200股。")],
            &"m4 goi1 bong1 ngo5 sap6 saam1 man1 gu1 ji6 baak3 gu2"
                .split_whitespace()
                .collect::<Vec<&str>>()
        ),
        vec![
            Word(normal_word("唔"), vec!["m4".into()]),
            Word(normal_word("該"), vec!["goi1".into()]),
            Punc("，".into()),
            Word(normal_word("幫"), vec!["bong1".into()]),
            Word(normal_word("我"), vec!["ngo5".into()]),
            Word(normal_word("13"), vec!["sap6".into(), "saam1".into()]),
            Word(normal_word("蚊"), vec!["man1".into()]),
            Word(bold_word("沽"), vec!["gu1".into()]),
            Word(normal_word("200"), vec!["ji6".into(), "baak3".into()]),
            Word(normal_word("股"), vec!["gu2".into()]),
            Punc("。".into()),
        ]
    );

    assert_eq!(
        match_ruby(
            &vec!["upgrade".into()],
            &vec![text("我 "), link("upgrade"), text(" 咗做 Win 10 之後。")],
            &"ngo5 ap1 gwei1 zo2 zou6 win1 sap6 zi1 hau6"
                .split_whitespace()
                .collect::<Vec<&str>>()
        ),
        vec![
            Word(normal_word("我"), vec!["ngo5".into()]),
            LinkedWord(vec![(
                bold_word("upgrade"),
                vec!["ap1".into(), "gwei1".into()]
            )]),
            Word(normal_word("咗"), vec!["zo2".into()]),
            Word(normal_word("做"), vec!["zou6".into()]),
            Word(normal_word("Win 10"), vec!["win1".into(), "sap6".into()]),
            Word(normal_word("之"), vec!["zi1".into()]),
            Word(normal_word("後"), vec!["hau6".into()]),
            Punc("。".into()),
        ]
    );

    // two full matches
    assert_eq!(
        match_ruby(
            &vec!["經理".into()],
            &vec![link("經理")],
            &vec!["ging1".into(), "lei5".into()]
        ),
        vec![LinkedWord(vec![
            (bold_word("經"), vec!["ging1".into()]),
            (bold_word("理"), vec!["lei5".into()])
        ])]
    );

    // one half match
    assert_eq!(
        match_ruby(
            &vec!["經理".into()],
            &vec![link("經理")],
            &vec!["ging1".into(), "lei".into()]
        ),
        vec![LinkedWord(vec![
            (bold_word("經"), vec!["ging1".into()]),
            (bold_word("理"), vec!["lei".into()])
        ])]
    );

    // two half matches
    assert_eq!(
        match_ruby(
            &vec!["經理".into()],
            &vec![link("經理")],
            &vec!["ging".into(), "lei".into()]
        ),
        vec![LinkedWord(vec![
            (bold_word("經"), vec!["ging".into()]),
            (bold_word("理"), vec!["lei".into()])
        ])]
    );
}

#[test]
fn test_parse_jyutping() {
    assert_eq!(
        parse_jyutping(&"ging1".to_string()),
        Some(JyutPing {
            initial: Some(JyutPingInitial::G),
            nucleus: JyutPingNucleus::I,
            coda: Some(JyutPingCoda::Ng),
            tone: Some(JyutPingTone::T1)
        })
    );

    assert_eq!(
        parse_jyutping(&"gwok3".to_string()),
        Some(JyutPing {
            initial: Some(JyutPingInitial::Gw),
            nucleus: JyutPingNucleus::O,
            coda: Some(JyutPingCoda::K),
            tone: Some(JyutPingTone::T3)
        })
    );

    assert_eq!(
        parse_jyutping(&"aa".to_string()),
        Some(JyutPing {
            initial: None,
            nucleus: JyutPingNucleus::Aa,
            coda: None,
            tone: None
        })
    );

    assert_eq!(
        parse_jyutping(&"a2".to_string()),
        Some(JyutPing {
            initial: None,
            nucleus: JyutPingNucleus::A,
            coda: None,
            tone: Some(JyutPingTone::T2)
        })
    );

    assert_eq!(
        parse_jyutping(&"a".to_string()),
        Some(JyutPing {
            initial: None,
            nucleus: JyutPingNucleus::A,
            coda: None,
            tone: None,
        })
    );

    assert_eq!(
        parse_jyutping(&"seoi5".to_string()),
        Some(JyutPing {
            initial: Some(JyutPingInitial::S),
            nucleus: JyutPingNucleus::Eo,
            coda: Some(JyutPingCoda::I),
            tone: Some(JyutPingTone::T5)
        })
    );

    assert_eq!(
        parse_jyutping(&"baau".to_string()),
        Some(JyutPing {
            initial: Some(JyutPingInitial::B),
            nucleus: JyutPingNucleus::Aa,
            coda: Some(JyutPingCoda::U),
            tone: None,
        })
    );
}

#[test]
fn test_compare_jyutping() {
    use super::search::compare_jyutping;

    // identical jyutpings
    assert_eq!(
        100,
        compare_jyutping(
            &JyutPing {
                initial: Some(JyutPingInitial::S),
                nucleus: JyutPingNucleus::Eo,
                coda: Some(JyutPingCoda::I),
                tone: Some(JyutPingTone::T5)
            },
            &JyutPing {
                initial: Some(JyutPingInitial::S),
                nucleus: JyutPingNucleus::Eo,
                coda: Some(JyutPingCoda::I),
                tone: Some(JyutPingTone::T5)
            }
        )
    );

    assert_eq!(
        100,
        compare_jyutping(
            &JyutPing {
                initial: None,
                nucleus: JyutPingNucleus::Aa,
                coda: None,
                tone: None
            },
            &JyutPing {
                initial: None,
                nucleus: JyutPingNucleus::Aa,
                coda: None,
                tone: None
            }
        )
    );

    // Initial: same category
    assert_eq!(
        84,
        compare_jyutping(
            &JyutPing {
                initial: Some(JyutPingInitial::F),
                nucleus: JyutPingNucleus::Eo,
                coda: Some(JyutPingCoda::I),
                tone: Some(JyutPingTone::T5)
            },
            &JyutPing {
                initial: Some(JyutPingInitial::S),
                nucleus: JyutPingNucleus::Eo,
                coda: Some(JyutPingCoda::I),
                tone: Some(JyutPingTone::T5)
            }
        )
    );

    // Initial: different category
    assert_eq!(
        60,
        compare_jyutping(
            &JyutPing {
                initial: Some(JyutPingInitial::T),
                nucleus: JyutPingNucleus::Eo,
                coda: Some(JyutPingCoda::I),
                tone: Some(JyutPingTone::T5)
            },
            &JyutPing {
                initial: Some(JyutPingInitial::S),
                nucleus: JyutPingNucleus::Eo,
                coda: Some(JyutPingCoda::I),
                tone: Some(JyutPingTone::T5)
            }
        )
    );

    // Initial: one has initial the other does not
    assert_eq!(
        60,
        compare_jyutping(
            &JyutPing {
                initial: Some(JyutPingInitial::T),
                nucleus: JyutPingNucleus::Eo,
                coda: Some(JyutPingCoda::I),
                tone: Some(JyutPingTone::T5)
            },
            &JyutPing {
                initial: None,
                nucleus: JyutPingNucleus::Eo,
                coda: Some(JyutPingCoda::I),
                tone: Some(JyutPingTone::T5)
            }
        )
    );

    // Nucleus: same category
    assert_eq!(
        96,
        compare_jyutping(
            &JyutPing {
                initial: Some(JyutPingInitial::S),
                nucleus: JyutPingNucleus::Eo,
                coda: Some(JyutPingCoda::I),
                tone: Some(JyutPingTone::T5)
            },
            &JyutPing {
                initial: Some(JyutPingInitial::S),
                nucleus: JyutPingNucleus::Oe,
                coda: Some(JyutPingCoda::I),
                tone: Some(JyutPingTone::T5)
            }
        )
    );

    // Nucleus: different roundedness
    assert_eq!(
        94,
        compare_jyutping(
            &JyutPing {
                initial: Some(JyutPingInitial::S),
                nucleus: JyutPingNucleus::I,
                coda: Some(JyutPingCoda::I),
                tone: Some(JyutPingTone::T5)
            },
            &JyutPing {
                initial: Some(JyutPingInitial::S),
                nucleus: JyutPingNucleus::Yu,
                coda: Some(JyutPingCoda::I),
                tone: Some(JyutPingTone::T5)
            }
        )
    );

    // Nucleus: different height
    assert_eq!(
        93,
        compare_jyutping(
            &JyutPing {
                initial: Some(JyutPingInitial::T),
                nucleus: JyutPingNucleus::I,
                coda: Some(JyutPingCoda::I),
                tone: Some(JyutPingTone::T5)
            },
            &JyutPing {
                initial: Some(JyutPingInitial::T),
                nucleus: JyutPingNucleus::E,
                coda: Some(JyutPingCoda::I),
                tone: Some(JyutPingTone::T5)
            }
        )
    );

    // Nucleus: different backness
    assert_eq!(
        93,
        compare_jyutping(
            &JyutPing {
                initial: Some(JyutPingInitial::T),
                nucleus: JyutPingNucleus::Yu,
                coda: Some(JyutPingCoda::I),
                tone: Some(JyutPingTone::T5)
            },
            &JyutPing {
                initial: Some(JyutPingInitial::T),
                nucleus: JyutPingNucleus::U,
                coda: Some(JyutPingCoda::I),
                tone: Some(JyutPingTone::T5)
            }
        )
    );

    // Coda: same category
    assert_eq!(
        94,
        compare_jyutping(
            &JyutPing {
                initial: Some(JyutPingInitial::S),
                nucleus: JyutPingNucleus::Eo,
                coda: Some(JyutPingCoda::P),
                tone: Some(JyutPingTone::T5)
            },
            &JyutPing {
                initial: Some(JyutPingInitial::S),
                nucleus: JyutPingNucleus::Eo,
                coda: Some(JyutPingCoda::T),
                tone: Some(JyutPingTone::T5)
            }
        )
    );

    // Coda: different category
    assert_eq!(
        76,
        compare_jyutping(
            &JyutPing {
                initial: Some(JyutPingInitial::S),
                nucleus: JyutPingNucleus::Eo,
                coda: Some(JyutPingCoda::P),
                tone: Some(JyutPingTone::T5)
            },
            &JyutPing {
                initial: Some(JyutPingInitial::S),
                nucleus: JyutPingNucleus::Eo,
                coda: Some(JyutPingCoda::I),
                tone: Some(JyutPingTone::T5)
            }
        )
    );

    // Coda: one has coda the other does not
    assert_eq!(
        76,
        compare_jyutping(
            &JyutPing {
                initial: Some(JyutPingInitial::T),
                nucleus: JyutPingNucleus::Eo,
                coda: Some(JyutPingCoda::I),
                tone: Some(JyutPingTone::T5)
            },
            &JyutPing {
                initial: Some(JyutPingInitial::T),
                nucleus: JyutPingNucleus::Eo,
                coda: None,
                tone: Some(JyutPingTone::T5)
            }
        )
    );

    // Tone: different tone
    assert_eq!(
        96,
        compare_jyutping(
            &JyutPing {
                initial: Some(JyutPingInitial::S),
                nucleus: JyutPingNucleus::Eo,
                coda: Some(JyutPingCoda::T),
                tone: Some(JyutPingTone::T4)
            },
            &JyutPing {
                initial: Some(JyutPingInitial::S),
                nucleus: JyutPingNucleus::Eo,
                coda: Some(JyutPingCoda::T),
                tone: Some(JyutPingTone::T5)
            }
        )
    );

    // Mixed:
    // Initial: different category
    // Coda: different category
    assert_eq!(
        36,
        compare_jyutping(
            &JyutPing {
                initial: Some(JyutPingInitial::S),
                nucleus: JyutPingNucleus::Eo,
                coda: Some(JyutPingCoda::P),
                tone: Some(JyutPingTone::T5)
            },
            &JyutPing {
                initial: Some(JyutPingInitial::T),
                nucleus: JyutPingNucleus::Eo,
                coda: Some(JyutPingCoda::I),
                tone: Some(JyutPingTone::T5)
            }
        )
    );
}
