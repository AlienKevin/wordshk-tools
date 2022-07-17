use super::dict::*;
use super::jyutping::*;
use super::parse::*;
use super::rich_dict;
use lip::assert_succeed;

fn text(string: &'static str) -> Segment {
    (SegmentType::Text, string.to_string())
}

fn link(string: &'static str) -> Segment {
    (SegmentType::Link, string.to_string())
}

fn text_word(word: rich_dict::Word) -> rich_dict::WordSegment {
    (SegmentType::Text, word)
}

fn link_word(word: rich_dict::Word) -> rich_dict::WordSegment {
    (SegmentType::Link, word)
}

fn simple_line(string: &'static str) -> Line {
    vec![text(string)]
}

fn simple_clause(string: &'static str) -> Clause {
    vec![simple_line(string)]
}

fn bold(string: &'static str) -> rich_dict::Text {
    (rich_dict::TextStyle::Bold, string.to_string())
}

fn normal(string: &'static str) -> rich_dict::Text {
    (rich_dict::TextStyle::Normal, string.to_string())
}

fn bold_word(string: &'static str) -> rich_dict::Word {
    rich_dict::Word(vec![bold(string)])
}

fn normal_word(string: &'static str) -> rich_dict::Word {
    rich_dict::Word(vec![normal(string)])
}

#[cfg(test)]
#[test]
fn test_parse_line() {
    // Need to delete a single space before '#' if previous character is a CJK
    // Need to delete a single space following a link (unconditional)
    assert_succeed(
        parse_line("yue"),
        "有個詞叫 #你好 係好常用嘅。",
        vec![text("有個詞叫"), link("你好"), text("係好常用嘅。")],
    );

    assert_succeed(
        parse_line("eng"),
        "particle #喇 laa3 before the particles #喎  wo3, #噃 bo3 or #可 ho2",
        vec![
            text("particle "),
            link("喇"),
            text(" laa3 before the particles "),
            link("喎"),
            text(" wo3, "),
            link("噃"),
            text(" bo3 or "),
            link("可"),
            text(" ho2"),
        ],
    );

    assert_succeed(
        parse_line("eng"),
        "less common than #等等 dang2 dang2",
        vec![
            text("less common than "),
            link("等等"),
            text(" dang2 dang2"),
        ],
    );

    assert_succeed(
        parse_line("eng"),
        "equavalent to #現 (jin6)",
        vec![text("equavalent to "), link("現"), text(" (jin6)")],
    );

    assert_succeed(
        parse_line("即 #未雨綢繆"),
        "即 #未雨綢繆",
        vec![text("即"), link("未雨綢繆")],
    );

    assert_succeed(
        parse_line("yue"),
        "#質素 同 #數量 嘅合稱",
        vec![link("質素"), text("同"), link("數量"), text("嘅合稱")],
    );

    assert_succeed(
        parse_line("eng"),
        "as a variant of #1 with a weaker",
        vec![text("as a variant of "), link("1"), text(" with a weaker")],
    );

    assert_succeed(
        parse_line("yue"),
        "eg. #咁 gam3, #勁 ging6 or #最 zeoi3",
        vec![
            text("eg. "),
            link("咁"),
            text(" gam3, "),
            link("勁"),
            text(" ging6 or "),
            link("最"),
            text(" zeoi3"),
        ],
    );
}

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
fn test_parse_pr_line() {
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

    // Test parentheses detection
    // i.e. check whether the content in the parentheses
    // is actually jyutping
    assert_succeed(
        parse_pr_line("yue"),
        "yue:p分別為聖。(約翰福音 17:19)",
        (simple_line("p分別為聖。(約翰福音 17:19)"), None),
    );

    assert_succeed(
        parse_pr_line("yue"),
        "yue:#質素 同 #數量 嘅合稱 (this is an side note with jyut6 ping3, not a pr)",
        (
            vec![
                link("質素"),
                text("同"),
                link("數量"),
                text("嘅合稱 (this is an side note with jyut6 ping3, not a pr)"),
            ],
            None,
        ),
    );

    assert_succeed(
        parse_pr_line("eng"),
        "eng:particle #喇 laa3 before the particles #喎  wo3, #噃 bo3 or #可 ho2(this is an side note, not a pr)",
        (vec![
            text("particle "),
            link("喇"),
            text(" laa3 before the particles "),
            link("喎"),
            text(" wo3, "),
            link("噃"),
            text(" bo3 or "),
            link("可"),
            text(" ho2(this is an side note, not a pr)"),
        ], None),
    );

    assert_succeed(
        parse_pr_line("yue"),
        "yue:eg. #咁 gam3, #勁 ging6 or #最 zeoi3(this is an side note, not a pr)",
        (
            vec![
                text("eg. "),
                link("咁"),
                text(" gam3, "),
                link("勁"),
                text(" ging6 or "),
                link("最"),
                text(" zeoi3(this is an side note, not a pr)"),
            ],
            None,
        ),
    );
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
                        nucleus: Some(JyutPingNucleus::I),
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
                        nucleus: Some(JyutPingNucleus::I),
                        coda: Some(JyutPingCoda::P),
                        tone: Some(JyutPingTone::T4),
                    },
                )])]),
            },
        ]);
        assert_succeed(
            parse_content(id, variants.clone(), true),
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
                published: true,
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
                    nucleus: Some(JyutPingNucleus::Aa),
                    coda: Some(JyutPingCoda::I),
                    tone: Some(JyutPingTone::T1),
                }),
                LaxJyutPingSegment::Standard(JyutPing {
                    initial: Some(JyutPingInitial::H),
                    nucleus: Some(JyutPingNucleus::Aa),
                    coda: Some(JyutPingCoda::I),
                    tone: Some(JyutPingTone::T1),
                }),
            ])]),
        }]);
        assert_succeed(
                parse_content(id, variants.clone(), true),
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
                    }],
                    published: true,
                })
            );
    }
}

#[test]
fn test_normalize() {
    use super::unicode::normalize;
    assert_eq!(normalize("Ｉ"), "i");
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
    assert!(is_latin('Ｉ'));

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
    assert_eq!(
        tokenize(&vec!["hi-fi".into()], "佢屋企套hi-fi值幾十萬"),
        vec![
            normal_word("佢"),
            normal_word("屋"),
            normal_word("企"),
            normal_word("套"),
            bold_word("hi-fi"),
            normal_word("值"),
            normal_word("幾"),
            normal_word("十"),
            normal_word("萬")
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
    assert_eq!(
        flatten_line(&vec!["hi-fi".into()], &vec![text("佢屋企套hi-fi值幾十萬")]),
        vec![
            text_word(normal_word("佢")),
            text_word(normal_word("屋")),
            text_word(normal_word("企")),
            text_word(normal_word("套")),
            text_word(bold_word("hi-fi")),
            text_word(normal_word("值")),
            text_word(normal_word("幾")),
            text_word(normal_word("十")),
            text_word(normal_word("萬"))
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

    // decimal number "1.5"
    assert_eq!(
        match_ruby(
            &vec!["volt".into()],
            &vec![text("筆芯電嘅電壓係1.5 volt。")],
            &vec![
                "bat1", "sam1", "din6", "ge3", "din6", "aat3", "hai6", "jat1", "dim2", "ng5",
                "wuk1"
            ]
        ),
        vec![
            Word(normal_word("筆"), vec!["bat1".into()]),
            Word(normal_word("芯"), vec!["sam1".into()]),
            Word(normal_word("電"), vec!["din6".into()]),
            Word(normal_word("嘅"), vec!["ge3".into()]),
            Word(normal_word("電"), vec!["din6".into()]),
            Word(normal_word("壓"), vec!["aat3".into()]),
            Word(normal_word("係"), vec!["hai6".into()]),
            Word(
                rich_dict::Word(vec![normal("1.5 "), bold("volt")]),
                vec!["jat1".into(), "dim2".into(), "ng5".into(), "wuk1".into()]
            ),
            Punc("。".into())
        ]
    );

    // typo in jyutping "能" nan4 should be nang4
    assert_eq!(
        match_ruby(
            &vec!["懷春".into()],
            &vec![text("能曾經有懷春嘅日子。")],
            &vec!["nan4", "cang4", "ging1", "jau5", "waai4", "ceon1", "ge3", "jat6", "zi2"]
        ),
        vec![
            Word(normal_word("能"), vec!["nan4".into()]),
            Word(normal_word("曾"), vec!["cang4".into()]),
            Word(normal_word("經"), vec!["ging1".into()]),
            Word(normal_word("有"), vec!["jau5".into()]),
            Word(bold_word("懷"), vec!["waai4".into()]),
            Word(bold_word("春"), vec!["ceon1".into()]),
            Word(normal_word("嘅"), vec!["ge3".into()]),
            Word(normal_word("日"), vec!["jat6".into()]),
            Word(normal_word("子"), vec!["zi2".into()]),
            Punc("。".into())
        ]
    );

    // typo in jyutping "曾" can4 should be cang4
    assert_eq!(
        match_ruby(
            &vec!["懷春".into()],
            &vec![text("能曾經有懷春嘅日子。")],
            &vec!["nang4", "can4", "ging1", "jau5", "waai4", "ceon1", "ge3", "jat6", "zi2"]
        ),
        vec![
            Word(normal_word("能"), vec!["nang4".into()]),
            Word(normal_word("曾"), vec!["can4".into()]),
            Word(normal_word("經"), vec!["ging1".into()]),
            Word(normal_word("有"), vec!["jau5".into()]),
            Word(bold_word("懷"), vec!["waai4".into()]),
            Word(bold_word("春"), vec!["ceon1".into()]),
            Word(normal_word("嘅"), vec!["ge3".into()]),
            Word(normal_word("日"), vec!["jat6".into()]),
            Word(normal_word("子"), vec!["zi2".into()]),
            Punc("。".into())
        ]
    );

    // typo in jyutping "能" nan4 should be nang4, "曾" can4 should be cang4
    assert_eq!(
        match_ruby(
            &vec!["懷春".into()],
            &vec![text("能曾經有懷春嘅日子。")],
            &vec!["nan4", "can4", "ging1", "jau5", "waai4", "ceon1", "ge3", "jat6", "zi2"]
        ),
        vec![
            Word(normal_word("能"), vec!["nan4".into()]),
            Word(normal_word("曾"), vec!["can4".into()]),
            Word(normal_word("經"), vec!["ging1".into()]),
            Word(normal_word("有"), vec!["jau5".into()]),
            Word(bold_word("懷"), vec!["waai4".into()]),
            Word(bold_word("春"), vec!["ceon1".into()]),
            Word(normal_word("嘅"), vec!["ge3".into()]),
            Word(normal_word("日"), vec!["jat6".into()]),
            Word(normal_word("子"), vec!["zi2".into()]),
            Punc("。".into())
        ]
    );

    // normalization of "Ｉ" FULLWIDTH LATIN CAPITAL LETTER I into "i" LATIN SMALL LETTER I
    assert_eq!(
        match_ruby(
            &vec!["II".into()],
            &vec![text("個ＩＩ，")],
            &vec!["go3", "aai1", "aai1"]
        ),
        vec![
            Word(normal_word("個"), vec!["go3".into()]),
            Word(bold_word("ＩＩ"), vec!["aai1".into(), "aai1".into()]),
            Punc("，".into())
        ]
    );

    // Normalization of uppercase "H" into lowercase "h"
    assert_eq!(
        match_ruby(
            &vec!["hello".into(), "哈佬".into()],
            &vec![text("Hello，好耐冇見。")],
            &vec!["haa1", "lou3", "hou2", "noi6", "mou5", "gin3"]
        ),
        vec![
            Word(bold_word("Hello"), vec!["haa1".into(), "lou3".into()]),
            Punc("，".into()),
            Word(normal_word("好"), vec!["hou2".into()]),
            Word(normal_word("耐"), vec!["noi6".into()]),
            Word(normal_word("冇"), vec!["mou5".into()]),
            Word(normal_word("見"), vec!["gin3".into()]),
            Punc("。".into())
        ]
    );

    // "-" HYPHEN-MINUS in the middle of a word
    assert_eq!(
        match_ruby(
            &vec!["hi-fi".into()],
            &vec![text("佢屋企套hi-fi值幾十萬")],
            &vec![
                "keoi5", "uk1", "kei2", "tou3", "haai1", "faai1", "zik6", "gei2", "sap6", "maan6"
            ]
        ),
        vec![
            Word(normal_word("佢"), vec!["keoi5".into()]),
            Word(normal_word("屋"), vec!["uk1".into()]),
            Word(normal_word("企"), vec!["kei2".into()]),
            Word(normal_word("套"), vec!["tou3".into()]),
            Word(bold_word("hi-fi"), vec!["haai1".into(), "faai1".into()]),
            Word(normal_word("值"), vec!["zik6".into()]),
            Word(normal_word("幾"), vec!["gei2".into()]),
            Word(normal_word("十"), vec!["sap6".into()]),
            Word(normal_word("萬"), vec!["maan6".into()])
        ]
    );

    // "FULLWIDTH COMMA" between english words
    // assert_eq!(
    //     match_ruby(
    //         &vec!["M".into()],
    //         &vec![text("好M唔M，M套？")],
    //         &vec![
    //             "hou2", "em1", "m4", "em1", "em1", "tou3"
    //         ]
    //     ),
    //     vec![
    //         Word(normal_word("好"), vec!["hou2".into()]),
    //         Word(bold_word("M"), vec!["em1".into()]),
    //         Word(normal_word("唔"), vec!["m4".into()]),
    //         Word(bold_word("M"), vec!["em1".into()]),
    //         Punc("，".into()),
    //         Word(bold_word("M"), vec!["em1".into()]),
    //         Word(normal_word("套"), vec!["tou3".into()]),
    //         Punc("？".into()),
    //     ]
    // );

    // one Chinese character two prs
    assert_eq!(
        match_ruby(
            &vec!["卅".into()],
            &vec![text("卅幾")],
            &vec!["saa1", "aa6", "gei2"]
        ),
        vec![
            Word(bold_word("卅"), vec!["saa1".into(), "aa6".into()]),
            Word(normal_word("幾"), vec!["gei2".into()]),
        ]
    );

    assert_eq!(
        match_ruby(
            &vec!["卅".into()],
            &vec![link("年卅晚")],
            &vec!["nin4", "saa1", "aa6", "maan5"]
        ),
        vec![LinkedWord(vec![
            (normal_word("年"), vec!["nin4".into()]),
            (bold_word("卅"), vec!["saa1".into(), "aa6".into()]),
            (normal_word("晚"), vec!["maan5".into()])
        ])]
    );
    assert_eq!(
        match_ruby(
            &vec!["卌".into()],
            &vec![text("卌二")],
            &vec!["se3", "aa6", "ji6"]
        ),
        vec![
            Word(bold_word("卌"), vec!["se3".into(), "aa6".into()]),
            Word(normal_word("二"), vec!["ji6".into()]),
        ]
    );

    // Q&A-style without ruby annotations for 甲 and 乙
    assert_eq!(
        match_ruby(
            &vec!["大家噉話"],
            &vec![text("甲：「身體健康！」乙：「大家噉話！」")],
            &vec!["san1", "tai2", "gin6", "hong1", "daai6", "gaa1", "gam2", "waa6"]
        ),
        vec![
            Word(normal_word("甲"), vec![]),
            Punc("：".into()),
            Punc("「".into()),
            Word(normal_word("身"), vec!["san1".into()]),
            Word(normal_word("體"), vec!["tai2".into()]),
            Word(normal_word("健"), vec!["gin6".into()]),
            Word(normal_word("康"), vec!["hong1".into()]),
            Punc("！".into()),
            Punc("」".into()),
            Word(normal_word("乙"), vec![]),
            Punc("：".into()),
            Punc("「".into()),
            Word(bold_word("大"), vec!["daai6".into()],),
            Word(bold_word("家"), vec!["gaa1".into()],),
            Word(bold_word("噉"), vec!["gam2".into()],),
            Word(bold_word("話"), vec!["waa6".into()],),
            Punc("！".into()),
            Punc("」".into()),
        ]
    );

    // Q&A-style with ruby annotations for 甲 and 乙
    assert_eq!(
        match_ruby(
            &vec!["囉"],
            &vec![
                text("甲：「"),
                link("方丈"),
                text("真係好小器呀！」乙：「係囉！」")
            ],
            &vec![
                "gaap3", "fong1", "zoeng6", "zan1", "hai6", "hou2", "siu2", "hei3", "aa3", "jyut6",
                "hai6", "lo1"
            ]
        ),
        vec![
            Word(normal_word("甲"), vec!["gaap3".into()]),
            Punc("：".into()),
            Punc("「".into()),
            LinkedWord(vec![
                (normal_word("方"), vec!["fong1".into()]),
                (normal_word("丈"), vec!["zoeng6".into()])
            ]),
            Word(normal_word("真"), vec!["zan1".into()]),
            Word(normal_word("係"), vec!["hai6".into()]),
            Word(normal_word("好"), vec!["hou2".into()]),
            Word(normal_word("小"), vec!["siu2".into()]),
            Word(normal_word("器"), vec!["hei3".into()]),
            Word(normal_word("呀"), vec!["aa3".into()]),
            Punc("！".into()),
            Punc("」".into()),
            Word(normal_word("乙"), vec!["jyut6".into()]),
            Punc("：".into()),
            Punc("「".into()),
            Word(normal_word("係"), vec!["hai6".into()],),
            Word(bold_word("囉"), vec!["lo1".into()],),
            Punc("！".into()),
            Punc("」".into()),
        ]
    );

    // Q&A-style without ruby annotations for 甲 and 乙
    assert_eq!(
        match_ruby(
            &vec!["囉"],
            &vec![
                text("甲：「"),
                link("方丈"),
                text("真係好小器呀！」乙：「係囉！」")
            ],
            &vec!["fong1", "zoeng6", "zan1", "hai6", "hou2", "siu2", "hei3", "aa3", "hai6", "lo1"]
        ),
        vec![
            Word(normal_word("甲"), vec![]),
            Punc("：".into()),
            Punc("「".into()),
            LinkedWord(vec![
                (normal_word("方"), vec!["fong1".into()]),
                (normal_word("丈"), vec!["zoeng6".into()])
            ]),
            Word(normal_word("真"), vec!["zan1".into()]),
            Word(normal_word("係"), vec!["hai6".into()]),
            Word(normal_word("好"), vec!["hou2".into()]),
            Word(normal_word("小"), vec!["siu2".into()]),
            Word(normal_word("器"), vec!["hei3".into()]),
            Word(normal_word("呀"), vec!["aa3".into()]),
            Punc("！".into()),
            Punc("」".into()),
            Word(normal_word("乙"), vec![]),
            Punc("：".into()),
            Punc("「".into()),
            Word(normal_word("係"), vec!["hai6".into()],),
            Word(bold_word("囉"), vec!["lo1".into()],),
            Punc("！".into()),
            Punc("」".into()),
        ]
    );
}

#[test]
fn test_parse_jyutping() {
    assert_eq!(
        parse_jyutping(&"ging1".to_string()),
        Some(JyutPing {
            initial: Some(JyutPingInitial::G),
            nucleus: Some(JyutPingNucleus::I),
            coda: Some(JyutPingCoda::Ng),
            tone: Some(JyutPingTone::T1)
        })
    );

    assert_eq!(
        parse_jyutping(&"gwok3".to_string()),
        Some(JyutPing {
            initial: Some(JyutPingInitial::Gw),
            nucleus: Some(JyutPingNucleus::O),
            coda: Some(JyutPingCoda::K),
            tone: Some(JyutPingTone::T3)
        })
    );

    assert_eq!(
        parse_jyutping(&"aa".to_string()),
        Some(JyutPing {
            initial: None,
            nucleus: Some(JyutPingNucleus::Aa),
            coda: None,
            tone: None
        })
    );

    assert_eq!(
        parse_jyutping(&"a2".to_string()),
        Some(JyutPing {
            initial: None,
            nucleus: Some(JyutPingNucleus::A),
            coda: None,
            tone: Some(JyutPingTone::T2)
        })
    );

    assert_eq!(
        parse_jyutping(&"a".to_string()),
        Some(JyutPing {
            initial: None,
            nucleus: Some(JyutPingNucleus::A),
            coda: None,
            tone: None,
        })
    );

    assert_eq!(
        parse_jyutping(&"seoi5".to_string()),
        Some(JyutPing {
            initial: Some(JyutPingInitial::S),
            nucleus: Some(JyutPingNucleus::Eo),
            coda: Some(JyutPingCoda::I),
            tone: Some(JyutPingTone::T5)
        })
    );

    assert_eq!(
        parse_jyutping(&"baau".to_string()),
        Some(JyutPing {
            initial: Some(JyutPingInitial::B),
            nucleus: Some(JyutPingNucleus::Aa),
            coda: Some(JyutPingCoda::U),
            tone: None,
        })
    );

    assert_eq!(
        parse_jyutping(&"ng".to_string()),
        Some(JyutPing {
            initial: Some(JyutPingInitial::Ng),
            nucleus: None,
            coda: None,
            tone: None
        })
    );

    assert_eq!(
        parse_jyutping(&"ng4".to_string()),
        Some(JyutPing {
            initial: Some(JyutPingInitial::Ng),
            nucleus: None,
            coda: None,
            tone: Some(JyutPingTone::T4)
        })
    );

    assert_eq!(
        parse_jyutping(&"ng5".to_string()),
        Some(JyutPing {
            initial: Some(JyutPingInitial::Ng),
            nucleus: None,
            coda: None,
            tone: Some(JyutPingTone::T5)
        })
    );

    assert_eq!(
        parse_jyutping(&"m".to_string()),
        Some(JyutPing {
            initial: Some(JyutPingInitial::M),
            nucleus: None,
            coda: None,
            tone: None
        })
    );

    assert_eq!(
        parse_jyutping(&"m4".to_string()),
        Some(JyutPing {
            initial: Some(JyutPingInitial::M),
            nucleus: None,
            coda: None,
            tone: Some(JyutPingTone::T4)
        })
    );

    // english
    assert_eq!(
        parse_jyutping(&"firework".to_string()),
        None
    );
    assert_eq!(
        parse_jyutping(&"good".to_string()),
        None
    );
    assert_eq!(
        parse_jyutping(&"manifest".to_string()),
        None
    );

    // nonstandard jyutping (simple)
    assert_eq!(
        parse_jyutping(&"!".to_string()),
        None
    );

    // nonstandard jyutping (complex)
    assert_eq!(
        parse_jyutping(&"!sdet6".to_string()),
        None
    );
}

#[test]
fn test_parse_pr() {
    assert_eq!(
        parse_pr(&"seoi5 ng".to_string()),
        LaxJyutPing(vec![
            LaxJyutPingSegment::Standard(JyutPing {
                initial: Some(JyutPingInitial::S),
                nucleus: Some(JyutPingNucleus::Eo),
                coda: Some(JyutPingCoda::I),
                tone: Some(JyutPingTone::T5)
            }),
            LaxJyutPingSegment::Standard(JyutPing {
                initial: Some(JyutPingInitial::Ng),
                nucleus: None,
                coda: None,
                tone: None
            })
        ])
    );

    assert_eq!(
        parse_pr(&"! sap6".to_string()),
        LaxJyutPing(vec![
            LaxJyutPingSegment::Nonstandard("!".into()),
            LaxJyutPingSegment::Standard(JyutPing {
                initial: Some(JyutPingInitial::S),
                nucleus: Some(JyutPingNucleus::A),
                coda: Some(JyutPingCoda::P),
                tone: Some(JyutPingTone::T6)
            })
        ]),
    );

    assert_eq!(
        parse_pr(&"sap6 !".to_string()),
        LaxJyutPing(vec![
            LaxJyutPingSegment::Standard(JyutPing {
                initial: Some(JyutPingInitial::S),
                nucleus: Some(JyutPingNucleus::A),
                coda: Some(JyutPingCoda::P),
                tone: Some(JyutPingTone::T6)
            }),
            LaxJyutPingSegment::Nonstandard("!".into()),
        ]),
    );

    assert_eq!(
        parse_pr(&"sap6 !sdet6".to_string()),
        LaxJyutPing(vec![
            LaxJyutPingSegment::Standard(JyutPing {
                initial: Some(JyutPingInitial::S),
                nucleus: Some(JyutPingNucleus::A),
                coda: Some(JyutPingCoda::P),
                tone: Some(JyutPingTone::T6)
            }),
            LaxJyutPingSegment::Nonstandard("!sdet6".into()),
        ]),
    );

    assert_eq!(
        parse_pr(&"!".to_string()),
        LaxJyutPing(vec![LaxJyutPingSegment::Nonstandard("!".into())]),
    );

    assert_eq!(
        parse_pr(&"!sdet6".to_string()),
        LaxJyutPing(vec![LaxJyutPingSegment::Nonstandard("!sdet6".into())]),
    );
}

#[test]
fn test_parse_jyutpings() {
    // all valid jyutpings
    assert_eq!(
        parse_jyutpings(&"jyut6 ping3".to_string()),
        Some(vec![JyutPing {
            initial: Some(JyutPingInitial::J),
            nucleus: Some(JyutPingNucleus::Yu),
            coda: Some(JyutPingCoda::T),
            tone: Some(JyutPingTone::T6)
        }, JyutPing {
            initial: Some(JyutPingInitial::P),
            nucleus: Some(JyutPingNucleus::I),
            coda: Some(JyutPingCoda::Ng),
            tone: Some(JyutPingTone::T3)
        }])
    );

    // a single invalid jyutping
    assert_eq!(
        parse_jyutpings(&"jyut6 pingg".to_string()),
        None
    );
}

#[test]
fn test_parse_continuous_jyutpings() {
    // valid jyutping
    assert_eq!(
        parse_continuous_prs(&"zinaan", Romanization::Jyutping),
        Some(vec![vec![JyutPing {
            initial: Some(JyutPingInitial::Z),
            nucleus: Some(JyutPingNucleus::I),
            coda: None,
            tone: None
        },JyutPing {
            initial: Some(JyutPingInitial::N),
            nucleus: Some(JyutPingNucleus::A),
            coda: None,
            tone: None
        },JyutPing {
            initial: None,
            nucleus: Some(JyutPingNucleus::A),
            coda: Some(JyutPingCoda::N),
            tone: None
        }],
        vec![JyutPing {
            initial: Some(JyutPingInitial::Z),
            nucleus: Some(JyutPingNucleus::I),
            coda: None,
            tone: None
        },JyutPing {
            initial: Some(JyutPingInitial::N),
            nucleus: Some(JyutPingNucleus::Aa),
            coda: Some(JyutPingCoda::N),
            tone: None
        }],
        vec![JyutPing {
            initial: Some(JyutPingInitial::Z),
            nucleus: Some(JyutPingNucleus::I),
            coda: Some(JyutPingCoda::N),
            tone: None
        },JyutPing {
            initial: None,
            nucleus: Some(JyutPingNucleus::A),
            coda: None,
            tone: None
        },JyutPing {
            initial: None,
            nucleus: Some(JyutPingNucleus::A),
            coda: Some(JyutPingCoda::N),
            tone: None
        }],
        vec![JyutPing {
            initial: Some(JyutPingInitial::Z),
            nucleus: Some(JyutPingNucleus::I),
            coda: Some(JyutPingCoda::N),
            tone: None
        },JyutPing {
            initial: None,
            nucleus: Some(JyutPingNucleus::Aa),
            coda: Some(JyutPingCoda::N),
            tone: None
        }]
        ])
    );

    assert_eq!(
        parse_continuous_prs(&"zi5naan", Romanization::Jyutping),
        Some(vec![vec![JyutPing {
            initial: Some(JyutPingInitial::Z),
            nucleus: Some(JyutPingNucleus::I),
            coda: None,
            tone: Some(JyutPingTone::T5)
        },JyutPing {
            initial: Some(JyutPingInitial::N),
            nucleus: Some(JyutPingNucleus::A),
            coda: None,
            tone: None
        },JyutPing {
            initial: None,
            nucleus: Some(JyutPingNucleus::A),
            coda: Some(JyutPingCoda::N),
            tone: None
        }],
        vec![JyutPing {
            initial: Some(JyutPingInitial::Z),
            nucleus: Some(JyutPingNucleus::I),
            coda: None,
            tone: Some(JyutPingTone::T5)
        },JyutPing {
            initial: Some(JyutPingInitial::N),
            nucleus: Some(JyutPingNucleus::Aa),
            coda: Some(JyutPingCoda::N),
            tone: None
        }]])
    );

    // invalid jyutping
    assert_eq!(
        parse_continuous_prs(&"zinagn", Romanization::Jyutping),
        None
    );

    // valid yale
    assert_eq!(
        parse_continuous_prs(&"yauji", Romanization::YaleNumbers),
        Some(vec![vec![JyutPing {
            initial: Some(JyutPingInitial::J),
            nucleus: Some(JyutPingNucleus::A),
            coda: Some(JyutPingCoda::U),
            tone: None
        },
        JyutPing {
            initial: Some(JyutPingInitial::Z),
            nucleus: Some(JyutPingNucleus::I),
            coda: None,
            tone: None
        }]])
    );

    assert_eq!(
        parse_continuous_prs(&"yau4", Romanization::YaleNumbers),
        Some(vec![vec![JyutPing {
            initial: Some(JyutPingInitial::J),
            nucleus: Some(JyutPingNucleus::A),
            coda: Some(JyutPingCoda::U),
            tone: Some(JyutPingTone::T4)
        }]])
    );
}

#[test]
fn test_looks_like_pr() {
    use super::jyutping::Romanization::*;

    assert!(
        looks_like_pr("ceon1 min4 bat1 gok3 hiu2 je6 loi4 fung1 jyu5 sing1", Jyutping)
    );
    assert!(
        looks_like_pr("ceon min bat1 gok3 hiu2 je6 loi4 fung1 jyu5 sing1", Jyutping)
    );

    assert!(
        looks_like_pr("cheun1 min4 bat1 gok3 hiu2 ye6 loi4 fung1 yu5 sing1", YaleNumbers)
    );
    assert!(
        looks_like_pr("cheun min bat1 gok3 hiu2 ye6 loi4 fung1 yu5 sing1", YaleNumbers)
    );

    assert!(
        looks_like_pr("tsoen1 min4 bat7 gok8 hiu2 je6 loi4 fung1 jy5 sing1", CantonesePinyin)
    );
    assert!(
        looks_like_pr("tsoen min bat7 gok8 hiu2 je6 loi4 fung1 jy5 sing1", CantonesePinyin)
    );

    assert!(
        looks_like_pr("chun1 min4 bat1 gok3 hiu2 ye6 loi4 fung1 yue5 sing1", SidneyLau)
    );
    assert!(
        looks_like_pr("chun min bat1 gok3 hiu2 ye6 loi4 fung1 yue5 sing1", SidneyLau)
    );
}

#[test]
fn test_convert_to_jyutpings() {
    use super::jyutping::Romanization::*;

    // Make sure the first data line of the TSV is not discarded
    assert_eq!(
        convert_to_jyutpings(&"a1", YaleNumbers),
        parse_jyutpings("aa1").map(|jyutpings| vec![jyutpings])
    );

    let expected_jyutpings1 = parse_jyutpings("ceon1 min4 bat1 gok3 hiu2").map(|jyutpings| vec![jyutpings]);
    let expected_jyutpings2 = parse_jyutpings("je6 loi4 fung1 jyu5 sing1").map(|jyutpings| vec![jyutpings]);
    
    assert_eq!(
        convert_to_jyutpings(&"cheun1 min4 bat1 gok3 hiu2", YaleNumbers),
        expected_jyutpings1
    );
    assert_eq!(
        convert_to_jyutpings(&"ye6 loi4 fung1 yu5 sing1", YaleNumbers),
        expected_jyutpings2
    );

    // assert_eq!(
    //     convert_to_jyutpings(&"cheūn mìhn bāt gok hiú", YaleDiacritics),
    //     expected_jyutpings1
    // );
    // assert_eq!(
    //     convert_to_jyutpings(&"yeh lòih fūng yúh sīng", YaleDiacritics),
    //     expected_jyutpings2
    // );

    assert_eq!(
        convert_to_jyutpings(&"tsoen1 min4 bat7 gok8 hiu2", CantonesePinyin),
        expected_jyutpings1
    );
    assert_eq!(
        convert_to_jyutpings(&"je6 loi4 fung1 jy5 sing1", CantonesePinyin),
        expected_jyutpings2
    );
    
    // assert_eq!(
    //     convert_to_jyutpings(&"cên1 min4 bed1 gog3 hiu2", Guangdong),
    //     expected_jyutpings1
    // );
    // assert_eq!(
    //     convert_to_jyutpings(&"yé6 loi4 fung1 yu5 xing1", Guangdong),
    //     expected_jyutpings2
    // );

    assert_eq!(
        convert_to_jyutpings(&"chun1 min4 bat1 gok3 hiu2", SidneyLau),
        expected_jyutpings1
    );
    assert_eq!(
        convert_to_jyutpings(&"ye6 loi4 fung1 yue5 sing1", SidneyLau),
        expected_jyutpings2
    );

    // assert_eq!(
    //     convert_to_jyutpings(&"tsʰɵn˥ miːn˨˩ pɐt˥ kɔːk˧ hiːu˧˥", Ipa),
    //     expected_jyutpings1
    // );
    // assert_eq!(
    //     convert_to_jyutpings(&"jɛː˨ lɔːi˨˩ fʊŋ˥ jyː˩˧ sɪŋ˥", Ipa),
    //     expected_jyutpings2
    // );

    let zi5naan4 = Some(vec![vec![JyutPing {
            initial: Some(JyutPingInitial::Z),
            nucleus: Some(JyutPingNucleus::I),
            coda: None,
            tone: Some(JyutPingTone::T5)
        },
        JyutPing {
            initial: Some(JyutPingInitial::N),
            nucleus: Some(JyutPingNucleus::Aa),
            coda: Some(JyutPingCoda::N),
            tone: Some(JyutPingTone::T4)
        }]
        ]);

    // continous jyutpings with tones
    assert_eq!(
        convert_to_jyutpings(&"zi5naan4", Jyutping),
        zi5naan4
    );

    // continous yale with tones
    assert_eq!(
        convert_to_jyutpings(&"ji5naan4", YaleNumbers),
        zi5naan4
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
                nucleus: Some(JyutPingNucleus::Eo),
                coda: Some(JyutPingCoda::I),
                tone: Some(JyutPingTone::T5)
            },
            &JyutPing {
                initial: Some(JyutPingInitial::S),
                nucleus: Some(JyutPingNucleus::Eo),
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
                nucleus: Some(JyutPingNucleus::Aa),
                coda: None,
                tone: None
            },
            &JyutPing {
                initial: None,
                nucleus: Some(JyutPingNucleus::Aa),
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
                nucleus: Some(JyutPingNucleus::Eo),
                coda: Some(JyutPingCoda::I),
                tone: Some(JyutPingTone::T5)
            },
            &JyutPing {
                initial: Some(JyutPingInitial::S),
                nucleus: Some(JyutPingNucleus::Eo),
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
                nucleus: Some(JyutPingNucleus::Eo),
                coda: Some(JyutPingCoda::I),
                tone: Some(JyutPingTone::T5)
            },
            &JyutPing {
                initial: Some(JyutPingInitial::S),
                nucleus: Some(JyutPingNucleus::Eo),
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
                nucleus: Some(JyutPingNucleus::Eo),
                coda: Some(JyutPingCoda::I),
                tone: Some(JyutPingTone::T5)
            },
            &JyutPing {
                initial: None,
                nucleus: Some(JyutPingNucleus::Eo),
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
                nucleus: Some(JyutPingNucleus::Eo),
                coda: Some(JyutPingCoda::I),
                tone: Some(JyutPingTone::T5)
            },
            &JyutPing {
                initial: Some(JyutPingInitial::S),
                nucleus: Some(JyutPingNucleus::Oe),
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
                nucleus: Some(JyutPingNucleus::I),
                coda: Some(JyutPingCoda::I),
                tone: Some(JyutPingTone::T5)
            },
            &JyutPing {
                initial: Some(JyutPingInitial::S),
                nucleus: Some(JyutPingNucleus::Yu),
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
                nucleus: Some(JyutPingNucleus::I),
                coda: Some(JyutPingCoda::I),
                tone: Some(JyutPingTone::T5)
            },
            &JyutPing {
                initial: Some(JyutPingInitial::T),
                nucleus: Some(JyutPingNucleus::E),
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
                nucleus: Some(JyutPingNucleus::Yu),
                coda: Some(JyutPingCoda::I),
                tone: Some(JyutPingTone::T5)
            },
            &JyutPing {
                initial: Some(JyutPingInitial::T),
                nucleus: Some(JyutPingNucleus::U),
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
                nucleus: Some(JyutPingNucleus::Eo),
                coda: Some(JyutPingCoda::P),
                tone: Some(JyutPingTone::T5)
            },
            &JyutPing {
                initial: Some(JyutPingInitial::S),
                nucleus: Some(JyutPingNucleus::Eo),
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
                nucleus: Some(JyutPingNucleus::Eo),
                coda: Some(JyutPingCoda::P),
                tone: Some(JyutPingTone::T5)
            },
            &JyutPing {
                initial: Some(JyutPingInitial::S),
                nucleus: Some(JyutPingNucleus::Eo),
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
                nucleus: Some(JyutPingNucleus::Eo),
                coda: Some(JyutPingCoda::I),
                tone: Some(JyutPingTone::T5)
            },
            &JyutPing {
                initial: Some(JyutPingInitial::T),
                nucleus: Some(JyutPingNucleus::Eo),
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
                nucleus: Some(JyutPingNucleus::Eo),
                coda: Some(JyutPingCoda::T),
                tone: Some(JyutPingTone::T4)
            },
            &JyutPing {
                initial: Some(JyutPingInitial::S),
                nucleus: Some(JyutPingNucleus::Eo),
                coda: Some(JyutPingCoda::T),
                tone: Some(JyutPingTone::T5)
            }
        )
    );

    assert_eq!(
        96,
        compare_jyutping(
            &JyutPing {
                initial: Some(JyutPingInitial::Ng),
                nucleus: None,
                coda: None,
                tone: Some(JyutPingTone::T5)
            },
            &JyutPing {
                initial: Some(JyutPingInitial::Ng),
                nucleus: None,
                coda: None,
                tone: None
            },
        )
    );

    assert_eq!(
        96,
        compare_jyutping(
            &JyutPing {
                initial: Some(JyutPingInitial::Ng),
                nucleus: None,
                coda: None,
                tone: None
            },
            &JyutPing {
                initial: Some(JyutPingInitial::Ng),
                nucleus: None,
                coda: None,
                tone: Some(JyutPingTone::T5)
            },
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
                nucleus: Some(JyutPingNucleus::Eo),
                coda: Some(JyutPingCoda::P),
                tone: Some(JyutPingTone::T5)
            },
            &JyutPing {
                initial: Some(JyutPingInitial::T),
                nucleus: Some(JyutPingNucleus::Eo),
                coda: Some(JyutPingCoda::I),
                tone: Some(JyutPingTone::T5)
            }
        )
    );
}

#[test]
fn test_to_lean_rich_dict() {
    use super::lean_rich_dict::{to_lean_rich_entry, LeanDef, LeanRichEntry, LeanVariant};
    {
        let id = 103022;
        let published = true;
        let variants = Variants(vec![
            Variant {
                word: "zip".to_string(),
                prs: LaxJyutPings(vec![
                    LaxJyutPing(vec![LaxJyutPingSegment::Standard(JyutPing {
                        initial: Some(JyutPingInitial::Z),
                        nucleus: Some(JyutPingNucleus::I),
                        coda: Some(JyutPingCoda::P),
                        tone: Some(JyutPingTone::T4),
                    })]),
                    LaxJyutPing(vec![LaxJyutPingSegment::Nonstandard("!".into())]),
                ]),
            },
            Variant {
                word: "jip".to_string(),
                prs: LaxJyutPings(vec![LaxJyutPing(vec![
                    LaxJyutPingSegment::Nonstandard("!".into()),
                    LaxJyutPingSegment::Standard(JyutPing {
                        initial: Some(JyutPingInitial::Z),
                        nucleus: Some(JyutPingNucleus::I),
                        coda: Some(JyutPingCoda::P),
                        tone: Some(JyutPingTone::T4),
                    }),
                ])]),
            },
        ]);
        let variants_simp = vec!["zip".to_string(), "jip".to_string()];
        let lean_variants = vec![
            LeanVariant {
                word: "zip".into(),
                prs: "zip4, !".into(),
            },
            LeanVariant {
                word: "jip".into(),
                prs: "! zip4".into(),
            },
        ];
        let lean_variants_simp = vec![
            LeanVariant {
                word: "zip".into(),
                prs: "zip4, !".into(),
            },
            LeanVariant {
                word: "jip".into(),
                prs: "! zip4".into(),
            },
        ];
        let entry = rich_dict::RichEntry {
            id,
            variants,
            variants_simp,
            poses: vec!["動詞".to_string(), "擬聲詞".to_string()],
            labels: vec![],
            sims: vec![],
            sims_simp: vec![],
            ants: vec![],
            ants_simp: vec![],
            refs: vec![],
            imgs: vec![],
            defs: vec![rich_dict::RichDef {
                yue: simple_clause("表現不屑而發出嘅聲音"),
                yue_simp: simple_clause("表现不屑而发出嘅声音"),
                eng: Some(simple_clause("tsk")),
                alts: vec![],
                egs: vec![],
            }],
            published,
        };

        let lean_entry = LeanRichEntry {
            id,
            variants: lean_variants,
            variants_simp: lean_variants_simp,
            poses: vec!["動詞".to_string(), "擬聲詞".to_string()],
            labels: vec![],
            sims: vec![],
            sims_simp: vec![],
            ants: vec![],
            ants_simp: vec![],
            defs: vec![LeanDef {
                yue: simple_clause("表現不屑而發出嘅聲音"),
                yue_simp: simple_clause("表现不屑而发出嘅声音"),
                eng: Some(simple_clause("tsk")),
                alts: vec![],
                egs: vec![],
            }],
            published,
        };
        assert_eq!(lean_entry, to_lean_rich_entry(&entry));
    }
}

#[test]
fn test_get_simplified_rich_line() {
    use rich_dict::{get_simplified_rich_line, RichLine, RubySegment, TextStyle, Word};
    // WordLine
    // empty edge case
    {
        let trad_rich_line = RichLine::Text(vec![]);
        let simp_rich_line = RichLine::Text(vec![]);
        let simp_line = "".to_string();
        assert_eq!(
            simp_rich_line,
            get_simplified_rich_line(&simp_line, &trad_rich_line)
        );
    }

    // single Chinese character
    {
        let trad_rich_line = RichLine::Text(vec![(
            SegmentType::Text,
            Word(vec![(TextStyle::Bold, "國".into())]),
        )]);
        let simp_rich_line = RichLine::Text(vec![(
            SegmentType::Text,
            Word(vec![(TextStyle::Bold, "国".into())]),
        )]);
        let simp_line = "国".to_string();
        assert_eq!(
            simp_rich_line,
            get_simplified_rich_line(&simp_line, &trad_rich_line)
        );
    }

    // a link with a single Chinese character
    {
        let trad_rich_line = RichLine::Text(vec![(
            SegmentType::Link,
            Word(vec![(TextStyle::Bold, "國".into())]),
        )]);
        let simp_rich_line = RichLine::Text(vec![(
            SegmentType::Link,
            Word(vec![(TextStyle::Bold, "国".into())]),
        )]);
        let simp_line = "国".to_string();
        assert_eq!(
            simp_rich_line,
            get_simplified_rich_line(&simp_line, &trad_rich_line)
        );
    }

    // a link with two Chinese characters
    {
        let trad_rich_line = RichLine::Text(vec![(
            SegmentType::Link,
            Word(vec![
                (TextStyle::Bold, "國".into()),
                (TextStyle::Bold, "家".into()),
            ]),
        )]);
        let simp_rich_line = RichLine::Text(vec![(
            SegmentType::Link,
            Word(vec![
                (TextStyle::Bold, "国".into()),
                (TextStyle::Bold, "家".into()),
            ]),
        )]);
        let simp_line = "国家".to_string();
        assert_eq!(
            simp_rich_line,
            get_simplified_rich_line(&simp_line, &trad_rich_line)
        );
    }

    // RubyLine
    // empty edge case
    {
        let trad_rich_line = RichLine::Ruby(vec![]);
        let simp_rich_line = RichLine::Ruby(vec![]);
        let simp_line = "".to_string();
        assert_eq!(
            simp_rich_line,
            get_simplified_rich_line(&simp_line, &trad_rich_line)
        );
    }

    // single Chinese character
    {
        let trad_rich_line = RichLine::Ruby(vec![RubySegment::Word(
            Word(vec![(TextStyle::Bold, "國".into())]),
            vec!["gwok3".into()],
        )]);
        let simp_rich_line = RichLine::Ruby(vec![RubySegment::Word(
            Word(vec![(TextStyle::Bold, "国".into())]),
            vec!["gwok3".into()],
        )]);
        let simp_line = "国".to_string();
        assert_eq!(
            simp_rich_line,
            get_simplified_rich_line(&simp_line, &trad_rich_line)
        );
    }

    // single phrase of Chinese characters
    {
        let trad_rich_line = RichLine::Ruby(vec![
            RubySegment::Word(
                Word(vec![(TextStyle::Bold, "乙".into())]),
                vec!["jyut6".into()],
            ),
            RubySegment::Word(
                Word(vec![(TextStyle::Bold, "等".into())]),
                vec!["dang2".into()],
            ),
            RubySegment::Punc("/".into()),
            RubySegment::Word(
                Word(vec![(TextStyle::Bold, "乙".into())]),
                vec!["jyut6".into()],
            ),
            RubySegment::Word(
                Word(vec![(TextStyle::Bold, "級".into())]),
                vec!["kap1".into()],
            ),
        ]);
        let simp_rich_line = RichLine::Ruby(vec![
            RubySegment::Word(
                Word(vec![(TextStyle::Bold, "乙".into())]),
                vec!["jyut6".into()],
            ),
            RubySegment::Word(
                Word(vec![(TextStyle::Bold, "等".into())]),
                vec!["dang2".into()],
            ),
            RubySegment::Punc("/".into()),
            RubySegment::Word(
                Word(vec![(TextStyle::Bold, "乙".into())]),
                vec!["jyut6".into()],
            ),
            RubySegment::Word(
                Word(vec![(TextStyle::Bold, "级".into())]),
                vec!["kap1".into()],
            ),
        ]);
        let simp_line = "乙等 / 乙级".to_string();
        assert_eq!(
            simp_rich_line,
            get_simplified_rich_line(&simp_line, &trad_rich_line)
        );
    }

    // single sentence of Chinese characters
    {
        let trad_rich_line = RichLine::Ruby(vec![
            RubySegment::Punc("「".into()),
            RubySegment::Word(
                Word(vec![(TextStyle::Bold, "國".into())]),
                vec!["gwok3".into()],
            ),
            RubySegment::Word(
                Word(vec![(TextStyle::Bold, "家".into())]),
                vec!["gaa1".into()],
            ),
            RubySegment::Punc("，".into()),
            RubySegment::Word(
                Word(vec![(TextStyle::Normal, "富".into())]),
                vec!["fu3".into()],
            ),
            RubySegment::Word(
                Word(vec![(TextStyle::Normal, "強".into())]),
                vec!["koeng4".into()],
            ),
            RubySegment::Punc("。".into()),
            RubySegment::Punc("」".into()),
        ]);
        let simp_rich_line = RichLine::Ruby(vec![
            RubySegment::Punc("「".into()),
            RubySegment::Word(
                Word(vec![(TextStyle::Bold, "国".into())]),
                vec!["gwok3".into()],
            ),
            RubySegment::Word(
                Word(vec![(TextStyle::Bold, "家".into())]),
                vec!["gaa1".into()],
            ),
            RubySegment::Punc("，".into()),
            RubySegment::Word(
                Word(vec![(TextStyle::Normal, "富".into())]),
                vec!["fu3".into()],
            ),
            RubySegment::Word(
                Word(vec![(TextStyle::Normal, "强".into())]),
                vec!["koeng4".into()],
            ),
            RubySegment::Punc("。".into()),
            RubySegment::Punc("」".into()),
        ]);
        let simp_line = "「国家，富强。」".to_string();
        assert_eq!(
            simp_rich_line,
            get_simplified_rich_line(&simp_line, &trad_rich_line)
        );
    }

    // simple sentence with latin characters
    {
        let trad_rich_line = RichLine::Ruby(vec![RubySegment::Word(
            Word(vec![(TextStyle::Bold, "keep fit".into())]),
            vec!["kip1".into(), "fit1".into()],
        )]);
        let simp_rich_line = trad_rich_line.clone();
        let simp_line = "keep fit".to_string();
        assert_eq!(
            simp_rich_line,
            get_simplified_rich_line(&simp_line, &trad_rich_line)
        );
    }

    {
        let trad_rich_line = RichLine::Ruby(vec![RubySegment::Word(
            Word(vec![(TextStyle::Bold, "keep".into()), (TextStyle::Normal, " fit".into())]),
            vec!["kip1".into(), "fit1".into()],
        )]);
        let simp_rich_line = trad_rich_line.clone();
        let simp_line = "keep fit".to_string();
        assert_eq!(
            simp_rich_line,
            get_simplified_rich_line(&simp_line, &trad_rich_line)
        );
    }

    // single sentence with a mixture of Chinese and latin characters
    {
        let trad_rich_line = RichLine::Ruby(vec![
            RubySegment::Punc("「".into()),
            RubySegment::Word(
                Word(vec![(TextStyle::Bold, "國".into())]),
                vec!["gwok3".into()],
            ),
            RubySegment::Word(
                Word(vec![(TextStyle::Bold, "家".into())]),
                vec!["gaa1".into()],
            ),
            RubySegment::Word(
                Word(vec![(TextStyle::Normal, "happy".into())]),
                vec!["hep1".into(), "pi2".into()],
            ),
            RubySegment::Punc("。".into()),
            RubySegment::Punc("」".into()),
        ]);
        let simp_rich_line = RichLine::Ruby(vec![
            RubySegment::Punc("「".into()),
            RubySegment::Word(
                Word(vec![(TextStyle::Bold, "国".into())]),
                vec!["gwok3".into()],
            ),
            RubySegment::Word(
                Word(vec![(TextStyle::Bold, "家".into())]),
                vec!["gaa1".into()],
            ),
            RubySegment::Word(
                Word(vec![(TextStyle::Normal, "happy".into())]),
                vec!["hep1".into(), "pi2".into()],
            ),
            RubySegment::Punc("。".into()),
            RubySegment::Punc("」".into()),
        ]);
        let simp_line = "「国家happy。」".to_string();
        assert_eq!(
            simp_rich_line,
            get_simplified_rich_line(&simp_line, &trad_rich_line)
        );
    }
    // single link
    {
        let trad_rich_line = RichLine::Ruby(vec![RubySegment::LinkedWord(vec![(
            Word(vec![(TextStyle::Bold, "國".into())]),
            vec!["gwok3".into()],
        )])]);
        let simp_rich_line = RichLine::Ruby(vec![RubySegment::LinkedWord(vec![(
            Word(vec![(TextStyle::Bold, "国".into())]),
            vec!["gwok3".into()],
        )])]);
        let simp_line = "国".to_string();
        assert_eq!(
            simp_rich_line,
            get_simplified_rich_line(&simp_line, &trad_rich_line)
        );
    }
    // link with two Chinese characters
    {
        let trad_rich_line = RichLine::Ruby(vec![RubySegment::LinkedWord(vec![
            (
                Word(vec![(TextStyle::Bold, "國".into())]),
                vec!["gwok3".into()],
            ),
            (
                Word(vec![(TextStyle::Bold, "家".into())]),
                vec!["gaa1".into()],
            ),
        ])]);
        let simp_rich_line = RichLine::Ruby(vec![RubySegment::LinkedWord(vec![
            (
                Word(vec![(TextStyle::Bold, "国".into())]),
                vec!["gwok3".into()],
            ),
            (
                Word(vec![(TextStyle::Bold, "家".into())]),
                vec!["gaa1".into()],
            ),
        ])]);
        let simp_line = "国家".to_string();
        assert_eq!(
            simp_rich_line,
            get_simplified_rich_line(&simp_line, &trad_rich_line)
        );
    }
    // link with latin characters
    {
        let trad_rich_line = RichLine::Ruby(vec![RubySegment::LinkedWord(vec![(
            Word(vec![(TextStyle::Bold, "keep fit".into())]),
            vec!["kip1".into(), "fit1".into()],
        )])]);
        let simp_rich_line = trad_rich_line.clone();
        let simp_line = "keep fit".to_string();
        assert_eq!(
            simp_rich_line,
            get_simplified_rich_line(&simp_line, &trad_rich_line)
        );
    }
    // link with mixed script
    {
        let trad_rich_line = RichLine::Ruby(vec![RubySegment::LinkedWord(vec![
            (
                Word(vec![(TextStyle::Bold, "keep".into())]),
                vec!["kip1".into()],
            ),
            (
                Word(vec![(TextStyle::Normal, "數".into())]),
                vec!["sou3".into()],
            ),
        ])]);
        let simp_rich_line = RichLine::Ruby(vec![RubySegment::LinkedWord(vec![
            (
                Word(vec![(TextStyle::Bold, "keep".into())]),
                vec!["kip1".into()],
            ),
            (
                Word(vec![(TextStyle::Normal, "数".into())]),
                vec!["sou3".into()],
            ),
        ])]);
        let simp_line = "keep数".to_string();
        assert_eq!(
            simp_rich_line,
            get_simplified_rich_line(&simp_line, &trad_rich_line)
        );
    }
    // embedded link with two Chinese characters
    {
        let trad_rich_line = RichLine::Ruby(vec![
            RubySegment::Punc("「".into()),
            RubySegment::Word(
                Word(vec![(TextStyle::Bold, "佢".into())]),
                vec!["keoi5".into()],
            ),
            RubySegment::Word(
                Word(vec![(TextStyle::Bold, "話".into())]),
                vec!["waa6".into()],
            ),
            RubySegment::LinkedWord(vec![
                (
                    Word(vec![(TextStyle::Bold, "國".into())]),
                    vec!["gwok3".into()],
                ),
                (
                    Word(vec![(TextStyle::Bold, "家".into())]),
                    vec!["gaa1".into()],
                ),
            ]),
            RubySegment::Word(
                Word(vec![(TextStyle::Normal, "富".into())]),
                vec!["fu3".into()],
            ),
            RubySegment::Word(
                Word(vec![(TextStyle::Normal, "強".into())]),
                vec!["koeng4".into()],
            ),
            RubySegment::Punc("。".into()),
            RubySegment::Punc("」".into()),
        ]);
        let simp_rich_line = RichLine::Ruby(vec![
            RubySegment::Punc("「".into()),
            RubySegment::Word(
                Word(vec![(TextStyle::Bold, "佢".into())]),
                vec!["keoi5".into()],
            ),
            RubySegment::Word(
                Word(vec![(TextStyle::Bold, "话".into())]),
                vec!["waa6".into()],
            ),
            RubySegment::LinkedWord(vec![
                (
                    Word(vec![(TextStyle::Bold, "国".into())]),
                    vec!["gwok3".into()],
                ),
                (
                    Word(vec![(TextStyle::Bold, "家".into())]),
                    vec!["gaa1".into()],
                ),
            ]),
            RubySegment::Word(
                Word(vec![(TextStyle::Normal, "富".into())]),
                vec!["fu3".into()],
            ),
            RubySegment::Word(
                Word(vec![(TextStyle::Normal, "强".into())]),
                vec!["koeng4".into()],
            ),
            RubySegment::Punc("。".into()),
            RubySegment::Punc("」".into()),
        ]);
        let simp_line = "「佢话国家富强。」".to_string();
        assert_eq!(
            simp_rich_line,
            get_simplified_rich_line(&simp_line, &trad_rich_line)
        );
    }
}

#[test]
fn test_get_simplified_variants() {
    use rich_dict::get_simplified_variants;
    {
        let trad_variants = Variants(vec![
            Variant {
                word: "這位是乾隆皇帝的乾兒子".to_string(),
                prs: LaxJyutPings(vec![LaxJyutPing(vec![LaxJyutPingSegment::Standard(
                    JyutPing {
                        initial: Some(JyutPingInitial::Z),
                        nucleus: Some(JyutPingNucleus::E),
                        coda: None,
                        tone: Some(JyutPingTone::T3),
                    },
                )])]),
            },
            Variant {
                word: "呢位係乾隆皇帝嘅養子".to_string(),
                prs: LaxJyutPings(vec![LaxJyutPing(vec![LaxJyutPingSegment::Standard(
                    JyutPing {
                        initial: Some(JyutPingInitial::N),
                        nucleus: Some(JyutPingNucleus::I),
                        coda: None,
                        tone: Some(JyutPingTone::T1),
                    },
                )])]),
            },
        ]);
        let simp_variants = Variants(vec![
            Variant {
                word: "这位是乾隆皇帝的干儿子".to_string(),
                prs: LaxJyutPings(vec![LaxJyutPing(vec![LaxJyutPingSegment::Standard(
                    JyutPing {
                        initial: Some(JyutPingInitial::Z),
                        nucleus: Some(JyutPingNucleus::E),
                        coda: None,
                        tone: Some(JyutPingTone::T3),
                    },
                )])]),
            },
            Variant {
                word: "呢位係乾隆皇帝嘅养子".to_string(),
                prs: LaxJyutPings(vec![LaxJyutPing(vec![LaxJyutPingSegment::Standard(
                    JyutPing {
                        initial: Some(JyutPingInitial::N),
                        nucleus: Some(JyutPingNucleus::I),
                        coda: None,
                        tone: Some(JyutPingTone::T1),
                    },
                )])]),
            },
        ]);
        let simp_variant_strings = vec![
            "这位是乾隆皇帝的干儿子".into(),
            "呢位係乾隆皇帝嘅养子".into(),
        ];
        assert_eq!(
            get_simplified_variants(&trad_variants, &simp_variant_strings),
            simp_variants
        );
    }
}

#[test]
fn test_create_combo_variants() {
    use super::search::{ComboVariant, create_combo_variants};
    {
        let trad_variants = Variants(vec![
            Variant {
                word: "這位是乾隆皇帝的乾兒子".to_string(),
                prs: LaxJyutPings(vec![LaxJyutPing(vec![LaxJyutPingSegment::Standard(
                    JyutPing {
                        initial: Some(JyutPingInitial::Z),
                        nucleus: Some(JyutPingNucleus::E),
                        coda: None,
                        tone: Some(JyutPingTone::T3),
                    },
                )])]),
            },
            Variant {
                word: "呢位係乾隆皇帝嘅養子".to_string(),
                prs: LaxJyutPings(vec![LaxJyutPing(vec![LaxJyutPingSegment::Standard(
                    JyutPing {
                        initial: Some(JyutPingInitial::N),
                        nucleus: Some(JyutPingNucleus::I),
                        coda: None,
                        tone: Some(JyutPingTone::T1),
                    },
                )])]),
            },
        ]);
        let combo_variants = vec![
            ComboVariant {
                word_trad: "這位是乾隆皇帝的乾兒子".to_string(),
                word_simp: "这位是乾隆皇帝的干儿子".to_string(),
                prs: LaxJyutPings(vec![LaxJyutPing(vec![LaxJyutPingSegment::Standard(
                    JyutPing {
                        initial: Some(JyutPingInitial::Z),
                        nucleus: Some(JyutPingNucleus::E),
                        coda: None,
                        tone: Some(JyutPingTone::T3),
                    },
                )])]),
            },
            ComboVariant {
                word_trad: "呢位係乾隆皇帝嘅養子".to_string(),
                word_simp: "呢位係乾隆皇帝嘅养子".to_string(),
                prs: LaxJyutPings(vec![LaxJyutPing(vec![LaxJyutPingSegment::Standard(
                    JyutPing {
                        initial: Some(JyutPingInitial::N),
                        nucleus: Some(JyutPingNucleus::I),
                        coda: None,
                        tone: Some(JyutPingTone::T1),
                    },
                )])]),
            },
        ];
        let simp_variant_strings = vec![
            "这位是乾隆皇帝的干儿子".into(),
            "呢位係乾隆皇帝嘅养子".into(),
        ];
        assert_eq!(
            create_combo_variants(&trad_variants, &simp_variant_strings),
            combo_variants
        );
    }
}

#[test]
fn test_replace_contents_in_word() {
    use super::rich_dict::{replace_contents_in_word, TextStyle::*, Word};
    // empty edge case
    {
        let target_word = Word(vec![]);
        let mut content_word = "".chars().peekable();
        let expected_word = Word(vec![]);
        assert_eq!(
            replace_contents_in_word(&target_word, &mut content_word),
            expected_word
        );
    }

    // a single Chinese character
    {
        let target_word = Word(vec![(Bold, "國".into())]);
        let mut content_word = "国".chars().peekable();
        let expected_word = Word(vec![(Bold, "国".into())]);
        assert_eq!(
            replace_contents_in_word(&target_word, &mut content_word),
            expected_word
        );
    }

    // some latin characters
    {
        let target_word = Word(vec![(Bold, "camp".into())]);
        let mut content_word = "camp".chars().peekable();
        let expected_word = Word(vec![(Bold, "camp".into())]);
        assert_eq!(
            replace_contents_in_word(&target_word, &mut content_word),
            expected_word
        );
    }

    // mixed script
    {
        let target_word = Word(vec![(Bold, "國".into()), (Bold, "camp".into())]);
        let mut content_word = "国camp".chars().peekable();
        let expected_word = Word(vec![(Bold, "国".into()), (Bold, "camp".into())]);
        assert_eq!(
            replace_contents_in_word(&target_word, &mut content_word),
            expected_word
        );
    }
}

#[test]
fn test_to_simplified() {
    use super::unicode::to_simplified;
    assert_eq!(to_simplified("乙等/乙級"), "乙等/乙级");
}
