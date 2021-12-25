use super::emit::{
    flatten_line, match_ruby, pr_to_string, pr_to_string_without_tone, prs_to_string,
    to_yue_lang_name, variants_to_words, word_to_string, RubySegment, TextStyle, Word, WordSegment,
};
use super::parse::{Clause, Dict, LaxJyutPingSegment, Line, PrLine, Segment, SegmentType};
use super::unicode;

use indoc::indoc;
use std::fs;

fn get_xml_start_tag(style: &TextStyle) -> &'static str {
    match style {
        TextStyle::Normal => "",
        TextStyle::Bold => "<b>",
    }
}

fn get_xml_end_tag(style: &TextStyle) -> &'static str {
    match style {
        TextStyle::Normal => "",
        TextStyle::Bold => "</b>",
    }
}

fn word_to_xml(word: &Word) -> String {
    word.iter()
        .map(|(style, seg)| {
            format!(
                "{start_tag}{content}{end_tag}",
                start_tag = get_xml_start_tag(style),
                content = xml_escape(seg),
                end_tag = get_xml_end_tag(style),
            )
        })
        .collect::<Vec<String>>()
        .join("")
}

/// Escape '<' and '&' in an XML string
fn xml_escape(s: &String) -> String {
    s.replace("<", "&lt;").replace("&", "&amp;")
}

fn segment_to_xml((seg_type, seg): &Segment) -> String {
    match seg_type {
        SegmentType::Text => xml_escape(seg),
        SegmentType::Link => link_to_xml(&xml_escape(&seg), &xml_escape(&seg)),
    }
}

fn word_segment_to_xml((seg_type, word): &WordSegment) -> String {
    match seg_type {
        SegmentType::Text => word_to_xml(word),
        SegmentType::Link => link_to_xml(&word_to_string(&word), &word_to_xml(&word)),
    }
}

fn link_to_xml(link: &String, word: &String) -> String {
    format!(
        r#"<a href="x-dictionary:d:{}:{dict_id}">{}</a>"#,
        link,
        word,
        dict_id = "wordshk"
    )
}

fn clause_to_xml_with_class_name(class_name: &str, clause: &Clause) -> String {
    format!(
        "<div class=\"{}\">{}</div>",
        class_name,
        clause
            .iter()
            .map(|line| {
                line.iter()
                    .map(segment_to_xml)
                    .collect::<Vec<String>>()
                    .join("")
            })
            .collect::<Vec<String>>()
            .join("\n")
    )
}

// Convert a [Line] (without Pr) to XML, highlighting variants
fn line_to_xml(variants: &Vec<&str>, line: &Line) -> String {
    format!(
        "<div class=\"{}\">{}</div>",
        "pr-clause",
        flatten_line(variants, line)
            .iter()
            .map(word_segment_to_xml)
            .collect::<Vec<String>>()
            .join("")
    )
}

/// Convert a [Clause] to an Apple Dictionary XML string
fn clause_to_xml(clause: &Clause) -> String {
    clause_to_xml_with_class_name("clause", clause)
}

/// Convert a [PrLine] to an Apple Dictionary XML string
fn pr_line_to_xml(variants: &Vec<&str>, (line, pr): &PrLine) -> String {
    match pr {
        Some(pr) => {
            let prs = unicode::to_words(pr);
            let ruby_line = match_ruby(variants, line, &prs);
            let mut output = "<ruby class=\"pr-clause\">".to_string();
            ruby_line.iter().for_each(|seg| match seg {
                RubySegment::LinkedWord(pairs) => {
                    let mut ruby = "<ruby>".to_string();
                    let mut word_str = String::new();
                    pairs.iter().for_each(|(word, prs)| {
                        ruby += &format!(
                            "\n<rb>{}</rb>\n<rt>{}</rt>",
                            word_to_xml(word),
                            prs.join(" ")
                        );
                        word_str += &word_to_string(word);
                    });
                    ruby += "\n</ruby>";
                    output += &format!("<rb>{}</rb><rt></rt>", &link_to_xml(&word_str, &ruby));
                }
                RubySegment::Word(word, prs) => {
                    output += &format!(
                        "\n<rb>{}</rb>\n<rt>{}</rt>",
                        word_to_xml(word),
                        prs.join(" ")
                    );
                }
                RubySegment::Punc(punc) => {
                    output += &format!("\n<rb>{}</rb>\n<rt></rt>", punc);
                }
            });
            output += "\n</ruby>";
            output
        }
        None => line_to_xml(variants, line),
    }
}

fn to_xml_badge_helper(is_emphasized: bool, tag: &String) -> String {
    format!(
        "<span class=\"badge{}\">{}</span>",
        if is_emphasized { "-em" } else { "-weak" },
        tag
    )
}

fn to_xml_badge_em(tag: &String) -> String {
    to_xml_badge_helper(true, tag)
}

fn to_xml_badge(tag: &String) -> String {
    to_xml_badge_helper(false, tag)
}

/// Convert a [Dict] to Apple Dictionary XML format
pub fn dict_to_xml(dict: Dict) -> String {
    let front_back_matter_filename = "apple_dict/front_back_matter.html";
    let front_back_matter = fs::read_to_string(front_back_matter_filename).expect(&format!(
        "Something went wrong when I tried to read {}",
        front_back_matter_filename
    ));

    let header = format!(
        r#"<?xml version="1.0" encoding="UTF-8"?>
<d:dictionary xmlns="http://www.w3.org/1999/xhtml" xmlns:d="http://www.apple.com/DTDs/DictionaryService-1.0.rng">
<d:entry id="front_back_matter" d:title="Front/Back Matter">
{}
</d:entry>
"#,
        front_back_matter
    );

    let entries = dict
        .iter()
        .map(|(_id, entry)| {
            let variant_words = variants_to_words(&entry.variants);
            let entry_str = format!(
                indoc! {r#"
                <d:entry id="{id}" d:title="{variant_0_word}">
                <div class="entry">
                {variants_index}
                <div class ="entry-head">
                {variants_word_pr}
                {tags}
                </div>
                <div>
                {defs}
                </div>
                </div>
                </d:entry>"#},
                id = entry.id,
                variant_0_word = entry.variants[0].word,
                variants_index = entry
                    .variants
                    .iter()
                    .map(|variant| {
                        format!(
                            indoc!{r#"<d:index d:value="{word}" d:pr="{prs}"/>
                            {pr_indices}"#},
                            word = variant.word,
                            prs = prs_to_string(&variant.prs),
                            pr_indices = variant.prs.iter().map(|pr| {
                                let word_and_pr = variant.word.clone() + " " + &pr_to_string(pr);
                                format!(r#"<d:index d:value="{pr}" d:title="{word_and_pr}" d:priority="2"/>{pr_without_tone}"#,
                                    pr = pr_to_string(pr),
                                    word_and_pr = word_and_pr,
                                    pr_without_tone = {
                                        if pr.iter().any(|pr_seg|
                                            if let LaxJyutPingSegment::Nonstandard(_) = pr_seg { true } else { false }
                                        ){
                                            "".to_string()
                                        } else {
                                            format!(r#"<d:index d:value="{pr_without_tone}" d:title="{word_and_pr}" d:priority="2"/>"#,
                                                pr_without_tone = pr_to_string_without_tone(pr),
                                                word_and_pr = word_and_pr
                                            )
                                        }
                                    })
                                }
                            ).collect::<Vec<String>>().join("\n")
                        )
                    })
                    .collect::<Vec<String>>()
                    .join("\n"),
                variants_word_pr = entry
                    .variants
                    .iter()
                    .map(|variant| {
                        format!(
                            indoc! {r#"<div>
                            <span d:priority="2"><h1>{}</h1></span>
                            <span class="prs"><span d:pr="JYUTPING">{}</span></span>
                            </div>"#},
                            variant.word,
                            prs_to_string(&variant.prs),
                        )
                    })
                    .collect::<Vec<String>>()
                    .join("\n"),
                tags = "<div class=\"tags\">\n".to_string()
            + &(if entry.poses.len() > 0 { format!("<span>詞性：{}</span>\n", entry.poses.iter().map(to_xml_badge_em).collect::<Vec<String>>().join("，")) } else { "".to_string() })
            + &(if entry.labels.len() > 0 { format!("<span> ｜ 標籤：{}</span>\n", entry.labels.iter().map(to_xml_badge).collect::<Vec<String>>().join("，")) } else { "".to_string() })
            + &(if entry.sims.len() > 0 { format!("<span> ｜ 近義：{}</span>\n", entry.sims.join("，")) } else { "".to_string() })
            + &(if entry.ants.len() > 0 { format!("<span> ｜ 反義：{}</span>\n", entry.ants.join("，")) } else { "".to_string() })
            // TODO: add refs 
            // TODO: add imgs
            + "</div>",
                defs = "<ol>\n".to_string()
                    + &entry
                        .defs
                        .iter()
                        .map(|def| {
                            "<li>\n".to_string()
                                + "<div class=\"def-head\">\n"
                                + &format!("<div class=\"def-yue\"> <div>【粵】</div> {} </div>\n", clause_to_xml(&def.yue))
                                + &def.eng.clone().map_or("".to_string(), |eng| {
                                    format!("<div class=\"def-eng\"> <div>【英】</div> {} </div>\n", clause_to_xml(&eng))
                                })
                                + &def
                                    .alts
                                    .iter()
                                    .map(|(lang, clause)| {
                                        format!(
                                            "<div class=\"def-alt\"> <div>【{lang_name}】</div> {clause} </div>\n",
                                            lang_name = to_yue_lang_name(*lang),
                                            clause = clause_to_xml(clause)
                                        )
                                    })
                                    .collect::<Vec<String>>()
                                    .join("")
                                + "</div>\n"
                                + &def
                                    .egs
                                    .iter()
                                    .map(|eg| {
                                        "<div class=\"eg\">\n".to_string()
                                        + &eg.zho.clone().map_or("".to_string(), |zho| {
                                            let clause = pr_line_to_xml(&variant_words, &zho);
                                            format!(
                                                "<div class=\"eg-clause\"> <div class=\"lang-tag-ch\">（中）</div> {} </div>\n",
                                                clause
                                            )
                                        }) + &eg.yue.clone().map_or("".to_string(), |yue| {
                                            let clause = pr_line_to_xml(&variant_words, &yue);
                                            format!(
                                                "<div class=\"eg-clause\"> <div class=\"lang-tag-ch\">（粵）</div> {} </div>\n",
                                                clause
                                            )
                                        }) + &eg.eng.clone().map_or("".to_string(), |eng| {
                                            format!("<div class=\"eg-clause\"> <div>（英）</div> <div class=\"eng-eg\">{}</div> </div>\n", clause_to_xml(&vec![eng]))
                                        })
                                        + "</div>"
                                    })
                                    .collect::<Vec<String>>()
                                    .join("\n")
                                + "</li>\n"
                        })
                        .collect::<Vec<String>>()
                        .join("\n")
                    + "</ol>"
            );

            entry_str
        })
        .collect::<Vec<String>>()
        .join("\n\n");
    header.to_string() + &entries + "\n</d:dictionary>\n"
}
