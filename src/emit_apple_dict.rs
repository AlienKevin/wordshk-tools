use super::dict::{Clause, LaxJyutPingSegment, Segment, SegmentType};
use super::rich_dict::{
    RichDef, RichDict, RichEntry, RichLine, RubySegment, TextStyle, Word, WordLine, WordSegment,
};

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
    word.0
        .iter()
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
fn xml_escape(s: &str) -> String {
    s.replace("<", "&lt;").replace("&", "&amp;")
}

fn word_segment_to_xml((seg_type, word): &WordSegment) -> String {
    match seg_type {
        SegmentType::Text => word_to_xml(word),
        SegmentType::Link => link_to_xml(&word.to_string(), &word_to_xml(&word)),
    }
}

fn link_to_xml(link: &str, word: &str) -> String {
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

fn segment_to_xml((seg_type, seg): &Segment) -> String {
    match seg_type {
        SegmentType::Text => xml_escape(seg),
        SegmentType::Link => link_to_xml(&xml_escape(&seg), &xml_escape(&seg)),
    }
}

// Convert a [WordLine] to XML, highlighting variants
fn word_line_to_xml(line: &WordLine) -> String {
    format!(
        "<div class=\"{}\">{}</div>",
        "word-clause",
        line.iter()
            .map(word_segment_to_xml)
            .collect::<Vec<String>>()
            .join("")
    )
}

/// Convert a [RichClause] to an Apple Dictionary XML string
fn clause_to_xml(clause: &Clause) -> String {
    clause_to_xml_with_class_name("clause", clause)
}

/// Convert a [RichLine] to an Apple Dictionary XML string
fn rich_line_to_xml(line: &RichLine) -> String {
    match line {
        RichLine::Ruby(ruby_line) => {
            let mut output = "<ruby class=\"pr-clause\">".to_string();
            ruby_line.iter().for_each(|seg| match seg {
                RubySegment::LinkedWord(pairs) => {
                    let mut ruby = "<ruby>".to_string();
                    let mut word_str = String::new();
                    pairs.iter().for_each(|(word, prs)| {
                        ruby += &format!(
                            "\n{}<rt>{}</rt>",
                            word_to_xml(word),
                            prs.join(" ")
                        );
                        word_str += &word.to_string();
                    });
                    ruby += "\n</ruby>";
                    output += &format!("{}<rt></rt>", &link_to_xml(&word_str, &ruby));
                }
                RubySegment::Word(word, prs) => {
                    output += &format!(
                        "\n{}<rt>{}</rt>",
                        word_to_xml(word),
                        prs.join(" ")
                    );
                }
                RubySegment::Punc(punc) => {
                    output += &format!("\n{}<rt></rt>", punc);
                }
            });
            output += "\n</ruby>";
            output
        }
        RichLine::Text(line) => word_line_to_xml(&line),
    }
}

fn to_xml_badge_helper(is_emphasized: bool, tag: &str) -> String {
    format!(
        "<span class=\"badge{}\">{}</span>",
        if is_emphasized { "-em" } else { "-weak" },
        tag
    )
}

fn to_xml_badge_em(tag: &str) -> String {
    to_xml_badge_helper(true, tag)
}

fn to_xml_badge(tag: &str) -> String {
    to_xml_badge_helper(false, tag)
}

fn rich_defs_to_xml(defs: &Vec<RichDef>) -> String {
    "<ol>\n".to_string() + &defs
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
                            lang_name = lang.to_yue_name(),
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
                            let clause = rich_line_to_xml(&zho);
                            format!(
                                "<div class=\"eg-clause\"> <div class=\"lang-tag-ch\">（中）</div> {} </div>\n",
                                clause
                            )
                        }) + &eg.yue.clone().map_or("".to_string(), |yue| {
                            let clause = rich_line_to_xml(&yue);
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
}

fn rich_entry_to_xml(entry: &RichEntry) -> String {
    {
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
                variant_0_word = entry.variants.0[0].word,
                variants_index = entry
                    .variants.0
                    .iter()
                    .map(|variant| {
                        format!(
                            indoc!{r#"<d:index d:value="{word}" d:pr="{prs}"/>
                            {pr_indices}"#},
                            word = variant.word,
                            prs = &variant.prs.to_string(),
                            pr_indices = variant.prs.0.iter().map(|pr| {
                                let word_and_pr = variant.word.clone() + " " + &pr.to_string();
                                format!(r#"<d:index d:value="{pr}" d:title="{word_and_pr}" d:priority="2"/>{pr_without_tone}"#,
                                    pr = pr.to_string(),
                                    word_and_pr = word_and_pr,
                                    pr_without_tone = {
                                        if pr.0.iter().any(|pr_seg|
                                            if let LaxJyutPingSegment::Nonstandard(_) = pr_seg { true } else { false }
                                        ){
                                            "".to_string()
                                        } else {
                                            format!(r#"<d:index d:value="{pr_without_tone}" d:title="{word_and_pr}" d:priority="2"/>"#,
                                                pr_without_tone = pr.to_string_without_tone(),
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
                    .variants.0
                    .iter()
                    .map(|variant| {
                        format!(
                            indoc! {r#"<div>
                            <span d:priority="2"><h1>{}</h1></span>
                            <span class="prs"><span d:pr="JYUTPING">{}</span></span>
                            </div>"#},
                            variant.word,
                            &variant.prs.to_string(),
                        )
                    })
                    .collect::<Vec<String>>()
                    .join("\n"),
                tags = "<div class=\"tags\">\n".to_string()
            + &(if entry.poses.len() > 0 { format!("<span>詞性：{}</span>\n", entry.poses.iter().map(|pos| to_xml_badge_em(pos)).collect::<Vec<String>>().join("，")) } else { "".to_string() })
            + &(if entry.labels.len() > 0 { format!("<span> ｜ 標籤：{}</span>\n", entry.labels.iter().map(|label| to_xml_badge(label)).collect::<Vec<String>>().join("，")) } else { "".to_string() })
            + &(if entry.sims.len() > 0 { format!("<span> ｜ 近義：{}</span>\n", entry.sims.join("，")) } else { "".to_string() })
            + &(if entry.ants.len() > 0 { format!("<span> ｜ 反義：{}</span>\n", entry.ants.join("，")) } else { "".to_string() })
            // TODO: add refs 
            // TODO: add imgs
            + "</div>",
                defs = rich_defs_to_xml(&entry.defs)
            );
        entry_str
    }
}

/// Convert a [RichDict] to Apple Dictionary XML format
pub fn rich_dict_to_xml(dict: RichDict) -> String {
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
        .map(|(_id, entry)| rich_entry_to_xml(entry))
        .collect::<Vec<String>>()
        .join("\n\n");
    header.to_string() + &entries + "\n</d:dictionary>\n"
}
