use super::dict::{Clause, Segment, SegmentType};
use super::rich_dict::{
    RichDef, RichEg, RichEntry, RichLine, RubySegment, TextStyle, Word, WordLine, WordSegment,
};

use indoc::indoc;

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
    format!(r#"<a href="{}">{}</a>"#, link, word,)
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

fn rich_eg_to_xml(eg: &RichEg) -> String {
    "<div class=\"eg\">\n".to_string()
        + &eg.zho.clone().map_or("".to_string(), |zho| {
            let clause = rich_line_to_xml(&zho);
            format!("<div>（中）{}</div>\n", clause)
        })
        + &eg.yue.clone().map_or("".to_string(), |yue| {
            let clause = rich_line_to_xml(&yue);
            format!("<div>（粵）{}</div>\n", clause)
        })
        + &eg.eng.clone().map_or("".to_string(), |eng| {
            format!("<div>（英）{}</div>\n", clause_to_xml(&vec![eng]))
        })
        + "</div>"
}

fn rich_defs_to_xml(defs: &Vec<RichDef>) -> String {
    "<ol>\n".to_string()
        + &defs
            .iter()
            .map(|def| {
                let mut egs_iter = def.egs.iter();
                "<li>\n".to_string()
                    + "<div class=\"def-head\">\n"
                    + &format!("<div>【粵】{}</div>\n", clause_to_xml(&def.yue))
                    + &def.eng.clone().map_or("".to_string(), |eng| {
                        format!("<div>【英】{}</div>\n", clause_to_xml(&eng))
                    })
                    + &def
                        .alts
                        .iter()
                        .map(|(lang, clause)| {
                            format!(
                                "<div>【{lang_name}】{clause}</div>\n",
                                lang_name = lang.to_yue_name(),
                                clause = clause_to_xml(clause)
                            )
                        })
                        .collect::<Vec<String>>()
                        .join("")
                    + "</div>\n"
                    + &egs_iter
                        .next()
                        .map(rich_eg_to_xml)
                        .unwrap_or("".to_string())
                    + &egs_iter
                        .next()
                        .map(rich_eg_to_xml)
                        .unwrap_or("".to_string())
                    + {
                        let hidden_egs = &egs_iter
                            .map(rich_eg_to_xml)
                            .collect::<Vec<String>>()
                            .join("\n");
                        &if hidden_egs.is_empty() {
                            "".to_string()
                        } else {
                            "<details>\n".to_string()
                                + "<summary>更多例句...</summary>"
                                + hidden_egs
                                + "</details>\n"
                        }
                    }
                    + "</li>\n"
            })
            .collect::<Vec<String>>()
            .join("\n")
        + "</ol>"
}

pub fn rich_entry_to_xml(entry: &RichEntry) -> String {
    {
        let entry_str = format!(
            indoc! {r#"
                <div class="entry">
                <div class ="entry-head">
                {variants_word_pr}
                {tags}
                </div>
                <div>
                {defs}
                </div>
                </div>"#},
            variants_word_pr = entry
                .variants
                .0
                .iter()
                .map(|variant| {
                    format!(
                        indoc! {r#"<div>
                            <h1>{}</h1>
                            <span class="prs">{}</span>
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
