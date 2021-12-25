use super::parse::{
    jyutping_to_string, jyutping_to_string_without_tone, AltLang, LaxJyutPing, LaxJyutPingSegment,
    Variant,
};
use super::rich_dict::Word;

/// Convert [AltLang] to a language name in Cantonese
pub fn to_yue_lang_name(lang: AltLang) -> String {
    match lang {
        AltLang::Jpn => "日文",
        AltLang::Kor => "韓文",
        AltLang::Por => "葡萄牙文",
        AltLang::Vie => "越南文",
        AltLang::Lat => "拉丁文",
        AltLang::Fra => "法文",
    }
    .to_string()
}

pub fn prs_to_string(prs: &Vec<LaxJyutPing>) -> String {
    prs.iter()
        .map(pr_to_string)
        .collect::<Vec<String>>()
        .join(", ")
}

pub fn pr_to_string(pr_segs: &LaxJyutPing) -> String {
    pr_segs
        .iter()
        .map(pr_segment_to_string)
        .collect::<Vec<String>>()
        .join(" ")
}

pub fn pr_to_string_without_tone(pr_segs: &LaxJyutPing) -> String {
    pr_segs
        .iter()
        .map(pr_segment_to_string_without_tone)
        .collect::<Vec<String>>()
        .join(" ")
}

pub fn pr_segment_to_string(pr: &LaxJyutPingSegment) -> String {
    match pr {
        LaxJyutPingSegment::Standard(pr) => jyutping_to_string(pr),
        LaxJyutPingSegment::Nonstandard(pr_str) => pr_str.clone(),
    }
}

pub fn pr_segment_to_string_without_tone(pr: &LaxJyutPingSegment) -> String {
    match pr {
        LaxJyutPingSegment::Standard(pr) => jyutping_to_string_without_tone(pr),
        LaxJyutPingSegment::Nonstandard(pr_str) => pr_str.clone(),
    }
}

pub fn variants_to_words(variants: &Vec<Variant>) -> Vec<&str> {
    variants.iter().map(|variant| &variant.word[..]).collect()
}

pub fn word_to_string(word: &Word) -> String {
    word.iter()
        .map(|(_, seg)| seg.clone())
        .collect::<Vec<String>>()
        .join("")
}
