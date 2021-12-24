use lazy_static::lazy_static;
use std::collections::HashSet;
use unicode_names2;
use unicode_segmentation::UnicodeSegmentation;

pub fn to_graphemes(s: &str) -> Vec<&str> {
    UnicodeSegmentation::graphemes(s, true).collect::<Vec<&str>>()
}

pub fn to_words(s: &str) -> Vec<&str> {
    UnicodeSegmentation::unicode_words(s).collect::<Vec<&str>>()
}

/// Test whether a character is a Chinese/English punctuation
pub fn is_punctuation(c: char) -> bool {
    PUNCTUATIONS.contains(&c)
}

/// Test if a character is latin small or capital letter
pub fn is_latin(c: char) -> bool {
    if let Some(name) = unicode_names2::name(c) {
        let name = format!("{}", name);
        name.starts_with("LATIN SMALL LETTER") || name.starts_with("LATIN CAPITAL LETTER")
    } else {
        false
    }
}

pub fn is_alphanumeric(c: char) -> bool {
    let cp = c as i32;
    (0x30 <= cp && cp < 0x40) || (0xFF10 <= cp && cp < 0xFF20) || is_latin(c)
}

pub fn is_cjk(c: char) -> bool {
    let cp = c as i32;
    (0x3400 <= cp && cp <= 0x4DBF)
        || (0x4E00 <= cp && cp <= 0x9FFF)
        || (0xF900 <= cp && cp <= 0xFAFF)
        || (0x20000 <= cp && cp <= 0x2FFFF)
}

pub fn test_g(f: fn(char) -> bool, g: &str) -> bool {
    if let Some(c) = g.chars().next() {
        g.chars().count() == 1 && f(c)
    } else {
        false
    }
}

lazy_static! {
    static ref PUNCTUATIONS: HashSet<char> = {
        HashSet::from([
            // Shared punctuations
            '@', '#', '$', '%', '^', '&', '*',
            // English punctuations
            '~', '`', '!',  '(', ')', '-', '_', '{', '}', '[', ']', '|', '\\', ':', ';',
            '"', '\'', '<', '>', ',', '.', '?', '/',
            // Chinese punctuations
            '～', '·', '！', '：', '；', '“', '”', '‘', '’', '【', '】', '（', '）',
            '「', '」', '《', '》', '？', '，', '。', '、', '／', '＋'
        ])
    };
}