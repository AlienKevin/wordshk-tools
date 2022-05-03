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
pub fn is_punc(c: char) -> bool {
    PUNCS.contains(&c)
}

/// Test whether a character is a Chinese punctuation
pub fn is_chinese_punc(c: char) -> bool {
    CHINESE_PUNCS
        .union(&SHARED_PUNCS)
        .copied()
        .collect::<HashSet<char>>()
        .contains(&c)
}

/// Test whether a character is an English punctuation
pub fn is_english_punc(c: char) -> bool {
    ENGLISH_PUNCS
        .union(&SHARED_PUNCS)
        .copied()
        .collect::<HashSet<char>>()
        .contains(&c)
}

/// Test if a character is latin small or capital letter
pub fn is_latin(c: char) -> bool {
    if let Some(name) = unicode_names2::name(c) {
        let name = format!("{}", name);
        name.contains("LATIN SMALL LETTER ") || name.contains("LATIN CAPITAL LETTER ")
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
    static ref PUNCS: HashSet<char> = {
        SHARED_PUNCS.union(&ENGLISH_PUNCS).copied().collect::<HashSet<char>>().union(&CHINESE_PUNCS).copied().collect()
    };

    static ref SHARED_PUNCS: HashSet<char> = {
        HashSet::from([
            '@', '#', '$', '%', '^', '&', '*', '·', '…', '‥', '—', '～'
        ])
    };

    static ref ENGLISH_PUNCS: HashSet<char> = {
        HashSet::from(['~', '`', '!',  '(', ')', '-', '_', '{', '}', '[', ']', '|', '\\', ':', ';',
            '"', '\'', '<', '>', ',', '.', '?', '/'])
    };

    static ref CHINESE_PUNCS: HashSet<char> = {
        HashSet::from([
            '！', '：', '；', '“', '”', '‘', '’', '【', '】', '（', '）',
            '「', '」', '﹁', '﹂', '『','』', '《', '》', '？', '，', '。', '、', '／', '＋',
            '〈','〉', '︿', '﹀', '［', '］', '‧',
            // Small Form Variants for Chinese National Standard CNS 11643
            '﹐', '﹑','﹒', '﹔', '﹕', '﹖', '﹗', '﹘', '﹙', '﹚', '﹛', '﹜', '﹝', '﹞', '﹟'])
    };
}
