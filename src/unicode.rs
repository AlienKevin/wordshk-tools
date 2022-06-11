use super::hk_variant_map_safe::HONG_KONG_VARIANT_MAP_SAFE;
use super::simp_to_trad::SIMP_TO_TRAD;
use super::variant_to_us_english::VARIANT_TO_US_ENGLISH;
use deunicode::deunicode;
use fast2s;
use itertools::Itertools;
use lazy_static::lazy_static;
use rust_stemmers::{Algorithm, Stemmer};
use std::collections::HashSet;
use unicode_names2;
use unicode_segmentation::UnicodeSegmentation;

pub fn remove_first_char(s: &str) -> String {
    let mut chars = s.chars();
    // skip the first char
    chars.next();
    chars.as_str().to_string()
}

pub fn remove_last_char(s: &str) -> String {
    let mut chars = s.chars();
    // skip the last char
    chars.next_back();
    chars.as_str().to_string()
}

pub fn normalize(s: &str) -> String {
    use unicode_normalization::UnicodeNormalization;
    s.nfkc().collect::<String>().to_lowercase()
}

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
    (0x30 <= cp && cp < 0x40) || is_latin(c)
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

pub fn is_multi_word(s: &str) -> bool {
    s.contains(char::is_whitespace)
}

/// Returns a 'HK variant' of the characters of the input text. The input is
/// assumed to be Chinese traditional. This variant list confirms to most
/// expectations of how characters should be written in Hong Kong, but does not
/// necessarily conform to any rigid standard. It may be fine tuned by editors
/// of words.hk. This is the "safe" version that is probably less controversial.
pub fn to_hk_safe_variant(str: &str) -> String {
    to_graphemes(str)
        .iter()
        .map(|g| {
            if test_g(is_cjk, g) {
                if g.chars().count() == 1 {
                    if let Some(c) = g.chars().next() {
                        return match HONG_KONG_VARIANT_MAP_SAFE.get(&c) {
                            Some(s) => s.to_string(),
                            None => g.to_string(),
                        };
                    }
                }
            }
            return g.to_string();
        })
        .join("")
}

pub fn to_traditional(str: &str) -> String {
    str.chars()
        .map(|c| match SIMP_TO_TRAD.get(&c) {
            Some(trad_char) => *trad_char,
            None => c,
        })
        .join("")
}

pub fn to_simplified(str: &str) -> String {
    fast2s::convert(str)
}

/// This function assumes that the input look like english words (i.e. some
/// west-Europe-alike language and just one word instead of many), and returns
/// a consistent form regardless of which variant of the word is given
pub fn normalize_english_word_for_search_index(word: &str) -> String {
    to_us_english(
        &deunicode(&word.to_lowercase())
            .split_whitespace()
            .join(" ")
            .chars()
            .filter(|c| c.is_alphabetic() || *c == ' ')
            .collect::<String>(),
    )
}

pub fn to_us_english(str: &str) -> String {
    str.split_whitespace()
        .map(|word| VARIANT_TO_US_ENGLISH.get(word).unwrap_or(&word).to_string())
        .join(" ")
}

pub fn american_english_stem(str: &str) -> String {
    let en_stemmer = Stemmer::create(Algorithm::English);
    en_stemmer.stem(str).into_owned()
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
