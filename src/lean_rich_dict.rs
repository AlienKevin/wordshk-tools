use crate::rich_dict::RichVariants;
use crate::search::Script;

use super::dict::{AltClause, Clause, EntryId, Line, Segment};
use super::rich_dict::{get_simplified_rich_line, RichDef, RichEg, RichEntry, RichLine};
use serde::Deserialize;
use serde::Serialize;

#[derive(Debug, PartialEq, Clone, Serialize, Deserialize)]
pub struct LeanRichEntry {
    pub id: EntryId,
    pub variants: Vec<LeanVariant>,
    pub variants_simp: Vec<LeanVariant>,
    pub poses: Vec<String>,
    pub labels: Vec<String>,
    pub sims: Vec<Segment>,
    pub sims_simp: Vec<String>,
    pub ants: Vec<Segment>,
    pub ants_simp: Vec<String>,
    pub defs: Vec<LeanDef>,
    pub published: bool,
}

/// A variant of a \[word\] with \[prs\] (pronounciations)
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LeanVariant {
    pub word: String,
    pub prs: String,
}

#[derive(Debug, PartialEq, Clone, Serialize, Deserialize)]
pub struct LeanDef {
    pub yue: Clause,
    pub yue_simp: Clause,
    pub eng: Option<Clause>,
    pub alts: Vec<AltClause>,
    pub egs: Vec<LeanEg>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LeanEg {
    pub zho: Option<RichLine>,
    pub zho_simp: Option<RichLine>,
    pub yue: Option<RichLine>,
    pub yue_simp: Option<RichLine>,
    pub eng: Option<Line>,
}

pub fn to_lean_rich_entry(entry: &RichEntry) -> LeanRichEntry {
    LeanRichEntry {
        id: entry.id,
        variants: to_lean_variants(&entry.variants, Script::Traditional),
        variants_simp: to_lean_variants(&entry.variants, Script::Simplified),
        poses: entry.poses.clone(),
        labels: entry.labels.clone(),
        sims: entry.sims.clone(),
        sims_simp: entry.sims_simp.clone(),
        ants: entry.ants.clone(),
        ants_simp: entry.ants_simp.clone(),
        defs: entry.defs.iter().map(to_lean_def).collect(),
        published: entry.published,
    }
}

fn to_lean_def(def: &RichDef) -> LeanDef {
    LeanDef {
        yue: def.yue.clone(),
        yue_simp: def.yue_simp.clone(),
        eng: def.eng.clone(),
        alts: def.alts.clone(),
        egs: def.egs.iter().map(to_lean_eg).collect(),
    }
}

fn to_lean_eg(eg: &RichEg) -> LeanEg {
    LeanEg {
        zho: eg.zho.clone(),
        zho_simp: eg
            .zho_simp
            .as_ref()
            .map(|zho_simp| get_simplified_rich_line(zho_simp, eg.zho.as_ref().unwrap())),
        yue: eg.yue.clone(),
        yue_simp: eg
            .yue_simp
            .as_ref()
            .map(|yue_simp| get_simplified_rich_line(yue_simp, eg.yue.as_ref().unwrap())),
        eng: eg.eng.clone(),
    }
}

fn to_lean_variants(variant: &RichVariants, script: Script) -> Vec<LeanVariant> {
    variant
        .0
        .iter()
        .map(|variant| LeanVariant {
            word: match script {
                Script::Traditional => variant.word.clone(),
                Script::Simplified => variant.word_simp.clone(),
            },
            prs: variant.prs.to_string(),
        })
        .collect()
}
