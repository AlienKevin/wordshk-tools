use std::collections::BTreeMap;
use std::io::Cursor;

use finalfusion::io::WriteEmbeddings;
use finalfusion::norms::NdNorms;
use finalfusion::prelude::*;
use finalfusion::storage::NdArray;
use finalfusion::vocab::SimpleVocab;
use ndarray::Array1;
use ndarray::ArrayViewMut1;
use ndarray::ArrayViewMut2;
use sif_embedding::SentenceEmbedder;
use sif_embedding::Sif;
use vtext::tokenize::Tokenizer;
use vtext::tokenize::VTextTokenizerParams;
use wordfreq_model::ModelKind;

use crate::english_index::tokenize_entry;
use crate::rich_dict::ArchivedRichDict;
use crate::unicode;
use anyhow::Result;

pub fn generate_english_embeddings<'a>(dict: &ArchivedRichDict) -> Result<(Vec<u8>, Vec<u8>)> {
    let mut phrase_map: BTreeMap<String, String> = BTreeMap::new();

    for (entry_id, entry) in dict.iter() {
        tokenize_entry(entry, unicode::normalize_english_word_for_embedding)
            .into_iter()
            .enumerate()
            .for_each(|(def_index, phrases)| {
                phrases
                    .into_iter()
                    .enumerate()
                    .for_each(|(phrase_index, (phrase, _))| {
                        let index: String = format!("{entry_id},{def_index},{phrase_index}");
                        phrase_map.insert(index, phrase);
                    })
            });
    }

    let tokenizer = VTextTokenizerParams::default().lang("en").build()?;
    let separator = sif_embedding::DEFAULT_SEPARATOR.to_string();

    let word_probs = wordfreq_model::load_wordfreq(ModelKind::LargeEn)?;

    let mut embeddings_reader = Cursor::new(include_bytes!("../data/glove.6B.300d.fifu"));
    let word_embeddings =
        Embeddings::<VocabWrap, StorageWrap>::read_embeddings(&mut embeddings_reader)?;

    let sif_model = Sif::new(&word_embeddings, &word_probs);

    let phrases: Vec<String> = phrase_map
        .values()
        .map(|phrase| {
            tokenizer
                .tokenize(&phrase)
                .collect::<Vec<_>>()
                .join(&separator)
                .to_lowercase()
        })
        .collect();

    let sif_model = sif_model.fit(&phrases)?;
    let mut phrase_embeddings = sif_model.embeddings(&phrases)?;

    let vocab: Vec<String> = phrase_map.keys().cloned().collect();

    let norms = l2_normalize_array(phrase_embeddings.view_mut());
    let embeddings = Embeddings::new(
        None,
        SimpleVocab::new(vocab),
        NdArray::new(phrase_embeddings),
        NdNorms::new(norms),
    );

    // Serialize the Finalfusion embeddings
    let mut cursor = Cursor::new(Vec::new());
    embeddings.write_embeddings(&mut cursor)?;
    let finalfusion_bytes = cursor.into_inner();

    Ok((sif_model.serialize()?, finalfusion_bytes))
}

fn l2_normalize(mut v: ArrayViewMut1<f32>) -> f32 {
    let norm = v.dot(&v).sqrt();

    if norm != 0. {
        v /= norm;
    }

    norm
}

fn l2_normalize_array(mut v: ArrayViewMut2<f32>) -> Array1<f32> {
    let mut norms = Vec::with_capacity(v.nrows());
    for embedding in v.outer_iter_mut() {
        norms.push(l2_normalize(embedding));
    }

    norms.into()
}
