use std::collections::BTreeMap;
use std::io::Cursor;

use fastembed::{EmbeddingBase, FlagEmbedding};
use finalfusion::io::WriteEmbeddings;
use finalfusion::norms::NdNorms;
use finalfusion::prelude::*;
use finalfusion::storage::NdArray;
use finalfusion::vocab::SimpleVocab;
use ndarray::{Array1, Array2, ArrayViewMut1, ArrayViewMut2};

use crate::english_index::split_entry_eng_defs_into_phrases;
use crate::rich_dict::ArchivedRichDict;
use anyhow::Result;

pub fn generate_english_embeddings<'a>(dict: &ArchivedRichDict) -> Result<Vec<u8>> {
    let mut phrase_map: BTreeMap<String, String> = BTreeMap::new();

    for (entry_id, entry) in dict.iter() {
        split_entry_eng_defs_into_phrases(entry, |x| x.to_string())
            .into_iter()
            .for_each(|(def_index, phrases)| {
                let index: String = format!("{entry_id},{def_index}");
                phrase_map
                    .entry(phrases.join("; "))
                    .and_modify(|indices| {
                        indices.push_str(&format!(";{}", index));
                    })
                    .or_insert(index);
            });
    }

    // With default InitOptions
    let model: FlagEmbedding = FlagEmbedding::try_new(Default::default())?;

    println!("Number of phrases: {}", phrase_map.len());

    let phrase_embeddings = model.embed(
        phrase_map
            .keys()
            .map(|phrase| format!("query: {phrase}"))
            .collect(),
        None,
    )?;
    let nrows = phrase_embeddings.len();
    let ncols = phrase_embeddings[0].len();
    // Flatten the Vec<Vec<f32>>
    let flat_vec: Vec<f32> = phrase_embeddings.into_iter().flatten().collect();

    // Create Array2 from the flattened Vec
    let mut phrase_embeddings = Array2::from_shape_vec((nrows, ncols), flat_vec).unwrap();

    let vocab: Vec<String> = phrase_map.values().cloned().collect();

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

    Ok(finalfusion_bytes)
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
