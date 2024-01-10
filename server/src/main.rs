use axum::{
    body::Body,
    extract::Query,
    http::HeaderValue,
    response::{IntoResponse, Response},
    routing::get,
    Json, Router,
};

use finalfusion::prelude::*;
use once_cell::sync::Lazy;
use rkyv::Deserialize;
use serde_json::json;
use std::collections::HashMap;
use std::fs::File;
use std::io::BufReader;
use std::io::Read;
use std::path::Path;
use tower_http::cors::{Any, CorsLayer};
use wordshk_tools::app_api::Api;
use wordshk_tools::dict::clause_to_string;
use wordshk_tools::dict::Clause;
use wordshk_tools::english_index::EnglishSearchRank;
use wordshk_tools::jyutping::Romanization;
use wordshk_tools::search::english_embedding_search;

// Define a struct to capture query parameters
#[derive(Debug, serde::Deserialize)]
struct SearchQuery {
    query: String,
}

const APP_DIR: &str = "data";

static API: Lazy<Api> = Lazy::new(|| {
    let romanization = Romanization::Jyutping;
    let api = unsafe { Api::load(APP_DIR, romanization) };
    api
});

static EMBEDDINGS: Lazy<Embeddings<VocabWrap, StorageWrap>> = Lazy::new(|| {
    let mut reader =
        BufReader::new(File::open(Path::new(APP_DIR).join("english_embeddings.fifu")).unwrap());
    let phrase_embeddings =
        Embeddings::<VocabWrap, StorageWrap>::mmap_embeddings(&mut reader).unwrap();
    phrase_embeddings
});

#[derive(Debug, serde::Serialize)]
struct SearchResult {
    entry_id: u32,
    variant: String,
    def_index: u32,
    score: u16,
    eng: String,
}

#[tokio::main]
async fn main() {
    // build our application with routes
    let app = Router::new()
        .route("/search", get(search))
        .route(
            "/.well-known/pki-validation/770DE1A4A87AD3AB4CDDD3664C481C37.txt",
            get(static_file),
        )
        .layer(CorsLayer::new().allow_origin(Any));

    // run our app with hyper, listening globally on port 3000
    let listener = tokio::net::TcpListener::bind("0.0.0.0:3000").await.unwrap();
    axum::serve(listener, app).await.unwrap();
}

async fn search(Query(params): Query<SearchQuery>) -> Json<Vec<SearchResult>> {
    let result = english_embedding_search(&EMBEDDINGS, unsafe { API.dict() }, &params.query);

    let mut results = vec![];

    for EnglishSearchRank {
        entry_id,
        def_index,
        matched_eng,
        score,
    } in result
    {
        let entry = unsafe { API.dict() }.get(&entry_id).unwrap();
        let variant = entry.variants.0[0].word.to_string();
        let def = &entry.defs[def_index as usize];
        let eng = def.eng.as_ref().unwrap();
        let eng: Clause = eng.deserialize(&mut rkyv::Infallible).unwrap();
        let eng = clause_to_string(&eng);
        results.push(SearchResult {
            entry_id,
            variant,
            def_index: def_index as u32,
            score: score as u16,
            eng,
        });
    }

    Json(results)
}

async fn static_file() -> impl IntoResponse {
    Response::builder()
        .header(axum::http::header::CONTENT_TYPE, "text/plain")
        .body(Body::from(include_str!(
            "../.well-known/pki-validation/770DE1A4A87AD3AB4CDDD3664C481C37.txt"
        )))
        .unwrap()
}
