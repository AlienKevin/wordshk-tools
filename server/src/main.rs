use axum::{extract::Query, routing::get, Json, Router};
use serde_json::json;
use std::collections::HashMap;

// Define a struct to capture query parameters
#[derive(Debug, serde::Deserialize)]
struct SearchQuery {
    query: String,
}

#[tokio::main]
async fn main() {
    // build our application with routes
    let app = Router::new().route("/search", get(search));

    // run our app with hyper, listening globally on port 3000
    let listener = tokio::net::TcpListener::bind("0.0.0.0:3000").await.unwrap();
    axum::serve(listener, app).await.unwrap();
}

async fn search(Query(params): Query<SearchQuery>) -> Json<HashMap<String, String>> {
    let mut response = HashMap::new();
    response.insert("entry_id".to_string(), "123".to_string());

    Json(response)
}
