use once_cell::sync::Lazy;
use wordshk_tools::{app_api::Api, jyutping::Romanization, search};

static API: Lazy<Api> =
    Lazy::new(|| unsafe { Api::load("../export_api/app_tmp", Romanization::Jyutping) });

use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn criterion_benchmark(c: &mut Criterion) {
    // Build the pr_indices first
    let _ = &*API;
    c.bench_function("pr_search", |b| {
        b.iter(|| {
            search::pr_search(
                &API.fst_pr_indices,
                unsafe { &API.dict() },
                black_box("gwong2 dung1 waa2"),
                Romanization::Jyutping,
            )
        })
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
