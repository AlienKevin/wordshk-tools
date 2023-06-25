use wordshk_tools::{dict::Dict, rich_dict::{enrich_dict, EnrichDictOptions}, parse::parse_dict, search, jyutping::Romanization};
use once_cell::sync::Lazy;

static VARIANTS_MAP: Lazy<search::VariantsMap> = Lazy::new(|| {
    let dict = parse_dict(include_str!("../../wordshk.csv").as_bytes()).expect("Failed to parse dict");
    let rich_dict = enrich_dict(
        &dict,
        &EnrichDictOptions {
            remove_dead_links: true,
        },
    );
    search::rich_dict_to_variants_map(&rich_dict)
});

use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn criterion_benchmark(c: &mut Criterion) {
    c.bench_function("pr_search", |b| b.iter(|| search::pr_search(&VARIANTS_MAP, black_box("gwong2 dung1 waa2"), Romanization::Jyutping)));
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
