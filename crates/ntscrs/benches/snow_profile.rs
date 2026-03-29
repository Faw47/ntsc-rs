use criterion::{Criterion, criterion_group, criterion_main};
use ntsc_rs::NtscEffect;
use ntsc_rs::yiq_fielding::{YiqField, YiqView};

fn snow_benchmark(c: &mut Criterion) {
    let width = 1920;
    let height = 1080;
    let mut buffer = vec![0.0; YiqView::buf_length_for((width, height), YiqField::Both)];

    c.bench_function("snow_effect", |b| {
        b.iter(|| {
            let mut yiq = YiqView::from_parts(&mut buffer, (width, height), YiqField::Both);
            let mut effect = NtscEffect::default();
            effect.snow_intensity = 1.0;
            effect.snow_anisotropy = 0.5;
            effect.apply_effect_to_yiq(&mut yiq, 0, [1.0, 1.0]);
        })
    });
}

criterion_group!(benches, snow_benchmark);
criterion_main!(benches);
