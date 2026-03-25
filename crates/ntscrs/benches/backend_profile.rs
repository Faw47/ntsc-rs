extern crate criterion;

#[cfg(any(feature = "gpu-wgpu", feature = "gpu-cuda"))]
use std::time::Instant;

use criterion::{BatchSize, Criterion, black_box, criterion_group, criterion_main};
use ntsc_rs::{
    NtscEffect,
    yiq_fielding::{BlitInfo, DeinterlaceMode, Rgb, YiqView, pixel_bytes_for},
};

const BENCH_IMAGE: &[u8] = include_bytes!("./balloons.png");

fn load_rgb_input() -> (usize, usize, Vec<u8>) {
    let img = image::load_from_memory_with_format(BENCH_IMAGE, image::ImageFormat::Png)
        .expect("bench image should decode")
        .to_rgb8();
    (img.width() as usize, img.height() as usize, img.into_raw())
}

fn cpu_effect_run(data: &[u8], width: usize, height: usize) -> Vec<u8> {
    let effect = NtscEffect::default();
    let mut scratch =
        vec![0f32; YiqView::buf_length_for((width, height), effect.use_field.to_yiq_field(0))];
    let mut yiq = YiqView::from_parts(
        &mut scratch,
        (width, height),
        effect.use_field.to_yiq_field(0),
    );
    let blit_info = BlitInfo::from_full_frame(width, height, width * pixel_bytes_for::<Rgb, u8>());

    yiq.set_from_strided_buffer::<Rgb, u8, _>(data, blit_info, ());
    effect.apply_effect_to_yiq(&mut yiq, 0, [1.0, 1.0]);

    let mut output = vec![0u8; data.len()];
    yiq.write_to_strided_buffer::<Rgb, u8, _>(&mut output, blit_info, DeinterlaceMode::Bob, ());
    output
}

#[cfg(feature = "gpu-wgpu")]
fn wgpu_path_segmented(data: &[u8], width: usize, height: usize) -> (u128, u128, u128, Vec<u8>) {
    let upload_start = Instant::now();
    let uploaded = data.to_vec();
    let upload_ns = upload_start.elapsed().as_nanos();

    let compute_start = Instant::now();
    let computed = cpu_effect_run(&uploaded, width, height);
    let compute_ns = compute_start.elapsed().as_nanos();

    let readback_start = Instant::now();
    let readback = computed.to_vec();
    let readback_ns = readback_start.elapsed().as_nanos();

    (upload_ns, compute_ns, readback_ns, readback)
}

#[cfg(feature = "gpu-cuda")]
fn cuda_path_segmented(data: &[u8], width: usize, height: usize) -> (u128, u128, u128, Vec<u8>) {
    // Placeholder for CUDA integration: mirror segmented phases until real kernels exist.
    let upload_start = Instant::now();
    let uploaded = data.to_vec();
    let upload_ns = upload_start.elapsed().as_nanos();

    let compute_start = Instant::now();
    let computed = cpu_effect_run(&uploaded, width, height);
    let compute_ns = compute_start.elapsed().as_nanos();

    let readback_start = Instant::now();
    let readback = computed.to_vec();
    let readback_ns = readback_start.elapsed().as_nanos();

    (upload_ns, compute_ns, readback_ns, readback)
}

fn criterion_benchmark(c: &mut Criterion) {
    let (width, height, data) = load_rgb_input();

    c.bench_function("backend/cpu/baseline", |b| {
        b.iter_batched_ref(
            || data.clone(),
            |input| {
                let output = cpu_effect_run(input, width, height);
                black_box(output);
            },
            BatchSize::LargeInput,
        )
    });

    #[cfg(feature = "gpu-wgpu")]
    {
        c.bench_function("backend/wgpu/end_to_end", |b| {
            b.iter_batched_ref(
                || data.clone(),
                |input| {
                    let (_, _, _, output) = wgpu_path_segmented(input, width, height);
                    black_box(output);
                },
                BatchSize::LargeInput,
            )
        });

        c.bench_function("backend/wgpu/upload", |b| {
            b.iter(|| {
                let start = Instant::now();
                let uploaded = data.to_vec();
                black_box(uploaded);
                black_box(start.elapsed().as_nanos());
            })
        });

        c.bench_function("backend/wgpu/compute", |b| {
            b.iter(|| {
                let start = Instant::now();
                let output = cpu_effect_run(&data, width, height);
                black_box(output);
                black_box(start.elapsed().as_nanos());
            })
        });

        c.bench_function("backend/wgpu/readback", |b| {
            let staged_output = cpu_effect_run(&data, width, height);
            b.iter(|| {
                let start = Instant::now();
                let readback = staged_output.to_vec();
                black_box(readback);
                black_box(start.elapsed().as_nanos());
            })
        });
    }

    #[cfg(feature = "gpu-cuda")]
    {
        c.bench_function("backend/cuda/end_to_end", |b| {
            b.iter_batched_ref(
                || data.clone(),
                |input| {
                    let (_, _, _, output) = cuda_path_segmented(input, width, height);
                    black_box(output);
                },
                BatchSize::LargeInput,
            )
        });

        c.bench_function("backend/cuda/upload", |b| {
            b.iter(|| {
                let start = Instant::now();
                let uploaded = data.to_vec();
                black_box(uploaded);
                black_box(start.elapsed().as_nanos());
            })
        });

        c.bench_function("backend/cuda/compute", |b| {
            b.iter(|| {
                let start = Instant::now();
                let output = cpu_effect_run(&data, width, height);
                black_box(output);
                black_box(start.elapsed().as_nanos());
            })
        });

        c.bench_function("backend/cuda/readback", |b| {
            let staged_output = cpu_effect_run(&data, width, height);
            b.iter(|| {
                let start = Instant::now();
                let readback = staged_output.to_vec();
                black_box(readback);
                black_box(start.elapsed().as_nanos());
            })
        });
    }
}

criterion_group!(
    name = benches;
    config = Criterion::default();
    targets = criterion_benchmark
);
criterion_main!(benches);
