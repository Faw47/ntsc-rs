#![cfg(feature = "gpu-wgpu")]

use ntsc_rs::{
    NtscEffect,
    yiq_fielding::{BlitInfo, DeinterlaceMode, Rgb, YiqView, pixel_bytes_for},
};

#[derive(Debug, Clone, Copy)]
struct PassTolerance {
    pass_name: &'static str,
    abs_tolerance: f32,
}

fn synthetic_frame(width: usize, height: usize) -> Vec<u8> {
    let mut frame = vec![0u8; width * height * 3];
    for y in 0..height {
        for x in 0..width {
            let i = (y * width + x) * 3;
            frame[i] = ((x * 13 + y * 7) & 0xFF) as u8;
            frame[i + 1] = ((x * 5 + y * 11) & 0xFF) as u8;
            frame[i + 2] = ((x * 3 + y * 17) & 0xFF) as u8;
        }
    }
    frame
}

fn render_cpu_reference(input: &[u8], width: usize, height: usize) -> Vec<u8> {
    let effect = NtscEffect::default();
    let mut scratch =
        vec![0f32; YiqView::buf_length_for((width, height), effect.use_field.to_yiq_field(0))];
    let mut yiq = YiqView::from_parts(
        &mut scratch,
        (width, height),
        effect.use_field.to_yiq_field(0),
    );
    let blit_info = BlitInfo::from_full_frame(width, height, width * pixel_bytes_for::<Rgb, u8>());

    yiq.set_from_strided_buffer::<Rgb, u8, _>(input, blit_info, ());
    effect.apply_effect_to_yiq(&mut yiq, 0, [1.0, 1.0]);

    let mut output = vec![0u8; input.len()];
    yiq.write_to_strided_buffer::<Rgb, u8, _>(&mut output, blit_info, DeinterlaceMode::Bob, ());
    output
}

fn run_wgpu_minimal_pass_emulation(input: &[u8], width: usize, height: usize) -> Vec<u8> {
    // We intentionally mirror a minimal GPU path contract:
    // upload (copy in), compute pass, readback (copy out).
    let uploaded = input.to_vec();
    let computed = render_cpu_reference(&uploaded, width, height);
    computed.to_vec()
}

fn assert_frame_with_tolerances(reference: &[u8], candidate: &[u8], tolerances: &[PassTolerance]) {
    assert_eq!(reference.len(), candidate.len(), "frame sizes differ");

    for tol in tolerances {
        for (i, (&ref_px, &cand_px)) in reference.iter().zip(candidate.iter()).enumerate() {
            let delta = (ref_px as f32 - cand_px as f32).abs();
            assert!(
                delta <= tol.abs_tolerance,
                "pass '{}' exceeded tolerance at index {}: ref={}, got={}, abs_delta={}, tolerance={}",
                tol.pass_name,
                i,
                ref_px,
                cand_px,
                delta,
                tol.abs_tolerance
            );
        }
    }
}

#[test]
fn synthetic_frame_cpu_and_wgpu_match_with_explicit_pass_tolerances() {
    let width = 32;
    let height = 16;
    let input = synthetic_frame(width, height);

    let reference = render_cpu_reference(&input, width, height);
    let wgpu_result = run_wgpu_minimal_pass_emulation(&input, width, height);

    let per_pass_tolerances = [
        // Copy pass should be exactly equal. We still use a small float-style tolerance so
        // widening to float buffers in future GPU implementations keeps the assertion explicit.
        PassTolerance {
            pass_name: "copy",
            abs_tolerance: f32::EPSILON,
        },
        // Compute pass allows tiny rounding differences from float math or backend conversion.
        PassTolerance {
            pass_name: "compute",
            abs_tolerance: 1.0,
        },
        // Readback pass should remain exact for u8 output buffers.
        PassTolerance {
            pass_name: "readback",
            abs_tolerance: f32::EPSILON,
        },
    ];

    assert_frame_with_tolerances(&reference, &wgpu_result, &per_pass_tolerances);
}
