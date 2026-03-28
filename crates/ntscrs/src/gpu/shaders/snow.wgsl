@group(0) @binding(0) var<storage, read_write> target_plane: array<f32>;

struct ShaderParams {
    width: u32,
    frame_num: u32,
    seed: u32,
    intensity: f32,
    anisotropy: f32,
    horizontal_scale: f32,
    _pad1: u32,
    _pad2: u32,
}
@group(1) @binding(0) var<uniform> params: ShaderParams;

const PI: f32 = 3.14159265359;
const SNOW_NOISE_INDEX: u32 = 6u;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let row_idx = global_id.x;
    let width = params.width;
    let height = arrayLength(&target_plane) / width;

    if (row_idx >= height) {
        return;
    }

    let base_h = mix_hash(vec2<u32>(0u, params.seed), vec2<u32>(0u, SNOW_NOISE_INDEX));
    let frame_h = mix_hash(base_h, vec2<u32>(0u, params.frame_num));
    let row_h = mix_hash(frame_h, vec2<u32>(0u, row_idx));

    var rng = xoshiro256_seed(row_h);

    let intensity = params.intensity;
    let anisotropy = params.anisotropy;

    let logistic_factor = exp((xoshiro256_next_f32(&rng) - intensity) / (intensity * (1.0 - intensity) * (1.0 - anisotropy)));
    var line_snow_intensity = anisotropy / (1.0 + logistic_factor) + intensity * (1.0 - anisotropy);

    line_snow_intensity *= 0.125;
    line_snow_intensity = clamp(line_snow_intensity, 0.0, 1.0);

    if (line_snow_intensity <= 0.0) {
        return;
    }

    // TRANSIENT_LEN_RANGE: 8.0..=64.0
    var pixel_idx = -64;
    let row_start = row_idx * width;

    loop {
        let jump = geometric_sample(&rng, line_snow_intensity);
        pixel_idx += i32(min(jump, 2147483647u));

        if (pixel_idx >= i32(width)) {
            break;
        }

        let transient_len = xoshiro256_next_f32_range(&rng, 8.0, 64.0) * params.horizontal_scale;
        let transient_freq = xoshiro256_next_f32_range(&rng, transient_len * 3.0, transient_len * 5.0);
        let pixel_idx_end = pixel_idx + i32(ceil(transient_len));

        // Use pixel_idx to instance a new transient RNG (acting as a perfect stand-in for rng.jump())
        var transient_rng = xoshiro256_seed(mix_hash(row_h, vec2<u32>(0u, bitcast<u32>(pixel_idx))));

        let start_i = max(0, pixel_idx);
        let end_i = min(i32(width), pixel_idx_end);

        for (var i = start_i; i < end_i; i++) {
            let x = f32(i - pixel_idx);
            let pow_val = 1.0 - (x / transient_len);
            let value = cos((x * PI) / transient_freq) * (pow_val * pow_val) * xoshiro256_next_f32_range(&transient_rng, -1.0, 2.0);
            target_plane[row_start + u32(i)] += value;
        }

        pixel_idx += 1;
    }
}
