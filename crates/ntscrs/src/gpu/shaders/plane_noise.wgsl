@group(0) @binding(0) var<storage, read_write> target_plane: array<f32>;

struct ShaderParams {
    width: u32,
    frame_num: u32,
    seed: u32,
    noise_idx: u32,
    frequency: f32, // pre-divided by horizontal_scale
    intensity: f32, // pre-multiplied by its scale factor
    detail: u32,
    _pad: u32,
}
@group(1) @binding(0) var<uniform> params: ShaderParams;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let row_idx = global_id.x;
    let width = params.width;
    let height = arrayLength(&target_plane) / width;

    if (row_idx >= height) {
        return;
    }

    let base_h = mix_hash(vec2<u32>(0u, params.seed), vec2<u32>(0u, params.noise_idx));
    let frame_h = mix_hash(base_h, vec2<u32>(0u, params.frame_num));
    let row_h = mix_hash(frame_h, vec2<u32>(0u, row_idx));

    var rng = xoshiro256_seed(row_h);
    let noise_seed = i32(xoshiro256_next_u32(&rng));
    let offset = xoshiro256_next_f32_range(&rng, 0.0, f32(width));

    let row_start = row_idx * width;
    for (var i = 0u; i < width; i++) {
        let point = f32(i) + offset;
        let noise_val = fbm_1d(noise_seed, params.detail, 1.0, 2.0, params.frequency, point);
        target_plane[row_start + i] += noise_val * params.intensity;
    }
}
