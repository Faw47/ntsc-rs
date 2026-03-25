@group(0) @binding(0) var<storage, read_write> y_plane: array<f32>;
@group(0) @binding(1) var<storage, read_write> i_plane: array<f32>;
@group(0) @binding(2) var<storage, read_write> q_plane: array<f32>;

struct Params {
    width: u32,
    frame_num: u32,
    phase_shift: u32,
    phase_offset: i32,

    filter_mode: u32,
    chroma_delay_horizontal: f32,
    chroma_delay_vertical: i32,
    seed: u32,

    composite_noise_frequency: f32,
    composite_noise_intensity: f32,
    composite_noise_detail: u32,
    luma_frequency: f32,

    luma_intensity: f32,
    luma_detail: u32,
    chroma_frequency: f32,
    chroma_intensity: f32,

    chroma_detail: u32,
    do_luma: u32,
    do_chroma: u32,
    _pad: u32,
}
@group(1) @binding(0) var<uniform> params: Params;

// Simple PCG random number generator
fn pcg_hash(input: u32) -> u32 {
    var state = input * 747796405u + 2891336453u;
    let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

fn rand_f32(state: ptr<function, u32>) -> f32 {
    *state = pcg_hash(*state);
    return f32(*state) / f32(0xffffffffu);
}

// 1D Simplex noise
fn permute(x: f32) -> f32 {
    return ((x * 34.0) + 1.0) * x - floor(((x * 34.0) + 1.0) * x / 289.0) * 289.0;
}

fn taylorInvSqrt(r: f32) -> f32 {
    return 1.79284291400159 - 0.85373472095314 * r;
}

fn simplex_1d(x: f32, seed: f32) -> f32 {
    let i0 = floor(x);
    let i1 = i0 + 1.0;
    let x0 = x - i0;
    let x1 = x0 - 1.0;

    var p0 = permute(i0 + seed);
    var p1 = permute(i1 + seed);

    var p = vec2<f32>(p0, p1);

    let fx = fract(p * (1.0 / 41.0)) * 2.0 - 1.0;
    let gy = abs(fx) - 0.5;
    let gx = fx - floor(fx + 0.5);

    var norm = vec2<f32>(taylorInvSqrt(gx.x * gx.x + gy.x * gy.x),
                         taylorInvSqrt(gx.y * gx.y + gy.y * gy.y));
    var g0 = vec2<f32>(gx.x, gy.x) * norm.x;
    var g1 = vec2<f32>(gx.y, gy.y) * norm.y;

    var w = vec2<f32>(1.0 - x0 * x0, 1.0 - x1 * x1);
    w = max(w, vec2<f32>(0.0));
    var w2 = w * w;
    var w4 = w2 * w2;

    var n = vec2<f32>(dot(g0, vec2<f32>(x0, 0.0)),
                      dot(g1, vec2<f32>(x1, 0.0)));

    return dot(w4, n) * 109.0;
}

fn fbm_1d(x: f32, octaves: u32, seed: f32) -> f32 {
    var v = 0.0;
    var a = 0.5;
    var shift = 100.0;
    var cur_x = x;
    for (var i = 0u; i < octaves; i++) {
        v += a * simplex_1d(cur_x, seed);
        cur_x = cur_x * 2.0 + shift;
        a *= 0.5;
    }
    return v;
}

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= arrayLength(&y_plane)) {
        return;
    }

    let width = params.width;
    let row_idx = index / width;
    let col_idx = index % width;

    if (params.do_luma != 0u) {
        var rng_state = pcg_hash(params.seed + row_idx * 3u);
        let noise_seed = f32(rng_state) / f32(0xffffffffu) * 289.0;
        rng_state = pcg_hash(rng_state);
        let offset = rand_f32(&rng_state) * f32(width);

        let x = (f32(col_idx) + offset) * params.luma_frequency;
        let noise = fbm_1d(x, params.luma_detail, noise_seed);
        y_plane[index] += noise * params.luma_intensity * 0.25;
    }

    if (params.do_chroma != 0u) {
        var rng_state_i = pcg_hash(params.seed + row_idx * 3u + 1u);
        let noise_seed_i = f32(rng_state_i) / f32(0xffffffffu) * 289.0;
        rng_state_i = pcg_hash(rng_state_i);
        let offset_i = rand_f32(&rng_state_i) * f32(width);

        var rng_state_q = pcg_hash(params.seed + row_idx * 3u + 2u);
        let noise_seed_q = f32(rng_state_q) / f32(0xffffffffu) * 289.0;
        rng_state_q = pcg_hash(rng_state_q);
        let offset_q = rand_f32(&rng_state_q) * f32(width);

        let x_i = (f32(col_idx) + offset_i) * params.chroma_frequency;
        let noise_i = fbm_1d(x_i, params.chroma_detail, noise_seed_i);
        i_plane[index] += noise_i * params.chroma_intensity * 0.25;

        let x_q = (f32(col_idx) + offset_q) * params.chroma_frequency;
        let noise_q = fbm_1d(x_q, params.chroma_detail, noise_seed_q);
        q_plane[index] += noise_q * params.chroma_intensity * 0.25;
    }
}
