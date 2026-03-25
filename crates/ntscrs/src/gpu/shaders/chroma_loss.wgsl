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
    luma_noise_frequency: f32,

    luma_noise_intensity: f32,
    luma_noise_detail: u32,
    chroma_noise_frequency: f32,
    chroma_noise_intensity: f32,

    chroma_noise_detail: u32,
    do_luma_noise: u32,
    do_chroma_noise: u32,
    chroma_loss_intensity: f32,
}
@group(1) @binding(0) var<uniform> params: Params;

fn pcg_hash(input: u32) -> u32 {
    var state = input * 747796405u + 2891336453u;
    let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

fn rand_f32(state: ptr<function, u32>) -> f32 {
    *state = pcg_hash(*state);
    return f32(*state) / f32(0xffffffffu);
}

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let row_idx = global_id.x;
    let width = params.width;
    let num_rows = arrayLength(&i_plane) / width;

    if (row_idx >= num_rows) {
        return;
    }

    // Seed is constructed per row
    var rng_state = pcg_hash(params.seed + row_idx);
    let sample = rand_f32(&rng_state);

    // Geometric distribution logic applied simply using independent rand checks
    if (sample < params.chroma_loss_intensity) {
        let row_start = row_idx * width;
        for (var i = 0u; i < width; i++) {
            let index = row_start + i;
            i_plane[index] = 0.0;
            q_plane[index] = 0.0;
        }
    }
}
