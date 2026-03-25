@group(0) @binding(0) var<storage, read_write> y_plane: array<f32>;
@group(0) @binding(1) var<storage, read> i_plane: array<f32>;
@group(0) @binding(2) var<storage, read> q_plane: array<f32>;

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
    _pad: u32,
}
@group(1) @binding(0) var<uniform> params: Params;

// Compute I_MULT and Q_MULT from phase
fn chroma_phase_shift(line_num: u32, frame_num: u32, phase_shift: u32, phase_offset: i32) -> u32 {
    if (phase_shift == 0u) {
        return 0u;
    } else if (phase_shift == 1u || phase_shift == 3u) {
        return u32((i32(frame_num) + phase_offset + i32(line_num >> 1u)) & 3i);
    } else if (phase_shift == 2u) {
        return u32((i32((frame_num + line_num) & 2u) + phase_offset) & 3i);
    }
    return 0u;
}

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= arrayLength(&y_plane)) {
        return;
    }

    let line_num = index / params.width;
    let xi = chroma_phase_shift(line_num * 2u, params.frame_num, params.phase_shift, params.phase_offset);

    let x = index % params.width;
    let phase = (x + xi) & 3u;

    var i_mult = 0.0;
    var q_mult = 0.0;

    if (phase == 0u) {
        i_mult = 1.0;
        q_mult = 0.0;
    } else if (phase == 1u) {
        i_mult = 0.0;
        q_mult = 1.0;
    } else if (phase == 2u) {
        i_mult = -1.0;
        q_mult = 0.0;
    } else if (phase == 3u) {
        i_mult = 0.0;
        q_mult = -1.0;
    }

    y_plane[index] = y_plane[index] + (i_plane[index] * i_mult) + (q_plane[index] * q_mult);
}
