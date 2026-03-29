@group(0) @binding(0) var<storage, read_write> y_plane: array<f32>;
@group(0) @binding(1) var<storage, read_write> i_plane: array<f32>;
@group(0) @binding(2) var<storage, read_write> q_plane: array<f32>;

struct ShaderParams {
    width: u32,
    frame_num: u32,
    seed_hi: u32,
    seed_lo: u32,

    noise_frequency: f32,
    noise_intensity: f32,
    noise_detail: u32,
    snow_anisotropy: f32,

    phase_shift: u32,
    phase_offset: i32,
    filter_mode: u32,
    chroma_delay_horizontal: f32,

    chroma_delay_vertical: i32,
    horizontal_scale: f32,
    vertical_scale: f32,
    mid_line_position: f32,

    mid_line_enabled: u32,
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
}
@group(1) @binding(0) var<uniform> params: ShaderParams;

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

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let col_idx = global_id.x;
    let row_idx = global_id.y;
    let width = params.width;
    let height = arrayLength(&y_plane) / width;

    if (col_idx >= width || row_idx >= height) {
        return;
    }

    let index = row_idx * width + col_idx;
    let xi = chroma_phase_shift(row_idx * 2u, params.frame_num, params.phase_shift, params.phase_offset);

    let phase = (col_idx + xi) & 3u;

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
