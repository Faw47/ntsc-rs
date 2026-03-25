@group(0) @binding(0) var<storage, read_write> y_plane: array<f32>;
@group(0) @binding(1) var<storage, read_write> i_plane: array<f32>;
@group(0) @binding(2) var<storage, read_write> q_plane: array<f32>;
@group(0) @binding(3) var<storage, read_write> scratch_plane: array<f32>;

struct Params {
    width: u32,
    frame_num: u32,
    phase_shift: u32,
    phase_offset: i32,
    filter_mode: u32, // 0: Box, 1: Notch, 2: OneLineComb, 3: TwoLineComb
}
@group(1) @binding(0) var<uniform> params: Params;

// Copied from chroma_into_luma
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

fn demodulate_chroma_line(index: u32, xi: u32, y: f32, modulated: f32, chroma_c: f32) {
    let offset_c = (index + xi) & 3u;
    var i_modulated = chroma_c;
    var q_modulated = chroma_c;

    // I_MULT_INV
    if (offset_c == 0u) {
        i_modulated *= -1.0;
        q_modulated *= 0.0;
    } else if (offset_c == 1u) {
        i_modulated *= 0.0;
        q_modulated *= -1.0;
    } else if (offset_c == 2u) {
        i_modulated *= 1.0;
        q_modulated *= 0.0;
    } else if (offset_c == 3u) {
        i_modulated *= 0.0;
        q_modulated *= 1.0;
    }

    let width = params.width;

    if (index % width < width - 1u) {
        let offset_r = (index + 1u + xi) & 3u;
        let chroma_r = y_plane[index + 1u] - scratch_plane[index + 1u];

        var i_r = chroma_r * 0.5;
        var q_r = chroma_r * 0.5;

        if (offset_r == 0u) { i_r *= -1.0; q_r *= 0.0; }
        else if (offset_r == 1u) { i_r *= 0.0; q_r *= -1.0; }
        else if (offset_r == 2u) { i_r *= 1.0; q_r *= 0.0; }
        else if (offset_r == 3u) { i_r *= 0.0; q_r *= 1.0; }

        i_modulated += i_r;
        q_modulated += q_r;
    }

    if (index % width > 0u) {
        let offset_l = (index - 1u + xi) & 3u;
        let chroma_l = y_plane[index - 1u] - scratch_plane[index - 1u];

        var i_l = chroma_l * 0.5;
        var q_l = chroma_l * 0.5;

        if (offset_l == 0u) { i_l *= -1.0; q_l *= 0.0; }
        else if (offset_l == 1u) { i_l *= 0.0; q_l *= -1.0; }
        else if (offset_l == 2u) { i_l *= 1.0; q_l *= 0.0; }
        else if (offset_l == 3u) { i_l *= 0.0; q_l *= 1.0; }

        i_modulated += i_l;
        q_modulated += q_l;
    }

    i_plane[index] = i_modulated;
    q_plane[index] = q_modulated;
}

@compute @workgroup_size(64, 1, 1)
fn demodulate_box(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= arrayLength(&y_plane)) {
        return;
    }

    let width = params.width;
    let line_num = index / width;
    let xi = chroma_phase_shift(line_num * 2u, params.frame_num, params.phase_shift, params.phase_offset);

    // Read from scratch which holds original modulated
    let cur_mod = scratch_plane[index];

    var m_left = 16.0 / 255.0;
    if (index % width > 0u) {
        m_left = scratch_plane[index - 1u];
    }

    var m_right1 = cur_mod;
    if (index % width < width - 1u) {
        m_right1 = scratch_plane[index + 1u];
    }

    var m_right2 = cur_mod;
    if (index % width < width - 2u) {
        m_right2 = scratch_plane[index + 2u];
    } else if (index % width < width - 1u) {
        m_right2 = scratch_plane[index + 1u];
    }

    let y = (m_left + cur_mod + m_right1 + m_right2) * 0.25;
    y_plane[index] = y;

    let chroma_c = y - cur_mod;

    demodulate_chroma_line(index, xi, y, cur_mod, chroma_c);
}
