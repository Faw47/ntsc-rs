@group(0) @binding(0) var<storage, read_write> y_plane: array<f32>;
@group(0) @binding(1) var<storage, read_write> i_plane: array<f32>;
@group(0) @binding(2) var<storage, read_write> q_plane: array<f32>;
@group(0) @binding(3) var<storage, read_write> scratch_plane: array<f32>;

struct Params {
    width: u32,
    frame_num: u32,
    phase_shift: u32,
    phase_offset: i32,
    filter_mode: u32,
    chroma_delay_horizontal: f32,
    chroma_delay_vertical: i32,
    _pad: u32,
}
@group(1) @binding(0) var<uniform> params: Params;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= arrayLength(&i_plane)) {
        return;
    }

    let width = params.width;
    let x = index % width;
    let y = index / width;
    let height = arrayLength(&i_plane) / width;

    // Apply vertical offset
    let dst_y = i32(y) - params.chroma_delay_vertical;
    if (dst_y < 0 || dst_y >= i32(height)) {
        i_plane[index] = 0.0;
        q_plane[index] = 0.0;
        return;
    }

    // Apply horizontal offset
    // This is continuous so we need to interpolate
    let src_x_float = f32(x) - params.chroma_delay_horizontal;

    // Nearest neighbor or basic linear interp for shift
    let src_x_left = i32(floor(src_x_float));
    let src_x_right = src_x_left + 1;
    let fract_x = src_x_float - f32(src_x_left);

    var i_val = 0.0;
    var q_val = 0.0;

    // Left sample
    if (src_x_left >= 0 && src_x_left < i32(width)) {
        let left_idx = u32(dst_y) * width + u32(src_x_left);
        i_val += scratch_plane[left_idx] * (1.0 - fract_x);
        q_val += scratch_plane[left_idx + arrayLength(&i_plane)] * (1.0 - fract_x);
        // Note: For simplicity and buffer passing we use scratch plane here assuming it holds
        // original I/Q data packed, or we use separate buffers. We will update the rust code to pass I and Q to scratch buffer
        // Let's assume we do this in multiple passes or we ping-pong.
        // For now, let's keep it simple: scratch buffer has I in first half, Q in second half.
    }

    // Right sample
    if (src_x_right >= 0 && src_x_right < i32(width)) {
        let right_idx = u32(dst_y) * width + u32(src_x_right);
        i_val += scratch_plane[right_idx] * fract_x;
        q_val += scratch_plane[right_idx + arrayLength(&i_plane)] * fract_x;
    }

    i_plane[index] = i_val;
    q_plane[index] = q_val;
}
