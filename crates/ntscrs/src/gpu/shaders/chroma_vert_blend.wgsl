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
    _pad: u32,
}
@group(1) @binding(0) var<uniform> params: Params;

// Chroma vertical blend takes the I and Q from the current row and averages it with the row above it.
// This needs to be done sequentially per row, OR ping-ponged. Since it just averages row N with row N-1,
// we can do it in parallel per pixel if we read from the original and write to a new buffer, OR we just
// iterate per row to do it in-place.
// The CPU implementation iterates sequentially top to bottom, meaning row N blends with row N-1 AFTER row N-1
// has already been blended. This makes it an IIR filter across rows!
// Wait, looking at the CPU code:
// yiq.i.chunks_exact_mut(width).for_each(|row| {
//     row.iter_mut().enumerate().for_each(|(index, i)| {
//         let c_i = *i;
//         *i = (delay_i[index] + c_i) * 0.5;
//         delay_i[index] = c_i;
//     });
// });
// Yes, `delay_i` stores the *original* `c_i` from the previous row, NOT the blended one.
// So it is an FIR filter: Out[y, x] = (In[y, x] + In[y-1, x]) * 0.5.
// This means we can parallelize it per pixel, as long as we read In[y-1, x] before overwriting it.
// But to avoid read-after-write hazards in a single pass without a scratch buffer, we need to read from
// a scratch buffer or process sequentially. Let's do sequentially per column since it's just 1 loop.
// So 1 thread per column.

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let col_idx = global_id.x;
    let width = params.width;
    if (col_idx >= width) {
        return;
    }

    let num_rows = arrayLength(&i_plane) / width;

    var delay_i = 0.0;
    var delay_q = 0.0;

    for (var y = 0u; y < num_rows; y++) {
        let index = y * width + col_idx;
        let c_i = i_plane[index];
        let c_q = q_plane[index];

        i_plane[index] = (delay_i + c_i) * 0.5;
        q_plane[index] = (delay_q + c_q) * 0.5;

        delay_i = c_i;
        delay_q = c_q;
    }
}
