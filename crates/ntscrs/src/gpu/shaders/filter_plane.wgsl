@group(0) @binding(0) var<storage, read_write> y_plane: array<f32>;

struct Params {
    width: u32,
    frame_num: u32,
    _pad1: u32,
    _pad2: u32,
}
@group(1) @binding(0) var<uniform> params: Params;

// Note: IIR recursive filters are hard to parallelize per-pixel because of dependencies.
// However, we can trivially parallelize across rows since rows are independent.
// Workgroup size is 1 per thread representing a row to keep logic simple without cross thread communication
// Or we map threads to rows: 64 threads, 1 row per thread.

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let row_idx = global_id.x;
    let width = params.width;
    let num_rows = arrayLength(&y_plane) / width;

    if (row_idx >= num_rows) {
        return;
    }

    // Example simplified filter_plane (Butterworth or constant k placeholder)
    // We iterate sequentially over the row.
    var prev1 = 0.0;
    var prev2 = 0.0;

    // In actual implementation we'd pass in numerator and denominator coefficients
    let row_start = row_idx * width;

    for (var i = 0u; i < width; i++) {
        let idx = row_start + i;
        let val = y_plane[idx];

        // do IIR calculation...
        // y_plane[idx] = ...
    }
}
