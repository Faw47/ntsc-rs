@group(0) @binding(0) var<storage, read_write> plane_to_filter: array<f32>;

// Ensure stride aligns to 16 bytes for uniform array
struct FilterParams {
    width: u32,
    frame_num: u32,
    num_len: u32,
    den_len: u32,
    // Using vec4<f32> to satisfy the 16-byte alignment requirement for array elements
    num: array<vec4<f32>, 16>,
    den: array<vec4<f32>, 16>,
    initial_condition: f32,
    initial_type: u32,
    delay: u32,
    _pad: u32,
}
@group(1) @binding(0) var<uniform> params: FilterParams;

// Direct Form II Transposed IIR filter. Matches filter_signal_in_place behavior on CPU.
@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let row_idx = global_id.x;
    let width = params.width;
    let num_rows = arrayLength(&plane_to_filter) / width;

    if (row_idx >= num_rows) {
        return;
    }

    let row_start = row_idx * width;

    // Determine initial condition
    var initial = 0.0;
    if (params.initial_type == 1u) {
        initial = params.initial_condition;
    } else if (params.initial_type == 2u) {
        initial = plane_to_filter[row_start];
    }

    // State array for Direct Form II Transposed
    var z = array<f32, 16>();

    // Pre-roll the filter state to match steady-state behavior with initial condition
    let steady_state_len = max(params.num_len, params.den_len);
    if (initial != 0.0 && steady_state_len > 0u) {
        // Simple approximation: feed the initial value in a few times
        // The real CPU code mathematically pre-computes steady state.
        // Here we just pump the initial condition 32 times.
        for (var k = 0u; k < 32u; k++) {
            let x = initial;
            let y = params.num[0].x * x + z[0];

            for (var i = 0u; i < steady_state_len - 1u; i++) {
                z[i] = params.num[i + 1u].x * x + z[i + 1u];
                if (i + 1u < params.den_len) {
                    z[i] += params.den[i + 1u].x * y;
                }
            }
        }
    }

    // Apply filter over row
    for (var i = 0u; i < width + params.delay; i++) {
        var x = initial;
        let read_idx = row_start + i;
        if (i < width) {
            x = plane_to_filter[read_idx];
        }

        let y = params.num[0].x * x + z[0];

        for (var j = 0u; j < steady_state_len - 1u; j++) {
            z[j] = params.num[j + 1u].x * x + z[j + 1u];
            if (j + 1u < params.den_len) {
                z[j] += params.den[j + 1u].x * y;
            }
        }

        // Output value with delay
        if (i >= params.delay) {
            let write_idx = row_start + (i - params.delay);
            plane_to_filter[write_idx] = y;
        }
    }
}
