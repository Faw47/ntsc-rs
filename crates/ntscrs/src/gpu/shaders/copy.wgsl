@group(0) @binding(0) var<storage, read_write> y_buffer: array<f32>;
@group(0) @binding(1) var<storage, read_write> i_buffer: array<f32>;
@group(0) @binding(2) var<storage, read_write> q_buffer: array<f32>;
@group(0) @binding(3) var<storage, read_write> scratch_buffer: array<f32>;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index < arrayLength(&y_buffer)) {
        // Just a no-op / identity operation to prove the pipeline executes
        y_buffer[index] = y_buffer[index];
    }
}
