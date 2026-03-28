@group(0) @binding(0) var<storage, read_write> target_plane: array<f32>;
@group(0) @binding(1) var<storage, read_write> scratch_plane: array<f32>;

struct Params {
    width: u32,
    frame_num: u32,
    seed: u32,
    noise_idx: u32,
    
    noise_frequency: f32, // Re-used for edge wave speed
    noise_intensity: f32,
    noise_detail: u32,
    snow_anisotropy: f32, // Reused for head_switching mid_line.jitter

    phase_shift: u32,     // 0 = edge wave, 1 = tracking, 2 = head_switching
    phase_offset: i32,    // affected_rows start index
    filter_mode: u32,     // affected_rows total
    chroma_delay_horizontal: f32, // used for head_switch shift or tracking wave_intensity

    chroma_delay_vertical: i32,
    horizontal_scale: f32,
    _pad1: u32,
    _pad2: u32,
}
@group(1) @binding(0) var<uniform> params: Params;

fn fract(x: f32) -> f32 { return x - floor(x); }

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let row_idx = global_id.x;
    let width = params.width;
    let height = arrayLength(&target_plane) / width;
    
    if (row_idx >= height) {
        return;
    }

    let start_row = u32(max(0, params.phase_offset));
    let num_rows = params.filter_mode;

    // Check if row is in affected range (except for edge wave which affects all rows)
    let is_edge_wave = params.phase_shift == 0u;
    if (!is_edge_wave && (row_idx < start_row || row_idx >= start_row + num_rows)) {
        return;
    }

    var shift = 0.0;
    
    if (params.phase_shift == 0u) { // vhs_edge_wave
        let base_h = mix_hash(vec2<u32>(0u, params.seed), vec2<u32>(0u, 5u)); // EDGE_WAVE
        var rng = xoshiro256_seed(base_h);
        let noise_seed = i32(xoshiro256_next_u32(&rng));
        let offset = xoshiro256_next_f32(&rng) * f32(height);
        
        let p = vec2<f32>(offset + f32(row_idx), f32(params.frame_num) * params.noise_frequency);
        let noise_val = fbm_2d(noise_seed, params.noise_detail, 0.70710678, 2.0, params.noise_frequency, p);
        
        shift = (noise_val / 0.022) * params.noise_intensity * 0.5 * params.horizontal_scale;
        
    } else if (params.phase_shift == 1u) { // tracking_noise
        let base_h = mix_hash(vec2<u32>(0u, params.seed), vec2<u32>(0u, 3u)); // TRACKING_NOISE
        let frame_h = mix_hash(base_h, vec2<u32>(0u, params.frame_num));
        var rng = xoshiro256_seed(frame_h);
        
        let noise_seed = i32(xoshiro256_next_u32(&rng));
        let offset = xoshiro256_next_f32(&rng) * f32(height);
        
        let idx = row_idx - start_row;
        let p = offset + f32(idx);
        let noise_val = simplex_1d(p * 0.5, noise_seed); // frequency = 0.5
        
        let intensity_scale = f32(idx) / f32(num_rows);
        shift = noise_val * intensity_scale * params.chroma_delay_horizontal * 0.25 * params.horizontal_scale;
        
    } else if (params.phase_shift == 2u) { // head_switching
        let base_h = mix_hash(vec2<u32>(0u, params.seed), vec2<u32>(0u, 2u)); // HEAD_SWITCHING
        let frame_h = mix_hash(base_h, vec2<u32>(0u, params.frame_num));
        
        let idx = row_idx - start_row; // note: reverse indexed from bottom in ntsc_rs
        let reversed_idx = num_rows - 1u - idx;
        
        let row_shift = params.chroma_delay_horizontal * pow((f32(reversed_idx) + params.noise_frequency) / f32(num_rows), 1.5);
        
        var rng = xoshiro256_seed(mix_hash(frame_h, vec2<u32>(0u, reversed_idx)));
        let noisy_mod = xoshiro256_next_f32(&rng) - 0.5;
        
        shift = (row_shift + noisy_mod) * params.horizontal_scale;
    }

    if (abs(shift) < 0.001) { return; }

    let row_start = row_idx * width;
    
    // Perform shifted copy via scratch buffer to prevent overwriting during copy
    for (var i = 0u; i < width; i++) {
        scratch_plane[row_start + i] = target_plane[row_start + i];
    }
    
    let shift_int = i32(floor(shift));
    let shift_frac = shift - f32(shift_int);
    
    for (var i = 0u; i < width; i++) {
        let dst_i = i;
        // left = src[i - shift_int - 1]
        let left_idx = i32(dst_i) - shift_int - 1;
        let right_idx = left_idx + 1;
        
        var left_val = 0.0;
        var right_val = 0.0;
        
        // Constant Boundary Handling (0.0) -> like the CPU implementation for these shifts
        if (params.phase_shift == 0u) {
            // Edge wave actually uses extend boundary in CPU ? No, CPU: `BoundaryHandling::Constant(0.0)`
        }
        
        if (left_idx >= 0 && left_idx < i32(width)) {
            left_val = scratch_plane[row_start + u32(left_idx)];
        }
        if (right_idx >= 0 && right_idx < i32(width)) {
            right_val = scratch_plane[row_start + u32(right_idx)];
        }
        
        target_plane[row_start + dst_i] = left_val * shift_frac + right_val * (1.0 - shift_frac);
    }
}
