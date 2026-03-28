// 64-bit unsigned integer represented as vec2<u32>(high, low)
fn u64_add(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
    let lower = a.y + b.y;
    let carry = u32(lower < a.y);
    let upper = a.x + b.x + carry;
    return vec2<u32>(upper, lower);
}

fn u64_xor(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
    return vec2<u32>(a.x ^ b.x, a.y ^ b.y);
}

fn u64_shl_17(x: vec2<u32>) -> vec2<u32> {
    return vec2<u32>((x.x << 17u) | (x.y >> 15u), x.y << 17u);
}

fn u64_shr(x: vec2<u32>, v: u32) -> vec2<u32> {
    if (v == 0u) { return x; }
    if (v < 32u) {
        return vec2<u32>(x.x >> v, (x.y >> v) | (x.x << (32u - v)));
    } else if (v == 32u) {
        return vec2<u32>(0u, x.x);
    } else {
        return vec2<u32>(0u, x.x >> (v - 32u));
    }
}

fn u64_rotl_23(x: vec2<u32>) -> vec2<u32> {
    let s_left = vec2<u32>((x.x << 23u) | (x.y >> 9u), x.y << 23u);
    let s_right = vec2<u32>(0u, x.x >> 9u); // Since 64 - 23 = 41 (>= 32) -> logic for rshift 41
    return vec2<u32>(s_left.x | s_right.x, s_left.y | s_right.y);
}

fn u64_rotl_45(x: vec2<u32>) -> vec2<u32> {
    let s_left = vec2<u32>(x.y << 13u, 0u); // Since 45 >= 32
    let s_right = vec2<u32>(x.x >> 19u, (x.y >> 19u) | (x.x << 13u)); // Since 64 - 45 = 19 (< 32)
    return vec2<u32>(s_left.x | s_right.x, s_left.y | s_right.y);
}

fn u64_mul(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
    let a_lo = a.y & 0xFFFFu;
    let a_hi = a.y >> 16u;
    let b_lo = b.y & 0xFFFFu;
    let b_hi = b.y >> 16u;

    let lo_lo = a_lo * b_lo;
    let hi_lo = a_hi * b_lo;
    let lo_hi = a_lo * b_hi;
    let hi_hi = a_hi * b_hi;

    let cross = (lo_lo >> 16u) + (hi_lo & 0xFFFFu) + lo_hi;
    let lower = (lo_lo & 0xFFFFu) | ((cross & 0xFFFFu) << 16u);
    let upper_carry = (cross >> 16u) + (hi_lo >> 16u) + hi_hi;

    let upper = a.x * b.y + a.y * b.x + upper_carry;
    return vec2<u32>(upper, lower);
}

struct SplitMix64 {
    state: vec2<u32>,
}

fn splitmix64_next(rng: ptr<function, SplitMix64>) -> vec2<u32> {
    (*rng).state = u64_add((*rng).state, vec2<u32>(0x9E3779B9u, 0x7F4A7C15u));
    var z = (*rng).state;
    z = u64_xor(z, u64_shr(z, 30u));
    z = u64_mul(z, vec2<u32>(0xBF58476Du, 0x1CE4E5B9u));
    z = u64_xor(z, u64_shr(z, 27u));
    z = u64_mul(z, vec2<u32>(0x94D049BBu, 0x133111EBu));
    z = u64_xor(z, u64_shr(z, 31u));
    return z;
}

struct Xoshiro256PlusPlus {
    s: array<vec2<u32>, 4>,
}

fn xoshiro256_seed(seed: vec2<u32>) -> Xoshiro256PlusPlus {
    var sm2 = SplitMix64(seed);
    var rng: Xoshiro256PlusPlus;
    rng.s[0] = splitmix64_next(&sm2);
    rng.s[1] = splitmix64_next(&sm2);
    rng.s[2] = splitmix64_next(&sm2);
    rng.s[3] = splitmix64_next(&sm2);
    return rng;
}

fn xoshiro256_next_u64(rng: ptr<function, Xoshiro256PlusPlus>) -> vec2<u32> {
    let result = u64_add(u64_rotl_23(u64_add((*rng).s[0], (*rng).s[3])), (*rng).s[0]);
    let t = u64_shl_17((*rng).s[1]);
    (*rng).s[2] = u64_xor((*rng).s[2], (*rng).s[0]);
    (*rng).s[3] = u64_xor((*rng).s[3], (*rng).s[1]);
    (*rng).s[1] = u64_xor((*rng).s[1], (*rng).s[2]);
    (*rng).s[0] = u64_xor((*rng).s[0], (*rng).s[3]);
    (*rng).s[2] = u64_xor((*rng).s[2], t);
    (*rng).s[3] = u64_rotl_45((*rng).s[3]);
    return result;
}

fn xoshiro256_next_u32(rng: ptr<function, Xoshiro256PlusPlus>) -> u32 {
    let val = xoshiro256_next_u64(rng);
    return val.y; // return lower 32 bits
}

fn xoshiro256_next_f32(rng: ptr<function, Xoshiro256PlusPlus>) -> f32 {
    let u = xoshiro256_next_u32(rng);
    // Float representation of (u >> 8) | 0x3f800000 - 1.0
    return bitcast<f32>((u >> 9u) | 0x3F800000u) - 1.0;
}

fn xoshiro256_next_f32_range(rng: ptr<function, Xoshiro256PlusPlus>, min_val: f32, max_val: f32) -> f32 {
    let v = xoshiro256_next_f32(rng);
    return min_val + v * (max_val - min_val);
}

// Geometric distribution simulator.
// Since WGSL doesn't have native log(), we use the WGSL log function.
// Expected probability p.
fn geometric_sample(rng: ptr<function, Xoshiro256PlusPlus>, p: f32) -> u32 {
    let lambda = log(1.0 - p);
    let rand_val = xoshiro256_next_f32(rng);
    let sample_val = log(rand_val) / lambda;
    return u32(floor(sample_val));
}

fn mix_hash(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
    // Equivalent of Seeder wrapping logic found in `ntsc.rs`
    var h = u64_add(a, b);
    h = u64_xor(h, u64_shr(h, 33u));
    h = u64_mul(h, vec2<u32>(0xFF51AFD7u, 0xED558CCDu));
    h = u64_xor(h, u64_shr(h, 33u));
    h = u64_mul(h, vec2<u32>(0xC4CEB9FEu, 0x1A85EC53u));
    h = u64_xor(h, u64_shr(h, 33u));
    return h;
}
