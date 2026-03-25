mod backend;
mod filter;
mod noise;
mod ntsc;
mod random;
pub mod settings;
mod shift;
mod thread_pool;
pub mod yiq_fielding;

use std::str::FromStr;

pub use settings::standard::{NtscEffect, NtscEffectFullSettings};
use yiq_fielding::YiqView;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum BackendPreference {
    #[default]
    Auto,
    Cpu,
    Wgpu,
    Cuda,
}

impl BackendPreference {
    pub fn from_env_var(var_name: &str) -> Option<Self> {
        let value = std::env::var(var_name).ok()?;
        Self::from_str(value.trim()).ok()
    }
}

impl FromStr for BackendPreference {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim().to_ascii_lowercase().as_str() {
            "auto" => Ok(Self::Auto),
            "cpu" => Ok(Self::Cpu),
            "wgpu" => Ok(Self::Wgpu),
            "cuda" => Ok(Self::Cuda),
            _ => Err(()),
        }
    }
}

pub fn apply_effect_to_yiq_with_backend_preference(
    effect: &NtscEffect,
    yiq: &mut YiqView,
    frame_num: usize,
    scale_factor: [f32; 2],
    _backend_preference: BackendPreference,
) {
    // Phase 1 fallback: keep existing behavior for all backend preferences.
    effect.apply_effect_to_yiq(yiq, frame_num, scale_factor);
}
