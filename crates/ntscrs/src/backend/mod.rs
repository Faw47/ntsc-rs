#[cfg(feature = "wgpu-backend")]
pub mod wgpu;
mod cpu;
#[cfg(feature = "wgpu")]
mod pipeline;
mod types;
#[cfg(feature = "wgpu")]
mod wgpu;

use crate::{settings::standard::NtscEffect, yiq_fielding::YiqView};

use self::cpu::CpuBackend;
#[cfg(feature = "wgpu")]
use self::wgpu::WgpuBackend;

pub use self::types::{
    BackendCapabilities, BackendInitError, BackendKind, BackendRunError, FrameDesc, FrameFormat,
    PlaneLayout,
};

#[derive(Debug)]
pub struct Backend {
    kind: BackendKind,
    cpu: CpuBackend,
    #[cfg(feature = "wgpu")]
    wgpu: Option<WgpuBackend>,
}

impl Backend {
    pub fn new(kind: BackendKind) -> Result<Self, BackendInitError> {
        let cpu = CpuBackend::new();
        match kind {
            BackendKind::Cpu => Ok(Self {
                kind,
                cpu,
                #[cfg(feature = "wgpu")]
                wgpu: None,
            }),
            #[cfg(feature = "wgpu")]
            BackendKind::Wgpu => Ok(Self {
                kind,
                cpu,
                wgpu: Some(WgpuBackend::new()),
            }),
            #[allow(unreachable_patterns)]
            _ => Err(BackendInitError::UnsupportedBackend(kind)),
        }
    }

    pub fn kind(&self) -> BackendKind {
        self.kind
    }

    pub fn capabilities(&self) -> BackendCapabilities {
        match self.kind {
            BackendKind::Cpu => CpuBackend::capabilities(),
            #[cfg(feature = "wgpu")]
            BackendKind::Wgpu => WgpuBackend::capabilities(),
        }
    }

    pub fn run(
        &self,
        effect: &NtscEffect,
        yiq: &mut YiqView,
        frame_num: usize,
        scale_factor: [f32; 2],
        frame_desc: FrameDesc,
    ) -> Result<(), BackendRunError> {
        match self.kind {
            BackendKind::Cpu => self
                .cpu
                .run(effect, yiq, frame_num, scale_factor, frame_desc),
            #[cfg(feature = "wgpu")]
            BackendKind::Wgpu => self
                .wgpu
                .as_ref()
                .ok_or(BackendRunError::PipelineNotInitialized)?
                .run(effect, yiq, frame_num, scale_factor, frame_desc),
        }
    }
}

pub fn default_backend() -> Backend {
    Backend::new(BackendKind::Cpu).expect("the CPU backend must always be available")
}

pub fn available_backends() -> &'static [BackendKind] {
    #[cfg(feature = "wgpu")]
    {
        &[BackendKind::Cpu, BackendKind::Wgpu]
    }
    #[cfg(not(feature = "wgpu"))]
    {
        &[BackendKind::Cpu]
    }
}
