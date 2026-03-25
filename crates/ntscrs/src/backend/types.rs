use std::fmt;

use crate::yiq_fielding::YiqField;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FrameFormat {
    YiqPlanarF32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PlaneLayout {
    ContiguousPlanar,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FrameDesc {
    pub dimensions: (usize, usize),
    pub field: YiqField,
    pub format: FrameFormat,
    pub plane_layout: PlaneLayout,
}

impl FrameDesc {
    pub fn yiq_planar_f32(dimensions: (usize, usize), field: YiqField) -> Self {
        Self {
            dimensions,
            field,
            format: FrameFormat::YiqPlanarF32,
            plane_layout: PlaneLayout::ContiguousPlanar,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BackendKind {
    Cpu,
    #[cfg(feature = "wgpu")]
    Wgpu,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BackendCapabilities {
    pub kind: BackendKind,
    pub supports_frame_format: FrameFormat,
    pub supports_plane_layout: PlaneLayout,
}

#[derive(Debug)]
pub enum BackendInitError {
    UnsupportedBackend(BackendKind),
    #[cfg(feature = "wgpu")]
    DeviceUnavailable,
}

impl fmt::Display for BackendInitError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedBackend(kind) => write!(f, "unsupported backend: {kind:?}"),
            #[cfg(feature = "wgpu")]
            Self::DeviceUnavailable => write!(f, "unable to initialize wgpu device"),
        }
    }
}

impl std::error::Error for BackendInitError {}

#[derive(Debug)]
pub enum BackendRunError {
    UnsupportedFrameFormat(FrameFormat),
    UnsupportedPlaneLayout(PlaneLayout),
    InvalidDimensions((usize, usize)),
    #[cfg(feature = "wgpu")]
    PipelineNotInitialized,
}

impl fmt::Display for BackendRunError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedFrameFormat(format) => {
                write!(f, "unsupported frame format: {format:?}")
            }
            Self::UnsupportedPlaneLayout(layout) => {
                write!(f, "unsupported plane layout: {layout:?}")
            }
            Self::InvalidDimensions(dimensions) => {
                write!(f, "invalid frame dimensions: {dimensions:?}")
            }
            #[cfg(feature = "wgpu")]
            Self::PipelineNotInitialized => write!(f, "wgpu pipeline not initialized"),
        }
    }
}

impl std::error::Error for BackendRunError {}
