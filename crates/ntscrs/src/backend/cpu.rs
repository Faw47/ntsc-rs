use crate::{settings::standard::NtscEffect, yiq_fielding::YiqView};

use super::types::{
    BackendCapabilities, BackendKind, BackendRunError, FrameDesc, FrameFormat, PlaneLayout,
};

#[derive(Debug, Default, Clone, Copy)]
pub(crate) struct CpuBackend;

impl CpuBackend {
    pub(crate) const fn new() -> Self {
        Self
    }

    pub(crate) fn capabilities() -> BackendCapabilities {
        BackendCapabilities {
            kind: BackendKind::Cpu,
            supports_frame_format: FrameFormat::YiqPlanarF32,
            supports_plane_layout: PlaneLayout::ContiguousPlanar,
        }
    }

    pub(crate) fn run(
        &self,
        effect: &NtscEffect,
        yiq: &mut YiqView,
        frame_num: usize,
        scale_factor: [f32; 2],
        frame_desc: FrameDesc,
    ) -> Result<(), BackendRunError> {
        if frame_desc.dimensions.0 == 0 || frame_desc.dimensions.1 == 0 {
            return Err(BackendRunError::InvalidDimensions(frame_desc.dimensions));
        }
        if frame_desc.format != FrameFormat::YiqPlanarF32 {
            return Err(BackendRunError::UnsupportedFrameFormat(frame_desc.format));
        }
        if frame_desc.plane_layout != PlaneLayout::ContiguousPlanar {
            return Err(BackendRunError::UnsupportedPlaneLayout(
                frame_desc.plane_layout,
            ));
        }

        effect.apply_effect_cpu_to_all_fields(yiq, frame_num, scale_factor);
        Ok(())
    }
}
