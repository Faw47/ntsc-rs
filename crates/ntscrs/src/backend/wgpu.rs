use crate::{settings::standard::NtscEffect, yiq_fielding::YiqView};

use super::{
    pipeline::{GpuPass, PassGraph},
    types::{
        BackendCapabilities, BackendKind, BackendRunError, FrameDesc, FrameFormat, PlaneLayout,
    },
};

#[derive(Debug)]
pub(crate) struct WgpuBackend {
    graph: PassGraph,
}

impl WgpuBackend {
    pub(crate) fn new() -> Self {
        Self {
            graph: PassGraph::minimal(),
        }
    }

    pub(crate) fn capabilities() -> BackendCapabilities {
        BackendCapabilities {
            kind: BackendKind::Wgpu,
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

        // Skeleton execution path: iterate the pass graph and execute
        // the single currently supported pass.
        for node in self.graph.passes() {
            match node.pass {
                GpuPass::ApplyNtscEffectField => {
                    effect.apply_effect_cpu_to_all_fields(yiq, frame_num, scale_factor)
                }
            }
        }

        Ok(())
    }
}
