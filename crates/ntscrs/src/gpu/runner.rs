use crate::{
    gpu::{BackendType, GpuBackend, GpuFrame},
    settings::standard::NtscEffect,
    yiq_fielding::YiqView,
};

/// The CPU reference implementation wrapper, adhering to the GpuBackend interface
/// so that the pipeline runner can transparently call it.
pub struct CpuBackend;

pub struct CpuFrame<'a> {
    pub yiq: YiqView<'a>,
}

impl<'a> GpuFrame for CpuFrame<'a> {
    fn download(&self, _dst: &mut YiqView) {
        // Since we process in-place on the CPU, a download is essentially a no-op
        // if they refer to the same buffers. However, to match the semantics where
        // `CpuFrame` might be an ephemeral wrapper, if it's different data, we'd copy.
        // For simplicity in the runner, we handle it by not copying at all if we are on CPU.
    }
}

impl GpuBackend for CpuBackend {
    type Frame = CpuFrame<'static>;

    fn upload_frame(&mut self, _src: &YiqView) -> Self::Frame {
        // Note: The CPU backend modifies in place. The runner handles this specially.
        unimplemented!("CPU backend handles frames in place");
    }

    fn apply_effect(
        &mut self,
        effect: &NtscEffect,
        frame: &mut Self::Frame,
        frame_num: usize,
        scale_factor: [f32; 2],
    ) {
        effect.apply_effect_to_yiq(&mut frame.yiq, frame_num, scale_factor);
    }
}

pub struct NtscEffectRunner {
    backend_type: BackendType,
    #[cfg(feature = "gpu-wgpu")]
    wgpu_backend: Option<crate::gpu::wgpu_backend::WgpuBackend>,
}

impl NtscEffectRunner {
    pub fn new(requested_backend: BackendType) -> Self {
        #[allow(unused_mut)]
        let mut actual_backend = BackendType::Cpu;

        #[cfg(feature = "gpu-wgpu")]
        let mut wgpu_backend = None;

        match requested_backend {
            BackendType::Cpu => {}
            #[cfg(feature = "gpu-wgpu")]
            BackendType::Wgpu | BackendType::Auto => {
                if let Some(backend) = crate::gpu::wgpu_backend::WgpuBackend::new() {
                    wgpu_backend = Some(backend);
                    actual_backend = BackendType::Wgpu;
                } else {
                    println!("ntsc-rs: Failed to initialize WGPU backend, falling back to CPU.");
                }
            }
            #[cfg(not(feature = "gpu-wgpu"))]
            BackendType::Auto => {}
        }

        Self {
            backend_type: actual_backend,
            #[cfg(feature = "gpu-wgpu")]
            wgpu_backend,
        }
    }

    pub fn active_backend(&self) -> BackendType {
        self.backend_type
    }

    pub fn apply_effect(
        &mut self,
        src: &mut YiqView,
        effect: &NtscEffect,
        frame_num: usize,
        scale_factor: [f32; 2],
    ) {
        match self.backend_type {
            BackendType::Cpu => {
                effect.apply_effect_to_yiq(src, frame_num, scale_factor);
            }
            #[cfg(feature = "gpu-wgpu")]
            BackendType::Wgpu => {
                let mut frame = self.wgpu_backend.as_mut().unwrap().upload_frame(src);
                self.wgpu_backend.as_mut().unwrap().apply_effect(effect, &mut frame, frame_num, scale_factor);
                frame.download(src);
            }
            BackendType::Auto => unreachable!("Auto should have resolved to a concrete backend"),
        }
    }
}
