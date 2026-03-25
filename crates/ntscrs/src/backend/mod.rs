#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BackendPreference {
    Auto,
    Cpu,
    Wgpu,
    Cuda,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RuntimeBackend {
    Cpu,
    Wgpu,
    Cuda,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BackendRejection {
    pub backend: RuntimeBackend,
    pub reason: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RuntimeBackendReport {
    pub selected_backend: RuntimeBackend,
    pub rejected_backends: Vec<BackendRejection>,
}

impl RuntimeBackendReport {
    pub fn new(selected_backend: RuntimeBackend) -> Self {
        Self {
            selected_backend,
            rejected_backends: Vec::new(),
        }
    }

    pub fn reject<S: Into<String>>(&mut self, backend: RuntimeBackend, reason: S) {
        self.rejected_backends.push(BackendRejection {
            backend,
            reason: reason.into(),
        });
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FrameDescriptor {
    pub width: usize,
    pub height: usize,
    pub is_interleaved: bool,
}

impl FrameDescriptor {
    fn accelerated_layout_supported(&self) -> bool {
        self.width > 0 && self.height > 0 && !self.is_interleaved
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RuntimeInitError {
    pub message: String,
    pub report: RuntimeBackendReport,
}

impl RuntimeInitError {
    fn new(message: impl Into<String>, report: RuntimeBackendReport) -> Self {
        Self {
            message: message.into(),
            report,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RuntimeBackendHandle {
    pub backend: RuntimeBackend,
    pub report: RuntimeBackendReport,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BackendManager {
    pub allow_cpu_fallback_on_override_failure: bool,
}

impl Default for BackendManager {
    fn default() -> Self {
        Self {
            allow_cpu_fallback_on_override_failure: true,
        }
    }
}

impl BackendManager {
    pub fn select_and_init(
        &self,
        preference: BackendPreference,
        frame_desc: FrameDescriptor,
        mut logger: impl FnMut(&str),
    ) -> Result<RuntimeBackendHandle, RuntimeInitError> {
        let mut report = RuntimeBackendReport::new(RuntimeBackend::Cpu);

        match preference {
            BackendPreference::Cpu => {
                report.selected_backend = RuntimeBackend::Cpu;
                return Ok(RuntimeBackendHandle {
                    backend: RuntimeBackend::Cpu,
                    report,
                });
            }
            BackendPreference::Wgpu => {
                return self.select_explicit_backend(
                    RuntimeBackend::Wgpu,
                    frame_desc,
                    logger,
                    report,
                );
            }
            BackendPreference::Cuda => {
                return self.select_explicit_backend(
                    RuntimeBackend::Cuda,
                    frame_desc,
                    logger,
                    report,
                );
            }
            BackendPreference::Auto => {}
        }

        if let Some(handle) =
            self.try_backend(RuntimeBackend::Wgpu, frame_desc, &mut report, &mut logger)
        {
            return Ok(handle);
        }

        if let Some(handle) =
            self.try_backend(RuntimeBackend::Cuda, frame_desc, &mut report, &mut logger)
        {
            return Ok(handle);
        }

        report.selected_backend = RuntimeBackend::Cpu;
        Ok(RuntimeBackendHandle {
            backend: RuntimeBackend::Cpu,
            report,
        })
    }

    fn select_explicit_backend(
        &self,
        backend: RuntimeBackend,
        frame_desc: FrameDescriptor,
        mut logger: impl FnMut(&str),
        mut report: RuntimeBackendReport,
    ) -> Result<RuntimeBackendHandle, RuntimeInitError> {
        if let Some(handle) = self.try_backend(backend, frame_desc, &mut report, &mut logger) {
            return Ok(handle);
        }

        if self.allow_cpu_fallback_on_override_failure {
            logger("backend override unavailable, falling back to CPU");
            report.selected_backend = RuntimeBackend::Cpu;
            return Ok(RuntimeBackendHandle {
                backend: RuntimeBackend::Cpu,
                report,
            });
        }

        Err(RuntimeInitError::new(
            "explicit backend override unavailable",
            report,
        ))
    }

    fn try_backend(
        &self,
        backend: RuntimeBackend,
        frame_desc: FrameDescriptor,
        report: &mut RuntimeBackendReport,
        logger: &mut impl FnMut(&str),
    ) -> Option<RuntimeBackendHandle> {
        if !frame_desc.accelerated_layout_supported() {
            report.reject(backend, "incompatible frame layout");
            return None;
        }

        let feature_enabled = match backend {
            RuntimeBackend::Cpu => true,
            RuntimeBackend::Wgpu => cfg!(feature = "gpu-wgpu"),
            RuntimeBackend::Cuda => cfg!(feature = "gpu-cuda"),
        };

        if !feature_enabled {
            report.reject(backend, "missing feature");
            return None;
        }

        if !self.backend_supported(backend) {
            report.reject(backend, "unsupported adapter");
            return None;
        }

        match self.initialize_backend(backend, frame_desc) {
            Ok(()) => {
                logger(match backend {
                    RuntimeBackend::Cpu => "initialized CPU backend",
                    RuntimeBackend::Wgpu => "initialized WGPU backend",
                    RuntimeBackend::Cuda => "initialized CUDA backend",
                });

                report.selected_backend = backend;
                Some(RuntimeBackendHandle {
                    backend,
                    report: report.clone(),
                })
            }
            Err(err) => {
                report.reject(backend, format!("init failure: {err}"));
                None
            }
        }
    }

    fn backend_supported(&self, backend: RuntimeBackend) -> bool {
        match backend {
            RuntimeBackend::Cpu => true,
            RuntimeBackend::Wgpu => false,
            RuntimeBackend::Cuda => false,
        }
    }

    fn initialize_backend(
        &self,
        backend: RuntimeBackend,
        _frame_desc: FrameDescriptor,
    ) -> Result<(), String> {
        match backend {
            RuntimeBackend::Cpu => Ok(()),
            RuntimeBackend::Wgpu => Ok(()),
            RuntimeBackend::Cuda => Ok(()),
        }
    }
}
