#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum GpuPass {
    ApplyNtscEffectField,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct PassNode {
    pub pass: GpuPass,
    pub label: &'static str,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct PassGraph {
    passes: &'static [PassNode],
}

impl PassGraph {
    pub(crate) const fn minimal() -> Self {
        Self {
            passes: &[PassNode {
                pass: GpuPass::ApplyNtscEffectField,
                label: "apply_ntsc_effect_field",
            }],
        }
    }

    pub(crate) fn passes(&self) -> &'static [PassNode] {
        self.passes
    }
}
