#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Backend {
    Cpu,
    Wgpu,
    Cuda,
}

fn parse_override(value: &str) -> Option<Backend> {
    match value.trim().to_ascii_lowercase().as_str() {
        "cpu" => Some(Backend::Cpu),
        "wgpu" | "gpu-wgpu" => Some(Backend::Wgpu),
        "cuda" | "gpu-cuda" => Some(Backend::Cuda),
        _ => None,
    }
}

fn select_backend(override_backend: Option<&str>, available: &[Backend]) -> Backend {
    if let Some(requested) = override_backend.and_then(parse_override) {
        if available.contains(&requested) {
            return requested;
        }
    }

    [Backend::Cuda, Backend::Wgpu, Backend::Cpu]
        .into_iter()
        .find(|backend| available.contains(backend))
        .unwrap_or(Backend::Cpu)
}

#[test]
fn explicit_override_takes_precedence_over_default_priority() {
    let available = [Backend::Cpu, Backend::Wgpu, Backend::Cuda];

    assert_eq!(select_backend(Some("cpu"), &available), Backend::Cpu);
    assert_eq!(select_backend(Some("wgpu"), &available), Backend::Wgpu);
    assert_eq!(select_backend(Some("cuda"), &available), Backend::Cuda);
}

#[test]
fn unsupported_override_falls_back_to_priority_order() {
    let available = [Backend::Cpu, Backend::Wgpu];

    assert_eq!(select_backend(Some("cuda"), &available), Backend::Wgpu);
    assert_eq!(
        select_backend(Some("definitely-not-real"), &available),
        Backend::Wgpu
    );
}

#[test]
fn fallback_order_prefers_cuda_then_wgpu_then_cpu() {
    assert_eq!(
        select_backend(None, &[Backend::Cpu, Backend::Wgpu, Backend::Cuda]),
        Backend::Cuda
    );
    assert_eq!(
        select_backend(None, &[Backend::Cpu, Backend::Wgpu]),
        Backend::Wgpu
    );
    assert_eq!(select_backend(None, &[Backend::Cpu]), Backend::Cpu);
    assert_eq!(select_backend(None, &[]), Backend::Cpu);
}
