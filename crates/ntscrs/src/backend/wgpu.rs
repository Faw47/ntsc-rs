use std::collections::HashMap;
use std::sync::Arc;

use bytemuck::{cast_slice, cast_slice_mut};
use wgpu::util::DeviceExt;

const WORKGROUP_SIZE: u32 = 256;
const TRIVIAL_PASS_ID: &str = "trivial_gain";
const TRIVIAL_SHADER_ID: &str = "shaders/trivial_gain.wgsl";

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GainParams {
    gain: f32,
    len: u32,
    _pad0: u32,
    _pad1: u32,
}

#[derive(Clone, Copy, Debug)]
pub struct PlaneLayout {
    pub width: u32,
    pub height: u32,
    pub stride: u32,
}

impl PlaneLayout {
    pub fn elem_count(self) -> usize {
        self.stride as usize * self.height as usize
    }

    pub fn byte_len(self) -> u64 {
        (self.elem_count() * std::mem::size_of::<f32>()) as u64
    }
}

#[derive(Clone, Copy, Debug)]
pub struct FrameLayout {
    pub y: PlaneLayout,
    pub i: PlaneLayout,
    pub q: PlaneLayout,
}

#[derive(Debug)]
pub struct CpuFrame {
    pub y: Vec<f32>,
    pub i: Vec<f32>,
    pub q: Vec<f32>,
}

pub struct GpuFrame {
    pub layout: FrameLayout,
    y_src: wgpu::Buffer,
    i_src: wgpu::Buffer,
    q_src: wgpu::Buffer,
    y_dst: wgpu::Buffer,
    i_dst: wgpu::Buffer,
    q_dst: wgpu::Buffer,
}

pub struct GpuBackend {
    pub instance: wgpu::Instance,
    pub adapter: wgpu::Adapter,
    pub device: Arc<wgpu::Device>,
    pub queue: Arc<wgpu::Queue>,
    bind_group_layout: wgpu::BindGroupLayout,
    pipeline_layout: wgpu::PipelineLayout,
    pipeline_cache: HashMap<String, wgpu::ComputePipeline>,
}

impl GpuBackend {
    pub fn new() -> Result<Self, String> {
        pollster::block_on(Self::new_async())
    }

    async fn new_async() -> Result<Self, String> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .map_err(|e| format!("failed to request adapter: {e}"))?;

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("ntscrs-gpu-device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                memory_hints: wgpu::MemoryHints::Performance,
                trace: wgpu::Trace::default(),
                experimental_features: wgpu::ExperimentalFeatures::disabled(),
            })
            .await
            .map_err(|e| format!("failed to request device: {e}"))?;

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("ntscrs-pass-bind-group-layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("ntscrs-compute-pipeline-layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        Ok(Self {
            instance,
            adapter,
            device: Arc::new(device),
            queue: Arc::new(queue),
            bind_group_layout,
            pipeline_layout,
            pipeline_cache: HashMap::new(),
        })
    }

    pub fn create_frame(&self, layout: FrameLayout) -> GpuFrame {
        GpuFrame {
            layout,
            y_src: self.make_plane_buffer("y-src", layout.y),
            i_src: self.make_plane_buffer("i-src", layout.i),
            q_src: self.make_plane_buffer("q-src", layout.q),
            y_dst: self.make_plane_buffer("y-dst", layout.y),
            i_dst: self.make_plane_buffer("i-dst", layout.i),
            q_dst: self.make_plane_buffer("q-dst", layout.q),
        }
    }

    fn make_plane_buffer(&self, label: &str, layout: PlaneLayout) -> wgpu::Buffer {
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: layout.byte_len(),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    fn pipeline_for_pass(
        &mut self,
        pass_id: &str,
        shader_id: &str,
        shader_source: &str,
    ) -> wgpu::ComputePipeline {
        self.pipeline_cache
            .entry(format!("{shader_id}:{pass_id}"))
            .or_insert_with(|| {
                let module = self
                    .device
                    .create_shader_module(wgpu::ShaderModuleDescriptor {
                        label: Some(shader_id),
                        source: wgpu::ShaderSource::Wgsl(shader_source.into()),
                    });

                self.device
                    .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                        label: Some(pass_id),
                        layout: Some(&self.pipeline_layout),
                        module: &module,
                        entry_point: Some("main"),
                        compilation_options: wgpu::PipelineCompilationOptions::default(),
                        cache: None,
                    })
            })
            .clone()
    }

    pub fn upload_cpu_planes(
        &self,
        frame: &GpuFrame,
        y: &[f32],
        i: &[f32],
        q: &[f32],
    ) -> Result<(), String> {
        Self::validate_plane_upload("Y", frame.layout.y, y)?;
        Self::validate_plane_upload("I", frame.layout.i, i)?;
        Self::validate_plane_upload("Q", frame.layout.q, q)?;

        self.queue.write_buffer(&frame.y_src, 0, cast_slice(y));
        self.queue.write_buffer(&frame.i_src, 0, cast_slice(i));
        self.queue.write_buffer(&frame.q_src, 0, cast_slice(q));
        Ok(())
    }

    fn validate_plane_upload(name: &str, layout: PlaneLayout, plane: &[f32]) -> Result<(), String> {
        if plane.len() != layout.elem_count() {
            return Err(format!(
                "{name} plane length mismatch: got {}, expected {}",
                plane.len(),
                layout.elem_count()
            ));
        }

        let bytes = layout.byte_len();
        if bytes % wgpu::COPY_BUFFER_ALIGNMENT != 0 {
            return Err(format!(
                "{name} plane byte length {bytes} is not aligned to COPY_BUFFER_ALIGNMENT ({})",
                wgpu::COPY_BUFFER_ALIGNMENT
            ));
        }

        Ok(())
    }

    pub fn run_minimal_pass_chain(&mut self, frame: &GpuFrame, gain: f32) {
        let shader_source = include_str!("shaders/trivial_gain.wgsl");
        let pipeline = self.pipeline_for_pass(TRIVIAL_PASS_ID, TRIVIAL_SHADER_ID, shader_source);

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("ntscrs-minimal-pass-encoder"),
            });

        self.dispatch_plane_pass(
            &mut encoder,
            &pipeline,
            &frame.y_src,
            &frame.y_dst,
            frame.layout.y,
            gain,
        );
        self.dispatch_plane_pass(
            &mut encoder,
            &pipeline,
            &frame.i_src,
            &frame.i_dst,
            frame.layout.i,
            gain,
        );
        self.dispatch_plane_pass(
            &mut encoder,
            &pipeline,
            &frame.q_src,
            &frame.q_dst,
            frame.layout.q,
            gain,
        );

        self.queue.submit(Some(encoder.finish()));
    }

    fn dispatch_plane_pass(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        pipeline: &wgpu::ComputePipeline,
        src: &wgpu::Buffer,
        dst: &wgpu::Buffer,
        layout: PlaneLayout,
        gain: f32,
    ) {
        let len = layout.width * layout.height;
        let dispatch_x = ceil_div(len, WORKGROUP_SIZE);

        let params = GainParams {
            gain,
            len,
            _pad0: 0,
            _pad1: 0,
        };
        let params_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("ntscrs-gain-params"),
                contents: cast_slice(&[params]),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("ntscrs-pass-bind-group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: src.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: dst.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("ntscrs-trivial-compute-pass"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(pipeline);
        cpass.set_bind_group(0, &bind_group, &[]);
        cpass.dispatch_workgroups(dispatch_x, 1, 1);
    }

    pub fn download_to_cpu_buffers(&self, frame: &GpuFrame) -> Result<CpuFrame, String> {
        let y = self.read_back_plane(&frame.y_dst, frame.layout.y, "Y")?;
        let i = self.read_back_plane(&frame.i_dst, frame.layout.i, "I")?;
        let q = self.read_back_plane(&frame.q_dst, frame.layout.q, "Q")?;

        Ok(CpuFrame { y, i, q })
    }

    fn read_back_plane(
        &self,
        source: &wgpu::Buffer,
        layout: PlaneLayout,
        plane_name: &str,
    ) -> Result<Vec<f32>, String> {
        let raw_len = layout.byte_len();
        if raw_len % wgpu::COPY_BUFFER_ALIGNMENT != 0 {
            return Err(format!(
                "{plane_name} plane copy size {raw_len} does not satisfy COPY_BUFFER_ALIGNMENT ({})",
                wgpu::COPY_BUFFER_ALIGNMENT
            ));
        }

        let padded_len = align_up(raw_len, wgpu::COPY_BUFFER_ALIGNMENT);
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ntscrs-readback-staging"),
            size: padded_len,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("ntscrs-readback-encoder"),
            });
        encoder.copy_buffer_to_buffer(source, 0, &staging, 0, raw_len);
        self.queue.submit(Some(encoder.finish()));

        let slice = staging.slice(..raw_len);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |res| {
            let _ = tx.send(res);
        });

        let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
        rx.recv()
            .map_err(|_| format!("{plane_name} plane map channel dropped"))?
            .map_err(|e| format!("{plane_name} plane map error: {e}"))?;

        let mapped = slice.get_mapped_range();
        let mut out = vec![0.0f32; layout.elem_count()];
        cast_slice_mut::<f32, u8>(&mut out).copy_from_slice(&mapped);
        drop(mapped);
        staging.unmap();
        Ok(out)
    }
}

fn align_up(value: u64, alignment: u64) -> u64 {
    if alignment == 0 {
        return value;
    }
    let remainder = value % alignment;
    if remainder == 0 {
        value
    } else {
        value + (alignment - remainder)
    }
}

fn ceil_div(value: u32, divisor: u32) -> u32 {
    if value == 0 {
        0
    } else {
        1 + ((value - 1) / divisor)
    }
}
