use crate::{
    gpu::{GpuBackend, GpuFrame},
    settings::standard::NtscEffect,
    yiq_fielding::YiqView,
};

use std::sync::Arc;
use wgpu::util::DeviceExt;

pub struct WgpuFrame {
    pub y_buffer: wgpu::Buffer,
    pub i_buffer: wgpu::Buffer,
    pub q_buffer: wgpu::Buffer,
    pub scratch_buffer: wgpu::Buffer,
    pub width: usize,
    pub height: usize,
    // Keep reference to the device/queue to easily do readbacks
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
}

impl GpuFrame for WgpuFrame {
    fn download(&self, dst: &mut YiqView) {
        // Readback logic here
        // We'll create staging buffers, copy from the GPU buffers to staging,
        // wait for the GPU, map the staging buffers, and copy into `dst`.
        assert_eq!(self.width, dst.dimensions.0);

        let size = (self.width * self.height * std::mem::size_of::<f32>()) as wgpu::BufferAddress;

        let staging_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback staging buffer"),
            size: size * 4, // 4 planes
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("download encoder"),
        });

        encoder.copy_buffer_to_buffer(&self.y_buffer, 0, &staging_buffer, 0, size);
        encoder.copy_buffer_to_buffer(&self.i_buffer, 0, &staging_buffer, size, size);
        encoder.copy_buffer_to_buffer(&self.q_buffer, 0, &staging_buffer, size * 2, size);
        encoder.copy_buffer_to_buffer(&self.scratch_buffer, 0, &staging_buffer, size * 3, size);

        self.queue.submit(Some(encoder.finish()));

        let buffer_slice = staging_buffer.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());

        // Wait for the GPU to finish
        self.device.poll(wgpu::Maintain::Wait);
        receiver.recv().unwrap().unwrap();

        let data = buffer_slice.get_mapped_range();

        // Split data into slices and copy
        let y_src = bytemuck::cast_slice(&data[0..(size as usize)]);
        let i_src = bytemuck::cast_slice(&data[(size as usize)..(size as usize * 2)]);
        let q_src = bytemuck::cast_slice(&data[(size as usize * 2)..(size as usize * 3)]);
        let scratch_src = bytemuck::cast_slice(&data[(size as usize * 3)..(size as usize * 4)]);

        dst.y[..y_src.len()].copy_from_slice(y_src);
        dst.i[..i_src.len()].copy_from_slice(i_src);
        dst.q[..q_src.len()].copy_from_slice(q_src);
        dst.scratch[..scratch_src.len()].copy_from_slice(scratch_src);
    }
}

pub struct WgpuBackend {
    pub device: Arc<wgpu::Device>,
    pub queue: Arc<wgpu::Queue>,
    copy_pipeline: wgpu::ComputePipeline,
    copy_bind_group_layout: wgpu::BindGroupLayout,
    chroma_into_luma_pipeline: wgpu::ComputePipeline,
    luma_into_chroma_box_pipeline: wgpu::ComputePipeline,
    chroma_delay_pipeline: wgpu::ComputePipeline,
    filter_plane_pipeline: wgpu::ComputePipeline,
    params_bind_group_layout: wgpu::BindGroupLayout,
}

impl WgpuBackend {
    pub fn new() -> Option<Self> {
        // Pollster block_on is needed to initialize async wgpu structs synchronously
        pollster::block_on(Self::init_async())
    }

    async fn init_async() -> Option<Self> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::PRIMARY,
            ..Default::default()
        });

        let adapter = instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            ..Default::default()
        }).await?;

        let (device, queue) = adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("ntsc-rs wgpu device"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::default(),
        }, None).await.ok()?;

        let copy_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("copy shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/copy.wgsl").into()),
        });

        let chroma_into_luma_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("chroma_into_luma shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/chroma_into_luma.wgsl").into()),
        });

        let luma_into_chroma_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("luma_into_chroma shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/luma_into_chroma.wgsl").into()),
        });

        let chroma_delay_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("chroma_delay shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/chroma_delay.wgsl").into()),
        });

        let filter_plane_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("filter_plane shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/filter_plane.wgsl").into()),
        });

        let copy_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("copy bind group layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
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
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let params_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("params bind group layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
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
            label: Some("copy pipeline layout"),
            bind_group_layouts: &[&copy_bind_group_layout],
            push_constant_ranges: &[],
        });

        let effect_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("effect pipeline layout"),
            bind_group_layouts: &[&copy_bind_group_layout, &params_bind_group_layout],
            push_constant_ranges: &[],
        });

        let copy_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("copy compute pipeline"),
            layout: Some(&pipeline_layout),
            module: &copy_shader,
            entry_point: "main",
            compilation_options: Default::default(),
        });

        let chroma_into_luma_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("chroma_into_luma pipeline"),
            layout: Some(&effect_pipeline_layout),
            module: &chroma_into_luma_shader,
            entry_point: "main",
            compilation_options: Default::default(),
        });

        let luma_into_chroma_box_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("luma_into_chroma box pipeline"),
            layout: Some(&effect_pipeline_layout),
            module: &luma_into_chroma_shader,
            entry_point: "demodulate_box",
            compilation_options: Default::default(),
        });

        let chroma_delay_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("chroma_delay pipeline"),
            layout: Some(&effect_pipeline_layout),
            module: &chroma_delay_shader,
            entry_point: "main",
            compilation_options: Default::default(),
        });

        let filter_plane_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("filter_plane pipeline"),
            layout: Some(&effect_pipeline_layout),
            module: &filter_plane_shader,
            entry_point: "main",
            compilation_options: Default::default(),
        });

        Some(Self {
            device: Arc::new(device),
            queue: Arc::new(queue),
            copy_pipeline,
            copy_bind_group_layout,
            chroma_into_luma_pipeline,
            luma_into_chroma_box_pipeline,
            chroma_delay_pipeline,
            filter_plane_pipeline,
            params_bind_group_layout,
        })
    }
}

impl GpuBackend for WgpuBackend {
    type Frame = WgpuFrame;

    fn upload_frame(&mut self, src: &YiqView) -> Self::Frame {
        let size = (src.dimensions.0 * src.num_rows() * std::mem::size_of::<f32>()) as wgpu::BufferAddress;

        let create_buffer = |label: &str, data: &[f32]| {
            let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.queue.write_buffer(&buffer, 0, bytemuck::cast_slice(&data[..data.len()]));
            buffer
        };

        let y_buffer = create_buffer("y_buffer", src.y);
        let i_buffer = create_buffer("i_buffer", src.i);
        let q_buffer = create_buffer("q_buffer", src.q);
        let scratch_buffer = create_buffer("scratch_buffer", src.scratch);

        WgpuFrame {
            y_buffer,
            i_buffer,
            q_buffer,
            scratch_buffer,
            width: src.dimensions.0,
            height: src.num_rows(),
            device: self.device.clone(),
            queue: self.queue.clone(),
        }
    }

    fn apply_effect(
        &mut self,
        effect: &NtscEffect,
        frame: &mut Self::Frame,
        frame_num: usize,
        _scale_factor: [f32; 2],
    ) {
        // Struct to hold uniform parameters for our passes
        #[repr(C)]
        #[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
        struct ShaderParams {
            width: u32,
            frame_num: u32,
            phase_shift: u32,
            phase_offset: i32,
            filter_mode: u32,
            chroma_delay_horizontal: f32,
            chroma_delay_vertical: i32,
            _pad: u32, // padding to 16 bytes alignment
        }

        let params = ShaderParams {
            width: frame.width as u32,
            frame_num: frame_num as u32,
            phase_shift: effect.video_scanline_phase_shift as u32,
            phase_offset: effect.video_scanline_phase_shift_offset,
            filter_mode: effect.chroma_demodulation as u32,
            chroma_delay_horizontal: effect.chroma_delay_horizontal,
            chroma_delay_vertical: effect.chroma_delay_vertical,
            _pad: 0,
        };

        let params_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("shader params buffer"),
            contents: bytemuck::cast_slice(&[params]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("compute encoder"),
        });

        // The layout of the main planes is the same for all passes
        let main_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("main bind group"),
            layout: &self.copy_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: frame.y_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: frame.i_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: frame.q_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: frame.scratch_buffer.as_entire_binding(),
                },
            ],
        });

        let params_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("params bind group"),
            layout: &self.params_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

        let num_pixels = frame.width * frame.height;
        let workgroups = (num_pixels.div_ceil(64)) as u32;

        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("chroma_into_luma pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.chroma_into_luma_pipeline);
            cpass.set_bind_group(0, &main_bind_group, &[]);
            cpass.set_bind_group(1, &params_bind_group, &[]);
            cpass.dispatch_workgroups(workgroups, 1, 1);
        }

        // Before demodulation, the original CPU path copies the Y channel into the scratch buffer.
        // We'll dispatch a quick copy pass for just that, or we could just use a copy_buffer_to_buffer.
        let size = (num_pixels * std::mem::size_of::<f32>()) as wgpu::BufferAddress;
        encoder.copy_buffer_to_buffer(&frame.y_buffer, 0, &frame.scratch_buffer, 0, size);

        if effect.chroma_demodulation == crate::settings::standard::ChromaDemodulationFilter::Box {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("luma_into_chroma_box pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.luma_into_chroma_box_pipeline);
            cpass.set_bind_group(0, &main_bind_group, &[]);
            cpass.set_bind_group(1, &params_bind_group, &[]);
            cpass.dispatch_workgroups(workgroups, 1, 1);
        }

        if effect.chroma_delay_horizontal != 0.0 || effect.chroma_delay_vertical != 0 {
            // First copy I/Q into scratch buffer
            encoder.copy_buffer_to_buffer(&frame.i_buffer, 0, &frame.scratch_buffer, 0, size);
            encoder.copy_buffer_to_buffer(&frame.q_buffer, 0, &frame.scratch_buffer, size, size);

            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("chroma_delay pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.chroma_delay_pipeline);
            cpass.set_bind_group(0, &main_bind_group, &[]);
            cpass.set_bind_group(1, &params_bind_group, &[]);
            cpass.dispatch_workgroups(workgroups, 1, 1);
        }

        // Example dispatch for the IIR filter passes
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("filter_plane pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.filter_plane_pipeline);
            cpass.set_bind_group(0, &main_bind_group, &[]);
            cpass.set_bind_group(1, &params_bind_group, &[]);
            // We dispatch one thread per row. 64 threads per workgroup.
            let rows = frame.height as u32;
            cpass.dispatch_workgroups(rows.div_ceil(64), 1, 1);
        }

        self.queue.submit(Some(encoder.finish()));
    }
}
