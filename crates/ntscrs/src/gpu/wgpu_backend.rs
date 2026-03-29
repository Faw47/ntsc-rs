use crate::{
    gpu::{GpuBackend, GpuFrame},
    noise_seeds,
    settings::standard::NtscEffect,
    yiq_fielding::YiqView,
};

use std::sync::Arc;
use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ShaderParams {
    pub width: u32,
    pub frame_num: u32,
    pub seed: u32,
    pub noise_idx: u32,

    pub noise_frequency: f32,
    pub noise_intensity: f32,
    pub noise_detail: u32,
    pub snow_anisotropy: f32,

    pub phase_shift: u32,
    pub phase_offset: i32,
    pub filter_mode: u32,
    pub chroma_delay_horizontal: f32,

    pub chroma_delay_vertical: i32,
    pub horizontal_scale: f32,
    pub vertical_scale: f32,
    pub _pad1: u32,
    pub _pad2: u32,
    /// Uniform block size must match WGSL `uniform` layout (rounded up to 16-byte boundary).
    pub _pad3: u32,
    pub _pad4: u32,
    pub _pad5: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct FilterCoeffs {
    pub num: [f32; 4],
    pub den: [f32; 4],
    pub z_initial: [f32; 4],
    pub delay: u32,
    pub filter_len: u32,
    pub plane_idx: u32,
    pub _pad1: u32,
}

pub struct WgpuFrame {
    pub y_buffer: wgpu::Buffer,
    pub i_buffer: wgpu::Buffer,
    pub q_buffer: wgpu::Buffer,
    pub scratch_buffer: wgpu::Buffer,
    pub staging_buffer: wgpu::Buffer,
    pub main_bind_group: wgpu::BindGroup,
    pub i_pass_bind_group: wgpu::BindGroup,
    pub q_pass_bind_group: wgpu::BindGroup,
    pub chroma_loss_bind_group: wgpu::BindGroup,
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

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("download encoder"),
            });

        encoder.copy_buffer_to_buffer(&self.y_buffer, 0, &self.staging_buffer, 0, size);
        encoder.copy_buffer_to_buffer(&self.i_buffer, 0, &self.staging_buffer, size, size);
        encoder.copy_buffer_to_buffer(&self.q_buffer, 0, &self.staging_buffer, size * 2, size);
        encoder.copy_buffer_to_buffer(
            &self.scratch_buffer,
            0,
            &self.staging_buffer,
            size * 3,
            size,
        );

        self.queue.submit(Some(encoder.finish()));

        let buffer_slice = self.staging_buffer.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());

        // Wait for the GPU to finish
        let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
        receiver.recv().unwrap().unwrap();

        {
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

        self.staging_buffer.unmap();
    }
}

#[allow(dead_code)]
pub struct WgpuBackend {
    pub device: Arc<wgpu::Device>,
    pub queue: Arc<wgpu::Queue>,
    copy_pipeline: wgpu::ComputePipeline,
    copy_bind_group_layout: wgpu::BindGroupLayout,
    chroma_into_luma_pipeline: wgpu::ComputePipeline,
    luma_into_chroma_box_pipeline: wgpu::ComputePipeline,
    chroma_delay_pipeline: wgpu::ComputePipeline,
    filter_plane_pipeline: wgpu::ComputePipeline,
    plane_noise_pipeline: wgpu::ComputePipeline,
    snow_pipeline: wgpu::ComputePipeline,
    vhs_shifting_pipeline: wgpu::ComputePipeline,
    chroma_loss_pipeline: wgpu::ComputePipeline,
    chroma_vert_blend_pipeline: wgpu::ComputePipeline,
    params_bind_group_layout: wgpu::BindGroupLayout,
    filter_coeffs_bind_group_layout: wgpu::BindGroupLayout,
    params_ring_buffer: Vec<(wgpu::Buffer, wgpu::BindGroup)>,
}

impl WgpuBackend {
    pub fn new() -> Option<Self> {
        // Pollster block_on is needed to initialize async wgpu structs synchronously
        pollster::block_on(Self::init_async())
    }

    async fn init_async() -> Option<Self> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::PRIMARY,
            ..Default::default()
        });

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                ..Default::default()
            })
            .await
            .ok()?;

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("ntsc-rs wgpu device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                ..Default::default()
            })
            .await
            .ok()?;

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

        let xoshiro_src = include_str!("shaders/xoshiro.wgsl");
        let simplex_src = include_str!("shaders/simplex.wgsl");

        let snow_src = format!("{}\n{}", xoshiro_src, include_str!("shaders/snow.wgsl"));
        let snow_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("snow shader"),
            source: wgpu::ShaderSource::Wgsl(snow_src.into()),
        });

        let plane_noise_src = format!(
            "{}\n{}\n{}",
            xoshiro_src,
            simplex_src,
            include_str!("shaders/plane_noise.wgsl")
        );
        let plane_noise_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("plane_noise shader"),
            source: wgpu::ShaderSource::Wgsl(plane_noise_src.into()),
        });

        let vhs_shift_src = format!(
            "{}\n{}\n{}",
            xoshiro_src,
            simplex_src,
            include_str!("shaders/vhs_shifting.wgsl")
        );
        let vhs_shift_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("vhs_shifting shader"),
            source: wgpu::ShaderSource::Wgsl(vhs_shift_src.into()),
        });

        let chroma_loss_src = format!(
            "{}\n{}",
            xoshiro_src,
            include_str!("shaders/chroma_loss_blend.wgsl")
        );
        let chroma_loss_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("chroma_loss_blend shader"),
            source: wgpu::ShaderSource::Wgsl(chroma_loss_src.into()),
        });

        let copy_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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

        let params_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("params bind group layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });

        let filter_coeffs_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("filter coeffs bind group layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("copy pipeline layout"),
            bind_group_layouts: &[&copy_bind_group_layout],
            push_constant_ranges: &[],
        });

        let effect_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("effect pipeline layout"),
                bind_group_layouts: &[&copy_bind_group_layout, &params_bind_group_layout],
                push_constant_ranges: &[],
            });

        let filter_plane_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("filter_plane pipeline layout"),
                bind_group_layouts: &[
                    &copy_bind_group_layout,
                    &params_bind_group_layout,
                    &filter_coeffs_bind_group_layout,
                ],
                push_constant_ranges: &[],
            });

        let copy_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("copy compute pipeline"),
            layout: Some(&pipeline_layout),
            module: &copy_shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let chroma_into_luma_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("chroma_into_luma pipeline"),
                layout: Some(&effect_pipeline_layout),
                module: &chroma_into_luma_shader,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });

        let luma_into_chroma_box_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("luma_into_chroma box pipeline"),
                layout: Some(&effect_pipeline_layout),
                module: &luma_into_chroma_shader,
                entry_point: Some("demodulate_box"),
                compilation_options: Default::default(),
                cache: None,
            });

        let chroma_delay_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("chroma_delay pipeline"),
                layout: Some(&effect_pipeline_layout),
                module: &chroma_delay_shader,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });

        let filter_plane_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("filter_plane pipeline"),
                layout: Some(&filter_plane_pipeline_layout),
                module: &filter_plane_shader,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });

        let plane_noise_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("plane_noise pipeline"),
                layout: Some(&effect_pipeline_layout),
                module: &plane_noise_shader,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });

        let snow_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("snow pipeline"),
            layout: Some(&effect_pipeline_layout),
            module: &snow_shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let vhs_shifting_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("vhs_shifting pipeline"),
                layout: Some(&effect_pipeline_layout),
                module: &vhs_shift_shader,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });

        let chroma_loss_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("chroma_loss pipeline"),
                layout: Some(&effect_pipeline_layout),
                module: &chroma_loss_shader,
                entry_point: Some("chroma_loss"),
                compilation_options: Default::default(),
                cache: None,
            });

        let chroma_vert_blend_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("chroma_vert_blend pipeline"),
                layout: Some(&effect_pipeline_layout),
                module: &chroma_loss_shader,
                entry_point: Some("chroma_vert_blend"),
                compilation_options: Default::default(),
                cache: None,
            });

        let params = ShaderParams {
            width: 0,
            frame_num: 0,
            seed: 0,
            noise_idx: 0,

            noise_frequency: 0.0,
            noise_intensity: 0.0,
            noise_detail: 0,
            snow_anisotropy: 0.0,

            phase_shift: 0,
            phase_offset: 0,
            filter_mode: 0,
            chroma_delay_horizontal: 0.0,

            chroma_delay_vertical: 0,
            horizontal_scale: 1.0,
            vertical_scale: 1.0,
            _pad1: 0,
            _pad2: 0,
            _pad3: 0,
            _pad4: 0,
            _pad5: 0,
        };

        let mut params_ring_buffer = Vec::new();
        for _ in 0..32 {
            let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("shader params buffer"),
                contents: bytemuck::cast_slice(&[params]),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });
            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("params bind group"),
                layout: &params_bind_group_layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buffer.as_entire_binding(),
                }],
            });
            params_ring_buffer.push((buffer, bind_group));
        }

        Some(Self {
            device: Arc::new(device),
            queue: Arc::new(queue),
            copy_pipeline,
            copy_bind_group_layout,
            chroma_into_luma_pipeline,
            luma_into_chroma_box_pipeline,
            chroma_delay_pipeline,
            filter_plane_pipeline,
            plane_noise_pipeline,
            snow_pipeline,
            vhs_shifting_pipeline,
            chroma_loss_pipeline,
            chroma_vert_blend_pipeline,
            params_bind_group_layout,
            filter_coeffs_bind_group_layout,
            params_ring_buffer,
        })
    }

    #[allow(dead_code)] // Reserved for filter parity with CPU pipeline
    fn dispatch_filter_plane<'a>(
        &'a self,
        encoder: &mut wgpu::CommandEncoder,
        frame: &WgpuFrame,
        params_bind_group: &'a wgpu::BindGroup,
        tf: &crate::filter::TransferFunction,
        initial: f32,
        delay: usize,
        plane_idx: u32,
    ) {
        let (num, den, z_initial) = tf.to_gpu_coeffs(initial);
        let filter_coeffs = FilterCoeffs {
            num,
            den,
            z_initial,
            delay: delay as u32,
            filter_len: tf.len() as u32,
            plane_idx,
            _pad1: 0,
        };

        let coeffs_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("filter coeffs buffer"),
                contents: bytemuck::cast_slice(&[filter_coeffs]),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let coeffs_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("filter coeffs bind group"),
            layout: &self.filter_coeffs_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: coeffs_buffer.as_entire_binding(),
            }],
        });

        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("filter_plane pass"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(&self.filter_plane_pipeline);
        cpass.set_bind_group(0, &frame.main_bind_group, &[]);
        cpass.set_bind_group(1, params_bind_group, &[]);
        cpass.set_bind_group(2, &coeffs_bind_group, &[]);

        let rows = frame.height as u32;
        cpass.dispatch_workgroups(rows.div_ceil(64), 1, 1);
    }

    fn get_params_bind_group<'a>(
        &'a self,
        params: &ShaderParams,
        ring_idx: &mut usize,
    ) -> &'a wgpu::BindGroup {
        let idx = *ring_idx % self.params_ring_buffer.len();
        *ring_idx += 1;
        let (buffer, bind_group) = &self.params_ring_buffer[idx];
        self.queue
            .write_buffer(buffer, 0, bytemuck::cast_slice(&[*params]));
        bind_group
    }
}

impl GpuBackend for WgpuBackend {
    type Frame = WgpuFrame;

    fn upload_frame(&mut self, src: &YiqView) -> Self::Frame {
        let size =
            (src.dimensions.0 * src.num_rows() * std::mem::size_of::<f32>()) as wgpu::BufferAddress;

        let create_buffer = |label: &str, data: &[f32]| {
            let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.queue
                .write_buffer(&buffer, 0, bytemuck::cast_slice(&data[..data.len()]));
            buffer
        };

        let y_buffer = create_buffer("y_buffer", src.y);
        let i_buffer = create_buffer("i_buffer", src.i);
        let q_buffer = create_buffer("q_buffer", src.q);
        let scratch_buffer = create_buffer("scratch_buffer", src.scratch);

        let staging_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback staging buffer"),
            size: size * 4, // 4 planes
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let main_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("main bind group"),
            layout: &self.copy_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: y_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: i_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: q_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: scratch_buffer.as_entire_binding(),
                },
            ],
        });

        let i_pass_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("i pass bind group"),
            layout: &self.copy_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: i_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: scratch_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: q_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: scratch_buffer.as_entire_binding(),
                },
            ],
        });

        let q_pass_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("q pass bind group"),
            layout: &self.copy_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: q_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: scratch_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: i_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: scratch_buffer.as_entire_binding(),
                },
            ],
        });

        let chroma_loss_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("chroma loss bind group"),
            layout: &self.copy_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: i_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: q_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: scratch_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: scratch_buffer.as_entire_binding(),
                },
            ],
        });

        WgpuFrame {
            y_buffer,
            i_buffer,
            q_buffer,
            scratch_buffer,
            staging_buffer,
            main_bind_group,
            i_pass_bind_group,
            q_pass_bind_group,
            chroma_loss_bind_group,
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
        let mut ring_idx = 0;
        let mut params = ShaderParams {
            width: frame.width as u32,
            frame_num: frame_num as u32,
            seed: effect.random_seed as u32,
            noise_idx: 0,

            noise_frequency: 0.0,
            noise_intensity: 0.0,
            noise_detail: 0,
            snow_anisotropy: 0.0,

            phase_shift: effect.video_scanline_phase_shift as u32,
            phase_offset: effect.video_scanline_phase_shift_offset,
            filter_mode: effect.chroma_demodulation as u32,
            chroma_delay_horizontal: effect.chroma_delay_horizontal,

            chroma_delay_vertical: effect.chroma_delay_vertical,
            horizontal_scale: effect
                .scale
                .as_ref()
                .map(|s| s.horizontal_scale)
                .unwrap_or(1.0),
            vertical_scale: effect
                .scale
                .as_ref()
                .map(|s| s.vertical_scale)
                .unwrap_or(1.0),
            _pad1: 0,
            _pad2: 0,
            _pad3: 0,
            _pad4: 0,
            _pad5: 0,
        };

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("compute encoder"),
            });

        let main_bind_group = &frame.main_bind_group;

        let num_pixels = frame.width * frame.height;
        let workgroups = (num_pixels.div_ceil(64)) as u32;

        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("chroma_into_luma pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.chroma_into_luma_pipeline);
            cpass.set_bind_group(0, main_bind_group, &[]);
            cpass.set_bind_group(1, self.get_params_bind_group(&params, &mut ring_idx), &[]);
            cpass.dispatch_workgroups(workgroups, 1, 1);
        }

        if effect.snow_intensity > 0.0 && params.horizontal_scale > 0.0 {
            params.noise_intensity = effect.snow_intensity * 0.01;
            params.snow_anisotropy = effect.snow_anisotropy;
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("snow pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.snow_pipeline);
            cpass.set_bind_group(0, main_bind_group, &[]);
            cpass.set_bind_group(1, self.get_params_bind_group(&params, &mut ring_idx), &[]);
            cpass.dispatch_workgroups(workgroups, 1, 1);
        }

        if let Some(crate::settings::HeadSwitchingSettings {
            height,
            offset,
            horiz_shift,
            mid_line: _,
        }) = &effect.head_switching
        {
            params.phase_shift = 2; // head_switching
            params.phase_offset = *offset as i32;
            params.filter_mode = *height as u32; // num_rows
            params.chroma_delay_horizontal = *horiz_shift;
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("head_switching pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.vhs_shifting_pipeline);
            cpass.set_bind_group(0, main_bind_group, &[]);
            cpass.set_bind_group(1, self.get_params_bind_group(&params, &mut ring_idx), &[]);
            cpass.dispatch_workgroups(workgroups, 1, 1);
        }

        if let Some(crate::settings::TrackingNoiseSettings {
            height,
            wave_intensity,
            ..
        }) = effect.tracking_noise
        {
            params.phase_shift = 1; // tracking_noise
            params.phase_offset = (frame
                .height
                .saturating_sub((height as f32 * params.vertical_scale).round() as usize))
                as i32;
            params.filter_mode = (height as f32 * params.vertical_scale).round() as u32; // num_rows
            params.chroma_delay_horizontal = wave_intensity;
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("tracking_noise pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.vhs_shifting_pipeline);
            cpass.set_bind_group(0, main_bind_group, &[]);
            cpass.set_bind_group(1, self.get_params_bind_group(&params, &mut ring_idx), &[]);
            cpass.dispatch_workgroups(workgroups, 1, 1);
        }

        let size = (num_pixels * std::mem::size_of::<f32>()) as wgpu::BufferAddress;
        encoder.copy_buffer_to_buffer(&frame.y_buffer, 0, &frame.scratch_buffer, 0, size);

        if effect.chroma_demodulation == crate::settings::standard::ChromaDemodulationFilter::Box {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("luma_into_chroma_box pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.luma_into_chroma_box_pipeline);
            cpass.set_bind_group(0, main_bind_group, &[]);
            cpass.set_bind_group(1, self.get_params_bind_group(&params, &mut ring_idx), &[]);
            cpass.dispatch_workgroups(workgroups, 1, 1);
        }

        if let Some(luma_noise) = &effect.luma_noise {
            params.noise_idx = noise_seeds::VIDEO_LUMA as u32;
            params.noise_frequency = luma_noise.frequency / params.horizontal_scale;
            params.noise_intensity = luma_noise.intensity;
            params.noise_detail = luma_noise.detail.try_into().unwrap_or_default();

            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("plane_noise luma pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.plane_noise_pipeline);
            cpass.set_bind_group(0, main_bind_group, &[]);
            cpass.set_bind_group(1, self.get_params_bind_group(&params, &mut ring_idx), &[]);
            cpass.dispatch_workgroups(workgroups, 1, 1);
        }

        if let Some(chroma_noise) = &effect.chroma_noise {
            params.noise_frequency = chroma_noise.frequency / params.horizontal_scale;
            params.noise_intensity = chroma_noise.intensity;
            params.noise_detail = chroma_noise.detail.try_into().unwrap_or_default();

            params.noise_idx = noise_seeds::VIDEO_CHROMA_I as u32;
            {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("plane_noise chroma i pass"),
                    timestamp_writes: None,
                });
                cpass.set_pipeline(&self.plane_noise_pipeline);
                cpass.set_bind_group(0, &frame.i_pass_bind_group, &[]);
                cpass.set_bind_group(1, self.get_params_bind_group(&params, &mut ring_idx), &[]);
                cpass.dispatch_workgroups(workgroups, 1, 1);
            }

            params.noise_idx = noise_seeds::VIDEO_CHROMA_Q as u32;
            {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("plane_noise chroma q pass"),
                    timestamp_writes: None,
                });
                cpass.set_pipeline(&self.plane_noise_pipeline);
                cpass.set_bind_group(0, &frame.q_pass_bind_group, &[]);
                cpass.set_bind_group(1, self.get_params_bind_group(&params, &mut ring_idx), &[]);
                cpass.dispatch_workgroups(workgroups, 1, 1);
            }
        }

        if effect.chroma_delay_horizontal != 0.0 || effect.chroma_delay_vertical != 0 {
            encoder.copy_buffer_to_buffer(&frame.i_buffer, 0, &frame.scratch_buffer, 0, size);
            encoder.copy_buffer_to_buffer(&frame.q_buffer, 0, &frame.scratch_buffer, size, size);

            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("chroma_delay pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.chroma_delay_pipeline);
            cpass.set_bind_group(0, main_bind_group, &[]);
            cpass.set_bind_group(1, self.get_params_bind_group(&params, &mut ring_idx), &[]);
            cpass.dispatch_workgroups(workgroups, 1, 1);
        }

        if let Some(vhs_settings) = &effect.vhs_settings {
            if let Some(edge_wave) = &vhs_settings.edge_wave {
                if edge_wave.intensity > 0.0 {
                    params.phase_shift = 0; // edge_wave
                    params.noise_frequency = edge_wave.speed;
                    params.noise_intensity = edge_wave.intensity;
                    params.noise_detail = edge_wave.detail.try_into().unwrap_or_default();
                    params.phase_offset = 0; // all rows

                    let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("vhs_edge_wave pass"),
                        timestamp_writes: None,
                    });
                    cpass.set_pipeline(&self.vhs_shifting_pipeline);
                    cpass.set_bind_group(0, main_bind_group, &[]);
                    cpass.set_bind_group(
                        1,
                        self.get_params_bind_group(&params, &mut ring_idx),
                        &[],
                    );
                    cpass.dispatch_workgroups(workgroups, 1, 1);
                }
            }

            if vhs_settings.chroma_loss > 0.0 {
                params.noise_frequency = vhs_settings.chroma_loss; // loss intensity
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("chroma_loss pass"),
                    timestamp_writes: None,
                });
                cpass.set_pipeline(&self.chroma_loss_pipeline);
                cpass.set_bind_group(0, &frame.chroma_loss_bind_group, &[]);
                cpass.set_bind_group(1, self.get_params_bind_group(&params, &mut ring_idx), &[]);
                cpass.dispatch_workgroups(1, 1, 1);
            }
        }

        if effect.chroma_vert_blend {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("chroma_vert_blend pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.chroma_vert_blend_pipeline);
            cpass.set_bind_group(0, &frame.chroma_loss_bind_group, &[]);
            cpass.set_bind_group(1, self.get_params_bind_group(&params, &mut ring_idx), &[]);
            cpass.dispatch_workgroups(frame.width.div_ceil(64) as u32, 1, 1);
        }

        self.queue.submit(Some(encoder.finish()));
    }
}
