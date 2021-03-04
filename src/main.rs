use ktx::KtxInfo;
use pilka_ash::ash::{
    version::{DeviceV1_0, InstanceV1_0},
    vk, RawDevice, VkBuffer, VkDevice, VkInstance,
};
use pilka_ash::*;
use std::mem::size_of;

use std::sync::Arc;

mod texture;
mod util;

#[macro_export]
macro_rules! offset_of {
    ($base:path, $field:ident) => {{
        #[allow(unused_unsafe)]
        unsafe {
            let b: $base = ::std::mem::zeroed();
            ((&b.$field as *const _ as isize) - (&b as *const _ as isize)) as _
        }
    }};
}

const VERTEX_BUFFER_BIND_ID: u32 = 0;

struct Graphics {
    device: Arc<RawDevice>,
    desc_set_layout: vk::DescriptorSetLayout,
    desc_set_pre_compute: vk::DescriptorSet,
    desc_set_post_compute: vk::DescriptorSet,
    pipline: vk::Pipeline,
    pipeline_layout: vk::PipelineLayout,
    semaphore: vk::Semaphore,
}

impl Drop for Graphics {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_pipeline(self.pipline, None);
            self.device
                .destroy_pipeline_layout(self.pipeline_layout, None);
            self.device
                .destroy_descriptor_set_layout(self.desc_set_layout, None);
            self.device.destroy_semaphore(self.semaphore, None);
        }
    }
}

struct Compute {
    device: Arc<RawDevice>,
    queue: vk::Queue,
    command_pool: vk::CommandPool,
    command_buffer: vk::CommandBuffer,
    semaphore: vk::Semaphore,
    desc_set_layout: vk::DescriptorSetLayout,
    desc_set_: vk::DescriptorSet,
    pipelines: Vec<vk::Pipeline>,
    pipeline_layout: vk::PipelineLayout,
    pipeline_index: usize, // 0
}

struct Vertices {
    input_state: vk::PipelineVertexInputStateCreateInfo,
    binding_desctiptions: Vec<vk::VertexInputBindingDescription>,
    attribute_description: Vec<vk::VertexInputAttributeDescription>,
}

impl Drop for Compute {
    fn drop(&mut self) {
        unsafe {
            for pipeline in &self.pipelines {
                self.device.destroy_pipeline(*pipeline, None);
            }

            self.device
                .destroy_pipeline_layout(self.pipeline_layout, None);
            self.device
                .destroy_descriptor_set_layout(self.desc_set_layout, None);
            self.device.destroy_semaphore(self.semaphore, None);
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let event_loop = winit::event_loop::EventLoop::new();

    let window = winit::window::WindowBuilder::new()
        .with_title("Pilka")
        .with_inner_size(winit::dpi::LogicalSize::new(
            f64::from(1280),
            f64::from(720),
        ))
        .build(&event_loop)?;

    let validation_layers = if cfg!(debug_assertions) {
        vec!["VK_LAYER_KHRONOS_validation\0"]
    } else {
        vec![]
    };
    // let validation_layers = vec!["VK_LAYER_KHRONOS_validation\0"];
    let extention_names = ash_window::ash_window::enumerate_required_extensions(&window)?;
    let instance = VkInstance::new(&validation_layers, &extention_names)?;

    let surface = instance.create_surface(&window)?;

    let (device, device_properties, queues) = instance.create_device_and_queues(Some(&surface))?;

    unsafe {
        dbg!(instance.get_physical_device_properties(device.physical_device));
    }

    let surface_resolution = surface.resolution(&device)?;

    let swapchain_loader = instance.create_swapchain_loader(&device);

    let swapchain = device.create_swapchain(swapchain_loader, &surface, &queues)?;

    let command_pool = device
        .create_vk_command_pool(queues.graphics_queue.index, swapchain.images.len() as u32)?;

    let render_pass = device.create_vk_render_pass(swapchain.format())?;

    let present_complete_semaphore = device.create_semaphore()?;
    let rendering_complete_semaphore = device.create_semaphore()?;

    let framebuffers = swapchain.create_framebuffers(
        (surface_resolution.width, surface_resolution.height),
        &render_pass,
        &device,
    )?;

    let (viewports, scissors, extent) = {
        let surface_resolution = surface.resolution(&device)?;
        (
            Box::new([vk::Viewport {
                x: 0.0,
                y: surface_resolution.height as f32,
                width: surface_resolution.width as f32,
                height: -(surface_resolution.height as f32),
                min_depth: 0.0,
                max_depth: 1.0,
            }]),
            Box::new([vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: surface_resolution,
            }]),
            surface_resolution,
        )
    };

    let pipeline_cache_create_info = vk::PipelineCacheCreateInfo::builder();
    let pipeline_cache =
        unsafe { device.create_pipeline_cache(&pipeline_cache_create_info, None) }?;

    let vertices = vec![
        vertex([1.0, 1.0, 0.0], [1.0, 1.0]),
        vertex([-1.0, 1.0, 0.0], [0.0, 1.0]),
        vertex([-1.0, -1.0, 0.0], [0.0, 0.0]),
        vertex([1.0, -1.0, 0.0], [1.0, 0.0]),
    ];

    let indices = vec![0, 1, 2, 2, 3, 0];
    let index_count = indices.len() as u64;

    let vertex_buffer = device.create_vk_buffer_from_slice(
        vk::BufferUsageFlags::VERTEX_BUFFER,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        &vertices,
    );
    let index_buffer = device.create_vk_buffer_from_slice(
        vk::BufferUsageFlags::INDEX_BUFFER,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        &indices,
    )?;

    let binding_desctiptions = vec![vk::VertexInputBindingDescription::builder()
        .binding(VERTEX_BUFFER_BIND_ID)
        .stride(size_of::<Vertex>() as _)
        .input_rate(vk::VertexInputRate::VERTEX)
        .build()];
    let attribute_description = vec![
        vk::VertexInputAttributeDescription::builder()
            .binding(VERTEX_BUFFER_BIND_ID)
            .location(0)
            .format(vk::Format::R32G32B32_SFLOAT)
            .offset(offset_of!(Vertex, pos))
            .build(),
        vk::VertexInputAttributeDescription::builder()
            .binding(VERTEX_BUFFER_BIND_ID)
            .location(1)
            .format(vk::Format::R32G32B32_SFLOAT)
            .offset(offset_of!(Vertex, uv))
            .build(),
    ];
    let input_state = vk::PipelineVertexInputStateCreateInfo::builder()
        .vertex_binding_descriptions(&binding_desctiptions)
        .vertex_attribute_descriptions(&attribute_description);

    let pool_sizes = vec![
        vk::DescriptorPoolSize::builder()
            .ty(vk::DescriptorType::UNIFORM_BUFFER)
            .descriptor_count(2)
            .build(),
        vk::DescriptorPoolSize::builder()
            .ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count(2)
            .build(),
        vk::DescriptorPoolSize::builder()
            .ty(vk::DescriptorType::STORAGE_IMAGE)
            .descriptor_count(2)
            .build(),
    ];
    let desc_pool_info = vk::DescriptorPoolCreateInfo::builder()
        .pool_sizes(&pool_sizes)
        .max_sets(3);
    let desc_pool = unsafe { device.create_descriptor_pool(&desc_pool_info, None) }?;

    let desc_set_layout = vec![
        vk::DescriptorSetLayoutBinding::builder()
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .stage_flags(vk::ShaderStageFlags::VERTEX)
            .binding(0)
            .descriptor_count(1)
            .build(),
        vk::DescriptorSetLayoutBinding::builder()
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT)
            .binding(1)
            .descriptor_count(1)
            .build(),
    ];

    let desc_layout = vk::DescriptorSetLayoutCreateInfo::builder().bindings(&desc_set_layout);
    let graphics_descriptor_set_layout =
        unsafe { device.create_descriptor_set_layout(&desc_layout, None) }?;

    let graphics_layouts = [graphics_descriptor_set_layout];
    let pipeline_layout_create_info =
        vk::PipelineLayoutCreateInfo::builder().set_layouts(&graphics_layouts);
    let pipeline_layout =
        unsafe { device.create_pipeline_layout(&pipeline_layout_create_info, None) };

    let desc_alloc_info = vk::DescriptorSetAllocateInfo::builder()
        .descriptor_pool(desc_pool)
        .set_layouts(&graphics_layouts);
    let graphics_pre_compute = unsafe { device.allocate_descriptor_sets(&desc_alloc_info) }?[0];
    let base_image_write_desc_sets = vec![
        vk::WriteDescriptorSet::builder()
            .dst_set(graphics_pre_compute)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .buffer_info(uniform_buffers_vs.descriptor)
            .build(),
        vk::WriteDescriptorSet::builder()
            .dst_set(graphics_pre_compute)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .buffer_info(texture_color_map.descriptor)
            .build(),
    ];
    unsafe { device.update_descriptor_sets(&base_image_write_desc_sets, &[]) };

    let graphics_post_compute = unsafe { device.allocate_descriptor_sets(&desc_alloc_info) }?[0];
    let write_desc_sets = vec![
        vk::WriteDescriptorSet::builder()
            .dst_set(graphics_post_compute)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .buffer_info(uniform_buffers_vs.descriptor)
            .build(),
        vk::WriteDescriptorSet::builder()
            .dst_set(graphics_post_compute)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .buffer_info(texture_color_map.descriptor)
            .build(),
    ];
    unsafe { device.update_descriptor_sets(&write_desc_sets, &[]) };

    let noop_stencil_state = vk::StencilOpState {
        fail_op: vk::StencilOp::KEEP,
        pass_op: vk::StencilOp::KEEP,
        depth_fail_op: vk::StencilOp::KEEP,
        compare_op: vk::CompareOp::ALWAYS,
        ..Default::default()
    };
    let depth_stencil = vk::PipelineDepthStencilStateCreateInfo {
        depth_test_enable: 1,
        depth_write_enable: 1,
        depth_compare_op: vk::CompareOp::LESS_OR_EQUAL,
        front: noop_stencil_state,
        back: noop_stencil_state,
        max_depth_bounds: 1.0,
        ..Default::default()
    };

    let input_assembly = vk::PipelineInputAssemblyStateCreateInfo {
        topology: vk::PrimitiveTopology::TRIANGLE_LIST,
        ..Default::default()
    };

    let rasterization = vk::PipelineRasterizationStateCreateInfo {
        front_face: vk::FrontFace::COUNTER_CLOCKWISE,
        line_width: 1.0,
        polygon_mode: vk::PolygonMode::FILL,
        cull_mode: vk::CullModeFlags::NONE,
        ..Default::default()
    };
    let multisample = vk::PipelineMultisampleStateCreateInfo {
        rasterization_samples: vk::SampleCountFlags::TYPE_1,
        ..Default::default()
    };

    let color_blend_attachments = Box::new([vk::PipelineColorBlendAttachmentState {
        blend_enable: 0,
        src_color_blend_factor: vk::BlendFactor::SRC_COLOR,
        dst_color_blend_factor: vk::BlendFactor::ONE_MINUS_DST_COLOR,
        color_blend_op: vk::BlendOp::ADD,
        src_alpha_blend_factor: vk::BlendFactor::ZERO,
        dst_alpha_blend_factor: vk::BlendFactor::ZERO,
        alpha_blend_op: vk::BlendOp::ADD,
        color_write_mask: vk::ColorComponentFlags::all(),
    }]);
    let color_blend = vk::PipelineColorBlendStateCreateInfo::builder()
        .logic_op(vk::LogicOp::CLEAR)
        .attachments(color_blend_attachments.as_ref())
        .build();

    let dynamic_state = Box::new([vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR]);
    let dynamic_state_info = vk::PipelineDynamicStateCreateInfo::builder()
        .dynamic_states(dynamic_state.as_ref())
        .build();

    let viewport = vk::PipelineViewportStateCreateInfo::builder()
        .viewports(&[vk::Viewport::default()])
        .scissors(&[vk::Rect2D::default()])
        .build();

    let pipeline_info = vk::GraphicsPipelineCreateInfo::builder()
        .stages(&shader_stages)
        .vertex_input_state(&input_state)
        .input_assembly_state(&input_assembly)
        .rasterization_state(&rasterization)
        .multisample_state(&multisample)
        .depth_stencil_state(&depth_stencil)
        .color_blend_state(&color_blend)
        .dynamic_state(&dynamic_state_info)
        .viewport_state(&viewport)
        .layout(pipeline_layout)
        .render_pass(render_pass.render_pass);

    let graphics_semaphore = unsafe { device.create_semaphore() }?;

    let set_layout_binding_c = vec![
        vk::DescriptorSetLayoutBinding::builder()
            .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .binding(0)
            .build(),
        vk::DescriptorSetLayoutBinding::builder()
            .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .binding(1)
            .build(),
    ];
    let descriptor_layout_info =
        vk::DescriptorSetLayoutCreateInfo::builder().bindings(&set_layout_binding_c);
    let desc_set_layout_c =
        [unsafe { device.create_descriptor_set_layout(&descriptor_layout_info, None) }?];
    let pipeline_layout_create_info =
        vk::PipelineLayoutCreateInfo::builder().set_layouts(&desc_set_layout_c);
    let compute_pipeline =
        unsafe { device.create_pipeline_layout(&pipeline_layout_create_info, None) }?;

    Ok(())
}

#[repr(C)]
struct Vertex {
    pos: [f32; 3],
    uv: [f32; 2],
}

impl Vertex {
    fn new(pos: [f32; 3], uv: [f32; 2]) -> Self {
        Self { pos, uv }
    }
}

fn vertex(pos: [f32; 3], uv: [f32; 2]) -> Vertex {
    Vertex { pos, uv }
}
