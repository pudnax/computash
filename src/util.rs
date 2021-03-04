use pilka_ash::ash::version::DeviceV1_0;
use pilka_ash::ash::vk;
use pilka_ash::ash::VkDevice;

pub fn find_memory_type_index(
    memory_req: &vk::MemoryRequirements,
    memory_prop: &vk::PhysicalDeviceMemoryProperties,
    flags: vk::MemoryPropertyFlags,
) -> Option<u32> {
    // Try to find an exactly matching memory flag
    let best_suitable_index =
        find_memorytype_index_f(memory_req, memory_prop, flags, |property_flags, flags| {
            property_flags == flags
        });
    if best_suitable_index.is_some() {
        return best_suitable_index;
    }
    // Otherwise find a memory flag that works
    find_memorytype_index_f(memory_req, memory_prop, flags, |property_flags, flags| {
        property_flags & flags == flags
    })
}

fn find_memorytype_index_f<F: Fn(vk::MemoryPropertyFlags, vk::MemoryPropertyFlags) -> bool>(
    memory_req: &vk::MemoryRequirements,
    memory_prop: &vk::PhysicalDeviceMemoryProperties,
    flags: vk::MemoryPropertyFlags,
    f: F,
) -> Option<u32> {
    let mut memory_type_bits = memory_req.memory_type_bits;
    for (index, ref memory_type) in memory_prop.memory_types.iter().enumerate() {
        if memory_type_bits & 1 == 1 && f(memory_type.property_flags, flags) {
            return Some(index as u32);
        }
        memory_type_bits >>= 1;
    }
    None
}

#[allow(clippy::clippy::too_many_arguments)]
pub fn set_image_layout(
    device: &VkDevice,
    cmd_buffer: vk::CommandBuffer,
    image: vk::Image,
    old_layout: vk::ImageLayout,
    new_layout: vk::ImageLayout,
    subresource_range: vk::ImageSubresourceRange,
    src_stage_mask: vk::PipelineStageFlags,
    dst_stage_mask: vk::PipelineStageFlags,
) {
    let mut image_memory_barrier = vk::ImageMemoryBarrier::builder()
        .old_layout(old_layout)
        .new_layout(new_layout)
        .image(image)
        .subresource_range(subresource_range);

    use vk::{AccessFlags, ImageLayout};
    image_memory_barrier.src_access_mask = match old_layout {
        ImageLayout::UNDEFINED => AccessFlags::empty(),
        ImageLayout::PREINITIALIZED => AccessFlags::HOST_WRITE,
        ImageLayout::COLOR_ATTACHMENT_OPTIMAL => AccessFlags::COLOR_ATTACHMENT_WRITE,
        ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL => {
            AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE
        }
        ImageLayout::TRANSFER_SRC_OPTIMAL => AccessFlags::TRANSFER_READ,
        ImageLayout::TRANSFER_DST_OPTIMAL => AccessFlags::TRANSFER_WRITE,
        ImageLayout::SHADER_READ_ONLY_OPTIMAL => AccessFlags::SHADER_READ,
        _ => AccessFlags::empty(),
    };

    image_memory_barrier.dst_access_mask = match new_layout {
        ImageLayout::TRANSFER_DST_OPTIMAL => AccessFlags::TRANSFER_WRITE,
        ImageLayout::TRANSFER_SRC_OPTIMAL => AccessFlags::TRANSFER_READ,
        ImageLayout::COLOR_ATTACHMENT_OPTIMAL => AccessFlags::COLOR_ATTACHMENT_WRITE,
        ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL => {
            image_memory_barrier.dst_access_mask | AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE
        }
        ImageLayout::SHADER_READ_ONLY_OPTIMAL => {
            if image_memory_barrier.src_access_mask.is_empty() {
                image_memory_barrier.src_access_mask =
                    AccessFlags::HOST_WRITE | AccessFlags::TRANSFER_WRITE;
            }
            AccessFlags::SHADER_READ
        }
        _ => AccessFlags::empty(),
    };

    let image_barriers = [image_memory_barrier.build()];
    unsafe {
        device.cmd_pipeline_barrier(
            cmd_buffer,
            src_stage_mask,
            dst_stage_mask,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &image_barriers,
        );
    }
}

pub fn set_image_layout_all_commands(
    device: &VkDevice,
    cmd_buffer: vk::CommandBuffer,
    image: vk::Image,
    old_layout: vk::ImageLayout,
    new_layout: vk::ImageLayout,
    subresource_range: vk::ImageSubresourceRange,
) {
    set_image_layout(
        device,
        cmd_buffer,
        image,
        old_layout,
        new_layout,
        subresource_range,
        vk::PipelineStageFlags::ALL_COMMANDS,
        vk::PipelineStageFlags::ALL_COMMANDS,
    );
}
