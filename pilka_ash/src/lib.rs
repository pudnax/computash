#![allow(dead_code)]

mod command_pool;
mod device;
mod image;
mod instance;
mod renderpass_and_pipeline;
mod shader_module;
mod surface;
mod swapchain;
pub mod utils;

pub mod ash_window {
    pub use ash_window;
}

pub mod ash {
    pub use ash::*;

    pub use crate::command_pool::*;
    pub use crate::device::*;
    pub use crate::image::*;
    pub use crate::instance::*;
    pub use crate::renderpass_and_pipeline::*;
    pub use crate::shader_module::*;
    pub use crate::surface::*;
    pub use crate::swapchain::*;
    pub use crate::utils as pilka_util;

    pub use raw_window_handle::HasRawWindowHandle;
    pub use shaderc;
}
