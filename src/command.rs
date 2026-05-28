use ash::vk;

use crate::device::{QueueCategory, RotexDevice};
use crate::error::{ErrorKind, RotexError, Severity};
use crate::pass::{Framebuffer, RenderPass};

pub struct RotexCommandBuffer {
    pub handle: vk::CommandBuffer,
}

impl RotexCommandBuffer {
    pub fn begin(
        &self,
        device: &RotexDevice,
        flags: vk::CommandBufferUsageFlags,
    ) -> Result<(), RotexError> {
        let begin_info = vk::CommandBufferBeginInfo::default().flags(flags);

        unsafe {
            device
                .device()
                .begin_command_buffer(self.handle, &begin_info)
        }
        .map_err(|err| RotexError {
            kind: ErrorKind::Vulkan(err),
            severity: Severity::Fatal,
        })
    }

    pub fn begin_render_pass(
        &self,
        device: &RotexDevice,
        render_pass: &RenderPass,
        framebuffer: &Framebuffer,
        clear_values: &[vk::ClearValue],
    ) {
        debug_assert_eq!(
            clear_values.len() as u32,
            render_pass.attachments().len() as u32,
            "Rotex Core Panic: The number of clear values does not match the Render Pass attachment count!"
        );

        let render_pass_info = vk::RenderPassBeginInfo::default()
            .render_pass(render_pass.handle())
            .framebuffer(framebuffer.handle())
            .render_area(vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: framebuffer.extent(),
            })
            .clear_values(clear_values);

        unsafe {
            device.device().cmd_begin_render_pass(
                self.handle,
                &render_pass_info,
                vk::SubpassContents::INLINE,
            );
        }
    }

    pub fn end_render_pass(&self, device: &RotexDevice) {
        unsafe {
            device.device().cmd_end_render_pass(self.handle);
        }
    }

    pub fn end(&self, device: &RotexDevice) -> Result<(), RotexError> {
        unsafe { device.device().end_command_buffer(self.handle) }.map_err(|err| RotexError {
            kind: ErrorKind::Vulkan(err),
            severity: Severity::Fatal,
        })
    }
}

pub struct RotexCommandPool {
    handle: vk::CommandPool,
}

impl RotexCommandPool {
    pub fn new(device: &RotexDevice) -> Result<Self, RotexError> {
        let graphics_queue = device
            .queues()
            .iter()
            .find(|q| q.category == QueueCategory::Graphics)
            .ok_or(RotexError {
                kind: ErrorKind::NoCompatibleDevice,
                severity: Severity::Fatal,
            })?;

        let pool_info = vk::CommandPoolCreateInfo::default()
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
            .queue_family_index(graphics_queue.family_index);

        let handle =
            unsafe { device.device().create_command_pool(&pool_info, None) }.map_err(|err| {
                RotexError {
                    kind: ErrorKind::Vulkan(err),
                    severity: Severity::Fatal,
                }
            })?;

        Ok(Self { handle })
    }

    pub fn allocate_buffers(
        &self,
        device: &RotexDevice,
        count: u32,
    ) -> Result<Vec<RotexCommandBuffer>, RotexError> {
        let alloc_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(self.handle)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(count);

        let handles =
            unsafe { device.device().allocate_command_buffers(&alloc_info) }.map_err(|err| {
                RotexError {
                    kind: ErrorKind::Vulkan(err),
                    severity: Severity::Fatal,
                }
            })?;

        Ok(handles
            .into_iter()
            .map(|handle| RotexCommandBuffer { handle })
            .collect())
    }

    pub fn destroy(&self, device: &RotexDevice) {
        unsafe {
            device.device().destroy_command_pool(self.handle, None);
        }
    }
}
