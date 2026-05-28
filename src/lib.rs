mod command;
mod device;
mod error;
mod pass;
mod shader;
mod swapchain;
mod sync;

pub use command::{RotexCommandBuffer, RotexCommandPool};
pub use device::{
    DeviceDescriptor, QueueAllocation, QueueCategory, QueueRequest, RotexAdapter, RotexDevice,
    RotexInstance,
};
pub use error::{ErrorKind, RotexError, Severity};
pub use pass::{
    Framebuffer, FramebufferBuilder, RenderPass, RenderPassBuilder, SubpassBlueprint,
};
pub use swapchain::{RotexSurface, RotexSwapchain};
pub use sync::{RotexFence, RotexSemaphore};

pub type RotexRenderPass = RenderPass;
pub type RotexFramebuffer = Framebuffer;
pub type RotexFramebufferBuilder = FramebufferBuilder;
