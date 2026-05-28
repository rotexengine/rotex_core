use ash::vk;

use crate::device::RotexDevice;
use crate::error::{ErrorKind, RotexError, Severity};

pub struct RotexSemaphore {
    pub handle: vk::Semaphore,
}

impl RotexSemaphore {
    pub fn new(device: &RotexDevice) -> Result<Self, RotexError> {
        let create_info = vk::SemaphoreCreateInfo::default();

        let handle =
            unsafe { device.device().create_semaphore(&create_info, None) }.map_err(|err| {
                RotexError {
                    kind: ErrorKind::Vulkan(err),
                    severity: Severity::Fatal,
                }
            })?;

        Ok(Self { handle })
    }

    pub fn destroy(&self, device: &RotexDevice) {
        unsafe {
            device.device().destroy_semaphore(self.handle, None);
        }
    }
}

pub struct RotexFence {
    handle: vk::Fence,
}

impl RotexFence {
    pub fn new(device: &RotexDevice, signaled: bool) -> Result<Self, RotexError> {
        let mut create_info = vk::FenceCreateInfo::default();

        if signaled {
            create_info = create_info.flags(vk::FenceCreateFlags::SIGNALED);
        }

        let handle =
            unsafe { device.device().create_fence(&create_info, None) }.map_err(|err| {
                RotexError {
                    kind: ErrorKind::Vulkan(err),
                    severity: Severity::Fatal,
                }
            })?;

        Ok(Self { handle })
    }

    pub fn handle(&self) -> vk::Fence {
        self.handle
    }

    pub fn wait(&self, device: &RotexDevice, timeout_ns: u64) -> Result<(), RotexError> {
        unsafe {
            device
                .device()
                .wait_for_fences(&[self.handle], true, timeout_ns)
        }
        .map_err(|err| RotexError {
            kind: ErrorKind::Vulkan(err),
            severity: Severity::Fatal,
        })
    }

    pub fn reset(&self, device: &RotexDevice) -> Result<(), RotexError> {
        unsafe { device.device().reset_fences(&[self.handle]) }.map_err(|err| RotexError {
            kind: ErrorKind::Vulkan(err),
            severity: Severity::Fatal,
        })
    }

    pub fn destroy(&self, device: &RotexDevice) {
        unsafe {
            device.device().destroy_fence(self.handle, None);
        }
    }
}
