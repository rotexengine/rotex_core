use ash::vk;

use crate::device::{QueueCategory, RotexDevice, RotexInstance};
use crate::error::{ErrorKind, RotexError, Severity};
use crate::sync::RotexSemaphore;

pub struct RotexSurface {
    loader: ash::khr::surface::Instance,
    surface: vk::SurfaceKHR,
}

impl RotexSurface {
    pub fn new(instance: &RotexInstance, surface: vk::SurfaceKHR) -> Self {
        let loader = ash::khr::surface::Instance::new(instance.entry(), instance.instance());
        Self { loader, surface }
    }

    pub fn loader(&self) -> &ash::khr::surface::Instance {
        &self.loader
    }

    pub fn handle(&self) -> vk::SurfaceKHR {
        self.surface
    }

    pub fn destroy(&mut self) {
        unsafe {
            self.loader.destroy_surface(self.surface, None);
        }
    }
}

pub struct RotexSwapchain {
    loader: ash::khr::swapchain::Device,
    swapchain: vk::SwapchainKHR,
    _surface: vk::SurfaceKHR,
    format: vk::Format,
    color_space: vk::ColorSpaceKHR,
    extent: vk::Extent2D,
    images: Vec<vk::Image>,
    image_views: Vec<vk::ImageView>,
}

impl RotexSwapchain {
    pub fn new(
        instance: &RotexInstance,
        device: &RotexDevice,
        surface: &RotexSurface,
    ) -> Result<Self, RotexError> {
        let queue = device
            .queues()
            .iter()
            .find(|allocation| allocation.category == QueueCategory::Graphics)
            .ok_or(RotexError {
                kind: ErrorKind::NoCompatibleDevice,
                severity: Severity::Fatal,
            })?;

        let surface_capabilities = unsafe {
            surface.loader().get_physical_device_surface_capabilities(
                device.physical_device(),
                surface.handle(),
            )
        }
        .map_err(|err| RotexError {
            kind: ErrorKind::Vulkan(err),
            severity: Severity::Fatal,
        })?;
        let surface_formats = unsafe {
            surface
                .loader()
                .get_physical_device_surface_formats(device.physical_device(), surface.handle())
        }
        .map_err(|err| RotexError {
            kind: ErrorKind::Vulkan(err),
            severity: Severity::Fatal,
        })?;
        let present_modes = unsafe {
            surface.loader().get_physical_device_surface_present_modes(
                device.physical_device(),
                surface.handle(),
            )
        }
        .map_err(|err| RotexError {
            kind: ErrorKind::Vulkan(err),
            severity: Severity::Fatal,
        })?;

        let preferred_format = [vk::Format::B8G8R8A8_SRGB, vk::Format::R8G8B8A8_SRGB];
        let preferred_color_space = vk::ColorSpaceKHR::SRGB_NONLINEAR;
        let surface_format = surface_formats
            .iter()
            .find(|format| {
                preferred_format.contains(&format.format)
                    && format.color_space == preferred_color_space
            })
            .or_else(|| surface_formats.first())
            .ok_or(RotexError {
                kind: ErrorKind::NoCompatibleDevice,
                severity: Severity::Fatal,
            })?;

        let present_mode = if present_modes.contains(&vk::PresentModeKHR::MAILBOX) {
            vk::PresentModeKHR::MAILBOX
        } else {
            vk::PresentModeKHR::FIFO
        };

        let extent = if surface_capabilities.current_extent.width != u32::MAX {
            surface_capabilities.current_extent
        } else {
            vk::Extent2D {
                width: surface_capabilities.min_image_extent.width,
                height: surface_capabilities.min_image_extent.height,
            }
        };

        let mut image_count = surface_capabilities.min_image_count + 1;
        if surface_capabilities.max_image_count > 0 {
            image_count = image_count.min(surface_capabilities.max_image_count);
        }

        let queue_family_indices = [queue.family_index];
        let swapchain_create_info = vk::SwapchainCreateInfoKHR::default()
            .surface(surface.handle())
            .min_image_count(image_count)
            .image_format(surface_format.format)
            .image_color_space(surface_format.color_space)
            .image_extent(extent)
            .image_array_layers(1)
            .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
            .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
            .queue_family_indices(&queue_family_indices)
            .pre_transform(surface_capabilities.current_transform)
            .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
            .present_mode(present_mode)
            .clipped(true);

        let swapchain_loader =
            ash::khr::swapchain::Device::new(&instance.instance(), &device.device());
        let swapchain = unsafe { swapchain_loader.create_swapchain(&swapchain_create_info, None) }
            .map_err(|err| RotexError {
                kind: ErrorKind::Vulkan(err),
                severity: Severity::Fatal,
            })?;
        let images =
            unsafe { swapchain_loader.get_swapchain_images(swapchain) }.map_err(|err| {
                RotexError {
                    kind: ErrorKind::Vulkan(err),
                    severity: Severity::Fatal,
                }
            })?;

        let image_views = images
            .iter()
            .map(|image| {
                let view_info = vk::ImageViewCreateInfo::default()
                    .image(*image)
                    .view_type(vk::ImageViewType::TYPE_2D)
                    .format(surface_format.format)
                    .subresource_range(
                        vk::ImageSubresourceRange::default()
                            .aspect_mask(vk::ImageAspectFlags::COLOR)
                            .level_count(1)
                            .layer_count(1),
                    );
                unsafe { device.device().create_image_view(&view_info, None) }.map_err(|err| {
                    RotexError {
                        kind: ErrorKind::Vulkan(err),
                        severity: Severity::Fatal,
                    }
                })
            })
            .collect::<Result<Vec<_>, _>>()?;

        Ok(Self {
            loader: swapchain_loader,
            swapchain,
            _surface: surface.handle(),
            format: surface_format.format,
            color_space: surface_format.color_space,
            extent,
            images,
            image_views,
        })
    }

    pub fn acquire_next_image(
        &self,
        semaphore: &RotexSemaphore,
    ) -> Result<(u32, bool), RotexError> {
        unsafe {
            self.loader.acquire_next_image(
                self.swapchain,
                u64::MAX,
                semaphore.handle,
                vk::Fence::null(),
            )
        }
        .map_err(|err| RotexError {
            kind: ErrorKind::Vulkan(err),
            severity: Severity::Fatal,
        })
    }

    pub fn present(
        &self,
        queue: vk::Queue,
        image_index: u32,
        wait_semaphore: &RotexSemaphore,
    ) -> Result<bool, RotexError> {
        let wait_semaphores = [wait_semaphore.handle];
        let swapchains = [self.swapchain];
        let image_indices = [image_index];

        let present_info = vk::PresentInfoKHR::default()
            .wait_semaphores(&wait_semaphores)
            .swapchains(&swapchains)
            .image_indices(&image_indices);

        unsafe { self.loader.queue_present(queue, &present_info) }.map_err(|err| RotexError {
            kind: ErrorKind::Vulkan(err),
            severity: Severity::Fatal,
        })
    }

    pub fn format(&self) -> vk::Format {
        self.format
    }

    pub fn color_space(&self) -> vk::ColorSpaceKHR {
        self.color_space
    }

    pub fn extent(&self) -> vk::Extent2D {
        self.extent
    }

    pub fn images(&self) -> &[vk::Image] {
        &self.images
    }

    pub fn image_views(&self) -> &[vk::ImageView] {
        &self.image_views
    }

    pub fn destroy(&mut self, device: &RotexDevice) {
        unsafe {
            for view in self.image_views.drain(..) {
                device.device().destroy_image_view(view, None);
            }
            self.loader.destroy_swapchain(self.swapchain, None);
        }
    }
}
