use ash::vk;
use rotex_core::{
    DeviceDescriptor, ErrorKind, QueueCategory, QueueRequest, RenderPassBuilder,
    RotexCommandBuffer, RotexCommandPool, RotexDevice, RotexError, RotexFence, RotexFramebuffer,
    RotexFramebufferBuilder, RotexInstance, RotexRenderPass, RotexSemaphore, RotexSurface,
    RotexSwapchain, Severity, SubpassBlueprint,
};
use winit::{
    application::ApplicationHandler,
    dpi::LogicalSize,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    raw_window_handle::{HasDisplayHandle, HasWindowHandle},
    window::{Window, WindowId},
};

struct App {
    command_pool: Option<RotexCommandPool>,
    command_buffer: Option<RotexCommandBuffer>,
    image_available: Option<RotexSemaphore>,
    render_finished: Vec<RotexSemaphore>,
    in_flight_fence: Option<RotexFence>,
    graphics_queue_index: u32,
    framebuffers: Vec<RotexFramebuffer>,
    render_pass: Option<RotexRenderPass>,
    swapchain: Option<RotexSwapchain>,
    rotex_surface: Option<RotexSurface>,
    device: Option<RotexDevice>,
    instance: Option<RotexInstance>,
    window: Option<Window>,
    swapchain_outdated: bool,
}

impl Default for App {
    fn default() -> Self {
        Self {
            command_pool: None,
            command_buffer: None,
            image_available: None,
            render_finished: Vec::new(),
            in_flight_fence: None,
            graphics_queue_index: 0,
            framebuffers: Vec::new(),
            render_pass: None,
            swapchain: None,
            rotex_surface: None,
            device: None,
            instance: None,
            window: None,
            swapchain_outdated: false,
        }
    }
}

fn vulkan_error(err: vk::Result) -> RotexError {
    RotexError {
        kind: ErrorKind::Vulkan(err),
        severity: Severity::Fatal,
    }
}

fn is_swapchain_out_of_date(err: &RotexError) -> bool {
    matches!(
        err.vk_result_code(),
        Some(code) if code == vk::Result::ERROR_OUT_OF_DATE_KHR.as_raw()
    )
}

fn record_clear_commands(
    device: &RotexDevice,
    command_buffer: &RotexCommandBuffer,
    render_pass: &RotexRenderPass,
    framebuffer: &RotexFramebuffer,
) -> Result<(), RotexError> {
    command_buffer.begin(device, vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT)?;

    let clear_values = [vk::ClearValue {
        color: vk::ClearColorValue {
            float32: [0.05, 0.05, 0.1, 1.0],
        },
    }];

    command_buffer.begin_render_pass(device, render_pass, framebuffer, &clear_values);
    command_buffer.end_render_pass(device);
    command_buffer.end(device)?;

    Ok(())
}

fn create_render_finished_semaphores(
    device: &RotexDevice,
    count: usize,
) -> Result<Vec<RotexSemaphore>, RotexError> {
    (0..count).map(|_| RotexSemaphore::new(device)).collect()
}

fn destroy_render_finished_semaphores(device: &RotexDevice, semaphores: &mut Vec<RotexSemaphore>) {
    for semaphore in semaphores.drain(..) {
        semaphore.destroy(device);
    }
}

fn recreate_swapchain(app: &mut App) -> Result<(), RotexError> {
    let instance = app.instance.as_ref().unwrap();
    let device = app.device.as_ref().unwrap();
    let surface = app.rotex_surface.as_ref().unwrap();
    let render_pass = app.render_pass.as_ref().unwrap();

    unsafe { device.device().device_wait_idle() }.map_err(vulkan_error)?;

    for framebuffer in app.framebuffers.drain(..) {
        framebuffer.destroy(device);
    }
    destroy_render_finished_semaphores(device, &mut app.render_finished);
    if let Some(mut swapchain) = app.swapchain.take() {
        swapchain.destroy(device);
    }

    let swapchain = RotexSwapchain::new(instance, device, surface)?;
    let render_pass_handle = render_pass.handle();

    for view in swapchain.image_views() {
        let framebuffer = RotexFramebufferBuilder::new()
            .with_attachment(*view)
            .with_extent(swapchain.extent().width, swapchain.extent().height)
            .build(device, render_pass_handle)
            .map_err(vulkan_error)?;
        app.framebuffers.push(framebuffer);
    }

    app.render_finished = create_render_finished_semaphores(device, swapchain.images().len())?;
    app.swapchain = Some(swapchain);
    app.swapchain_outdated = false;
    Ok(())
}

fn draw_frame(app: &mut App) -> Result<(), RotexError> {
    if app.swapchain_outdated {
        recreate_swapchain(app)?;
    }

    let device = app.device.as_ref().unwrap();
    let swapchain = app.swapchain.as_ref().unwrap();
    let render_pass = app.render_pass.as_ref().unwrap();
    let command_buffer = app.command_buffer.as_ref().unwrap();
    let image_available = app.image_available.as_ref().unwrap();
    let in_flight_fence = app.in_flight_fence.as_ref().unwrap();

    in_flight_fence.wait(device, u64::MAX)?;

    let image_index = match swapchain.acquire_next_image(image_available) {
        Ok((index, _suboptimal)) => index,
        Err(err) if is_swapchain_out_of_date(&err) => {
            app.swapchain_outdated = true;
            return Ok(());
        }
        Err(err) => return Err(err),
    };

    let render_finished = &app.render_finished[image_index as usize];

    in_flight_fence.reset(device)?;

    unsafe {
        device
            .device()
            .reset_command_buffer(command_buffer.handle, vk::CommandBufferResetFlags::empty())
    }
    .map_err(vulkan_error)?;

    record_clear_commands(
        device,
        command_buffer,
        render_pass,
        &app.framebuffers[image_index as usize],
    )?;

    let wait_semaphores = [image_available.handle];
    let wait_stages = [vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
    let command_buffers = [command_buffer.handle];
    let signal_semaphores = [render_finished.handle];

    let submit_info = vk::SubmitInfo::default()
        .wait_semaphores(&wait_semaphores)
        .wait_dst_stage_mask(&wait_stages)
        .command_buffers(&command_buffers)
        .signal_semaphores(&signal_semaphores);

    let graphics_queue = device.get_queue(app.graphics_queue_index, 0);

    unsafe {
        device
            .device()
            .queue_submit(graphics_queue, &[submit_info], in_flight_fence.handle())
    }
    .map_err(vulkan_error)?;

    match swapchain.present(graphics_queue, image_index, render_finished) {
        Ok(_suboptimal) => Ok(()),
        Err(err) if is_swapchain_out_of_date(&err) => {
            app.swapchain_outdated = true;
            Ok(())
        }
        Err(err) => Err(err),
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_none() {
            match event_loop.create_window(
                Window::default_attributes()
                    .with_title("Rotex")
                    .with_inner_size(LogicalSize::new(800.0, 600.0))
                    .with_decorations(true)
                    .with_visible(true),
            ) {
                Ok(window) => {
                    self.window = Some(window);
                }
                Err(error) => {
                    eprintln!("failed to create window: {error}");
                    event_loop.exit();
                    return;
                }
            }
        }

        if self.device.is_some() {
            return;
        }

        let window = self
            .window
            .as_ref()
            .expect("window must exist after resumed");
        let instance = self
            .instance
            .as_ref()
            .expect("instance must be set before resumed");
        let adapter = match instance.enumerate_adapters().into_iter().next() {
            Some(adapter) => adapter,
            None => {
                eprintln!("no Vulkan adapters found");
                event_loop.exit();
                return;
            }
        };
        let descriptor = DeviceDescriptor {
            required_features: vk::PhysicalDeviceFeatures::default(),
            enable_swapchain: true,
            queues: vec![
                QueueRequest {
                    category: QueueCategory::Graphics,
                    count: 1,
                },
                QueueRequest {
                    category: QueueCategory::Transfer,
                    count: 1,
                },
            ],
        };
        match adapter.request_device(instance, descriptor) {
            Ok(device) => {
                self.graphics_queue_index = device
                    .queues()
                    .iter()
                    .find(|allocation| allocation.category == QueueCategory::Graphics)
                    .expect("graphics queue must exist")
                    .family_index;
                self.device = Some(device);
            }
            Err(error) => {
                eprintln!("failed to initialize Vulkan device: {error}");
                event_loop.exit();
                return;
            }
        }

        let surface = match unsafe {
            ash_window::create_surface(
                instance.entry(),
                instance.instance(),
                window.display_handle().unwrap().as_raw(),
                window.window_handle().unwrap().as_raw(),
                None,
            )
        } {
            Ok(surface) => surface,
            Err(error) => {
                eprintln!("failed to create surface: {error:?}");
                event_loop.exit();
                return;
            }
        };

        let rotex_surface = RotexSurface::new(instance, surface);
        let device = self.device.as_ref().unwrap();
        let swapchain = match RotexSwapchain::new(instance, device, &rotex_surface) {
            Ok(swapchain) => swapchain,
            Err(error) => {
                eprintln!("failed to create swapchain: {error}");
                event_loop.exit();
                return;
            }
        };

        self.render_pass = match RenderPassBuilder::new()
            .with_attachment(
                vk::AttachmentDescription::default()
                    .format(swapchain.format())
                    .samples(vk::SampleCountFlags::TYPE_1)
                    .load_op(vk::AttachmentLoadOp::CLEAR)
                    .store_op(vk::AttachmentStoreOp::STORE)
                    .initial_layout(vk::ImageLayout::UNDEFINED)
                    .final_layout(vk::ImageLayout::PRESENT_SRC_KHR),
            )
            .with_subpass(SubpassBlueprint {
                color_attachments: vec![0],
                depth_attachment: None,
            })
            .build(device)
        {
            Ok(render_pass) => Some(render_pass),
            Err(error) => {
                eprintln!("failed to create render pass: {error:?}");
                event_loop.exit();
                return;
            }
        };

        let render_pass_handle = self.render_pass.as_ref().unwrap().handle();
        self.framebuffers = Vec::with_capacity(swapchain.image_views().len());

        for view in swapchain.image_views() {
            let framebuffer = match RotexFramebufferBuilder::new()
                .with_attachment(*view)
                .with_extent(swapchain.extent().width, swapchain.extent().height)
                .build(device, render_pass_handle)
            {
                Ok(framebuffer) => framebuffer,
                Err(error) => {
                    eprintln!("failed to create framebuffer: {error:?}");
                    event_loop.exit();
                    return;
                }
            };
            self.framebuffers.push(framebuffer);
        }

        let command_pool = match RotexCommandPool::new(device) {
            Ok(pool) => pool,
            Err(error) => {
                eprintln!("failed to create command pool: {error}");
                event_loop.exit();
                return;
            }
        };
        let mut command_buffers = match command_pool.allocate_buffers(device, 1) {
            Ok(buffers) => buffers,
            Err(error) => {
                eprintln!("failed to allocate command buffer: {error}");
                event_loop.exit();
                return;
            }
        };

        let image_available = match RotexSemaphore::new(device) {
            Ok(semaphore) => semaphore,
            Err(error) => {
                eprintln!("failed to create image-available semaphore: {error}");
                event_loop.exit();
                return;
            }
        };
        let render_finished =
            match create_render_finished_semaphores(device, swapchain.images().len()) {
                Ok(semaphores) => semaphores,
                Err(error) => {
                    eprintln!("failed to create render-finished semaphores: {error}");
                    event_loop.exit();
                    return;
                }
            };
        let in_flight_fence = match RotexFence::new(device, true) {
            Ok(fence) => fence,
            Err(error) => {
                eprintln!("failed to create in-flight fence: {error}");
                event_loop.exit();
                return;
            }
        };

        self.command_pool = Some(command_pool);
        self.command_buffer = command_buffers.pop();
        self.image_available = Some(image_available);
        self.render_finished = render_finished;
        self.in_flight_fence = Some(in_flight_fence);
        self.rotex_surface = Some(rotex_surface);
        self.swapchain = Some(swapchain);

        window.request_redraw();
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, id: WindowId, event: WindowEvent) {
        if self.window.as_ref().map(|window| window.id()) != Some(id) {
            return;
        }

        match event {
            WindowEvent::CloseRequested => {
                println!("The close button was pressed; stopping");
                event_loop.exit();
            }
            WindowEvent::Resized(_size) => {
                self.swapchain_outdated = true;
                if let Some(window) = self.window.as_ref() {
                    window.request_redraw();
                }
            }
            WindowEvent::RedrawRequested => {
                if self.device.is_none() {
                    return;
                }

                if let Err(error) = draw_frame(self) {
                    eprintln!("failed to draw frame: {error}");
                    event_loop.exit();
                    return;
                }

                if let Some(window) = self.window.as_ref() {
                    window.request_redraw();
                }
            }
            _ => (),
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let event_loop = EventLoop::new()?;
    let extensions =
        ash_window::enumerate_required_extensions(event_loop.display_handle()?.as_raw())?;
    let instance = RotexInstance::new(extensions).map_err(|err| {
        std::io::Error::new(
            std::io::ErrorKind::Other,
            format!("failed to initialize Vulkan instance: {err}"),
        )
    })?;
    event_loop.set_control_flow(ControlFlow::Poll);
    let mut app = App {
        instance: Some(instance),
        ..Default::default()
    };
    event_loop.run_app(&mut app)?;

    if let Some(device) = app.device.as_ref() {
        if let Some(fence) = app.in_flight_fence.as_ref() {
            let _ = fence.wait(device, u64::MAX);
        }
        unsafe {
            let _ = device.device().device_wait_idle();
        }
    }

    if let Some(pool) = app.command_pool.take() {
        if let Some(device) = app.device.as_ref() {
            pool.destroy(device);
        }
    }
    if let Some(semaphore) = app.image_available.take() {
        if let Some(device) = app.device.as_ref() {
            semaphore.destroy(device);
        }
    }
    if let Some(device) = app.device.as_ref() {
        destroy_render_finished_semaphores(device, &mut app.render_finished);
    }
    if let Some(fence) = app.in_flight_fence.take() {
        if let Some(device) = app.device.as_ref() {
            fence.destroy(device);
        }
    }

    if let Some(device) = app.device.as_ref() {
        for framebuffer in app.framebuffers.drain(..) {
            framebuffer.destroy(device);
        }
        if let Some(render_pass) = app.render_pass.take() {
            render_pass.destroy(device);
        }
        if let Some(mut swapchain) = app.swapchain.take() {
            swapchain.destroy(device);
        }
    }
    if let Some(mut surface) = app.rotex_surface.take() {
        surface.destroy();
    }
    if let Some(mut device) = app.device.take() {
        device.destroy();
    }
    if let Some(mut instance) = app.instance.take() {
        instance.destroy();
    }

    Ok(())
}
