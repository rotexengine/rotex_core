use rotex_core::{CoreInstance, DeviceContext};
use winit::{
    application::ApplicationHandler,
    dpi::LogicalSize,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    window::{Window, WindowId},
};

#[derive(Default)]
struct App {
    window: Option<Window>,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
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
            }
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, id: WindowId, event: WindowEvent) {
        let window = self
            .window
            .as_ref()
            .expect("window must exist after resumed");

        if id != window.id() {
            return;
        }

        match event {
            WindowEvent::CloseRequested => {
                println!("The close button was pressed; stopping");
                event_loop.exit();
            }
            WindowEvent::Resized(_size) => {
                window.request_redraw();
            }
            WindowEvent::RedrawRequested => {
                window.request_redraw();
            }
            _ => (),
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let core = CoreInstance::new().map_err(|err| {
        std::io::Error::new(
            std::io::ErrorKind::Other,
            format!("failed to initialize Vulkan engine: {err}"),
        )
    })?;
    let _context = DeviceContext::new(core, None).map_err(|err| {
        std::io::Error::new(
            std::io::ErrorKind::Other,
            format!("failed to initialize device context: {err}"),
        )
    })?;
    let event_loop = EventLoop::new()?;
    event_loop.set_control_flow(ControlFlow::Poll);
    let mut app = App::default();
    event_loop.run_app(&mut app)?;

    Ok(())
}
