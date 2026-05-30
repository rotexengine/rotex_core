use rotex_core::{
    CullMode, DeviceDescriptor, Extent2D, GraphicsContext, FrameDescriptor, IndexFormat, InstanceDescriptor,
    MaterialDescriptor, MaterialId, MeshDescriptor, MeshId, MeshInstanceDescriptor, PassDescriptor,
    ResourceBatchCreate, ResourceCreateDescriptor, ResourceHandle, SceneDescriptor, SurfaceDescriptor,
    VertexAttribute, VertexBufferLayout, VertexFormat,
};
use winit::{
    application::ApplicationHandler,
    dpi::LogicalSize,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    raw_window_handle::{HasDisplayHandle, HasWindowHandle},
    window::{Window, WindowAttributes, WindowId},
};

#[derive(Default)]
struct App {
    window: Option<Window>,
    graphics_context: Option<GraphicsContext>,
    scene: Option<SceneDescriptor>,
    frame: Option<FrameDescriptor>,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct ColoredVertex {
    position: [f32; 3],
    color: [f32; 3],
}

fn window_extent(window: &Window) -> Extent2D {
    let size = window.inner_size();
    Extent2D {
        width: size.width.max(1),
        height: size.height.max(1),
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }

        let attrs = WindowAttributes::default()
            .with_title("Rotex Frontend Triangle")
            .with_inner_size(LogicalSize::new(900.0, 600.0));
        let window = event_loop.create_window(attrs).expect("window");
        let display_handle = window.display_handle().expect("display").as_raw();
        let window_handle = window.window_handle().expect("window").as_raw();
        let mut instance_descriptor = InstanceDescriptor::default();
        #[cfg(not(target_arch = "wasm32"))]
        {
            instance_descriptor.required_instance_extensions = ash_window::enumerate_required_extensions(
                display_handle,
            )
            .expect("required instance extensions")
            .iter()
            .map(|extension| unsafe { std::ffi::CStr::from_ptr(*extension) })
            .map(|extension| extension.to_string_lossy().into_owned())
            .collect();
        }
        let mut graphics_context =
            pollster::block_on(GraphicsContext::new(instance_descriptor, DeviceDescriptor::default()))
        .expect("triangle graphics_context");
        graphics_context
            .attach_surface(SurfaceDescriptor {
                display_handle,
                window_handle,
                extent: window_extent(&window),
            })
            .expect("attach surface");
        let resources = graphics_context
            .create_resources(ResourceBatchCreate::new(vec![
                ResourceCreateDescriptor::Mesh(mesh_from_vertices(
                    &[
                        ColoredVertex {
                            position: [0.0, -0.6, 0.0],
                            color: [1.0, 0.2, 0.3],
                        },
                        ColoredVertex {
                            position: [0.6, 0.6, 0.0],
                            color: [0.2, 1.0, 0.3],
                        },
                        ColoredVertex {
                            position: [-0.6, 0.6, 0.0],
                            color: [0.2, 0.4, 1.0],
                        },
                    ],
                    &[0, 1, 2],
                )),
                ResourceCreateDescriptor::Material(MaterialDescriptor {
                    vertex_shader_spv: include_bytes!(concat!(env!("OUT_DIR"), "/triangle.vert.spv"))
                        .to_vec(),
                    vertex_entry: "main".to_string(),
                    fragment_shader_spv: include_bytes!(concat!(env!("OUT_DIR"), "/triangle.frag.spv"))
                        .to_vec(),
                    fragment_entry: "main".to_string(),
                    enable_depth: false,
                    cull_mode: CullMode::Back,
                    texture: None,
                }),
            ]))
            .expect("triangle resources");
        let mesh_id = expect_mesh(resources.handles[0]);
        let material_id = expect_material(resources.handles[1]);
        let scene = SceneDescriptor::new(vec![MeshInstanceDescriptor::new(mesh_id, material_id)]);
        let frame = FrameDescriptor::new(vec![
            PassDescriptor::new("main").with_clear_color([0.06, 0.06, 0.09, 1.0]),
        ]);

        window.request_redraw();
        self.window = Some(window);
        self.graphics_context = Some(graphics_context);
        self.scene = Some(scene);
        self.frame = Some(frame);
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => {
                if let Some(graphics_context) = self.graphics_context.as_mut() {
                    if let Err(err) = graphics_context.resize(Extent2D {
                        width: size.width.max(1),
                        height: size.height.max(1),
                    }) {
                        eprintln!("resize failed: {err}");
                        event_loop.exit();
                    }
                }
            }
            WindowEvent::RedrawRequested => {
                if let (Some(graphics_context), Some(scene), Some(frame)) =
                    (self.graphics_context.as_mut(), self.scene.as_ref(), self.frame.as_ref())
                {
                    if let Err(err) = graphics_context.render(scene, frame) {
                        eprintln!("render failed: {err}");
                        event_loop.exit();
                    }
                }
                if let Some(window) = self.window.as_ref() {
                    window.request_redraw();
                }
            }
            _ => {}
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let event_loop = EventLoop::new()?;
    event_loop.set_control_flow(ControlFlow::Poll);
    let mut app = App::default();
    event_loop.run_app(&mut app)?;

    if let Some(graphics_context) = app.graphics_context.take() {
        graphics_context.destroy();
    }
    Ok(())
}

fn expect_mesh(handle: ResourceHandle) -> MeshId {
    match handle {
        ResourceHandle::Mesh(id) => id,
        _ => panic!("expected mesh handle"),
    }
}

fn expect_material(handle: ResourceHandle) -> MaterialId {
    match handle {
        ResourceHandle::Material(id) => id,
        _ => panic!("expected material handle"),
    }
}

fn mesh_from_vertices(vertices: &[ColoredVertex], indices: &[u32]) -> MeshDescriptor {
    MeshDescriptor {
        vertex_data: vertices_as_bytes(vertices),
        vertex_layout: colored_vertex_layout(),
        index_data: indices_as_bytes(indices),
        index_format: IndexFormat::Uint32,
        index_count: indices.len() as u32,
    }
}

fn colored_vertex_layout() -> VertexBufferLayout {
    VertexBufferLayout {
        array_stride: 24,
        attributes: vec![
            VertexAttribute {
                location: 0,
                format: VertexFormat::Float32x3,
                offset: 0,
            },
            VertexAttribute {
                location: 1,
                format: VertexFormat::Float32x3,
                offset: 12,
            },
        ],
    }
}

fn vertices_as_bytes(vertices: &[ColoredVertex]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(vertices.len() * 24);
    for vertex in vertices {
        for component in vertex.position {
            bytes.extend_from_slice(&component.to_le_bytes());
        }
        for component in vertex.color {
            bytes.extend_from_slice(&component.to_le_bytes());
        }
    }
    bytes
}

fn indices_as_bytes(indices: &[u32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(indices.len() * 4);
    for index in indices {
        bytes.extend_from_slice(&index.to_le_bytes());
    }
    bytes
}
