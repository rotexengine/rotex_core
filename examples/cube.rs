use rotex_core::{
    CullMode, DeviceDescriptor, Extent2D, FrameDescriptor, GraphicsContext, IndexFormat, InstanceDescriptor,
    MaterialDescriptor, MaterialId, MeshDescriptor, MeshId, MeshInstanceDescriptor, PassDescriptor,
    ResourceBatchCreate, ResourceBatchUpdate, ResourceCreateDescriptor, ResourceHandle, TextureDescriptor,
    TextureFormat, TextureId,
    ResourceUpdateDescriptor, SceneDescriptor, SurfaceDescriptor, VertexAttribute, VertexBufferLayout,
    VertexFormat,
};
use web_time::Instant;
use winit::{
    application::ApplicationHandler,
    dpi::LogicalSize,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    raw_window_handle::{HasDisplayHandle, HasWindowHandle},
    window::{Window, WindowAttributes, WindowId},
};
#[cfg(target_arch = "wasm32")]
use wasm_bindgen_futures::spawn_local;
#[cfg(target_arch = "wasm32")]
use winit::platform::web::{EventLoopExtWebSys, WindowAttributesExtWebSys};

#[repr(C)]
#[derive(Clone, Copy)]
struct ColoredVertex {
    position: [f32; 3],
    color: [f32; 3],
}

#[derive(Default)]
struct App {
    window: Option<Window>,
    graphics_context: Option<GraphicsContext>,
    scene: Option<SceneDescriptor>,
    frame: Option<FrameDescriptor>,
    mesh_id: Option<MeshId>,
    base_positions: Vec<[f32; 3]>,
    colors: Vec<[f32; 3]>,
    indices: Vec<u32>,
    start_time: Option<Instant>,
}

impl App {
    #[cfg(target_arch = "wasm32")]
    fn with_graphics_context(graphics_context: GraphicsContext) -> Self {
        Self {
            graphics_context: Some(graphics_context),
            ..Default::default()
        }
    }
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
            .with_title("Rotex Frontend Cube")
            .with_inner_size(LogicalSize::new(900.0, 700.0));
        #[cfg(target_arch = "wasm32")]
        let attrs = attrs.with_append(true);
        let window = match event_loop.create_window(attrs) {
            Ok(window) => window,
            Err(err) => {
                eprintln!("window creation failed: {err}");
                event_loop.exit();
                return;
            }
        };
        let display_handle = match window.display_handle() {
            Ok(handle) => handle.as_raw(),
            Err(err) => {
                eprintln!("display handle unavailable: {err}");
                event_loop.exit();
                return;
            }
        };
        let window_handle = match window.window_handle() {
            Ok(handle) => handle.as_raw(),
            Err(err) => {
                eprintln!("window handle unavailable: {err}");
                event_loop.exit();
                return;
            }
        };

        #[cfg(not(target_arch = "wasm32"))]
        if self.graphics_context.is_none() {
            let mut desc = InstanceDescriptor::default();
            let required_extensions = match ash_window::enumerate_required_extensions(display_handle) {
                Ok(exts) => exts,
                Err(err) => {
                    eprintln!("failed to enumerate required instance extensions: {err}");
                    event_loop.exit();
                    return;
                }
            };
            desc.required_instance_extensions = required_extensions
                .iter()
                .map(|extension| unsafe { std::ffi::CStr::from_ptr(*extension) })
                .map(|extension| extension.to_string_lossy().into_owned())
                .collect();
            match pollster::block_on(GraphicsContext::new(desc, DeviceDescriptor::default())) {
                Ok(graphics_context) => self.graphics_context = Some(graphics_context),
                Err(err) => {
                    eprintln!("cube graphics_context initialization failed: {err}");
                    event_loop.exit();
                    return;
                }
            }
        }

        let Some(mut graphics_context) = self.graphics_context.take() else {
            eprintln!("cube graphics_context missing during resume");
            event_loop.exit();
            return;
        };
        if let Err(err) = graphics_context.attach_surface(SurfaceDescriptor {
                display_handle,
                window_handle,
                extent: window_extent(&window),
            }) {
            eprintln!("attach surface failed: {err}");
            event_loop.exit();
            return;
        }

        let (positions, colors, indices) = cube_geometry();
        let initial_vertices = build_vertices(&positions, &colors, 0.0);
        let resources = match graphics_context.create_resources(ResourceBatchCreate::new(vec![
                ResourceCreateDescriptor::Mesh(MeshDescriptor {
                    vertex_data: vertices_as_bytes(&initial_vertices),
                    vertex_layout: colored_vertex_layout(),
                    index_data: indices_as_bytes(&indices),
                    index_format: IndexFormat::Uint32,
                    index_count: indices.len() as u32,
                }),
                ResourceCreateDescriptor::Texture(TextureDescriptor {
                    width: 1,
                    height: 1,
                    format: TextureFormat::Rgba8Unorm,
                    data: vec![255, 255, 255, 255],
                }),
            ])) {
            Ok(resources) => resources,
            Err(err) => {
                eprintln!("cube mesh/texture creation failed: {err}");
                event_loop.exit();
                return;
            }
        };
        if resources.handles.len() < 2 {
            eprintln!(
                "cube mesh/texture creation returned insufficient handles: {}",
                resources.handles.len()
            );
            event_loop.exit();
            return;
        }
        let Some(mesh_id) = expect_mesh(resources.handles[0]) else {
            eprintln!("expected mesh handle at resources[0]");
            event_loop.exit();
            return;
        };
        let Some(texture_id) = expect_texture(resources.handles[1]) else {
            eprintln!("expected texture handle at resources[1]");
            event_loop.exit();
            return;
        };

        let material_resources = match graphics_context.create_resources(ResourceBatchCreate::new(vec![
                ResourceCreateDescriptor::Material(MaterialDescriptor {
                    vertex_shader_spv: include_bytes!(concat!(env!("OUT_DIR"), "/cube.vert.spv")).to_vec(),
                    vertex_entry: "main".to_string(),
                    fragment_shader_spv: include_bytes!(concat!(env!("OUT_DIR"), "/cube.frag.spv")).to_vec(),
                    fragment_entry: "main".to_string(),
                    enable_depth: true,
                    cull_mode: CullMode::Back,
                    texture: Some(texture_id),
                }),
            ])) {
            Ok(resources) => resources,
            Err(err) => {
                eprintln!("cube material creation failed: {err}");
                event_loop.exit();
                return;
            }
        };
        if material_resources.handles.is_empty() {
            eprintln!(
                "cube material creation returned insufficient handles: {}",
                material_resources.handles.len()
            );
            event_loop.exit();
            return;
        }
        let Some(material_id) = expect_material(material_resources.handles[0]) else {
            eprintln!("expected material handle at material_resources[0]");
            event_loop.exit();
            return;
        };
        let scene = SceneDescriptor::new(vec![MeshInstanceDescriptor::new(mesh_id, material_id)]);
        let frame = FrameDescriptor::new(vec![
            PassDescriptor::new("main")
                .with_clear_color([0.02, 0.02, 0.04, 1.0])
                .with_clear_depth(Some(1.0)),
        ]);

        window.request_redraw();
        self.window = Some(window);
        self.graphics_context = Some(graphics_context);
        self.scene = Some(scene);
        self.frame = Some(frame);
        self.mesh_id = Some(mesh_id);
        self.base_positions = positions;
        self.colors = colors;
        self.indices = indices;
        self.start_time = Some(Instant::now());
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
                if let (Some(graphics_context), Some(scene), Some(frame), Some(mesh_id), Some(start_time)) = (
                    self.graphics_context.as_mut(),
                    self.scene.as_ref(),
                    self.frame.as_ref(),
                    self.mesh_id,
                    self.start_time.as_ref(),
                ) {
                    let t = start_time.elapsed().as_secs_f32();
                    let vertices = build_vertices(&self.base_positions, &self.colors, t);
                    let update = ResourceBatchUpdate::new(vec![ResourceUpdateDescriptor::Mesh {
                        id: mesh_id,
                        vertex_data: vertices_as_bytes(&vertices),
                        vertex_layout: colored_vertex_layout(),
                        index_data: indices_as_bytes(&self.indices),
                        index_format: IndexFormat::Uint32,
                        index_count: self.indices.len() as u32,
                    }]);
                    if let Err(err) = graphics_context.update_resources(update) {
                        eprintln!("resource update failed: {err}");
                        event_loop.exit();
                        return;
                    }
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

#[cfg(not(target_arch = "wasm32"))]
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

#[cfg(target_arch = "wasm32")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let event_loop = EventLoop::new()?;
    event_loop.set_control_flow(ControlFlow::Poll);
    spawn_local(async move {
        match GraphicsContext::new(InstanceDescriptor::default(), DeviceDescriptor::default()).await {
            Ok(graphics_context) => event_loop.spawn_app(App::with_graphics_context(graphics_context)),
            Err(err) => eprintln!("cube graphics_context initialization failed: {err}"),
        }
    });
    Ok(())
}

fn cube_geometry() -> (Vec<[f32; 3]>, Vec<[f32; 3]>, Vec<u32>) {
    let positions = vec![
        [-0.6, -0.6, -0.6],
        [0.6, -0.6, -0.6],
        [0.6, 0.6, -0.6],
        [-0.6, 0.6, -0.6],
        [-0.6, -0.6, 0.6],
        [0.6, -0.6, 0.6],
        [0.6, 0.6, 0.6],
        [-0.6, 0.6, 0.6],
    ];
    let colors = vec![
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 1.0, 0.0],
        [1.0, 0.0, 1.0],
        [0.0, 1.0, 1.0],
        [1.0, 0.5, 0.0],
        [0.4, 0.2, 1.0],
    ];
    let indices = vec![
        // back (-Z)
        0, 2, 1, 2, 0, 3,
        // front (+Z)
        4, 5, 6, 6, 7, 4,
        // left (-X)
        0, 4, 7, 7, 3, 0,
        // right (+X)
        1, 6, 5, 6, 1, 2,
        // top (+Y)
        3, 6, 2, 6, 3, 7,
        // bottom (-Y)
        0, 1, 5, 5, 4, 0,
    ];
    (positions, colors, indices)
}

fn build_vertices(base_positions: &[[f32; 3]], colors: &[[f32; 3]], time: f32) -> Vec<ColoredVertex> {
    base_positions
        .iter()
        .zip(colors.iter())
        .map(|(position, color)| {
            let mut p = *position;
            rotate_vertex(&mut p, time);
            ColoredVertex {
                position: p,
                color: *color,
            }
        })
        .collect()
}

fn rotate_vertex(position: &mut [f32; 3], time: f32) {
    let angle_y = time;
    let angle_x = time * 0.7;
    let (sy, cy) = angle_y.sin_cos();
    let (sx, cx) = angle_x.sin_cos();

    let x = position[0] * cy + position[2] * sy;
    let z = -position[0] * sy + position[2] * cy;
    let y = position[1] * cx - z * sx;
    let z2 = position[1] * sx + z * cx;

    position[0] = x * 0.7;
    position[1] = y * 0.7;
    position[2] = (z2 * 0.45) + 0.5;
}

fn expect_mesh(handle: ResourceHandle) -> Option<MeshId> {
    match handle {
        ResourceHandle::Mesh(id) => Some(id),
        _ => None,
    }
}

fn expect_material(handle: ResourceHandle) -> Option<MaterialId> {
    match handle {
        ResourceHandle::Material(id) => Some(id),
        _ => None,
    }
}

fn expect_texture(handle: ResourceHandle) -> Option<TextureId> {
    match handle {
        ResourceHandle::Texture(id) => Some(id),
        _ => None,
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
