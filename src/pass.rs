use ash::vk;

use crate::device::RotexDevice;

#[derive(Debug, Clone)]
pub struct SubpassBlueprint {
    pub color_attachments: Vec<u32>,
    pub depth_attachment: Option<u32>,
}

pub struct RenderPass {
    render_pass: vk::RenderPass,
    attachments: Vec<vk::AttachmentDescription>,
}

impl RenderPass {
    pub fn handle(&self) -> vk::RenderPass {
        self.render_pass
    }

    pub fn attachments(&self) -> &[vk::AttachmentDescription] {
        &self.attachments
    }

    pub fn destroy(&self, device: &RotexDevice) {
        unsafe {
            device.device().destroy_render_pass(self.render_pass, None);
        }
    }
}

pub struct RenderPassBuilder {
    attachments: Vec<vk::AttachmentDescription>,
    subpasses: Vec<SubpassBlueprint>,
    dependencies: Vec<vk::SubpassDependency>,
}

impl RenderPassBuilder {
    pub fn new() -> Self {
        Self {
            attachments: Vec::new(),
            subpasses: Vec::new(),
            dependencies: Vec::new(),
        }
    }

    pub fn with_attachment(mut self, attachment: vk::AttachmentDescription) -> Self {
        self.attachments.push(attachment);
        self
    }

    pub fn with_subpass(mut self, subpass: SubpassBlueprint) -> Self {
        self.subpasses.push(subpass);
        self
    }

    pub fn with_dependency(mut self, dependency: vk::SubpassDependency) -> Self {
        self.dependencies.push(dependency);
        self
    }

    pub fn build(self, device: &RotexDevice) -> Result<RenderPass, vk::Result> {
        let mut all_color_refs: Vec<Vec<vk::AttachmentReference>> =
            Vec::with_capacity(self.subpasses.len());
        let mut all_depth_refs: Vec<Option<vk::AttachmentReference>> =
            Vec::with_capacity(self.subpasses.len());

        for blueprint in &self.subpasses {
            let color_refs: Vec<vk::AttachmentReference> = blueprint
                .color_attachments
                .iter()
                .map(|&idx| {
                    vk::AttachmentReference::default()
                        .attachment(idx)
                        .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                })
                .collect();
            all_color_refs.push(color_refs);

            let depth_ref = blueprint.depth_attachment.map(|idx| {
                vk::AttachmentReference::default()
                    .attachment(idx)
                    .layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
            });
            all_depth_refs.push(depth_ref);
        }

        let mut vk_subpasses: Vec<vk::SubpassDescription> =
            Vec::with_capacity(self.subpasses.len());

        for i in 0..self.subpasses.len() {
            let mut subpass_desc = vk::SubpassDescription::default()
                .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
                .color_attachments(&all_color_refs[i]);

            if let Some(depth_ref) = &all_depth_refs[i] {
                subpass_desc = subpass_desc.depth_stencil_attachment(depth_ref);
            }

            vk_subpasses.push(subpass_desc);
        }

        let render_pass_info = vk::RenderPassCreateInfo::default()
            .attachments(&self.attachments)
            .subpasses(&vk_subpasses)
            .dependencies(&self.dependencies);

        let render_pass = unsafe {
            device
                .device()
                .create_render_pass(&render_pass_info, None)?
        };
        Ok(RenderPass {
            render_pass,
            attachments: self.attachments,
        })
    }
}

pub struct Framebuffer {
    framebuffer: vk::Framebuffer,
    extent: vk::Extent2D,
}

impl Framebuffer {
    pub fn handle(&self) -> vk::Framebuffer {
        self.framebuffer
    }

    pub fn extent(&self) -> vk::Extent2D {
        self.extent
    }

    pub fn destroy(&self, device: &RotexDevice) {
        unsafe {
            device.device().destroy_framebuffer(self.framebuffer, None);
        }
    }
}

pub struct FramebufferBuilder {
    attachments: Vec<vk::ImageView>,
    width: u32,
    height: u32,
    layers: u32,
}

impl FramebufferBuilder {
    pub fn new() -> Self {
        Self {
            attachments: Vec::new(),
            width: 0,
            height: 0,
            layers: 1,
        }
    }

    pub fn with_attachment(mut self, attachment: vk::ImageView) -> Self {
        self.attachments.push(attachment);
        self
    }

    pub fn with_extent(mut self, width: u32, height: u32) -> Self {
        self.width = width;
        self.height = height;
        self
    }

    pub fn with_layers(mut self, layers: u32) -> Self {
        self.layers = layers;
        self
    }

    pub fn build(
        self,
        device: &RotexDevice,
        render_pass: vk::RenderPass,
    ) -> Result<Framebuffer, vk::Result> {
        let framebuffer_info = vk::FramebufferCreateInfo::default()
            .render_pass(render_pass)
            .attachments(&self.attachments)
            .width(self.width)
            .height(self.height)
            .layers(self.layers);

        let framebuffer = unsafe { device.device().create_framebuffer(&framebuffer_info, None) }?;
        Ok(Framebuffer {
            framebuffer,
            extent: vk::Extent2D {
                width: self.width,
                height: self.height,
            },
        })
    }
}
