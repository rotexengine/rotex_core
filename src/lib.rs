use std::ffi::CStr;

use ash::vk;
use thiserror::Error;

unsafe extern "system" fn vulkan_debug_utils_callback(
    message_severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    message_type: vk::DebugUtilsMessageTypeFlagsEXT,
    p_callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT,
    _p_user_data: *mut std::ffi::c_void,
) -> vk::Bool32 {
    if p_callback_data.is_null() {
        return vk::FALSE;
    }

    let message = unsafe { std::ffi::CStr::from_ptr((*p_callback_data).p_message) };
    let severity = format!("{:?}", message_severity).to_lowercase();
    let ty = format!("{:?}", message_type).to_lowercase();
    println!("[Debug][{}][{}] {:?}", severity, ty, message);
    vk::FALSE
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Severity {
    Fatal,
    Warning,
}

#[derive(Error, Debug)]
pub enum ErrorKind {
    #[error("Vulkan API error: {0:?}")]
    Vulkan(vk::Result),

    #[error("Missing Vulkan Layer")]
    MissingLayer(String),

    #[error("No compatible physical device found")]
    NoCompatibleDevice,

    #[error("Failed to load Vulkan library")]
    Loading(#[from] ash::LoadingError),
}

#[derive(Debug)]
pub struct EngineError {
    pub kind: ErrorKind,
    pub severity: Severity,
}

impl std::fmt::Display for EngineError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.kind {
            ErrorKind::Vulkan(err) => write!(
                f,
                "[{:?}] Vulkan API error: {:?} (code {})",
                self.severity,
                err,
                err.as_raw()
            ),
            other => write!(f, "[{:?}] {}", self.severity, other),
        }
    }
}

impl std::error::Error for EngineError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.kind)
    }
}

impl EngineError {
    pub fn vk_result_code(&self) -> Option<i32> {
        match &self.kind {
            ErrorKind::Vulkan(err) => Some(err.as_raw()),
            _ => None,
        }
    }
}

pub struct CoreInstance {
    pub instance: ash::Instance,
    debug_utils: Option<ash::ext::debug_utils::Instance>,
    debug_messenger: Option<vk::DebugUtilsMessengerEXT>,
}

impl CoreInstance {
    pub fn new() -> Result<Self, EngineError> {
        Self::init_vk_instance()
    }

    pub fn instance(&self) -> &ash::Instance {
        &self.instance
    }

    fn init_vk_instance() -> Result<Self, EngineError> {
        let enable_validation = cfg!(debug_assertions);
        let entry = ash::Entry::linked();
        let enginename = std::ffi::CString::new("Rotex").unwrap();
        let appname = std::ffi::CString::new("Rotex").unwrap();
        let app_info = vk::ApplicationInfo::default()
            .application_name(&appname)
            .engine_name(&enginename)
            .application_version(vk::make_api_version(0, 0, 1, 0))
            .engine_version(vk::make_api_version(0, 0, 1, 0))
            .api_version(vk::make_api_version(0, 1, 4, 0));
        let layer_names: Vec<std::ffi::CString> = if enable_validation {
            vec![std::ffi::CString::new("VK_LAYER_KHRONOS_validation").unwrap()]
        } else {
            Vec::new()
        };
        let layer_name_pointers: Vec<*const i8> = layer_names
            .iter()
            .map(|layer_name| layer_name.as_ptr())
            .collect();
        let extension_name_pointers: Vec<*const i8> = if enable_validation {
            vec![
                ash::ext::debug_utils::NAME.as_ptr(),
                ash::khr::surface::NAME.as_ptr(),
                ash::khr::xlib_surface::NAME.as_ptr(),
            ]
        } else {
            Vec::new()
        };

        let mut debugcreateinfo = vk::DebugUtilsMessengerCreateInfoEXT::default()
            .message_severity(
                vk::DebugUtilsMessageSeverityFlagsEXT::WARNING
                    | vk::DebugUtilsMessageSeverityFlagsEXT::VERBOSE
                    | vk::DebugUtilsMessageSeverityFlagsEXT::INFO
                    | vk::DebugUtilsMessageSeverityFlagsEXT::ERROR,
            )
            .message_type(
                vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                    | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE
                    | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION,
            )
            .pfn_user_callback(Some(vulkan_debug_utils_callback));

        let mut instance_create_info = vk::InstanceCreateInfo::default()
            .application_info(&app_info)
            .enabled_layer_names(layer_name_pointers.as_ref())
            .enabled_extension_names(extension_name_pointers.as_ref());
        if enable_validation {
            instance_create_info = instance_create_info.push_next(&mut debugcreateinfo);
        }
        if enable_validation {
            dbg!(&instance_create_info);
        }
        let instance =
            unsafe { entry.create_instance(&instance_create_info, None) }.map_err(|err| {
                EngineError {
                    kind: ErrorKind::Vulkan(err),
                    severity: Severity::Fatal,
                }
            })?;

        let (debug_utils, debug_messenger) = if enable_validation {
            let debug_utils = ash::ext::debug_utils::Instance::new(&entry, &instance);
            let debug_messenger =
                unsafe { debug_utils.create_debug_utils_messenger(&debugcreateinfo, None) };
            match debug_messenger {
                Ok(debug_messenger) => (Some(debug_utils), Some(debug_messenger)),
                Err(err) => {
                    unsafe {
                        instance.destroy_instance(None);
                    }
                    return Err(EngineError {
                        kind: ErrorKind::Vulkan(err),
                        severity: Severity::Fatal,
                    });
                }
            }
        } else {
            (None, None)
        };
        Ok(Self {
            instance,
            debug_utils,
            debug_messenger,
        })
    }
}

impl Drop for CoreInstance {
    fn drop(&mut self) {
        unsafe {
            if let (Some(debug_utils), Some(debug_messenger)) =
                (self.debug_utils.as_ref(), self.debug_messenger)
            {
                debug_utils.destroy_debug_utils_messenger(debug_messenger, None);
            }
            self.instance.destroy_instance(None)
        }
    }
}

pub struct SurfaceSupport<'a> {
    pub loader: &'a ash::khr::surface::Instance,
    pub surface: vk::SurfaceKHR,
}

pub struct DeviceContext {
    physical_device: vk::PhysicalDevice,
    graphics_queue_index: u32,
    transfer_queue_index: u32,
    device: ash::Device,
}

impl DeviceContext {
    pub fn new(
        core: CoreInstance,
        surface_support: Option<SurfaceSupport<'_>>,
    ) -> Result<Self, EngineError> {
        let needs_swapchain = surface_support.is_some();
        let (physical_device, graphics_queue_index, transfer_queue_index) =
            Self::pick_physical_device(&core, surface_support)?;
        let device = Self::create_logical_device(
            core,
            physical_device,
            graphics_queue_index,
            transfer_queue_index,
            needs_swapchain,
        )?;
        Ok(DeviceContext {
            physical_device,
            graphics_queue_index,
            transfer_queue_index,
            device,
        })
    }

    pub fn physical_device(&self) -> vk::PhysicalDevice {
        self.physical_device
    }

    pub fn graphics_queue_index(&self) -> u32 {
        self.graphics_queue_index
    }

    pub fn transfer_queue_index(&self) -> u32 {
        self.transfer_queue_index
    }

    pub fn device(&self) -> &ash::Device {
        &self.device
    }

    fn pick_physical_device(
        core: &CoreInstance,
        surface_support: Option<SurfaceSupport<'_>>,
    ) -> Result<(vk::PhysicalDevice, u32, u32), EngineError> {
        let devices = unsafe { core.instance.enumerate_physical_devices() }.map_err(|err| {
            EngineError {
                kind: ErrorKind::Vulkan(err),
                severity: Severity::Fatal,
            }
        })?;

        let mut best_device = None;
        let mut best_score = 0u64;
        for device in devices {
            let props = unsafe { core.instance.get_physical_device_properties(device) };
            let features = unsafe { core.instance.get_physical_device_features(device) };
            let memory_props =
                unsafe { core.instance.get_physical_device_memory_properties(device) };
            let queue_families = unsafe {
                core.instance
                    .get_physical_device_queue_family_properties(device)
            };
            let graphics_queue_index = if let Some(surface_support) = surface_support.as_ref() {
                let mut selected = None;
                for (index, family) in queue_families.iter().enumerate() {
                    if !family.queue_flags.contains(vk::QueueFlags::GRAPHICS) {
                        continue;
                    }
                    let supports_present = unsafe {
                        surface_support.loader.get_physical_device_surface_support(
                            device,
                            index as u32,
                            surface_support.surface,
                        )
                    }
                    .map_err(|err| EngineError {
                        kind: ErrorKind::Vulkan(err),
                        severity: Severity::Fatal,
                    })?;
                    if supports_present {
                        selected = Some(index as u32);
                        break;
                    }
                }
                selected
            } else {
                queue_families
                    .iter()
                    .enumerate()
                    .find(|(_, family)| family.queue_flags.contains(vk::QueueFlags::COMPUTE))
                    .map(|(index, _)| index as u32)
            };
            let dedicated_transfer_queue_index = queue_families
                .iter()
                .enumerate()
                .find(|(_, family)| {
                    family.queue_flags.contains(vk::QueueFlags::TRANSFER)
                        && !family.queue_flags.contains(vk::QueueFlags::GRAPHICS)
                        && !family.queue_flags.contains(vk::QueueFlags::COMPUTE)
                })
                .map(|(index, _)| index as u32);
            let graphics_queue_index = match graphics_queue_index {
                Some(index) => index,
                None => {
                    continue;
                }
            };
            let transfer_queue_index =
                dedicated_transfer_queue_index.unwrap_or(graphics_queue_index);

            let needs_swapchain = surface_support.is_some();
            if needs_swapchain {
                let extensions = unsafe {
                    core.instance
                        .enumerate_device_extension_properties(device)
                }
                .map_err(|err| EngineError {
                    kind: ErrorKind::Vulkan(err),
                    severity: Severity::Fatal,
                })?;
                let has_swapchain = extensions.iter().any(|ext| unsafe {
                    CStr::from_ptr(ext.extension_name.as_ptr()) == vk::KHR_SWAPCHAIN_NAME
                });
                if !has_swapchain {
                    continue;
                }
            }

            let mut score = 0u64;
            let device_local_bytes = memory_props.memory_heaps
                [..memory_props.memory_heap_count as usize]
                .iter()
                .filter(|heap| heap.flags.contains(vk::MemoryHeapFlags::DEVICE_LOCAL))
                .map(|heap| heap.size)
                .sum::<u64>();
            let device_local_gb = device_local_bytes / (1024 * 1024 * 1024);
            score += device_local_gb * 100;
            score += props.limits.max_image_dimension2_d as u64;
            score += props.limits.max_image_dimension3_d as u64;
            score += props.limits.max_bound_descriptor_sets as u64 * 10;
            score += (props.limits.max_sampler_anisotropy * 10.0) as u64;
            if features.sampler_anisotropy == vk::TRUE {
                score += 100;
            }
            if features.geometry_shader == vk::TRUE {
                score += 50;
            }
            if dedicated_transfer_queue_index.is_some() {
                score += 200;
            }
            match props.device_type {
                vk::PhysicalDeviceType::DISCRETE_GPU => score += 1000,
                vk::PhysicalDeviceType::INTEGRATED_GPU => score += 500,
                vk::PhysicalDeviceType::VIRTUAL_GPU => score += 200,
                vk::PhysicalDeviceType::CPU => score += 100,
                _ => {}
            }
            if score > best_score {
                best_score = score;
                best_device = Some((device, graphics_queue_index, transfer_queue_index));
            }
        }

        best_device.ok_or(EngineError {
            kind: ErrorKind::NoCompatibleDevice,
            severity: Severity::Fatal,
        })
    }

    fn create_logical_device(
        core: CoreInstance,
        physical_device: vk::PhysicalDevice,
        graphics_queue_index: u32,
        transfer_queue_index: u32,
        enable_swapchain: bool,
    ) -> Result<ash::Device, EngineError> {
        let priorities = [1.0f32];
        let mut queue_family_indices = vec![graphics_queue_index];
        if transfer_queue_index != graphics_queue_index {
            queue_family_indices.push(transfer_queue_index);
        }
        let queue_create_infos: Vec<vk::DeviceQueueCreateInfo> = queue_family_indices
            .into_iter()
            .map(|queue_family_index| {
                vk::DeviceQueueCreateInfo::default()
                    .queue_family_index(queue_family_index)
                    .queue_priorities(&priorities)
            })
            .collect();
        let device_extensions: Vec<*const i8> = if enable_swapchain {
            vec![vk::KHR_SWAPCHAIN_NAME.as_ptr()]
        } else {
            Vec::new()
        };
        let device_create_info = vk::DeviceCreateInfo::default()
            .queue_create_infos(&queue_create_infos)
            .enabled_extension_names(&device_extensions);

        unsafe { core.instance.create_device(physical_device, &device_create_info, None) }
            .map_err(|err| EngineError {
                kind: ErrorKind::Vulkan(err),
                severity: Severity::Fatal,
            })
    }
}

impl Drop for DeviceContext {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_device(None);
        }
    }
}
