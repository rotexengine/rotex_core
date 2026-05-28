use std::collections::BTreeMap;
use std::ffi::CStr;

use ash::vk;

use crate::error::{ErrorKind, RotexError, Severity};

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

pub struct RotexInstance {
    entry: ash::Entry,
    instance: ash::Instance,
    _debug_utils: Option<ash::ext::debug_utils::Instance>,
    _debug_messenger: Option<vk::DebugUtilsMessengerEXT>,
}

impl RotexInstance {
    pub fn new(extensions: &[*const i8]) -> Result<Self, RotexError> {
        let enable_validation = cfg!(debug_assertions);
        let entry = ash::Entry::linked();
        let appname = std::ffi::CString::new("Rotex").unwrap();
        let enginename = std::ffi::CString::new("Rotex").unwrap();
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
        let mut extension_name_pointers = extensions.to_vec();
        if enable_validation {
            extension_name_pointers.push(ash::ext::debug_utils::NAME.as_ptr());
        }

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
                RotexError {
                    kind: ErrorKind::Vulkan(err),
                    severity: Severity::Fatal,
                }
            })?;

        let (_debug_utils, _debug_messenger) = if enable_validation {
            let debug_utils = ash::ext::debug_utils::Instance::new(&entry, &instance);
            let debug_messenger =
                unsafe { debug_utils.create_debug_utils_messenger(&debugcreateinfo, None) };
            match debug_messenger {
                Ok(debug_messenger) => (Some(debug_utils), Some(debug_messenger)),
                Err(err) => {
                    unsafe {
                        instance.destroy_instance(None);
                    }
                    return Err(RotexError {
                        kind: ErrorKind::Vulkan(err),
                        severity: Severity::Fatal,
                    });
                }
            }
        } else {
            (None, None)
        };

        Ok(Self {
            entry,
            instance,
            _debug_utils,
            _debug_messenger,
        })
    }

    pub fn entry(&self) -> &ash::Entry {
        &self.entry
    }

    pub fn instance(&self) -> &ash::Instance {
        &self.instance
    }

    pub fn enumerate_adapters(&self) -> Vec<RotexAdapter> {
        let devices = unsafe { self.instance.enumerate_physical_devices() }.unwrap_or_else(|err| {
            eprintln!("failed to enumerate physical devices: {err:?}");
            Vec::new()
        });

        devices
            .into_iter()
            .map(|handle| {
                let props = unsafe { self.instance.get_physical_device_properties(handle) };
                let name = unsafe { CStr::from_ptr(props.device_name.as_ptr()) }
                    .to_string_lossy()
                    .into_owned();
                RotexAdapter::new(handle, name, props.device_type, props.limits)
            })
            .collect()
    }

    pub fn destroy(&mut self) {
        unsafe {
            if let (Some(debug_utils), Some(debug_messenger)) =
                (self._debug_utils.as_ref(), self._debug_messenger)
            {
                debug_utils.destroy_debug_utils_messenger(debug_messenger, None);
            }
            self._debug_messenger = None;
            self._debug_utils = None;
            self.instance.destroy_instance(None);
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QueueCategory {
    Graphics,
    Compute,
    Transfer,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QueueRequest {
    pub category: QueueCategory,
    pub count: u32,
}

#[derive(Debug, Clone)]
pub struct DeviceDescriptor {
    pub required_features: vk::PhysicalDeviceFeatures,
    pub enable_swapchain: bool,
    pub queues: Vec<QueueRequest>,
}

pub struct RotexAdapter {
    handle: vk::PhysicalDevice,
    name: String,
    device_type: vk::PhysicalDeviceType,
    limits: vk::PhysicalDeviceLimits,
}

impl RotexAdapter {
    pub(crate) fn new(
        handle: vk::PhysicalDevice,
        name: String,
        device_type: vk::PhysicalDeviceType,
        limits: vk::PhysicalDeviceLimits,
    ) -> Self {
        Self {
            handle,
            name,
            device_type,
            limits,
        }
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn device_type(&self) -> vk::PhysicalDeviceType {
        self.device_type
    }

    pub fn limits(&self) -> &vk::PhysicalDeviceLimits {
        &self.limits
    }

    pub fn physical_device(&self) -> vk::PhysicalDevice {
        self.handle
    }

    pub fn request_device(
        &self,
        instance: &RotexInstance,
        desc: DeviceDescriptor,
    ) -> Result<RotexDevice, RotexError> {
        let queue_families = unsafe {
            instance
                .instance()
                .get_physical_device_queue_family_properties(self.handle)
        };
        let graphics_index = queue_families
            .iter()
            .enumerate()
            .find(|(_, family)| family.queue_flags.contains(vk::QueueFlags::GRAPHICS))
            .map(|(index, _)| index as u32);
        let compute_index = queue_families
            .iter()
            .enumerate()
            .find(|(_, family)| family.queue_flags.contains(vk::QueueFlags::COMPUTE))
            .map(|(index, _)| index as u32);
        let transfer_any_index = queue_families
            .iter()
            .enumerate()
            .find(|(_, family)| family.queue_flags.contains(vk::QueueFlags::TRANSFER))
            .map(|(index, _)| index as u32);
        let transfer_dedicated_index = queue_families
            .iter()
            .enumerate()
            .find(|(_, family)| {
                family.queue_flags.contains(vk::QueueFlags::TRANSFER)
                    && !family.queue_flags.contains(vk::QueueFlags::GRAPHICS)
                    && !family.queue_flags.contains(vk::QueueFlags::COMPUTE)
            })
            .map(|(index, _)| index as u32);

        if desc.enable_swapchain {
            let extensions = unsafe {
                instance
                    .instance()
                    .enumerate_device_extension_properties(self.handle)
            }
            .map_err(|err| RotexError {
                kind: ErrorKind::Vulkan(err),
                severity: Severity::Fatal,
            })?;
            let has_swapchain = extensions.iter().any(|ext| unsafe {
                CStr::from_ptr(ext.extension_name.as_ptr()) == vk::KHR_SWAPCHAIN_NAME
            });
            if !has_swapchain {
                return Err(RotexError {
                    kind: ErrorKind::NoCompatibleDevice,
                    severity: Severity::Fatal,
                });
            }
        }

        let mut allocations = Vec::new();
        for request in desc.queues {
            if request.count == 0 {
                continue;
            }
            let family_index = match request.category {
                QueueCategory::Graphics => graphics_index,
                QueueCategory::Compute => compute_index,
                QueueCategory::Transfer => transfer_dedicated_index
                    .or(graphics_index)
                    .or(transfer_any_index)
                    .or(compute_index),
            };
            let family_index = match family_index {
                Some(index) => index,
                None => {
                    return Err(RotexError {
                        kind: ErrorKind::NoCompatibleDevice,
                        severity: Severity::Fatal,
                    });
                }
            };
            allocations.push(QueueAllocation {
                category: request.category,
                family_index,
                count: request.count,
            });
        }

        if allocations.is_empty() {
            return Err(RotexError {
                kind: ErrorKind::NoCompatibleDevice,
                severity: Severity::Fatal,
            });
        }

        let mut queue_priorities: BTreeMap<u32, Vec<f32>> = BTreeMap::new();
        for allocation in &allocations {
            let entry = queue_priorities
                .entry(allocation.family_index)
                .or_insert_with(Vec::new);
            entry.extend(std::iter::repeat(1.0).take(allocation.count as usize));
        }

        for (family_index, priorities) in queue_priorities.iter_mut() {
            let max_supported = queue_families[*family_index as usize].queue_count as usize;
            if priorities.len() > max_supported {
                priorities.truncate(max_supported);
            }
        }

        let mut priorities_store = Vec::new();
        let mut queue_layouts = Vec::new();
        for (family_index, priorities) in queue_priorities {
            priorities_store.push(priorities);
            let idx = priorities_store.len() - 1;
            queue_layouts.push((family_index, idx));
        }

        let queue_create_infos: Vec<vk::DeviceQueueCreateInfo> = queue_layouts
            .into_iter()
            .map(|(family_index, idx)| {
                vk::DeviceQueueCreateInfo::default()
                    .queue_family_index(family_index)
                    .queue_priorities(&priorities_store[idx])
            })
            .collect();

        let device_extensions: Vec<*const i8> = if desc.enable_swapchain {
            vec![vk::KHR_SWAPCHAIN_NAME.as_ptr()]
        } else {
            Vec::new()
        };
        let device_create_info = vk::DeviceCreateInfo::default()
            .queue_create_infos(&queue_create_infos)
            .enabled_extension_names(&device_extensions)
            .enabled_features(&desc.required_features);

        let device = unsafe {
            instance
                .instance()
                .create_device(self.handle, &device_create_info, None)
        }
        .map_err(|err| RotexError {
            kind: ErrorKind::Vulkan(err),
            severity: Severity::Fatal,
        })?;

        Ok(RotexDevice {
            handle: self.handle,
            device,
            queues: allocations,
        })
    }
}

#[derive(Debug, Clone)]
pub struct QueueAllocation {
    pub category: QueueCategory,
    pub family_index: u32,
    pub count: u32,
}

pub struct RotexDevice {
    handle: vk::PhysicalDevice,
    device: ash::Device,
    queues: Vec<QueueAllocation>,
}

impl RotexDevice {
    pub fn device(&self) -> &ash::Device {
        &self.device
    }

    pub fn physical_device(&self) -> vk::PhysicalDevice {
        self.handle
    }

    pub fn queues(&self) -> &[QueueAllocation] {
        &self.queues
    }

    pub fn get_queue(&self, family_index: u32, queue_index: u32) -> vk::Queue {
        unsafe { self.device.get_device_queue(family_index, queue_index) }
    }

    pub fn destroy(&mut self) {
        unsafe {
            self.device.destroy_device(None);
        }
    }
}
