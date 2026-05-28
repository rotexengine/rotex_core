use ash::vk;
use rotex_core::{DeviceDescriptor, QueueCategory, QueueRequest, RotexAdapter, RotexDevice, RotexInstance};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut instance: RotexInstance = RotexInstance::new(&[]).map_err(|err| {
        std::io::Error::new(
            std::io::ErrorKind::Other,
            format!("failed to initialize Vulkan instance: {err}"),
        )
    })?;

    let adapter: RotexAdapter = instance
        .enumerate_adapters()
        .into_iter()
        .next()
        .ok_or_else(|| {
            std::io::Error::new(std::io::ErrorKind::Other, "no Vulkan adapters found")
        })?;

    let descriptor = DeviceDescriptor {
        required_features: vk::PhysicalDeviceFeatures::default(),
        enable_swapchain: false,
        queues: vec![
            QueueRequest {
                category: QueueCategory::Compute,
                count: 1,
            },
            QueueRequest {
                category: QueueCategory::Transfer,
                count: 1,
            },
        ],
    };

    let mut device: RotexDevice = adapter.request_device(&instance, descriptor).map_err(|err| {
        std::io::Error::new(
            std::io::ErrorKind::Other,
            format!("failed to initialize logical device: {err}"),
        )
    })?;

    println!("Headless device initialized on adapter: {}", adapter.name());

    device.destroy();
    instance.destroy();
    Ok(())
}
