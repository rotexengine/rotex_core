use ash::vk;
use thiserror::Error;

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
pub struct RotexError {
    pub kind: ErrorKind,
    pub severity: Severity,
}

impl std::fmt::Display for RotexError {
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

impl std::error::Error for RotexError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.kind)
    }
}

impl RotexError {
    pub fn vk_result_code(&self) -> Option<i32> {
        match &self.kind {
            ErrorKind::Vulkan(err) => Some(err.as_raw()),
            _ => None,
        }
    }
}
