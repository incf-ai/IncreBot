//! The GPU backend that shares nicely (gpu-plan.md): inference on a Radeon
//! through raw Vulkan, preempted as soon as another program starts using the
//! card.

pub mod manager;
pub mod model;
pub mod monitor;
pub mod policy;
pub mod vk;

pub use model::{GpuModel, Interrupted};
pub use vk::{Device, PciAddr};
