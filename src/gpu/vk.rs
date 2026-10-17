//! Raw Vulkan through `ash` (gpu-plan.md §1): one [`Device`] per R9700, chosen by
//! PCI address; buffers with device addresses (the shaders take pointers in push
//! constants, so there are no descriptor sets); and a tracked allocator, whose
//! byte count the VRAM monitor subtracts from the card's total.

use std::ffi::CStr;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, anyhow, bail, ensure};
use ash::vk;

/// Size of the push-constant block every pipeline declares.
const PUSH_BYTES: u32 = 128;

/// How fast the kernel releases freed VRAM, for [`Tracker::pending_release`]:
/// measured ~12 ms per GiB (gpu-plan.md, milestone 1); this assumes 30 ms so a
/// slow release isn't mistaken for another program.
const RELEASE_PER_GIB: Duration = Duration::from_millis(30);

/// A PCI address such as `0000:06:00.0`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct PciAddr {
    pub domain: u32,
    pub bus: u32,
    pub device: u32,
    pub function: u32,
}

impl std::str::FromStr for PciAddr {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self> {
        let bad = || anyhow!("bad PCI address {s:?} (expected e.g. 0000:06:00.0)");
        let (domain, rest) = s.split_once(':').ok_or_else(bad)?;
        let (bus, rest) = rest.split_once(':').ok_or_else(bad)?;
        let (device, function) = rest.split_once('.').ok_or_else(bad)?;
        let hex = |v: &str| u32::from_str_radix(v, 16).map_err(|_| bad());
        Ok(Self {
            domain: hex(domain)?,
            bus: hex(bus)?,
            device: hex(device)?,
            function: hex(function)?,
        })
    }
}

impl std::fmt::Display for PciAddr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{:04x}:{:02x}:{:02x}.{:x}",
            self.domain, self.bus, self.device, self.function
        )
    }
}

/// Our VRAM on one card: what we hold, and what we freed recently that the kernel
/// may still be releasing (gpu-plan.md, finding 2).
#[derive(Default)]
pub struct Tracker {
    held: AtomicU64,
    freed: Mutex<Vec<(u64, Instant)>>,
}

impl Tracker {
    /// Device-local bytes we hold. Counted before the allocation is made, so the
    /// card's total never shows an allocation we haven't counted.
    pub fn held(&self) -> u64 {
        self.held.load(Ordering::Relaxed)
    }

    /// Freed bytes the kernel may not have released yet, assuming it releases them
    /// at [`RELEASE_PER_GIB`].
    pub fn pending_release(&self, now: Instant) -> u64 {
        let mut freed = self.freed.lock().unwrap_or_else(|e| e.into_inner());
        let left = |&(bytes, at): &(u64, Instant)| {
            let gone = now.saturating_duration_since(at).as_secs_f64()
                / RELEASE_PER_GIB.as_secs_f64()
                * (1u64 << 30) as f64;
            bytes.saturating_sub(gone as u64)
        };
        freed.retain(|f| left(f) > 0);
        freed.iter().map(left).sum()
    }

    fn add(&self, bytes: u64) {
        self.held.fetch_add(bytes, Ordering::Relaxed);
    }

    fn remove(&self, bytes: u64, freed: bool) {
        self.held.fetch_sub(bytes, Ordering::Relaxed);
        if freed {
            self.freed
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .push((bytes, Instant::now()));
        }
    }
}

/// The compute pipelines, one per shader.
pub struct Pipelines {
    pub matvec: vk::Pipeline,
    pub mmq_q4k: vk::Pipeline,
    pub mmq_q6k: vk::Pipeline,
    pub quantize_q8k: vk::Pipeline,
    pub rmsnorm: vk::Pipeline,
    pub rope_store: vk::Pipeline,
    pub swiglu: vk::Pipeline,
    pub flash_attn: vk::Pipeline,
    pub attention_split: vk::Pipeline,
    pub attention_merge: vk::Pipeline,
}

macro_rules! spirv {
    ($name:literal) => {
        include_bytes!(concat!(env!("OUT_DIR"), "/", $name, ".spv")).as_slice()
    };
}

fn vulkan() -> Result<&'static (ash::Entry, ash::Instance)> {
    static VK: OnceLock<std::result::Result<(ash::Entry, ash::Instance), String>> = OnceLock::new();
    VK.get_or_init(|| {
        // SAFETY: loading the system Vulkan loader; the instance lives for the
        // whole process.
        unsafe {
            let entry = ash::Entry::load().map_err(|e| format!("loading libvulkan: {e}"))?;
            let app = vk::ApplicationInfo::default()
                .application_name(c"increbot")
                .api_version(vk::API_VERSION_1_3);
            let instance = entry
                .create_instance(
                    &vk::InstanceCreateInfo::default().application_info(&app),
                    None,
                )
                .map_err(|e| format!("creating the Vulkan instance: {e}"))?;
            Ok((entry, instance))
        }
    })
    .as_ref()
    .map_err(|e| anyhow!("{e}"))
}

/// Whether the shaders were compiled into this binary (`build.rs` needs `glslc`).
pub fn shaders_built() -> bool {
    !spirv!("rmsnorm").is_empty()
}

/// One GPU: the logical device, its compute queue, pipelines and allocator.
pub struct Device {
    pub pci: PciAddr,
    pub name: String,
    pub(crate) raw: ash::Device,
    queue: Mutex<vk::Queue>,
    pub(crate) queue_family: u32,
    mem_vram: u32,
    mem_host: u32,
    pub(crate) layout: vk::PipelineLayout,
    pub(crate) pipes: Pipelines,
    pub tracker: Arc<Tracker>,
    /// Nanoseconds per timestamp tick.
    timestamp_ns: f32,
}

impl Device {
    /// Opens the GPU at `pci`. Needs RADV-class features (buffer device address,
    /// 8/16-bit storage, timeline semaphores) and resizable BAR (all VRAM
    /// host-visible), which the R9700s have.
    pub fn open(pci: PciAddr) -> Result<Arc<Self>> {
        ensure!(
            shaders_built(),
            "this binary was built without glslc, so it has no GPU shaders"
        );
        let (_, instance) = vulkan()?;
        // SAFETY: Vulkan calls on handles we own; every struct chain lives until the
        // call that reads it returns.
        unsafe {
            let pd = instance
                .enumerate_physical_devices()?
                .into_iter()
                .find(|&pd| pci_of(instance, pd) == Some(pci))
                .with_context(|| format!("no Vulkan device at PCI {pci}"))?;
            let props = instance.get_physical_device_properties(pd);
            let name = CStr::from_ptr(props.device_name.as_ptr())
                .to_string_lossy()
                .into_owned();

            // Prefer a compute-only queue family (async compute on AMD).
            let families = instance.get_physical_device_queue_family_properties(pd);
            let compute =
                |f: &vk::QueueFamilyProperties| f.queue_flags.contains(vk::QueueFlags::COMPUTE);
            let queue_family = families
                .iter()
                .position(|f| compute(f) && !f.queue_flags.contains(vk::QueueFlags::GRAPHICS))
                .or_else(|| families.iter().position(compute))
                .context("no compute queue")? as u32;

            let mut f11 = vk::PhysicalDeviceVulkan11Features::default()
                .storage_buffer16_bit_access(true)
                .storage_push_constant16(true);
            let mut f12 = vk::PhysicalDeviceVulkan12Features::default()
                .buffer_device_address(true)
                .timeline_semaphore(true)
                .shader_float16(true)
                .shader_int8(true)
                .storage_buffer8_bit_access(true);
            let mut f2 = vk::PhysicalDeviceFeatures2::default()
                .features(
                    vk::PhysicalDeviceFeatures::default()
                        .shader_int64(true)
                        .shader_int16(true),
                )
                .push_next(&mut f11)
                .push_next(&mut f12);
            let priorities = [1.0];
            let queues = [vk::DeviceQueueCreateInfo::default()
                .queue_family_index(queue_family)
                .queue_priorities(&priorities)];
            let raw = instance
                .create_device(
                    pd,
                    &vk::DeviceCreateInfo::default()
                        .queue_create_infos(&queues)
                        .push_next(&mut f2),
                    None,
                )
                .with_context(|| format!("creating a Vulkan device on {pci} ({name})"))?;
            let queue = raw.get_device_queue(queue_family, 0);

            let mem = instance.get_physical_device_memory_properties(pd);
            let types = &mem.memory_types[..mem.memory_type_count as usize];
            let find = |want: vk::MemoryPropertyFlags, not: vk::MemoryPropertyFlags| {
                types
                    .iter()
                    .position(|t| {
                        t.property_flags.contains(want) && !t.property_flags.intersects(not)
                    })
                    .map(|i| i as u32)
            };
            use vk::MemoryPropertyFlags as M;
            let mem_vram = find(
                M::DEVICE_LOCAL | M::HOST_VISIBLE | M::HOST_COHERENT,
                M::DEVICE_COHERENT_AMD,
            )
            .with_context(|| format!("{pci}: no host-visible VRAM (enable resizable BAR)"))?;
            let mem_host = find(
                M::HOST_VISIBLE | M::HOST_COHERENT | M::HOST_CACHED,
                M::DEVICE_LOCAL | M::DEVICE_COHERENT_AMD,
            )
            .or_else(|| find(M::HOST_VISIBLE | M::HOST_COHERENT, M::DEVICE_LOCAL))
            .context("no host-visible system memory type")?;

            let range = [vk::PushConstantRange {
                stage_flags: vk::ShaderStageFlags::COMPUTE,
                offset: 0,
                size: PUSH_BYTES,
            }];
            let layout = raw.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default().push_constant_ranges(&range),
                None,
            )?;
            let pipe = |code: &[u8]| -> Result<vk::Pipeline> {
                let words = ash::util::read_spv(&mut std::io::Cursor::new(code))?;
                let module = raw.create_shader_module(
                    &vk::ShaderModuleCreateInfo::default().code(&words),
                    None,
                )?;
                let info = vk::ComputePipelineCreateInfo::default()
                    .stage(
                        vk::PipelineShaderStageCreateInfo::default()
                            .stage(vk::ShaderStageFlags::COMPUTE)
                            .module(module)
                            .name(c"main"),
                    )
                    .layout(layout);
                let p = raw
                    .create_compute_pipelines(vk::PipelineCache::null(), &[info], None)
                    .map_err(|(_, e)| e)?;
                raw.destroy_shader_module(module, None);
                Ok(p[0])
            };
            let pipes = Pipelines {
                matvec: pipe(spirv!("matvec"))?,
                mmq_q4k: pipe(spirv!("mmq_q4k"))?,
                mmq_q6k: pipe(spirv!("mmq_q6k"))?,
                quantize_q8k: pipe(spirv!("quantize_q8k"))?,
                rmsnorm: pipe(spirv!("rmsnorm"))?,
                rope_store: pipe(spirv!("rope_store"))?,
                swiglu: pipe(spirv!("swiglu"))?,
                flash_attn: pipe(spirv!("flash_attn"))?,
                attention_split: pipe(spirv!("attention_split"))?,
                attention_merge: pipe(spirv!("attention_merge"))?,
            };
            Ok(Arc::new(Self {
                pci,
                name,
                raw,
                queue: Mutex::new(queue),
                queue_family,
                mem_vram,
                mem_host,
                layout,
                pipes,
                tracker: Arc::default(),
                timestamp_ns: props.limits.timestamp_period,
            }))
        }
    }

    /// A VRAM buffer, mapped for direct CPU writes (resizable BAR). Counted by the
    /// tracker.
    pub fn vram(self: &Arc<Self>, bytes: u64) -> Result<Buffer> {
        Buffer::new(self, bytes, true)
    }

    /// A buffer in system memory (host-cached), for reading results back. Not VRAM,
    /// so not tracked (gpu-plan.md, finding 1).
    pub fn host(self: &Arc<Self>, bytes: u64) -> Result<Buffer> {
        Buffer::new(self, bytes, false)
    }

    pub(crate) fn submit(
        &self,
        cb: vk::CommandBuffer,
        sem: vk::Semaphore,
        value: u64,
    ) -> Result<()> {
        let values = [value];
        let mut timeline =
            vk::TimelineSemaphoreSubmitInfo::default().signal_semaphore_values(&values);
        let (cbs, sems) = ([cb], [sem]);
        let submit = vk::SubmitInfo::default()
            .command_buffers(&cbs)
            .signal_semaphores(&sems)
            .push_next(&mut timeline);
        let queue = self.queue.lock().unwrap_or_else(|e| e.into_inner());
        // SAFETY: the queue is externally synchronized by the mutex.
        unsafe { self.raw.queue_submit(*queue, &[submit], vk::Fence::null()) }
            .context("vkQueueSubmit")
    }
}

impl Drop for Device {
    fn drop(&mut self) {
        // SAFETY: every Buffer holds an Arc<Device>, so none are left; the queue is
        // drained first.
        unsafe {
            let _ = self.raw.device_wait_idle();
            let p = &self.pipes;
            for pipe in [
                p.matvec,
                p.mmq_q4k,
                p.mmq_q6k,
                p.quantize_q8k,
                p.rmsnorm,
                p.rope_store,
                p.swiglu,
                p.flash_attn,
                p.attention_split,
                p.attention_merge,
            ] {
                self.raw.destroy_pipeline(pipe, None);
            }
            self.raw.destroy_pipeline_layout(self.layout, None);
            self.raw.destroy_device(None);
        }
    }
}

/// The PCI address of a physical device, if it reports one.
unsafe fn pci_of(instance: &ash::Instance, pd: vk::PhysicalDevice) -> Option<PciAddr> {
    // SAFETY: caller passes a valid physical device of `instance`.
    unsafe {
        let has_ext = instance
            .enumerate_device_extension_properties(pd)
            .ok()?
            .iter()
            .any(|e| e.extension_name_as_c_str() == Ok(c"VK_EXT_pci_bus_info"));
        if !has_ext {
            return None;
        }
        let mut pci = vk::PhysicalDevicePCIBusInfoPropertiesEXT::default();
        let mut props = vk::PhysicalDeviceProperties2::default().push_next(&mut pci);
        instance.get_physical_device_properties2(pd, &mut props);
        Some(PciAddr {
            domain: pci.pci_domain,
            bus: pci.pci_bus,
            device: pci.pci_device,
            function: pci.pci_function,
        })
    }
}

/// One allocation holding one buffer, persistently mapped. Freed on drop.
pub struct Buffer {
    dev: Arc<Device>,
    pub(crate) raw: vk::Buffer,
    mem: vk::DeviceMemory,
    pub size: u64,
    pub addr: u64,
    ptr: *mut u8,
    vram: bool,
}

// SAFETY: the mapping is plain memory; access is coordinated by the owners (the
// GPU is idle whenever the CPU touches a buffer).
unsafe impl Send for Buffer {}
unsafe impl Sync for Buffer {}

impl Buffer {
    fn new(dev: &Arc<Device>, bytes: u64, vram: bool) -> Result<Self> {
        let bytes = bytes.max(16);
        let raw_dev = &dev.raw;
        if vram {
            dev.tracker.add(bytes);
        }
        // SAFETY: standard buffer + dedicated memory creation; on failure everything
        // created so far is destroyed.
        let made = unsafe {
            (|| -> Result<(vk::Buffer, vk::DeviceMemory, u64, *mut u8)> {
                use vk::BufferUsageFlags as U;
                let buf = raw_dev.create_buffer(
                    &vk::BufferCreateInfo::default().size(bytes).usage(
                        U::STORAGE_BUFFER
                            | U::TRANSFER_SRC
                            | U::TRANSFER_DST
                            | U::SHADER_DEVICE_ADDRESS,
                    ),
                    None,
                )?;
                let req = raw_dev.get_buffer_memory_requirements(buf);
                let ty = if vram { dev.mem_vram } else { dev.mem_host };
                if req.memory_type_bits & (1 << ty) == 0 {
                    raw_dev.destroy_buffer(buf, None);
                    bail!("memory type {ty} not allowed for buffers");
                }
                let mut flags = vk::MemoryAllocateFlagsInfo::default()
                    .flags(vk::MemoryAllocateFlags::DEVICE_ADDRESS);
                let mem = match raw_dev.allocate_memory(
                    &vk::MemoryAllocateInfo::default()
                        .allocation_size(req.size)
                        .memory_type_index(ty)
                        .push_next(&mut flags),
                    None,
                ) {
                    Ok(m) => m,
                    Err(e) => {
                        raw_dev.destroy_buffer(buf, None);
                        return Err(e.into());
                    }
                };
                let bound = raw_dev.bind_buffer_memory(buf, mem, 0).and_then(|()| {
                    raw_dev.map_memory(mem, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty())
                });
                let ptr = match bound {
                    Ok(p) => p.cast::<u8>(),
                    Err(e) => {
                        raw_dev.destroy_buffer(buf, None);
                        raw_dev.free_memory(mem, None);
                        return Err(e.into());
                    }
                };
                let addr = raw_dev
                    .get_buffer_device_address(&vk::BufferDeviceAddressInfo::default().buffer(buf));
                Ok((buf, mem, addr, ptr))
            })()
        };
        match made {
            Ok((raw, mem, addr, ptr)) => Ok(Self {
                dev: dev.clone(),
                raw,
                mem,
                size: bytes,
                addr,
                ptr,
                vram,
            }),
            Err(e) => {
                if vram {
                    dev.tracker.remove(bytes, false);
                }
                Err(e.context(format!(
                    "allocating {} MiB of {} on {}",
                    bytes >> 20,
                    if vram { "VRAM" } else { "host memory" },
                    dev.pci
                )))
            }
        }
    }

    /// The mapping as bytes. The GPU must not be using the buffer.
    #[allow(clippy::mut_from_ref)]
    pub fn bytes_mut(&self) -> &mut [u8] {
        // SAFETY: a live mapping of `size` bytes; see the type's Sync note.
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.size as usize) }
    }

    pub fn bytes(&self) -> &[u8] {
        // SAFETY: as above.
        unsafe { std::slice::from_raw_parts(self.ptr, self.size as usize) }
    }

    /// Copies `data` in at byte `offset`.
    pub fn write<T: Copy>(&self, offset: usize, data: &[T]) {
        let src =
            unsafe { std::slice::from_raw_parts(data.as_ptr().cast::<u8>(), size_of_val(data)) };
        self.bytes_mut()[offset..offset + src.len()].copy_from_slice(src);
    }

    /// `len` values of `T` starting at byte `offset`, in place. `T` must be a
    /// plain number type whose alignment `offset` respects.
    pub fn slice<T: Copy>(&self, offset: usize, len: usize) -> &[T] {
        let bytes = &self.bytes()[offset..offset + len * size_of::<T>()];
        assert!(bytes.as_ptr().cast::<T>().is_aligned(), "misaligned slice");
        // SAFETY: in bounds and aligned (checked); T is plain data.
        unsafe { std::slice::from_raw_parts(bytes.as_ptr().cast::<T>(), len) }
    }

    /// `len` values of `T` starting at byte `offset`.
    pub fn read<T: Copy + Default>(&self, offset: usize, len: usize) -> Vec<T> {
        let mut out = vec![T::default(); len];
        let n = len * size_of::<T>();
        // SAFETY: `out` has room for `n` bytes; the source range is checked.
        unsafe {
            std::ptr::copy_nonoverlapping(
                self.bytes()[offset..offset + n].as_ptr(),
                out.as_mut_ptr().cast::<u8>(),
                n,
            );
        }
        out
    }
}

impl Drop for Buffer {
    fn drop(&mut self) {
        // SAFETY: the owner guarantees the GPU is done with the buffer.
        unsafe {
            self.dev.raw.destroy_buffer(self.raw, None);
            self.dev.raw.free_memory(self.mem, None);
        }
        if self.vram {
            self.dev.tracker.remove(self.size, true);
        }
    }
}

/// Builds a push-constant block in std430 layout: addresses are 8-byte aligned,
/// scalars 4-byte.
#[derive(Default)]
pub struct Push(Vec<u8>);

impl Push {
    pub fn addr(mut self, a: u64) -> Self {
        while !self.0.len().is_multiple_of(8) {
            self.0.push(0);
        }
        self.0.extend(a.to_le_bytes());
        self
    }

    pub fn u32(mut self, v: u32) -> Self {
        self.0.extend(v.to_le_bytes());
        self
    }

    pub fn f32(mut self, v: f32) -> Self {
        self.0.extend(v.to_le_bytes());
        self
    }

    pub fn bytes(&self) -> &[u8] {
        assert!(
            self.0.len() <= PUSH_BYTES as usize,
            "push constants too large"
        );
        &self.0
    }
}

/// Command recording and timeline-semaphore submission for one thread, with a
/// small ring of command buffers so at most `ring` submissions are in flight.
pub struct Commands {
    dev: Arc<Device>,
    pool: vk::CommandPool,
    bufs: Vec<vk::CommandBuffer>,
    /// The timeline value each ring slot last signalled.
    slot_value: Vec<u64>,
    sem: vk::Semaphore,
    /// The last value submitted.
    submitted: u64,
    cur: Option<usize>,
    /// With `INCREBOT_PROFILE`: a timestamp query pool per ring slot and the
    /// label of each interval recorded into it.
    prof: Option<(Vec<vk::QueryPool>, Vec<Vec<&'static str>>)>,
}

/// Timestamps per command buffer when profiling.
const MAX_QUERIES: u32 = 256;

// SAFETY: used by one thread at a time (owned by the model, behind a mutex).
unsafe impl Send for Commands {}

impl Commands {
    pub fn new(dev: &Arc<Device>, ring: usize) -> Result<Self> {
        // SAFETY: plain object creation on our device.
        unsafe {
            let pool = dev.raw.create_command_pool(
                &vk::CommandPoolCreateInfo::default()
                    .queue_family_index(dev.queue_family)
                    .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
                None,
            )?;
            let bufs = dev.raw.allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_pool(pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(ring as u32),
            )?;
            let mut ty = vk::SemaphoreTypeCreateInfo::default()
                .semaphore_type(vk::SemaphoreType::TIMELINE)
                .initial_value(0);
            let sem = dev
                .raw
                .create_semaphore(&vk::SemaphoreCreateInfo::default().push_next(&mut ty), None)?;
            let prof = if crate::model::profile::enabled() {
                let mut pools = Vec::new();
                for _ in 0..ring {
                    pools.push(
                        dev.raw.create_query_pool(
                            &vk::QueryPoolCreateInfo::default()
                                .query_type(vk::QueryType::TIMESTAMP)
                                .query_count(MAX_QUERIES),
                            None,
                        )?,
                    );
                }
                Some((pools, vec![Vec::new(); ring]))
            } else {
                None
            };
            Ok(Self {
                prof,
                dev: dev.clone(),
                pool,
                slot_value: vec![0; bufs.len()],
                bufs,
                sem,
                submitted: 0,
                cur: None,
            })
        }
    }

    /// Starts recording into the next ring slot, first waiting for its previous
    /// submission to finish.
    pub fn begin(&mut self) -> Result<()> {
        assert!(self.cur.is_none(), "begin while recording");
        let slot = (self.submitted as usize) % self.bufs.len();
        self.wait(self.slot_value[slot])?;
        self.collect(slot);
        let cb = self.bufs[slot];
        // SAFETY: the slot's previous submission has completed.
        unsafe {
            self.dev
                .raw
                .reset_command_buffer(cb, vk::CommandBufferResetFlags::empty())?;
            self.dev.raw.begin_command_buffer(
                cb,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )?;
            if let Some((pools, _)) = &self.prof {
                self.dev
                    .raw
                    .cmd_reset_query_pool(cb, pools[slot], 0, MAX_QUERIES);
            }
        }
        self.cur = Some(slot);
        self.stamp("")?;
        // Order against everything submitted before (compute and copies).
        self.barrier_all()?;
        Ok(())
    }

    /// The command buffer being recorded. Errors (rather than panics) because
    /// `dispatch`/`copy` are called from `gpu/model.rs`, across the module
    /// boundary: a future caller forgetting `begin()` fails that one job
    /// with a clear error instead of panicking the inference thread.
    fn cb(&self) -> Result<vk::CommandBuffer> {
        self.cur.map(|s| self.bufs[s]).context("not recording")
    }

    /// When profiling, ends the interval since the last stamp, labelled `label`.
    fn stamp(&mut self, label: &'static str) -> Result<()> {
        let slot = self.cur.context("not recording")?;
        let cb = self.bufs[slot];
        let Some((pools, labels)) = &mut self.prof else {
            return Ok(());
        };
        let n = labels[slot].len() as u32;
        if n >= MAX_QUERIES {
            return Ok(());
        }
        // SAFETY: recording into our command buffer; the query was reset in begin.
        unsafe {
            self.dev.raw.cmd_write_timestamp(
                cb,
                vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                pools[slot],
                n,
            );
        }
        labels[slot].push(label);
        Ok(())
    }

    /// Adds a finished slot's timestamps to the profile.
    fn collect(&mut self, slot: usize) {
        let Some((pools, labels)) = &mut self.prof else {
            return;
        };
        let labels = std::mem::take(&mut labels[slot]);
        if labels.len() < 2 {
            return;
        }
        let mut ticks = vec![0u64; labels.len()];
        // SAFETY: the slot's submission has completed, so every query is written.
        let ok = unsafe {
            self.dev.raw.get_query_pool_results(
                pools[slot],
                0,
                &mut ticks,
                vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT,
            )
        };
        if ok.is_err() {
            return;
        }
        for i in 1..labels.len() {
            let ns = ticks[i].saturating_sub(ticks[i - 1]) as f64 * self.dev.timestamp_ns as f64;
            crate::model::profile::add(labels[i], Duration::from_nanos(ns as u64));
        }
    }

    /// Records a dispatch followed by a compute → compute barrier. `label` names
    /// it in the profile.
    pub fn dispatch(
        &mut self,
        label: &'static str,
        pipe: vk::Pipeline,
        push: Push,
        groups: [u32; 3],
    ) -> Result<()> {
        let cb = self.cb()?;
        let d = &self.dev.raw;
        // SAFETY: recording into our command buffer with our pipeline layout.
        unsafe {
            d.cmd_bind_pipeline(cb, vk::PipelineBindPoint::COMPUTE, pipe);
            d.cmd_push_constants(
                cb,
                self.dev.layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                push.bytes(),
            );
            d.cmd_dispatch(cb, groups[0], groups[1], groups[2]);
            let b = [vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE)];
            d.cmd_pipeline_barrier(
                cb,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &b,
                &[],
                &[],
            );
        }
        self.stamp(label)
    }

    /// Records a buffer-to-buffer copy (after a barrier on earlier writes).
    pub fn copy(&mut self, src: &Buffer, src_off: u64, dst: &Buffer, dst_off: u64, bytes: u64) -> Result<()> {
        if bytes == 0 {
            return Ok(());
        }
        let cb = self.cb()?;
        // SAFETY: recording into our command buffer; ranges are the caller's.
        unsafe {
            let b = [vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                .dst_access_mask(vk::AccessFlags::TRANSFER_READ)];
            self.dev.raw.cmd_pipeline_barrier(
                cb,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &b,
                &[],
                &[],
            );
            self.dev.raw.cmd_copy_buffer(
                cb,
                src.raw,
                dst.raw,
                &[vk::BufferCopy {
                    src_offset: src_off,
                    dst_offset: dst_off,
                    size: bytes,
                }],
            );
        }
        self.stamp("gpu_kv_readback")
    }

    fn barrier_all(&mut self) -> Result<()> {
        let cb = self.cb()?;
        let b = [vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::MEMORY_WRITE)
            .dst_access_mask(vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE)];
        // SAFETY: recording into our command buffer.
        unsafe {
            self.dev.raw.cmd_pipeline_barrier(
                cb,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::PipelineStageFlags::ALL_COMMANDS | vk::PipelineStageFlags::HOST,
                vk::DependencyFlags::empty(),
                &b,
                &[],
                &[],
            );
        }
        Ok(())
    }

    /// Ends recording and submits; returns the timeline value it will signal.
    pub fn submit(&mut self) -> Result<u64> {
        // Make this buffer's writes visible to the host (results, read-back copies).
        self.barrier_all()?;
        let slot = self.cur.take().context("not recording")?;
        let cb = self.bufs[slot];
        // SAFETY: ending our own recording.
        unsafe { self.dev.raw.end_command_buffer(cb)? };
        let value = self.submitted + 1;
        self.dev.submit(cb, self.sem, value)?;
        self.submitted = value;
        self.slot_value[slot] = value;
        Ok(value)
    }

    /// Waits until submission `value` (and everything before it) has finished.
    pub fn wait(&self, value: u64) -> Result<()> {
        if value == 0 {
            return Ok(());
        }
        let (sems, values) = ([self.sem], [value]);
        // SAFETY: waiting on our own timeline semaphore.
        unsafe {
            self.dev.raw.wait_semaphores(
                &vk::SemaphoreWaitInfo::default()
                    .semaphores(&sems)
                    .values(&values),
                u64::MAX,
            )
        }
        .context("waiting for the GPU (device lost?)")
    }

    /// Waits for everything submitted.
    pub fn wait_all(&mut self) -> Result<()> {
        if let Some(slot) = self.cur.take() {
            // Abandon a half-recorded buffer.
            // SAFETY: not submitted, so not in use.
            unsafe {
                let _ = self.dev.raw.end_command_buffer(self.bufs[slot]);
            }
            if let Some((_, labels)) = &mut self.prof {
                labels[slot].clear();
            }
        }
        self.wait(self.submitted)?;
        for slot in 0..self.bufs.len() {
            self.collect(slot);
        }
        Ok(())
    }
}

impl Drop for Commands {
    fn drop(&mut self) {
        let _ = self.wait_all();
        // SAFETY: nothing in flight any more.
        unsafe {
            if let Some((pools, _)) = &self.prof {
                for &p in pools {
                    self.dev.raw.destroy_query_pool(p, None);
                }
            }
            self.dev.raw.destroy_semaphore(self.sem, None);
            self.dev.raw.destroy_command_pool(self.pool, None);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pci_addresses_round_trip() {
        let a: PciAddr = "0000:06:00.0".parse().unwrap();
        assert_eq!((a.domain, a.bus, a.device, a.function), (0, 6, 0, 0));
        assert_eq!(a.to_string(), "0000:06:00.0");
        let b: PciAddr = "0001:a3:1f.7".parse().unwrap();
        assert_eq!(b.to_string(), "0001:a3:1f.7");
        assert!("06:00.0".parse::<PciAddr>().is_err());
        assert!("0000:zz:00.0".parse::<PciAddr>().is_err());
    }

    #[test]
    fn push_constants_align_addresses() {
        let p = Push::default().addr(1).u32(2).addr(3).f32(1.0);
        assert_eq!(p.bytes().len(), 8 + 4 + 4 + 8 + 4);
        assert_eq!(&p.bytes()[16..24], &3u64.to_le_bytes());
    }

    #[test]
    fn pending_release_decays() {
        let t = Tracker::default();
        t.add(2 << 30);
        t.remove(2 << 30, true);
        assert_eq!(t.held(), 0);
        let now = Instant::now();
        assert!(t.pending_release(now) > (1 << 30));
        assert_eq!(t.pending_release(now + 3 * RELEASE_PER_GIB), 0);
    }
}
