//! First uploads of fresh textures on a copy queue of their own.
//!
//! Off by default. With `VICTORY_UPLOAD_AHEAD=1`, a texture no list has used
//! yet is detiled and copied by a worker as soon as a descriptor names it;
//! the slot stays hidden until the copy's fence has run. Conflicting normal
//! uploads wait for that fence instead of racing it. `VICTORY_SHOW_AFTER=1`
//! holds other new texture slots until their upload list has run, and
//! `VICTORY_SHOW_NEW_LATER=1` holds them for one more look.

use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicU8, AtomicU64, Ordering::AcqRel, Ordering::Acquire, Ordering::Relaxed, Ordering::Release};
use std::sync::{Arc, Condvar, Mutex, OnceLock};

use windows::Win32::Foundation::HANDLE;
use windows::Win32::Graphics::Direct3D12::*;
use windows::core::Interface;

use super::{Staging, Upload};
use crate::object::This;
use crate::resource::HostTexture;

pub(crate) const FRESH: u8 = 0;
const NORMAL: u8 = 1;
const ARMED: u8 = 2;
const COPYING: u8 = 3;
const CANCELED: u8 = 4;
const READY: u8 = 5;

const BATCH_BYTES: u64 = 16 << 20;
const BATCH_COUNT: usize = 64;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Status {
    Unavailable,
    Pending,
    Ready,
}

fn state(phase: u8) -> Status {
    match phase {
        ARMED | COPYING | CANCELED => Status::Pending,
        READY => Status::Ready,
        _ => Status::Unavailable,
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Limits {
    max_bytes: u64,
    workers: usize,
}

/// Cap and worker count for the adapter's video budget and available RAM:
/// small adapters get one worker and 16-32 MiB in flight, large ones two
/// workers and up to 96 MiB.
fn limits_for(video_budget_mb: f64, available_ram_gb: f64) -> Limits {
    if available_ram_gb <= 8.0 || video_budget_mb < 2048.0 {
        Limits { max_bytes: 16 << 20, workers: 1 }
    } else if available_ram_gb <= 16.0 || video_budget_mb < 4096.0 {
        Limits { max_bytes: 32 << 20, workers: 1 }
    } else if available_ram_gb < 24.0 || video_budget_mb < 8192.0 {
        Limits { max_bytes: 64 << 20, workers: 1 }
    } else {
        Limits { max_bytes: 96 << 20, workers: 2 }
    }
}

fn limits() -> Limits {
    static LIMITS: OnceLock<Limits> = OnceLock::new();
    *LIMITS.get_or_init(|| {
        let video = crate::host::video_memory().map_or(0.0, |m| m[0].1 + m[1].1);
        let mut status = windows_sys::Win32::System::SystemInformation::MEMORYSTATUSEX::default();
        // SAFETY: a valid out pointer of the right size.
        let ram = if unsafe { windows_sys::Win32::System::SystemInformation::GlobalMemoryStatusEx(&mut status) } != 0 {
            status.ullAvailPhys as f64 / 1e9
        } else {
            0.0
        };
        limits_for(video, ram)
    })
}

fn flag(name: &str) -> bool {
    std::env::var_os(name).is_some_and(|v| v == "1")
}

pub(crate) fn enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| flag("VICTORY_UPLOAD_AHEAD"))
}

pub(crate) fn show_after() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| flag("VICTORY_SHOW_AFTER"))
}

fn show_new_later() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| flag("VICTORY_SHOW_NEW_LATER"))
}

pub(crate) fn fresh() -> AtomicU8 {
    AtomicU8::new(FRESH)
}

/// The submitted upload, held until its fence so no normal upload can reuse
/// the staging bytes or the destination before the copy has finished.
pub(crate) struct Ready {
    staging: Arc<Staging>,
    fingerprint: blake3::Hash,
}

struct Job {
    staging: Arc<Staging>,
    copy: Arc<HostTexture>,
    upload: Upload,
    fingerprint: blake3::Hash,
}

// SAFETY: the staging buffer and host texture stay alive until the fence;
// D3D12 command objects are free-threaded.
unsafe impl Send for Job {}

struct Queue {
    jobs: VecDeque<Job>,
    bytes: u64,
    accepting: bool,
}

static QUEUE: (Mutex<Queue>, Condvar) = (
    Mutex::new(Queue { jobs: VecDeque::new(), bytes: 0, accepting: true }),
    Condvar::new(),
);
static WAITING: Mutex<Vec<usize>> = Mutex::new(Vec::new());
static LANDED: Condvar = Condvar::new();

fn admit(bytes: u64, queued: u64, cap: u64) -> bool {
    bytes <= BATCH_BYTES && queued <= cap.saturating_sub(bytes)
}

fn enqueue(job: Job) -> bool {
    let bytes = staging_bytes(&job.staging);
    let mut queue = QUEUE.0.lock().unwrap_or_else(|e| e.into_inner());
    if !queue.accepting || !admit(bytes, queue.bytes, limits().max_bytes) {
        return false;
    }
    let key = Arc::as_ptr(&job.copy) as usize;
    queue.bytes += bytes;
    queue.jobs.push_back(job);
    WAITING.lock().unwrap_or_else(|e| e.into_inner()).push(key);
    QUEUE.1.notify_one();
    true
}

fn staging_bytes(staging: &Staging) -> u64 {
    staging.size
}

/// A normal upload path found a copy on its way: let it land before the
/// list records its own copy of the same texture.
pub(crate) fn wait(copy: &HostTexture) {
    let key = copy as *const HostTexture as usize;
    let mut waiting = WAITING.lock().unwrap_or_else(|e| e.into_inner());
    let started = std::time::Instant::now();
    while waiting.contains(&key) && started.elapsed().as_millis() < 1000 {
        waiting = LANDED.wait_timeout(waiting, std::time::Duration::from_millis(50)).unwrap_or_else(|e| e.into_inner()).0;
    }
}

/// Claims the next upload of `copy` for the ordinary path. A pending one
/// waits in [`wait`] rather than recording a competing copy.
pub(crate) fn claim_normal(copy: &HostTexture) -> Status {
    if !enabled() {
        return Status::Unavailable;
    }
    match copy.ahead.compare_exchange(FRESH, NORMAL, AcqRel, Acquire) {
        Ok(_) => Status::Unavailable,
        Err(READY) => status(copy),
        Err(phase) => state(phase),
    }
}

/// Enrolls the first upload of a fresh texture before a descriptor exposes
/// it. `first` is the view's most detailed mip; only level 0 is taken.
pub(crate) fn request(resource: This, copy: &Arc<HostTexture>, first: u32) -> Status {
    if !enabled() || first != 0 || !eligible(resource, copy) || !start_worker() {
        REJECTED.fetch_add(1, Relaxed);
        return claim_normal(copy);
    }
    match copy.ahead.compare_exchange(FRESH, ARMED, AcqRel, Acquire) {
        Ok(_) => {}
        Err(READY) => return status(copy),
        Err(phase) => return state(phase),
    }
    let Some(job) = prepare(resource, copy.clone()) else {
        failed(copy);
        return Status::Unavailable;
    };
    let bytes = staging_bytes(&job.staging);
    if copy.ahead.compare_exchange(ARMED, COPYING, AcqRel, Acquire).is_err()
        || !enqueue(job)
    {
        failed(copy);
        return Status::Unavailable;
    }
    REQUESTED.fetch_add(1, Relaxed);
    let _ = bytes;
    Status::Pending
}

fn eligible(resource: This, copy: &HostTexture) -> bool {
    if !copy.fresh()
        || copy.host_written.load(Acquire)
        || copy.state.load(Acquire) != D3D12_RESOURCE_STATE_COMMON.0 as u32
        || copy.staging.lock().unwrap_or_else(|e| e.into_inner()).is_some()
        || super::is_host_written(resource)
        || crate::resource::alias_source(resource).is_some()
        || crate::resource::texture_memory(resource).is_none()
    {
        return false;
    }
    // SAFETY: a live host texture.
    let desc = unsafe { copy.texture.GetDesc() };
    desc.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE2D
        && desc.DepthOrArraySize == 1
        && desc.SampleDesc.Count == 1
        && desc.Flags.0 & 7 == 0
        && matches!(desc.Format.0, 70..=84 | 94..=99)
}

fn prepare(resource: This, copy: Arc<HostTexture>) -> Option<Job> {
    let staging = super::staging(resource, &copy.texture)?;
    if !super::committed(staging.memory, staging.size as usize) {
        return None;
    }
    let before = fingerprint(&staging)?;
    let upload = staging.refresh()?;
    if fingerprint(&staging)? != before {
        staging.reset();
        return None;
    }
    Some(Job { staging, copy, upload, fingerprint: before })
}

fn fingerprint(staging: &Staging) -> Option<blake3::Hash> {
    let len = staging.layout.size as usize;
    if !super::committed(staging.memory, len) {
        return None;
    }
    // SAFETY: committed memory of the layout's size.
    Some(blake3::hash(unsafe { std::slice::from_raw_parts(staging.memory, len) }))
}

fn staging_reset(staging: &Staging) {
    staging.reset();
}

fn failed(copy: &HostTexture) {
    let mut ready = copy.ahead_ready.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(ready) = ready.take() {
        staging_reset(&ready.staging);
    }
    drop(ready);
    copy.ahead.store(NORMAL, Release);
    REJECTED.fetch_add(1, Relaxed);
}

fn clear_ready(copy: &HostTexture) {
    let mut ready = copy.ahead_ready.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(ready) = ready.take() {
        staging_reset(&ready.staging);
    }
    copy.ahead.store(NORMAL, Release);
}

fn checked_ready(copy: &HostTexture, ready: &mut Option<Ready>) -> bool {
    if copy.ahead.load(Acquire) != READY {
        return false;
    }
    let live = ready.as_ref().is_some_and(|r| fingerprint(&r.staging) == Some(r.fingerprint));
    if live {
        true
    } else {
        clear_ready(copy);
        false
    }
}

pub(crate) fn status(copy: &HostTexture) -> Status {
    let phase = copy.ahead.load(Acquire);
    if phase != READY {
        return state(phase);
    }
    let mut ready = copy.ahead_ready.lock().unwrap_or_else(|e| e.into_inner());
    if checked_ready(copy, &mut ready) {
        Status::Ready
    } else {
        Status::Unavailable
    }
}

/// Publishes the descriptor only while the ready upload still matches title
/// memory. A write during the copy loses the publication and the ordinary
/// upload path takes over.
pub(crate) fn while_ready(copy: &HostTexture, publish: impl FnOnce()) -> bool {
    if copy.ahead.load(Acquire) != READY {
        return false;
    }
    let mut ready = copy.ahead_ready.lock().unwrap_or_else(|e| e.into_inner());
    if !checked_ready(copy, &mut ready) {
        return false;
    }
    publish();
    true
}

/// The descriptor now names this texture: all later updates use the normal
/// upload path, and the staging buffer can return to its pool.
/// The resource a texture view names, using the host copy made for a placed
/// view when needed.
pub(crate) fn view_resource(view: &crate::descriptor::View) -> Option<This> {
    let view = crate::descriptor::host::placed_texture(view).unwrap_or_else(|| view.clone());
    match view {
        crate::descriptor::View::Shader(resource, _) => Some(resource as This),
        _ => None,
    }
}

pub(crate) fn retire(copy: &HostTexture) -> bool {
    let mut ready = copy.ahead_ready.lock().unwrap_or_else(|e| e.into_inner());
    if !checked_ready(copy, &mut ready) {
        return false;
    }
    ready.take();
    copy.ahead.store(NORMAL, Release);
    true
}

/// A descriptor revision was superseded before its copy landed: keep it
/// hidden and fall back to the normal upload path.
pub(crate) fn discard(copy: &HostTexture) {
    let phase = copy.ahead.load(Acquire);
    let next = cancel_phase(phase);
    if next != phase {
        copy.ahead.store(next, Release);
    }
    if phase == READY {
        clear_ready(copy);
    }
}

fn cancel_phase(phase: u8) -> u8 {
    match phase {
        ARMED => NORMAL,
        COPYING => CANCELED,
        _ => phase,
    }
}

fn publish_phase(phase: u8) -> u8 {
    if phase == COPYING { READY } else { phase }
}

fn start_worker() -> bool {
    static STARTED: OnceLock<bool> = OnceLock::new();
    *STARTED.get_or_init(|| {
        if !enabled() {
            return false;
        }
        let Some(host) = crate::host::host() else { return false };
        // SAFETY: plain object creation.
        let queue: ID3D12CommandQueue = match unsafe {
            host.device.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC {
                Type: D3D12_COMMAND_LIST_TYPE_COPY,
                ..Default::default()
            })
        } {
            Ok(queue) => queue,
            Err(error) => {
                tracing::warn!(%error, "texture ahead copy queue is unavailable");
                return false;
            }
        };
        let limits = limits();
        for n in 0..limits.workers {
            let Ok(worker) = Worker::new(&host.device, queue.clone()) else { return false };
            if std::thread::Builder::new()
                .name(format!("texture ahead copy {n}"))
                .spawn(move || worker.run())
                .is_err()
            {
                return false;
            }
        }
        true
    }) && QUEUE.0.lock().unwrap_or_else(|e| e.into_inner()).accepting
}

struct Worker {
    queue: ID3D12CommandQueue,
    allocator: ID3D12CommandAllocator,
    list: ID3D12GraphicsCommandList,
    fence: ID3D12Fence,
}

// SAFETY: the queue is free-threaded; the other objects belong to this
// worker's thread alone.
unsafe impl Send for Worker {}

enum Copied {
    Complete,
    Rejected(windows::core::Error),
    Uncertain(windows::core::Error),
}

impl Worker {
    fn new(device: &ID3D12Device, queue: ID3D12CommandQueue) -> windows::core::Result<Self> {
        // SAFETY: typed object creation, each output owned by this worker.
        unsafe {
            let allocator = device.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COPY)?;
            let list: ID3D12GraphicsCommandList =
                device.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_COPY, &allocator, None)?;
            list.Close()?;
            Ok(Self {
                queue,
                allocator,
                list,
                fence: device.CreateFence(0, D3D12_FENCE_FLAG_NONE)?,
            })
        }
    }

    fn run(self) {
        for value in 1.. {
            let (jobs, bytes) = next_batch();
            match self.copy(&jobs, value) {
                Copied::Complete => finish(jobs, true),
                Copied::Rejected(error) => {
                    tracing::warn!(%error, "texture ahead copies could not be recorded");
                    finish(jobs, false);
                }
                Copied::Uncertain(error) => {
                    tracing::error!(%error, "texture ahead copy lost its fence; submitted slots stay hidden");
                    let queued = {
                        let mut queue = QUEUE.0.lock().unwrap_or_else(|e| e.into_inner());
                        queue.accepting = false;
                        std::mem::take(&mut queue.jobs)
                    };
                    finish(queued.into(), false);
                    // The GPU may still read these resources. Keep them and
                    // the command objects alive; never reuse this worker.
                    std::mem::forget(jobs);
                    std::mem::forget(self);
                    return;
                }
            }
            let mut queue = QUEUE.0.lock().unwrap_or_else(|e| e.into_inner());
            queue.bytes = queue.bytes.saturating_sub(bytes);
        }
    }

    fn copy(&self, jobs: &[Job], value: u64) -> Copied {
        // SAFETY: this worker owns its allocator and list; all earlier work
        // has passed the fence. Destinations stay in COMMON around a copy.
        unsafe {
            let commands = (|| -> windows::core::Result<ID3D12CommandList> {
                self.allocator.Reset()?;
                self.list.Reset(&self.allocator, None)?;
                for job in jobs {
                    record(&self.list, job);
                }
                self.list.Close()?;
                self.list.cast()
            })();
            let commands = match commands {
                Ok(commands) => commands,
                Err(error) => return Copied::Rejected(error),
            };
            self.queue.ExecuteCommandLists(&[Some(commands)]);
            if let Err(error) = self.queue.Signal(&self.fence, value) {
                return Copied::Uncertain(error);
            }
            let wait = self.fence.SetEventOnCompletion(value, HANDLE::default());
            let seen = self.fence.GetCompletedValue();
            if seen != u64::MAX && seen >= value {
                Copied::Complete
            } else {
                Copied::Uncertain(wait.err().unwrap_or_else(windows::core::Error::empty))
            }
        }
    }
}

/// # Safety
/// `list` is an open copy list; every job in it lives until its fence.
unsafe fn record(list: &ID3D12GraphicsCommandList, job: &Job) {
    let (buffer, at) = (&job.upload.0, job.upload.1);
    for &i in &job.upload.2 {
        let mut footprint = job.staging.footprints[i as usize];
        footprint.Offset += at;
        let to = D3D12_TEXTURE_COPY_LOCATION {
            // SAFETY: borrowed by the location, which never releases it.
            pResource: unsafe { std::mem::transmute_copy(&job.copy.texture) },
            Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: i },
        };
        let from = D3D12_TEXTURE_COPY_LOCATION {
            // SAFETY: as above.
            pResource: unsafe { std::mem::transmute_copy(buffer) },
            Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { PlacedFootprint: footprint },
        };
        // SAFETY: an open copy list, live resources in copy states.
        unsafe { list.CopyTextureRegion(&to, 0, 0, 0, &from, None) };
    }
}

fn next_batch() -> (Vec<Job>, u64) {
    let mut queue = QUEUE.0.lock().unwrap_or_else(|e| e.into_inner());
    while queue.jobs.is_empty() {
        queue = QUEUE.1.wait(queue).unwrap_or_else(|e| e.into_inner());
    }
    let mut jobs = Vec::new();
    let mut bytes = 0;
    while jobs.len() < BATCH_COUNT
        && let Some(job) = queue.jobs.front()
        && bytes + staging_bytes(&job.staging) <= BATCH_BYTES
    {
        bytes += staging_bytes(&job.staging);
        jobs.push(queue.jobs.pop_front().expect("first queued upload"));
    }
    (jobs, bytes)
}

fn finish(jobs: Vec<Job>, copied: bool) {
    for job in jobs {
        let key = Arc::as_ptr(&job.copy) as usize;
        if copied && job.copy.ahead.load(Acquire) == COPYING {
            let phase = publish_phase(job.copy.ahead.load(Acquire));
            job.copy.ahead.store(phase, Release);
            if phase == READY {
                let mut ready = job.copy.ahead_ready.lock().unwrap_or_else(|e| e.into_inner());
                *ready = Some(Ready {
                    staging: job.staging.clone(),
                    fingerprint: job.fingerprint,
                });
            } else {
                staging_reset(&job.staging);
            }
            COMPLETED.fetch_add(1, Relaxed);
        } else {
            staging_reset(&job.staging);
            job.copy.ahead.store(cancel_phase(job.copy.ahead.load(Acquire)), Release);
            REJECTED.fetch_add(1, Relaxed);
        }
        let mut waiting = WAITING.lock().unwrap_or_else(|e| e.into_inner());
        waiting.retain(|&p| p != key);
        LANDED.notify_all();
    }
}

static REQUESTED: AtomicU64 = AtomicU64::new(0);
static COMPLETED: AtomicU64 = AtomicU64::new(0);
static REJECTED: AtomicU64 = AtomicU64::new(0);

/// Slot gates for `VICTORY_SHOW_AFTER` and `VICTORY_SHOW_NEW_LATER`. Keys
/// include the view generation, so a rewritten descriptor gets its own gate.
#[derive(Default)]
struct Gate {
    seen: u32,
    shown: bool,
}

static GATES: OnceLock<Mutex<HashMap<(u32, u64), Gate>>> = OnceLock::new();

fn gates() -> &'static Mutex<HashMap<(u32, u64), Gate>> {
    GATES.get_or_init(|| Mutex::new(HashMap::new()))
}

fn slot_key(handle: u64) -> u64 {
    // SAFETY: descriptor slots are four dwords in title memory.
    let [tag, index, generation] = unsafe { (handle as *const [u32; 3]).read_unaligned() };
    if tag != super::super::descriptor::VIEW_TAG {
        return 0;
    }
    u64::from(generation) << 32 | u64::from(index)
}

pub(crate) fn hold_slot(index: u32, handle: u64, list: Option<&crate::host::HostList>) {
    if !show_after() && !show_new_later() {
        return;
    }
    let Some(view) = crate::descriptor::view(handle) else { return };
    if view_resource(&view).is_none() {
        return;
    }
    let key = slot_key(handle);
    if key == 0 {
        return;
    }
    let mut map = gates().lock().unwrap_or_else(|e| e.into_inner());
    let gate = map.entry((index, key)).or_default();
    if gate.shown {
        return;
    }
    if show_new_later() {
        gate.seen += 1;
        if gate.seen > 1 {
            gate.shown = true;
            return;
        }
    }
    if let Some(list) = list
        && show_after()
    {
        list.defer_host(move || {
            gates().lock().unwrap_or_else(|e| e.into_inner()).remove(&(index, key));
        });
    }
}

pub(crate) fn slot_hidden(index: u32, handle: u64) -> bool {
    let key = slot_key(handle);
    key != 0 && gates().lock().unwrap_or_else(|e| e.into_inner()).get(&(index, key)).is_some_and(|g| !g.shown)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn state_transitions_only_publish_after_the_copy_fence() {
        assert_eq!(state(FRESH), Status::Unavailable);
        assert_eq!(state(NORMAL), Status::Unavailable);
        assert_eq!(state(ARMED), Status::Pending);
        assert_eq!(state(COPYING), Status::Pending);
        assert_eq!(state(CANCELED), Status::Pending);
        assert_eq!(state(READY), Status::Ready);
        assert_eq!(cancel_phase(ARMED), NORMAL);
        assert_eq!(cancel_phase(COPYING), CANCELED);
        assert_eq!(publish_phase(COPYING), READY);
        assert_eq!(publish_phase(ARMED), ARMED);
        assert_eq!(publish_phase(CANCELED), CANCELED);
    }

    #[test]
    fn staging_byte_cap_is_enforced() {
        let cap = 32 << 20;
        assert!(admit(16 << 20, 0, cap));
        assert!(admit(16 << 20, 16 << 20, cap));
        assert!(!admit(16 << 20, 17 << 20, cap));
        assert!(!admit(32 << 20, 1, cap));
        assert!(admit(16 << 20, 0, 16 << 20));
    }

    #[test]
    fn cancelled_or_fence_lost_jobs_never_become_visible() {
        for phase in [ARMED, COPYING, CANCELED] {
            let after = cancel_phase(phase);
            assert_ne!(after, READY, "phase {phase} became visible");
        }
        let abandoned = cancel_phase(COPYING);
        assert_eq!(abandoned, CANCELED);
        assert_eq!(state(abandoned), Status::Pending);
        assert_eq!(publish_phase(abandoned), CANCELED);
    }

    #[test]
    fn limits_scale_with_capability() {
        assert_eq!(limits_for(1024.0, 8.0), Limits { max_bytes: 16 << 20, workers: 1 });
        assert_eq!(limits_for(3072.0, 12.0), Limits { max_bytes: 32 << 20, workers: 1 });
        assert_eq!(limits_for(6144.0, 20.0), Limits { max_bytes: 64 << 20, workers: 1 });
        assert_eq!(limits_for(12288.0, 32.0), Limits { max_bytes: 96 << 20, workers: 2 });
    }
}
