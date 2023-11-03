use crate::gs::array::GPUArray;
use crate::gs::registry::ProgramRegistry;
use metal::*;
use std::cmp::min;

const RADIX: u64 = 16;
const GROUP_SIZE: u32 = 256;
const BLOCK_COUNT: u32 = 8;
type Histogram = [u32; RADIX as usize];

pub struct SortStorage {
    n: u32,
    downsweep_args: GPUArray<u32>,
    histograms: GPUArray<Histogram>,
}

impl Drop for SortStorage {
    fn drop(&mut self) {}
}

fn block_count(n: u32) -> u32 {
    // How many keys a single thread processes
    (n + BLOCK_COUNT - 1) / BLOCK_COUNT
}

fn group_count(n: u32) -> u32 {
    (block_count(n) + GROUP_SIZE - 1) / GROUP_SIZE
}

pub fn allocate_sort_storage(device: &Device, n: u32) -> SortStorage {
    SortStorage {
        n,
        downsweep_args: GPUArray::<u32>::empty(device, 3, None),
        histograms: GPUArray::<Histogram>::empty(device, group_count(n) as u64, None),
    }
}

fn sort_downsweep(
    registry: &ProgramRegistry,
    storage: &mut SortStorage,
    cb: &CommandBufferRef,
    keys: &GPUArray<u32>,
    shift: u32,
) {
    let n = keys.len() as u32;
    assert!(n <= storage.n);

    storage.downsweep_args.get_mut_slice()[0] = n as u32;
    storage.downsweep_args.get_mut_slice()[1] = BLOCK_COUNT as u32;
    storage.downsweep_args.get_mut_slice()[2] = shift;

    storage.downsweep_args.sync(cb);
    keys.sync(cb);

    let (_, pipeline_state) = registry.get_function("sort", "sort_downsweep_u32").unwrap();

    let encoder = cb.compute_command_encoder_with_descriptor(ComputePassDescriptor::new());
    encoder.set_compute_pipeline_state(&pipeline_state);
    encoder.set_buffer(0, Some(&keys), 0);
    encoder.set_buffer(1, Some(&storage.histograms), 0);
    encoder.set_buffer(2, Some(&storage.downsweep_args), 0);
    encoder.set_buffer(3, Some(&storage.downsweep_args), 4);
    encoder.set_buffer(4, Some(&storage.downsweep_args), 8);

    encoder.dispatch_thread_groups(
        MTLSize::new(group_count(n).into(), 1, 1),
        MTLSize::new(GROUP_SIZE.into(), 1, 1),
    );
    encoder.end_encoding();

    storage.histograms.sync(cb);
}

pub fn _sort<T>(
    registry: &ProgramRegistry,
    _storage: &SortStorage,
    command_buffer: &CommandBufferRef,
    a: &GPUArray<T>,
    b: &GPUArray<T>,
    out: &GPUArray<T>,
) {
    assert!(a.len() == b.len() && b.len() == out.len());
    let n = a.len();
    if n == 0 {
        return;
    }

    let (_, pipeline_state) = registry.get_function("sum", "sum").unwrap();

    let descriptor = ComputePassDescriptor::new();

    let encoder = command_buffer.compute_command_encoder_with_descriptor(descriptor);
    encoder.set_compute_pipeline_state(&pipeline_state);
    encoder.set_buffer(0, Some(&a), 0);
    encoder.set_buffer(1, Some(&b), 0);
    encoder.set_buffer(2, Some(&out), 0);

    let grid_size = MTLSize::new(n, 1, 1);
    let thread_group_size = MTLSize::new(
        min(n, pipeline_state.max_total_threads_per_threadgroup()),
        1,
        1,
    );

    encoder.dispatch_threads(grid_size, thread_group_size);
    encoder.end_encoding();
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gs;
    use crate::gs::array::GPUArray;
    use objc::rc::autoreleasepool;

    #[test]
    fn test_sort_downsweep() {
        autoreleasepool(|| {
            let device = Device::system_default().expect("No device found");
            let registry = gs::registry::create_default_registry(&device, None);
            let queue = device.new_command_queue();

            let n = 1024u32;

            let mut storage = allocate_sort_storage(&device, n);

            let keys: Vec<u32> = (0..n).rev().collect();
            let keys = GPUArray::<u32>::from_data(&device, &keys, None);

            let cb = queue.new_command_buffer();
            sort_downsweep(&registry, &mut storage, &cb, &keys, 0u32);
            cb.commit();
            cb.wait_until_completed();

            println!("{:?}", storage.histograms.get_slice());

            assert!(false);
        })
    }

    #[test]
    fn test_reduce() {
        autoreleasepool(|| {
            let device = Device::system_default().expect("No device found");
            let registry = gs::registry::create_default_registry(&device, None);
            let queue = device.new_command_queue();

            let (_, pipeline_state) = registry.get_function("sort", "reduce_test").unwrap();

            let group_size = 32;

            let mut args =
                GPUArray::<u32>::empty(&device, 2, Some(MTLResourceOptions::StorageModeShared));

            for n in [1, 16, 24, 32] {
                let cb = queue.new_command_buffer();

                args.get_mut_slice()[1] = n;

                let encoder =
                    cb.compute_command_encoder_with_descriptor(ComputePassDescriptor::new());
                encoder.set_compute_pipeline_state(&pipeline_state);
                encoder.set_buffer(0, Some(&args), 0);
                encoder.set_buffer(1, Some(&args), 4);
                encoder.set_threadgroup_memory_length(0, group_size * 4);
                encoder
                    .dispatch_thread_groups(MTLSize::new(1, 1, 1), MTLSize::new(group_size, 1, 1));
                encoder.end_encoding();

                cb.commit();
                cb.wait_until_completed();

                println!("{:?}", args.get_slice());
                assert!(args.get_slice()[0] == n);
            }
        })
    }
}
