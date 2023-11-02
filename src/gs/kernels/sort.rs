use crate::gs::array::GPUArray;
use crate::gs::registry::ProgramRegistry;
use metal::*;
use std::cmp::min;

pub struct SortStorage {
    n: u64,
}

impl Drop for SortStorage {
    fn drop(&mut self) {}
}

pub fn allocate_sort_storage(n: u64) -> SortStorage {
    SortStorage { n }
}

pub fn sort<T>(
    registry: &ProgramRegistry,
    storage: &SortStorage,
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
