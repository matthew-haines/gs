mod gs;
use gs::array::GPUArray;
use gs::kernels::sum::sum;
use metal::*;
use objc::rc::autoreleasepool;
use std::time::{Duration, Instant};

fn timeit<F, T>(f: F) -> (T, Duration)
where
    F: FnOnce() -> T,
{
    let start = Instant::now();
    (f(), start.elapsed())
}

fn main() {
    autoreleasepool(|| {
        let device = Device::system_default().expect("No device found");
        let registry = gs::registry::create_default_registry(&device, None);
        let queue = device.new_command_queue();

        let n = 1024 * 1024 * 256 as usize;

        let a = GPUArray::<f32>::from_data(
            &device,
            &vec![1f32; n],
            Some(MTLResourceOptions::StorageModeManaged),
        );
        let b = GPUArray::<f32>::from_data(
            &device,
            &vec![1f32; n],
            Some(MTLResourceOptions::StorageModeManaged),
        );
        let c = GPUArray::<f32>::empty(
            &device,
            n as u64,
            Some(MTLResourceOptions::StorageModeManaged),
        );

        for _i in 0..8 {
            let command_buffer = queue.new_command_buffer();

            a.sync(command_buffer);
            b.sync(command_buffer);
            c.sync(command_buffer);

            let (_, dur) = timeit(|| sum(&registry, &command_buffer, &a, &b, &c));

            a.sync(command_buffer);
            b.sync(command_buffer);
            c.sync(command_buffer);

            command_buffer.commit();
            command_buffer.wait_until_completed();
            println!("{}", dur.as_micros());
        }

        assert!(c.get_slice().iter().all(|&x| x == 2.0f32));
    })
}
