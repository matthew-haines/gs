use metal::*;
use objc::rc::autoreleasepool;
use std::cmp::min;
use std::collections::HashMap;
use std::marker::PhantomData;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

struct Library {
    functions: HashMap<String, (Function, ComputePipelineState)>,
}

impl Library {
    fn new(device: &Device, lib: metal::Library) -> Library {
        let mut functions = HashMap::<String, (Function, ComputePipelineState)>::new();
        for function_name in lib.function_names().iter() {
            let function = lib.get_function(function_name, None).unwrap();
            let pipeline_state = device
                .new_compute_pipeline_state_with_function(&function)
                .unwrap();

            functions.insert(function_name.to_string(), (function, pipeline_state));
        }

        Library { functions }
    }
}

struct ProgramRegistry<'a> {
    device: &'a Device,
    libraries: HashMap<String, Library>,
}

impl<'a> ProgramRegistry<'_> {
    fn new(device: &'a Device) -> ProgramRegistry<'a> {
        ProgramRegistry {
            device,
            libraries: HashMap::new(),
        }
    }

    fn register_library(
        &mut self,
        library_name: &str,
        path: &Path,
        compile_options: Option<&CompileOptions>,
    ) {
        let src = std::fs::read_to_string(path).unwrap();
        let mtl_lib = self
            .device
            .new_library_with_source(&src, compile_options.unwrap_or(&CompileOptions::new()))
            .unwrap();
        self.libraries
            .insert(library_name.to_owned(), Library::new(self.device, mtl_lib));
    }

    fn get_function(
        &self,
        library_name: &str,
        function_name: &str,
    ) -> Option<&(Function, ComputePipelineState)> {
        self.libraries
            .get(library_name)?
            .functions
            .get(function_name)
    }
}

static NAMES: &'static [&'static str] = &["sum"];

fn default_registry_path() -> PathBuf {
    if let Ok(path) = std::env::var("REGISTRY_PATH") {
        PathBuf::from(path)
    } else {
        PathBuf::from(".")
    }
}

fn create_default_registry<'a>(device: &'a Device, folder: Option<&Path>) -> ProgramRegistry<'a> {
    let mut registry = ProgramRegistry::new(&device);

    let resolved_folder = if let Some(path) = folder {
        PathBuf::from(path)
    } else {
        default_registry_path()
    };

    for name in NAMES {
        let file_path = resolved_folder.join(format!("{}.metal", name));
        registry.register_library(name, &file_path, None);
    }

    registry
}

struct GPUArray<T> {
    buffer: Buffer,
    phantom: PhantomData<T>,
}

fn array_bytes<T>(n: u64) -> u64 {
    // Naive, ignores padding/alignment
    return n * std::mem::size_of::<T>() as u64;
}

impl<T> GPUArray<T> {
    fn from_data(
        device: &Device,
        data: &[T],
        resource_options: Option<MTLResourceOptions>,
    ) -> Self {
        let length = array_bytes::<T>(data.len() as u64);

        let buffer = device.new_buffer_with_data(
            unsafe { std::mem::transmute(data.as_ptr()) },
            length,
            resource_options.unwrap_or(MTLResourceOptions::StorageModeShared),
        );

        GPUArray {
            buffer,
            phantom: PhantomData,
        }
    }

    fn empty(device: &Device, n: u64, resource_options: Option<MTLResourceOptions>) -> Self {
        let length = array_bytes::<T>(n);
        let buffer = device.new_buffer(
            length,
            resource_options.unwrap_or(MTLResourceOptions::StorageModeShared),
        );

        GPUArray {
            buffer,
            phantom: PhantomData,
        }
    }

    fn len(&self) -> u64 {
        self.buffer.length() / std::mem::size_of::<T>() as u64
    }

    fn get_slice(&self) -> &[T] {
        unsafe {
            std::slice::from_raw_parts(self.buffer.contents().cast::<T>(), self.len() as usize)
        }
    }
}

impl<T> std::ops::Deref for GPUArray<T> {
    type Target = Buffer;

    fn deref(&self) -> &Self::Target {
        &self.buffer
    }
}

fn sum<T>(
    registry: &ProgramRegistry,
    command_queue: &CommandQueue,
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

    let command_buffer = command_queue.new_command_buffer();

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

    command_buffer.commit();
    command_buffer.wait_until_completed();
}

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
        let registry = create_default_registry(&device, None);
        let queue = device.new_command_queue();

        let n = 1024 * 1024 * 256 as usize;

        let a = GPUArray::<f32>::from_data(&device, &vec![1f32; n], None);
        let b = GPUArray::<f32>::from_data(&device, &vec![1f32; n], None);
        let c = GPUArray::<f32>::empty(&device, n as u64, None);

        let (_, dur) = timeit(|| sum(&registry, &queue, &a, &b, &c));
        println!("{}", dur.as_micros());

        assert!(c.get_slice().iter().all(|&x| x == 2.0f32));
    })
}
