use metal::*;
use std::marker::PhantomData;

pub struct GPUArray<T> {
    buffer: Buffer,
    phantom: PhantomData<T>,
}

fn array_bytes<T>(n: u64) -> u64 {
    // Naive, ignores padding/alignment
    return n * std::mem::size_of::<T>() as u64;
}

impl<T> GPUArray<T> {
    pub fn from_data(
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

    pub fn empty(device: &Device, n: u64, resource_options: Option<MTLResourceOptions>) -> Self {
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

    pub fn len(&self) -> u64 {
        self.buffer.length() / std::mem::size_of::<T>() as u64
    }

    pub fn get_slice(&self) -> &[T] {
        unsafe {
            std::slice::from_raw_parts(self.buffer.contents().cast::<T>(), self.len() as usize)
        }
    }

    pub fn get_mut_slice(&mut self) -> &mut [T] {
        unsafe {
            std::slice::from_raw_parts_mut(self.buffer.contents().cast::<T>(), self.len() as usize)
        }
    }

    pub fn sync(&self, command_buffer: &CommandBufferRef) {
        if self.buffer.resource_options() == MTLResourceOptions::StorageModeManaged {
            let blit_command_encoder = command_buffer.new_blit_command_encoder();
            blit_command_encoder.synchronize_resource(&self.buffer);
            blit_command_encoder.end_encoding();
        }
    }
}

impl<T> std::ops::Deref for GPUArray<T> {
    type Target = Buffer;

    fn deref(&self) -> &Self::Target {
        &self.buffer
    }
}
