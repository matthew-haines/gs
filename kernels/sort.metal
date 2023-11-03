#include <metal_compute>

template <typename T>
void group_reduce(device T& out, threadgroup T* data, const uint idx, const uint n) {
  // https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf

  threadgroup_barrier(metal::mem_flags::mem_threadgroup);

  for (uint s = 1; s < n; s *= 2) {
    if (idx % (2 * s) == 0 && (idx + s) < n) {
      data[idx] += data[idx + s];
    }
    threadgroup_barrier(metal::mem_flags::mem_threadgroup);
  }
   
  if (idx == 0) {
    out = data[0];
  }
}

kernel void reduce_test(
  device uint& out,
  device const uint& n,
  uint thread_index [[thread_position_in_threadgroup]],
  uint group_size [[ threads_per_threadgroup ]],
  threadgroup uint* shared_memory [[ threadgroup(0) ]]
  ) {
  shared_memory[thread_index] = 1;
  group_reduce(out, shared_memory, thread_index, n);
}

template <typename T, uint n>
struct Vector {
  T data[n];

  thread T& operator[](uint i) {
    return data[i];
  }

  thread const T& operator[](uint i) const {
    return data[i];
  }

  threadgroup T& operator[](uint i) threadgroup {
    return data[i];
  }

  threadgroup const T& operator[](uint i) const threadgroup {
    return data[i];
  }

  thread Vector<T, n>& operator+=(const thread Vector<T, n>& b) {
    for (uint i = 0; i < n; ++i)
      data[i] += b[i];
  }

  threadgroup Vector<T, n>& operator+=(const threadgroup Vector<T, n>& b) threadgroup {
    for (uint i = 0; i < n; ++i)
      data[i] += b[i];
  }
};

template <uint block_size, uint radix>
kernel void sort_downsweep(
  device const uint* data,
  device Vector<uint, radix>* out_histograms,
  device const uint& n,
  device const uint& block_count,
  device const uint& shift,
  uint thread_index [[thread_position_in_threadgroup]],
  uint group_index [[threadgroup_position_in_grid]],
  uint group_size [[ threads_per_threadgroup ]],
  threadgroup Vector<uint, radix>* shared_memory [[ threadgroup(0) ]]) {
  Vector<uint, radix> histogram{};

  const uint bitshift = metal::popcount(radix - 1) * shift;
  const uint mask = (radix - 1) << bitshift;

  for (uint block_index = 0; block_index < block_count; ++block_index) {
    uint index = thread_index + block_size * block_index + block_size * group_size * group_index;
    if (index >= n) {
      continue;
    }

    uint item = data[index];
    uint key = (item & mask) >> bitshift;
    ++histogram[key];
  }

  shared_memory[thread_index] = histogram;
  group_reduce(out_histograms[group_index], shared_memory, thread_index, n);
}

template [[host_name("sort_downsweep_u32")]] kernel void sort_downsweep<8, 4>(device const uint*, device Vector<uint, 4>*, device const uint&, device const uint&, device const uint&, uint, uint, uint, threadgroup Vector<uint, 4>*);
