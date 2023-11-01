kernel void sum(
  device const float* inA,
  device const float* inB,
  device float* result,
  uint index [[thread_position_in_grid]]) {
  result[index] = inA[index] + inB[index];
}

// template <uint block_size, uint radix>
// kernel void sort_reduce(
//   device const uint* data,
//   uint block_count,
//   uint group_index [[threadgroup_position_in_grid]],
//   uint thread_index [[thread_position_in_threadgroup]],
//   uint group_size [[ threads_per_threadgroup ]]) {
//   
//   uint histogram[radix];
// 
//   for (uint block_index = 0; block_index < block_count; ++block_index) {
//     uint index = thread_index + block_size * block_index + block_size * group_size * group_index;
//     uint item = data[index];
//     uint key = item & (radix - 1);
//     ++histogram[key];
//   }
// 
// 
// }
// 
// template sort_reduce<8, 4>;
