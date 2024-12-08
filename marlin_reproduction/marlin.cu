#ifndef MARLIN_CUDA_KERNEL_CUH
#define MARLIN_CUDA_KERNEL_CUH

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <iostream>
#include <algorithm>

constexpr int ceildiv(int a, int b) {
  return (a + b - 1) / b;
}


// Instances of `Vec` are used to organize groups of >>registers<<, as needed for instance as inputs to tensor core
// operations. Consequently, all corresponding index accesses must be compile-time constants, which is why we
// extensively use `#pragma unroll` throughout the kernel code to guarantee this.
template <typename T, int n>
struct Vec {
  T elems[n];
  __device__ T& operator[](int i) {
    return elems[i];
  }
};

using I4 = Vec<int, 4>;

// Matrix fragments for tensor core instructions; their precise layout is documented here: 
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#matrix-fragments-for-mma-m16n8k16-with-floating-point-type
using FragA = Vec<half2, 4>;
using FragB = Vec<half2, 2>;
using FragC = Vec<float, 4>;
using FragS = Vec<half2, 1>; // quantization scales

// Predicated asynchronous global->shared copy; used for inputs A where we apply predication to handle batchsizes that
// are not multiples of 16.
__device__ inline void cp_async4_pred(void* smem_ptr, const void* glob_ptr, bool pred = true) {
  const int BYTES = 16;
  uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  asm volatile(
    "{\n"
    "   .reg .pred p;\n"
    "   setp.ne.b32 p, %0, 0;\n"
    "   @p cp.async.cg.shared.global [%1], [%2], %3;\n"
    "}\n" :: "r"((int) pred), "r"(smem), "l"(glob_ptr), "n"(BYTES)
  );
}

// Asynchronous global->shared copy with a cache hint indicating that the values may be evicted immediately; used for
// quantized weights B, which are only accessed precisely once and should thus not pollute the L2 cache which we need
// for inputs A and outputs C. 
__device__ inline void cp_async4_stream(void* smem_ptr, const void* glob_ptr) {
  const int BYTES = 16;
  uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  asm volatile(
    "{\n"
    "   .reg .b64 p;\n"
    "   createpolicy.fractional.L2::evict_first.b64 p, 1.0;"
    "   cp.async.cg.shared.global.L2::cache_hint [%0], [%1], %2, p;\n"
    "}\n" :: "r"(smem), "l"(glob_ptr), "n"(BYTES)
  );
}

// Async copy fence.
__device__ inline void cp_async_fence() {
  asm volatile("cp.async.commit_group;\n" ::);
}

// Wait until at most `n` async copy stages are still pending.
template <int n>
__device__ inline void cp_async_wait() {
  asm volatile("cp.async.wait_group %0;\n" :: "n"(n));
}

// m16n8k16 tensor core mma instruction with fp16 inputs and fp32 output/accumulation.
__device__ inline void mma(const FragA& a_frag, const FragB& frag_b, FragC& frag_c) {
  const uint32_t* a = reinterpret_cast<const uint32_t*>(&a_frag);
  const uint32_t* b = reinterpret_cast<const uint32_t*>(&frag_b);
  float* c = reinterpret_cast<float*>(&frag_c);
  asm volatile(
    "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
    : "=f"(c[0]), "=f"(c[1]), "=f"(c[2]), "=f"(c[3])
    :  "r"(a[0]),  "r"(a[1]),  "r"(a[2]),  "r"(a[3]),  "r"(b[0]),  "r"(b[1]),
       "f"(c[0]),  "f"(c[1]),  "f"(c[2]),  "f"(c[3])
  );
}

// Instruction for loading a full 16x16 matrix fragment of operand A from shared memory, directly in tensor core layout.
__device__ inline void ldsm4(FragA& frag_a, const void* smem_ptr) {
  uint32_t* a = reinterpret_cast<uint32_t*>(&frag_a);
  uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  asm volatile(
    "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
    : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3]) : "r"(smem)
  );
}

// Lookup-table based 3-input logical operation; explicitly used for dequantization as the compiler does not seem to
// automatically recognize it in all cases. 
template <int lut>
__device__ inline int lop3(int a, int b, int c) {
  int res;
  asm volatile(
    "lop3.b32 %0, %1, %2, %3, %4;\n"
    : "=r"(res) : "r"(a), "r"(b), "r"(c), "n"(lut)
  );
  return res;
}

// Efficiently dequantize an int32 value into a full B-fragment of 4 fp16 values.
// We mostly follow the strategy in the link below, with some small changes:
// https://github.com/NVIDIA/FasterTransformer/blob/main/src/fastertransformer/cutlass_extensions/include/cutlass_extensions/interleaved_numeric_conversion.h
__device__ inline FragB dequant(int q) {
    const int LOW_BIT_MASK = 0x000f000f;
    const int HIGH_BIT_MASK = 0x00f000f0;
    const int MAGIC_NUM = 0b01100100000000000110010000000000;
    
    const int LUT = (0xf0 & 0xcc) | 0xaa;
    int low_val = lop3<LUT>(q, LOW_BIT_MASK, MAGIC_NUM);
    int high_val = lop3<LUT>(q, HIGH_BIT_MASK, MAGIC_NUM);
    
    // This is the half2 {1032, 1032} represented as an integer.
    const uint32_t FP16_TOP_MAGIC_NUM = 0x64086408;
    // This is the half2 {1 / 16, 1 / 16} represented as an integer.
    const uint32_t ONE_SIXTEENTH = 0x2c002c00;
    // This is the half2 {-72, -72} represented as an integer.
    const uint32_t NEG_72 = 0xd480d480;
    
    FragB res; 
    asm volatile("sub.f16x2 %0, %1, %2;\n" : 
        "=r"((reinterpret_cast<uint32_t*>(&res))[0]) : "r"(low_val), "r"(FP16_TOP_MAGIC_NUM));
    asm volatile("hfma.f16x2 %0, %1, %2, %3;\n" : 
        "=r"((reinterpret_cast<uint32_t*>(&res))[1]) : "r"(high_val), "r"(ONE_SIXTEENTH), "r"(NEG_72));
    return res;
}

#define BLOCK_PER_SM 4
#define M_BLOCK_SIZE 16
#define K_BLOCK_SIZE 8 
#define N_BLOCK_SIZE 8
#define THREADS 256
#define STAGES 4
#define SHARED_MEM 96 * 1024



__device__ inline void wait_for_stage() {
  cp_async_wait<STAGES - 2>();
  __syncthreads();
}


template <
  // const int threads, // number of threads in a threadblock
  const int thread_m_blocks, // number of 16x16 blocks in the m dimension (batchsize) of the threadblock 
  const int thread_n_blocks, // same for n dimension (output) 
  const int thread_k_blocks // same for k dimension (reduction)
  // const int stages, // number of stages for the async global->shared fetch pipeline
  // const int group_blocks = -1 // number of consecutive 16x16 blocks with a separate quantization scale
>
__global__ void Marlin(
  const int4* __restrict__ A, // fp16 input matrix of shape mxk 
  const int4* __restrict__ B, // 4bit quantized weight matrix of shape kxn 
        int4* __restrict__ C, // fp16 output buffer of shape mxn
  const int4* __restrict__ s, // fp16 quantization scales of shape (k/groupsize)xn 
  int  total_m, // batch dimension m
  int  total_n, // output dimension n
  int  total_k, // reduction dimension k
  int* workspace // extra global storage for barrier synchronization 
) {

  // instead of having each thread block process a "stripe", just have it do
  // an entire column of the B matrix.

  
  int num_parallel = 1;
  
  int num_tiles_k = total_k / (16 * thread_k_blocks);
  int num_tiles_n = total_n / (16 * thread_n_blocks);
  
  // Each threadblock handles one n tile (column)
  int total_tiles_per_threadblock = num_tiles_k;
  
  // Current row starts at 0 for each threadblock
  int cur_row = 0;
  // Each threadblock gets assigned a different column
  // hackey solution right now
  int cur_col = blockIdx.x % num_tiles_n;
  if (blockIdx.x >= num_tiles_n) {
    return; 
  }
  

  int a_global_stride = total_k / 8; 
  constexpr int a_global_read_delta_outer = K_BLOCK_SIZE * thread_k_blocks / 8;
  int a_global_read_delta_inner = a_global_stride * (THREADS / a_global_read_delta_outer);

  int a_global_read_index = a_global_stride * (threadIdx.x / a_global_read_delta_outer) 
              + (threadIdx.x % a_global_read_delta_outer);
  
  constexpr int a_shared_stride = 16 * thread_k_blocks / 8;
  constexpr int a_shared_write_delta = a_shared_stride * (THREADS / a_global_read_delta_outer);
  int a_shared_write_index = a_shared_stride * (threadIdx.x / a_global_read_delta_outer) + 
              (threadIdx.x % a_global_read_delta_outer); 
  if (threadIdx.x < 10 && blockIdx.x == 0) {
    printf("threadIdx.x: %d, a_shared_stride: %d, a_shared_write_delta: %d, a_shared_write_index: %d\n", 
           threadIdx.x, a_shared_stride, a_shared_write_delta, a_shared_write_index);
  }
  constexpr int a_total_tile_size = a_shared_stride * 16 * thread_m_blocks;
  constexpr int a_shared_write_iters = ceildiv(a_total_tile_size, a_shared_write_delta);


  int b_global_stride = 16 * total_n / 32; 
  constexpr int b_shared_stride = 32 * thread_n_blocks / 4;
  int b_global_read_delta_outer = b_global_stride * thread_k_blocks;
  int b_global_read_delta_inner = b_global_stride * (THREADS / b_shared_stride);
  constexpr int b_shared_write_delta = THREADS;
  constexpr int b_total_tile_size = b_shared_stride * thread_k_blocks;
  constexpr int b_shared_write_iters = ceildiv(b_total_tile_size, b_shared_write_delta);

  int b_global_read_index = b_global_stride * (threadIdx.x / b_shared_stride) 
              + (threadIdx.x % b_shared_stride);
  b_global_read_index += b_shared_stride * cur_col;
  b_global_read_index += b_global_read_delta_outer * cur_row;
  int b_shared_write_index = threadIdx.x; 

  int a_shared_write_indices[a_shared_write_iters];
  #pragma unroll
  for (int i = 0; i < a_shared_write_iters; i++)
    a_shared_write_indices[i] = a_shared_write_index + a_shared_stride * i;
  

  // Shared memory storage for global fetch pipelines. 
  extern __shared__ int4 smem[];
  int4* smem_a = smem;
  int4* smem_b = smem_a + (STAGES * a_total_tile_size);
  int4* smem_s = smem_b + (STAGES * b_total_tile_size);

  half2* smem_a_half = reinterpret_cast<half2*>(smem_a);
  // Zero out smem_a
  if (threadIdx.x < a_total_tile_size) {
    #pragma unroll
    for (int i = 0; i < STAGES; i++) {
      smem_a_half[threadIdx.x + i * a_total_tile_size] = {0, 0};
    }
  }
  __syncthreads();


  // Registers
  FragA frag_a[2][thread_m_blocks];
  I4 frag_b_quant[2];
  FragC frag_c[thread_m_blocks][4][2];
  FragS frag_s[2][4];

  auto zero_accumulators = [&] () {
    #pragma unroll
    for (int i = 0; i < thread_m_blocks * 4 * 2 * 4; i++) {
      reinterpret_cast<float*>(frag_c)[i] = 0;
    }
  };

  auto fetch_to_smem_a = [&] (int pipe, int a_off, bool pred = true) {
    if (pred) { 
      int4* smem_a_cur = smem_a + pipe * a_total_tile_size;
      #pragma unroll
      for (int i = 0; i < a_shared_write_iters; i++) {
        cp_async4_pred(
          &smem_a_cur[a_shared_write_indices[i]],
          &A[a_global_read_index + 
            a_global_read_delta_outer * a_off + 
            a_global_read_delta_inner * i],
          true
        );
      }
    }
  };
  
  #pragma unroll
  for (int i = 0; i < STAGES - 1; i++)
    fetch_to_smem_a(i, i, true);
  zero_accumulators();
  wait_for_stage();
    
  // Print all elements of smem_a
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    for (int i = 0; i < STAGES * a_total_tile_size; i++) {
      half2* smem_a_cur = reinterpret_cast<half2*>(smem_a);
      printf("smem_a[%d]: %f %f\n", i, 
             __half2float(smem_a_cur[i].x), __half2float(smem_a_cur[i].y));
    }
  }
  __syncthreads(); // Ensure all threads complete printing before continuing

}


#define LAUNCH_MARLIN_KERNEL(M, N, K) \
  if ( \
    thread_m_blocks == M && \
    thread_n_blocks == N && \
    thread_k_blocks == K \
  ) { \
    cudaFuncSetAttribute( \
      Marlin<M, N, K>, \
      cudaFuncAttributeMaxDynamicSharedMemorySize, \
      SHARED_MEM \
    ); \
    dim3 block_size(sms, 1, 1); \
    dim3 thread_size(THREADS, 1, 1); \
    Marlin<M, N, K><<<block_size, thread_size, SHARED_MEM, stream>>>( \
      A_ptr, B_ptr, C_ptr, s_ptr, \
      cur_m, total_n, total_k, \
      workspace_ptr \
    ); \
  }


int marlin_cuda(
  const void* A,
  const void* B,
        void* C,
        void* s,
  int total_m,
  int total_n,
  int total_k,
  void* workspace,
  int groupsize = -1,
  int dev = 0,
  cudaStream_t stream = 0,
  int thread_k = -1,
  int thread_n = -1,
  int sms = -1,
  int max_par = 16
) {
  int num_m_blocks = ceildiv(total_m, M_BLOCK_SIZE);
  int pad_m = M_BLOCK_SIZE * num_m_blocks - total_m;
  if (sms == -1)
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
  if (thread_k == -1) thread_k = 64; 
  if (thread_n == -1) thread_n = 256;

  int thread_k_blocks = thread_k /  16;
  int thread_n_blocks = thread_n /  16; 


  const int4* A_ptr = (const int4*) A;
  const int4* B_ptr = (const int4*) B;
  int4* C_ptr = (int4*) C;
  const int4* s_ptr = (const int4*) s;
  int* workspace_ptr = (int*) workspace;
  
  for (int i_iter = 0; i_iter < num_m_blocks; i_iter += 4) {
    int thread_m_blocks = BLOCK_PER_SM;
    int cur_m = M_BLOCK_SIZE * BLOCK_PER_SM;
    int par = 1; 
    // if (num_m_blocks - i_iter > BLOCK_PER_SM) {
    //   par = std::min(num_m_blocks - i_iter / BLOCK_PER_SM, max_par);
    //   cur_m = M_BLOCK_SIZE * BLOCK_PER_SM * par;
    //   i_iter += BLOCK_PER_SM * (par - 1);
    //   thread_m_blocks = BLOCK_PER_SM;
    //   printf("thread_m_blocks: %d, par: %d, cur_m: %d\n", thread_m_blocks, par, cur_m);
    // }
    
    LAUNCH_MARLIN_KERNEL(4, 16, 4)

    A_ptr += M_BLOCK_SIZE * thread_m_blocks * (total_k / K_BLOCK_SIZE) * par;
    C_ptr += M_BLOCK_SIZE * thread_m_blocks * (total_n / N_BLOCK_SIZE) * par;
  }

  return 0;
  
}

#endif
