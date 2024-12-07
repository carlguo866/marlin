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
  const int LO = 0x000f000f;
  const int HI = 0x00f000f0;
  const int EX = 0x64006400;
  // Guarantee that the `(a & b) | c` operations are LOP3s.
  int lo = lop3<(0xf0 & 0xcc) | 0xaa>(q, LO, EX);
  int hi = lop3<(0xf0 & 0xcc) | 0xaa>(q, HI, EX);
  // We want signed int4 outputs, hence we fuse the `-8` symmetric zero point directly into `SUB` and `ADD`.
  const int SUB = 0x64086408;
  const int MUL = 0x2c002c00;
  const int ADD = 0xd480d480;
  FragB frag_b;
  frag_b[0] = __hsub2(
    *reinterpret_cast<half2*>(&lo),
    *reinterpret_cast<const half2*>(&SUB)
  );
  frag_b[1] = __hfma2(
    *reinterpret_cast<half2*>(&hi),
    *reinterpret_cast<const half2*>(&MUL), *reinterpret_cast<const half2*>(&ADD)
  );
  return frag_b;
}

#define BLOCK_PER_SM 4
#define M_BLOCK_SIZE 16
#define K_BLOCK_SIZE 8 
#define N_BLOCK_SIZE 8
#define THREADS 256
#define STAGES 4
#define SHARED_MEM 96 * 1024




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
  
  int total_tiles_per_threadblock = ceildiv(num_tiles_k * num_tiles_n * num_parallel, gridDim.x);
  
  int cur_row = (cur_block * blockIdx.x) % num_tiles_k;
  int cur_col = (cur_block * blockIdx.x) / num_tiles_k;
  int a_gl_stride = prob_k / 8; 
  constexpr int a_sh_stride = 16 * thread_k_blocks / 8; 
  constexpr int a_gl_rd_delta_o = 16 * thread_k_blocks / 8;
  int a_gl_rd_delta_i = a_gl_stride * (threads / a_gl_rd_delta_o);
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
