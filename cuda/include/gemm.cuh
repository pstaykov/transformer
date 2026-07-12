#pragma once
#include "common.cuh"
#include "kernels.cuh"

// Row-major GEMM on top of column-major cuBLAS, using the standard identity
// C_rowmajor = A_rowmajor * B_rowmajor  <=>  C_colmajor^T = (A*B)^T = B^T*A^T,
// computed by cuBLAS as a column-major product with A and B swapped. This is
// the same trick used by e.g. Caffe's caffe_gpu_gemm.
//
// Semantics match classic (S)GEMM: C(M,N) = op(A) * op(B), where op(X) = X if
// Trans::N else X^T. `A`/`B` are passed as they are physically stored in
// row-major memory (i.e. if transA==T, A is physically stored as a (K,M)
// row-major array, and op(A) = A^T is the logical (M,K) matrix).
enum class Trans { N, T };

// `fp8`: when true AND the global g_fp8_enabled switch is on, both operands are
// snapped to E4M3 (via the fp8_scratch_* buffers) before the fp32 accumulate -
// this is set only at the call sites that use fp8.matmul/qmatmul in the numpy
// reference (Linear forward + its dX backward, and every attention GEMM), never
// on the weight-gradient GEMMs (those are plain np.tensordot in Python).
inline void gemm(cublasHandle_t handle, Trans transA, Trans transB,
                  int M, int N, int K, float alpha,
                  const float* A, const float* B, float beta, float* C,
                  bool fp8 = false) {
    if (fp8 && g_fp8_enabled) {
        float* Aq = fp8_scratch_a((size_t)M * K);
        float* Bq = fp8_scratch_b((size_t)K * N);
        launch_quantize_e4m3(A, Aq, M * K);
        launch_quantize_e4m3(B, Bq, K * N);
        A = Aq;
        B = Bq;
    }
    int lda = (transA == Trans::N) ? K : M;
    int ldb = (transB == Trans::N) ? N : K;
    cublasOperation_t opA = (transA == Trans::N) ? CUBLAS_OP_N : CUBLAS_OP_T;
    cublasOperation_t opB = (transB == Trans::N) ? CUBLAS_OP_N : CUBLAS_OP_T;
    CUBLAS_CHECK(cublasSgemm(handle, opB, opA, N, M, K, &alpha, B, ldb, A, lda, &beta, C, N));
}

// Strided-batched variant, same row-major convention, batchCount independent
// (M,K)x(K,N)->(M,N) products at fixed strides between batches. The operand
// buffers are tightly packed (stride == one matrix), so quantizing
// stride*batchCount elements covers every batch.
inline void gemm_batched(cublasHandle_t handle, Trans transA, Trans transB,
                          int M, int N, int K, float alpha,
                          const float* A, long long strideA,
                          const float* B, long long strideB,
                          float beta, float* C, long long strideC,
                          int batchCount, bool fp8 = false) {
    if (fp8 && g_fp8_enabled) {
        size_t nA = (size_t)strideA * batchCount;
        size_t nB = (size_t)strideB * batchCount;
        float* Aq = fp8_scratch_a(nA);
        float* Bq = fp8_scratch_b(nB);
        launch_quantize_e4m3(A, Aq, (int)nA);
        launch_quantize_e4m3(B, Bq, (int)nB);
        A = Aq;
        B = Bq;
    }
    int lda = (transA == Trans::N) ? K : M;
    int ldb = (transB == Trans::N) ? N : K;
    cublasOperation_t opA = (transA == Trans::N) ? CUBLAS_OP_N : CUBLAS_OP_T;
    cublasOperation_t opB = (transB == Trans::N) ? CUBLAS_OP_N : CUBLAS_OP_T;
    CUBLAS_CHECK(cublasSgemmStridedBatched(
        handle, opB, opA, N, M, K, &alpha,
        B, ldb, strideB,
        A, lda, strideA,
        &beta, C, N, strideC,
        batchCount));
}
