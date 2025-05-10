#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <chrono>
#include <iostream>
#include <vector>
#include <stdexcept>
#include <cstdint>
#include <functional>
#include <algorithm>
#include <cmath>

#include <arm_neon.h>


void inline matrix_multiply_neon(
    float32_t *A, 
    float32_t *B, 
    float32_t *C, 
    uint32_t n, 
    uint32_t m, 
    uint32_t k
) {
    /* 
     * Multiply matrices A and B, store the result in C. 
     * It is the user's responsibility to make sure the matrices are compatible.
     */     

    int A_idx;
    int B_idx;
    int C_idx;
    
    // these are the columns of a 4x4 sub matrix of A
    float32x4_t A0;
    float32x4_t A1;
    float32x4_t A2;
    float32x4_t A3;
    
    // these are the columns of a 4x4 sub matrix of B
    float32x4_t B0;
    float32x4_t B1;
    float32x4_t B2;
    float32x4_t B3;
    
    // these are the columns of a 4x4 sub matrix of C
    float32x4_t C0;
    float32x4_t C1;
    float32x4_t C2;
    float32x4_t C3;
    
    for (int i_idx=0; i_idx<n; i_idx+=4) {
        for (int j_idx=0; j_idx<m; j_idx+=4) {
            // Zero accumulators before matrix op
            C0 = vmovq_n_f32(0);
            C1 = vmovq_n_f32(0);
            C2 = vmovq_n_f32(0);
            C3 = vmovq_n_f32(0);
            for (int k_idx=0; k_idx<k; k_idx+=4) {
                // Compute base index to 4x4 block
                A_idx = i_idx + n*k_idx;
                B_idx = k*j_idx + k_idx;
                
                // Load most current A values in row 
                A0 = vld1q_f32(A+A_idx);
                A1 = vld1q_f32(A+A_idx+n);
                A2 = vld1q_f32(A+A_idx+2*n);
                A3 = vld1q_f32(A+A_idx+3*n);
                
                // Multiply accumulate in 4x1 blocks, i.e. each column in C
                B0 = vld1q_f32(B+B_idx);
                C0 = vfmaq_laneq_f32(C0, A0, B0, 0);
                C0 = vfmaq_laneq_f32(C0, A1, B0, 1);
                C0 = vfmaq_laneq_f32(C0, A2, B0, 2);
                C0 = vfmaq_laneq_f32(C0, A3, B0, 3);
                
                B1 = vld1q_f32(B+B_idx+k);
                C1 = vfmaq_laneq_f32(C1, A0, B1, 0);
                C1 = vfmaq_laneq_f32(C1, A1, B1, 1);
                C1 = vfmaq_laneq_f32(C1, A2, B1, 2);
                C1 = vfmaq_laneq_f32(C1, A3, B1, 3);
                
                B2 = vld1q_f32(B+B_idx+2*k);
                C2 = vfmaq_laneq_f32(C2, A0, B2, 0);
                C2 = vfmaq_laneq_f32(C2, A1, B2, 1);
                C2 = vfmaq_laneq_f32(C2, A2, B2, 2);
                C2 = vfmaq_laneq_f32(C2, A3, B2, 3);
                
                B3 = vld1q_f32(B+B_idx+3*k);
                C3 = vfmaq_laneq_f32(C3, A0, B3, 0);
                C3 = vfmaq_laneq_f32(C3, A1, B3, 1);
                C3 = vfmaq_laneq_f32(C3, A2, B3, 2);
                C3 = vfmaq_laneq_f32(C3, A3, B3, 3);
            }
            // Compute base index for stores
            C_idx = n*j_idx + i_idx;
            vst1q_f32(C+C_idx, C0);
            vst1q_f32(C+C_idx+n, C1);
            vst1q_f32(C+C_idx+2*n, C2);
            vst1q_f32(C+C_idx+3*n, C3);
        }
    }
}


void matrix_multiply(
    const std::vector<float>& A, 
    const std::vector<float>& B, 
    std::vector<float>& C,
    uint32_t n,  // rows in A (and rows in C)
    uint32_t m,  // columns in B (and columns in C)
    uint32_t k   // columns in A and rows in B
) {
    // Validate input matrix dimensions
    if (A.size() != n * k) {
        throw std::invalid_argument("Matrix A dimensions do not match expected size n*k");
    }
    if (B.size() != k * m) {
        throw std::invalid_argument("Matrix B dimensions do not match expected size k*m");
    }
    if (C.size() != n * m) {
        C.resize(n * m, 0.0f);
    }

    // For empty matrices, return immediately
    if (n == 0 || m == 0 || k == 0) {
        if (!C.empty()) {
            std::fill(C.begin(), C.end(), 0.0f);
        }
        return;
    }

    // Round up dimensions to multiples of 4 for NEON processing
    const uint32_t n_padded = ((n + 3) / 4) * 4;
    const uint32_t m_padded = ((m + 3) / 4) * 4;
    const uint32_t k_padded = ((k + 3) / 4) * 4;

    // If dimensions are already multiples of 4, no need for padding
    if (n == n_padded && m == m_padded && k == k_padded) {
        // Initialize C to zeros before computation
        std::fill(C.begin(), C.end(), 0.0f);
        matrix_multiply_neon(const_cast<float*>(A.data()), const_cast<float*>(B.data()), C.data(), n, m, k);
        return;
    }

    // Create padded matrices
    std::vector<float> A_padded(n_padded * k_padded, 0.0f);
    std::vector<float> B_padded(k_padded * m_padded, 0.0f);

    // Copy data from A to A_padded (column-major format)
    for (uint32_t col = 0; col < k; ++col) {
        for (uint32_t row = 0; row < n; ++row) {
            A_padded[col * n_padded + row] = A[col * n + row];
        }
    }

    // Copy data from B to B_padded (column-major format)
    for (uint32_t col = 0; col < m; ++col) {
        for (uint32_t row = 0; row < k; ++row) {
            B_padded[col * k_padded + row] = B[col * k + row];
        }
    }

    // Create padded result matrix
    std::vector<float> C_padded(n_padded * m_padded, 0.0f);

    // Perform matrix multiplication using NEON optimized function
    matrix_multiply_neon(A_padded.data(), B_padded.data(), C_padded.data(), n_padded, m_padded, k_padded);

    // Extract the relevant portion of the result
    for (uint32_t col = 0; col < m; ++col) {
        for (uint32_t row = 0; row < n; ++row) {
            C[col * n + row] = C_padded[col * n_padded + row];
        }
    }
}


std::vector<float> matrix_multiply(
    const std::vector<float>& A,
    const std::vector<float>& B,
    uint32_t n,  // rows in A (and rows in C)
    uint32_t m,  // columns in B (and columns in C)
    uint32_t k   // columns in A and rows in B
) {
    std::vector<float> C(n * m, 0.0f);
    matrix_multiply(A, B, C, n, m, k);
    return C;
}


std::vector<float> naive_matrix_multiply(
    const std::vector<float>& A,
    const std::vector<float>& B,
    uint32_t n, uint32_t m, uint32_t k
) {
    // Perform naive matrix multiplication for verification
    std::vector<float> C_expected(n * m, 0.0f);
    
    // C[i,j] = sum(A[i,p] * B[p,j]) for all p from 0 to k-1
    for (uint32_t j = 0; j < m; ++j) {       // column of C and B
        for (uint32_t i = 0; i < n; ++i) {   // row of C and A
            float sum = 0.0f;
            for (uint32_t p = 0; p < k; ++p) { // column of A, row of B
                sum += A[p * n + i] * B[j * k + p];
            }
            C_expected[j * n + i] = sum;
        }
    }
    
    return C_expected;
}


bool verify_matrix_multiply(
    const std::vector<float>& C,
    const std::vector<float>& C_expected,
    float epsilon = 1e-5
) {
    // Check if the results match
    for (uint32_t i = 0; i < C.size(); ++i) {
        if (std::abs(C[i] - C_expected[i]) > epsilon) {
            std::cout << "Mismatch at index " << i << ": " 
                      << C[i] << " vs expected " << C_expected[i] << std::endl;
            return false;
        }
    }
    
    return true;
}


void matrix_init_rand(std::vector<float>& M, uint32_t numvals) {
        // Ensure the vector has enough capacity before filling
        if (M.size() != numvals) {
                M.resize(numvals);
        }
        
        // Fill with random values
        for (int i=0; i<numvals; i++) {
                M[i] = (float)rand()/(float)(RAND_MAX);
        }
}
