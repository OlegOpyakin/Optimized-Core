#include <vector>
#include <algorithm>

#define TILE_SIZE 32U

#pragma once

std::vector<float> naive_matrix_multiply(
    const std::vector<float>& A,
    const std::vector<float>& B,
    uint32_t n, uint32_t m, uint32_t k
) {
    std::vector<float> C(n * m, 0.0f);
    
    // C[i,j] = sum(A[i,p] * B[p,j]) for all p from 0 to k-1
    // A[i,p] is stored at index [p*n+i] (column-major)
    // B[p,j] is stored at index [j*k+p] (row-major for B transpose)
    // C[i,j] is stored at index [j*n+i] (column-major)
    for (uint32_t j = 0; j < m; ++j) {
        for (uint32_t i = 0; i < n; ++i) {
            float sum = 0.0f;
            for (uint32_t p = 0; p < k; ++p) {
                sum += A[p * n + i] * B[j * k + p];
            }
            C[j * n + i] = sum;
        }
    }
    return C;
}

/*
std::vector<float> matrix_multiply_tiled(
    const std::vector<float>& A,
    const std::vector<float>& B,
    uint32_t n,  // rows in A (and rows in C)
    uint32_t k   // columns in A and rows in B
    uint32_t m,  // columns in B (and columns in C)
) {
    std::vector<float> C(n * m, 0.0f);

    //matrix_mult_wiki_block(A.data(), B.data(), C.data(), n, m, k);

    const int block_size = 64 / sizeof(float); // 64 = common cache line size
    for(int i=0; i<N; i++) {
        for(int j=0; j<K; j++) {
            C[K*i + j] = 0;
        }
    }
    for (int i0 = 0; i0 < N; i0 += block_size) {
        int imax = i0 + block_size > N ? N : i0 + block_size;

        for (int j0 = 0; j0 < M; j0 += block_size) {
            int jmax = j0 + block_size > M ? M : j0 + block_size;

            for (int k0 = 0; k0 < K; k0 += block_size) {
                int kmax = k0 + block_size > K ? K : k0 + block_size;

                for (int j1 = j0; j1 < jmax; ++j1) {
                    int sj = M * j1;

                    for (int i1 = i0; i1 < imax; ++i1) {
                        int mi = M * i1;
                        int ki = K * i1;
                        int kij = ki + j1;

                        for (int k1 = k0; k1 < kmax; ++k1) {
                            C[kij] += A[mi + k1] * B[sj + k1];
                        }
                    }
                }
            }
        }
    }

    return C;
}

void matrix_multiply_tiled(
    const std::vector<float>& A,
    const std::vector<float>& B,
    std::vector<float>& C,
    uint32_t n,  // rows in A (and rows in C)
    uint32_t m,  // columns in B (and columns in C)
    uint32_t k   // columns in A and rows in B
) {
    C = matrix_multiply_tiled(A, B, n, m, k);
    return;
}
*/