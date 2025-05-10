#include <fstream>
#include <iostream>
#include <vector>

#pragma once

bool verify_matrix_multiply(
    const std::vector<float>& C,
    const std::vector<float>& C_expected,
    float epsilon = 1e-2
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


void saveDataToFile(const std::vector<uint64_t>& time_neon,
                    const std::vector<uint64_t>& time_classic,
                    const std::vector<uint64_t>& result_matrix_size,
                    const std::string& filename = "benchmark_data.txt") {
    std::ofstream out_file(filename);
    if (!out_file.is_open()) {
        std::cerr << "Error opening file for writing!" << std::endl;
        return;
    }
    
    // Записываем заголовок (опционально)
    //out_file << "MatrixSize NeonTime ClassicTime\n";
    
    // Проверяем, что все векторы одинакового размера
    size_t n = time_neon.size();
    if (time_classic.size() != n || result_matrix_size.size() != n) {
        std::cerr << "Error: Vector sizes don't match!" << std::endl;
        return;
    }
    
    // Записываем данные построчно
    for (size_t i = 0; i < n; ++i) {
        out_file << result_matrix_size[i] << " " 
                 << time_neon[i] << " " 
                 << time_classic[i] << "\n";
    }
    
    out_file.close();
    std::cout << "Data saved to " << filename << std::endl;
}



/*
void matrix_multiply_tiled(
    std::vector<float>& A, 
    std::vector<float>& B, 
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

    uint32_t num_of_created_matrices_n = n / BLOCK_SIZE;
    uint32_t num_of_created_matrices_m = m / BLOCK_SIZE;
    uint32_t num_of_created_matrices_k = k / BLOCK_SIZE;

    // if matrices are too small no need for tiling
    if(num_of_created_matrices_n == 0 && num_of_created_matrices_m == 0 && num_of_created_matrices_k == 0){
        matrix_multiply_fast_neon(A, B, C, n, m, k);
        return;
    }

    // now create cutted out matrices
    std::vector<float> A_tile(BLOCK_SIZE * BLOCK_SIZE, 0.0f);
    std::vector<float> B_tile(BLOCK_SIZE * BLOCK_SIZE, 0.0f);
    std::vector<float> C_tile(BLOCK_SIZE * BLOCK_SIZE, 0.0f);

    for (uint32_t i = 0; i < n; i += BLOCK_SIZE) {
        uint32_t actual_tile_height = std::min(BLOCK_SIZE, n - i);
        
        for (uint32_t j = 0; j < m; j += BLOCK_SIZE) {
            uint32_t actual_tile_width = std::min(BLOCK_SIZE, m - j);
            
            // Reset C tile
            std::fill(C_tile.begin(), C_tile.end(), 0.0f);
            
            for (uint32_t t = 0; t < k; t += BLOCK_SIZE) {
                uint32_t actual_tile_depth = std::min(BLOCK_SIZE, k - t);
                
                // Extract tile from A
                for (uint32_t ki = 0; ki < actual_tile_depth; ++ki) {
                    for (uint32_t ii = 0; ii < actual_tile_height; ++ii) {
                        A_tile[ki * BLOCK_SIZE + ii] = A[(t + ki) * n + (i + ii)];
                    }
                }
                
                // Extract tile from B
                for (uint32_t ji = 0; ji < actual_tile_width; ++ji) {
                    for (uint32_t ki = 0; ki < actual_tile_depth; ++ki) {
                        B_tile[ji * BLOCK_SIZE + ki] = B[(j + ji) * k + (t + ki)];
                    }
                }
                
                // Compute partial product for this tile
                std::vector<float> partial_C_tile(BLOCK_SIZE * BLOCK_SIZE, 0.0f);
                
                // Round up dimensions to multiples of 4 for NEON processing
                const uint32_t h_padded = ((actual_tile_height + 3) / 4) * 4;
                const uint32_t w_padded = ((actual_tile_width + 3) / 4) * 4;
                const uint32_t d_padded = ((actual_tile_depth + 3) / 4) * 4;
                
                // Padded matrices if necessary
                std::vector<float> A_padded(h_padded * d_padded, 0.0f);
                std::vector<float> B_padded(d_padded * w_padded, 0.0f);
                std::vector<float> C_padded(h_padded * w_padded, 0.0f);
                
                // Fill padded A
                for (uint32_t col = 0; col < actual_tile_depth; ++col) {
                    for (uint32_t row = 0; row < actual_tile_height; ++row) {
                        A_padded[col * h_padded + row] = A_tile[col * BLOCK_SIZE + row];
                    }
                }
                
                // Fill padded B
                for (uint32_t col = 0; col < actual_tile_width; ++col) {
                    for (uint32_t row = 0; row < actual_tile_depth; ++row) {
                        B_padded[col * d_padded + row] = B_tile[col * BLOCK_SIZE + row];
                    }
                }
                
                // Use NEON optimized function
                matrix_multiply_neon(A_padded.data(), B_padded.data(), C_padded.data(), h_padded, w_padded, d_padded);
                
                // Accumulate results to C_tile
                for (uint32_t ji = 0; ji < actual_tile_width; ++ji) {
                    for (uint32_t ii = 0; ii < actual_tile_height; ++ii) {
                        C_tile[ji * BLOCK_SIZE + ii] += C_padded[ji * h_padded + ii];
                    }
                }
            }
            
            // Copy results from C_tile to the final C matrix
            for (uint32_t ji = 0; ji < actual_tile_width; ++ji) {
                for (uint32_t ii = 0; ii < actual_tile_height; ++ii) {
                    C[(j + ji) * n + (i + ii)] = C_tile[ji * BLOCK_SIZE + ii];
                }
            }
        }
    }
    return;
}
*/