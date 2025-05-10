#include "matrix_multiply_neon.h"
#include "func.h"
//#include "naive_matrix_multiply.h"
#include <cassert>


int main() try{
    std::vector<float> A;           //[n*k];
    std::vector<float> B;           //[k*m];
    std::vector<float> C_neon;      //[n*m];
    std::vector<float> C_classic;   //[n*m];
    std::vector<float> C_tiled;     //[n*m];

    std::vector<uint32_t> n; // rows in A
    std::vector<uint32_t> m; // cols in B
    std::vector<uint32_t> k; // cols in A and rows in B

    std::vector<uint64_t> time_neon;            // microseconds
    std::vector<uint64_t> time_classic;         // microseconds
    //std::vector<uint64_t> time_tiled;         // microseconds
    std::vector<uint64_t> result_matrix_size;   // uint64_t

    uint32_t counter = 14;
    n = {50, 100, 200, 300, 400, 600, 700, 1000, 1300, 1600, 2000, 2500, 3000, 5000};
    m = {51, 101, 201, 303, 405, 603, 701, 1005, 1301, 1603, 2001, 2501, 3003, 5001};
    k = {10, 20, 30, 50, 70, 100, 130, 160, 200, 240, 300, 400, 500, 1000};

    for(uint32_t i = 0; i < counter; ++i){
        matrix_init_rand(A, n[i]*k[i]);
        matrix_init_rand(B, k[i]*m[i]);

        // Fast neon mult
        auto start = std::chrono::high_resolution_clock::now();
        C_neon = matrix_multiply_fast_neon(A, B, n[i], m[i], k[i]);
        auto end = std::chrono::high_resolution_clock::now();

        time_neon.push_back(std::chrono::duration_cast<std::chrono::microseconds>(end - start).count());

        // Classic mult
        start = std::chrono::high_resolution_clock::now(); 
        C_classic = naive_matrix_multiply(A, B, n[i], m[i], k[i]);
        end = std::chrono::high_resolution_clock::now(); 

        time_classic.push_back(std::chrono::duration_cast<std::chrono::microseconds>(end - start).count());
        
        // Tiled mul
        //start = std::chrono::high_resolution_clock::now(); 
        //C_tiled = matrix_multiply_tiled(A, B, n[i], m[i], k[i]);
        //end = std::chrono::high_resolution_clock::now(); 

        //time_tiled.push_back(std::chrono::duration_cast<std::chrono::microseconds>(end - start).count());

        result_matrix_size.push_back(n[i]*m[i]); // save matrix size

        // Verify results
        if(!verify_matrix_multiply(C_neon, C_classic)) throw std::runtime_error("C_neon mismatch C_classic");
        if(!verify_matrix_multiply(C_tiled, C_classic)) throw std::runtime_error("C_tiled mismatch C_classic");
        if(!verify_matrix_multiply(C_tiled, C_neon)) throw std::runtime_error("C_tiled mismatch C_neon");

        C_neon.clear();
        C_classic.clear();
        C_tiled.clear();

        std::cout << "Level " << i << " passed\n";
        std::cout << "Time neon " << time_neon[i] << "\nTime classic " << time_classic[i] << /*"\nTime tiled " << time_tiled[i] <<*/ "\n";
        std::cout << "Size " << result_matrix_size[i] << "\n\n";
    }

    printf("Tests passed\n");

    saveDataToFile(time_neon, time_classic, result_matrix_size);
}
catch (const std::exception& e){
    std::cerr << "Error: " << e.what() << "\n";
}
catch (...){
    std::cerr << "Unknown exception\n";
}