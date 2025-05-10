#include "kernel.h"
#include <cassert>
#include <fstream>

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

// saveDataToFile(time_neon, time_classic, result_matrix_size);

int main() try{
    std::vector<float> A;           //[n*k];
    std::vector<float> B;           //[k*m];
    std::vector<float> C_neon;      //[n*m];
    std::vector<float> C_classic;   //[n*m];

    std::vector<uint32_t> n; // rows in A
    std::vector<uint32_t> m; // cols in B
    std::vector<uint32_t> k; // cols in A and rows in B

    std::vector<uint64_t> time_neon;    // microseconds
    std::vector<uint64_t> time_classic; // microseconds
    std::vector<uint64_t> result_matrix_size;

    /*
    uint32_t counter = 0;
    for(uint32_t i = 1; i < 1000; i = i*2){
        //std::cout << i << " " << i*2 << " " << i*3 << "\n";
        n.push_back(i);
        m.push_back(i*2);
        k.push_back(i*3);
        ++counter;
    }
    std::cout << counter << "\n\n";
    */

    uint32_t counter = 14;
    n = {50, 100, 200, 300, 400, 600, 700, 1000, 1300, 1600, 2000, 2500, 3000, 5000};
    m = {51, 101, 201, 303, 405, 603, 701, 1005, 1301, 1603, 2001, 2501, 3003, 5001};
    k = {10, 20, 30, 50, 70, 100, 130, 160, 200, 240, 300, 400, 500, 1000};

    for(uint32_t i = 0; i < counter; ++i){
        matrix_init_rand(A, n[i]*k[i]);
        matrix_init_rand(B, k[i]*m[i]);

        auto start = std::chrono::high_resolution_clock::now();
        C_neon = matrix_multiply(A, B, n[i], m[i], k[i]);
        auto end = std::chrono::high_resolution_clock::now();

        time_neon.push_back(std::chrono::duration_cast<std::chrono::microseconds>(end - start).count());

        start = std::chrono::high_resolution_clock::now(); 
        C_classic = naive_matrix_multiply(A, B, n[i], m[i], k[i]);
        end = std::chrono::high_resolution_clock::now(); 

        time_classic.push_back(std::chrono::duration_cast<std::chrono::microseconds>(end - start).count());

        result_matrix_size.push_back(n[i]*m[i]);

        if(!verify_matrix_multiply(C_neon, C_classic)) std::abort();

        C_neon.clear();
        C_classic.clear();

        std::cout << "Level " << i << " passed\n";
        std::cout << "Time neon " << time_neon[i] << "\nTime classic " << time_classic[i] << "\n";
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