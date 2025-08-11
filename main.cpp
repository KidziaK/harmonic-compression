#include "compression_lib.h"
#include <iostream>
#include <string>
#include <chrono>

int main(int argc, char* argv[]) {
    if (argc != 6) {
        std::cerr << "Usage: " << argv[0] << " <input_path> <output_path> <l_max> <n> <num_batches>" << std::endl;
        return 1;
    }

    std::string input_path = argv[1];
    std::string output_path = argv[2];
    int l_max = std::stoi(argv[3]);
    int n = std::stoi(argv[4]);
    size_t num_batches = std::stoul(argv[5]);

    auto start_time = std::chrono::high_resolution_clock::now();

    try {
        process_point_cloud(input_path, output_path, l_max, n, num_batches);
    } catch (const std::exception& e) {
        std::cerr << "An error occurred: " << e.what() << std::endl;
        return 1;
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cout << "\nTotal execution time: " << duration.count() << " ms" << std::endl;

    return 0;
}