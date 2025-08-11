#include "compression_lib.h"
#include "utility.h"
#include <iostream>
#include <vector>
#include <string>
#include <stdexcept>
#include <iomanip>
#include <filesystem>
#include <chrono>

namespace fs = std::filesystem;

int main() {
    std::cout << "--- Running Compression Tests ---" << std::endl;
    std::string data_directory = "../data";
    std::vector<std::string> test_files;

    try {
        for (const auto& entry : fs::directory_iterator(data_directory)) {
            if (entry.is_regular_file() && entry.path().extension() == ".e57") {
                test_files.push_back(entry.path().string());
            }
        }
    } catch (const fs::filesystem_error& e) {
        std::cerr << "Error accessing data directory: " << e.what() << std::endl;
        return 1;
    }

    if (test_files.empty()) {
        std::cerr << "No .e57 files found in " << data_directory << std::endl;
        return 1;
    }

    const int l_max = 2;
    const int n = 16;
    const size_t batches_to_process = 500;
    const size_t points_to_process = batches_to_process * n;
    bool all_tests_passed = true;

    std::cout << std::fixed << std::setprecision(6);

    for (const auto& input_file : test_files) {
        std::cout << "\n============================================================================================" << std::endl;
        std::cout << "TESTING FILE: " << fs::path(input_file).filename().string() << std::endl;
        std::cout << "--------------------------------------------------------------------------------------------" << std::endl;
        std::cout << "| l_max | Chamfer Distance | Hausdorff Distance | Time (s) | Points/sec | Status |" << std::endl;
        std::cout << "|-------|------------------|--------------------|----------|------------|--------|" << std::endl;

        PC original_pc_full = PC::from_e57(input_file);
        if (original_pc_full.x_coords.size() < points_to_process) {
            std::cout << "| " << std::setw(5) << l_max << " | "
                      << std::setw(16) << "N/A" << " | "
                      << std::setw(18) << "N/A" << " | "
                      << std::setw(8) << "N/A" << " | "
                      << std::setw(10) << "N/A" << " | "
                      << std::setw(6) << "SKIP" << " |" << std::endl;
            std::cerr << "Skipping file: Not enough points for " << batches_to_process << " batches." << std::endl;
            continue;
        }

        PC_T<float> original_subset;
        original_subset.x_coords.assign(original_pc_full.x_coords.begin(), original_pc_full.x_coords.begin() + points_to_process);
        original_subset.y_coords.assign(original_pc_full.y_coords.begin(), original_pc_full.y_coords.begin() + points_to_process);
        original_subset.z_coords.assign(original_pc_full.z_coords.begin(), original_pc_full.z_coords.begin() + points_to_process);

        std::string output_file = "reconstructed_" + fs::path(input_file).stem().string() + ".e57";
        double chamfer = -1.0, hausdorff = -1.0;
        double duration_s = -1.0;
        double pps = 0.0;
        bool test_passed = true;

        try {
            auto start_time = std::chrono::high_resolution_clock::now();

            process_point_cloud(input_file, output_file, l_max, n, batches_to_process);

            auto end_time = std::chrono::high_resolution_clock::now();
            auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
            duration_s = duration_ms.count() / 1000.0;
            if (duration_s > 0) {
                pps = points_to_process / duration_s;
            }

            PC reconstructed_pc_full = PC::from_e57(output_file);
            PC_T<float> reconstructed_subset;
            reconstructed_subset.x_coords.assign(reconstructed_pc_full.x_coords.begin(), reconstructed_pc_full.x_coords.begin() + points_to_process);
            reconstructed_subset.y_coords.assign(reconstructed_pc_full.y_coords.begin(), reconstructed_pc_full.y_coords.begin() + points_to_process);
            reconstructed_subset.z_coords.assign(reconstructed_pc_full.z_coords.begin(), reconstructed_pc_full.z_coords.begin() + points_to_process);

            calculate_distances(original_subset, reconstructed_subset, chamfer, hausdorff);

        } catch (const std::exception& e) {
            std::cerr << "Test for " << fs::path(input_file).filename().string() << " failed with error: " << e.what() << std::endl;
            test_passed = false;
            all_tests_passed = false;
        }

        std::cout << "| " << std::setw(5) << l_max << " | "
                  << std::setw(16) << chamfer << " | "
                  << std::setw(18) << hausdorff << " | "
                  << std::setw(8) << duration_s << " | "
                  << std::setw(10) << static_cast<long>(pps) << " | "
                  << std::setw(6) << (test_passed ? "PASS" : "FAIL") << " |" << std::endl;
        std::cout << "--------------------------------------------------------------------------------------------" << std::endl;
    }

    if (all_tests_passed) {
        std::cout << "\n--- All tests completed successfully ---" << std::endl;
        return 0;
    } else {
        std::cout << "\n--- Some tests FAILED ---" << std::endl;
        return 1;
    }
}