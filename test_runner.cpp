#include "compression_lib.h"
#include "utility.h"
#include <iostream>
#include <vector>
#include <string>
#include <stdexcept>
#include <iomanip>

int main() {
    std::cout << "--- Running Compression Tests ---" << std::endl;
    std::string input_file = "../data/StSulpice-Cloud-50mm.e57";

    PC original_pc_full = PC::from_e57(input_file);

    std::vector<int> l_max_values = {2, 4, 8};
    int n = 16;
    size_t batches_to_process = 50;
    size_t points_to_process = batches_to_process * n;
    bool all_tests_passed = true;

    PC_T<float> original_subset;
    original_subset.x_coords.assign(original_pc_full.x_coords.begin(), original_pc_full.x_coords.begin() + points_to_process);
    original_subset.y_coords.assign(original_pc_full.y_coords.begin(), original_pc_full.y_coords.begin() + points_to_process);
    original_subset.z_coords.assign(original_pc_full.z_coords.begin(), original_pc_full.z_coords.begin() + points_to_process);

    std::cout << std::fixed << std::setprecision(8);
    std::cout << "------------------------------------------------------------------" << std::endl;
    std::cout << "Test Parameters:" << std::endl;
    std::cout << "  Input File: " << input_file << std::endl;
    std::cout << "  Batch Size (n): " << n << std::endl;
    std::cout << "  Batches to Process: " << batches_to_process << " (" << points_to_process << " points)" << std::endl;
    std::cout << "------------------------------------------------------------------" << std::endl;
    std::cout << "| l_max | Chamfer Distance | Hausdorff Distance | Status |" << std::endl;
    std::cout << "|-------|------------------|--------------------|--------|" << std::endl;


    for (int l_max : l_max_values) {
        std::string output_file = "reconstructed_lmax_" + std::to_string(l_max) + ".e57";
        double chamfer = -1.0, hausdorff = -1.0;
        bool test_passed = true;

        try {
            process_point_cloud(input_file, output_file, l_max, n, batches_to_process);

            PC reconstructed_pc_full = PC::from_e57(output_file);
            PC_T<float> reconstructed_subset;
            reconstructed_subset.x_coords.assign(reconstructed_pc_full.x_coords.begin(), reconstructed_pc_full.x_coords.begin() + points_to_process);
            reconstructed_subset.y_coords.assign(reconstructed_pc_full.y_coords.begin(), reconstructed_pc_full.y_coords.begin() + points_to_process);
            reconstructed_subset.z_coords.assign(reconstructed_pc_full.z_coords.begin(), reconstructed_pc_full.z_coords.begin() + points_to_process);

            calculate_distances(original_subset, reconstructed_subset, chamfer, hausdorff);

        } catch (const std::exception& e) {
            std::cerr << "[FAIL] Test for l_max = " << l_max << " with error: " << e.what() << std::endl;
            test_passed = false;
            all_tests_passed = false;
        }

        std::cout << "| " << std::setw(5) << l_max << " | "
                  << std::setw(16) << chamfer << " | "
                  << std::setw(18) << hausdorff << " | "
                  << std::setw(6) << (test_passed ? "PASS" : "FAIL") << " |" << std::endl;
    }
    std::cout << "------------------------------------------------------------------" << std::endl;

    if (all_tests_passed) {
        std::cout << "\n--- All tests completed successfully ---" << std::endl;
        return 0;
    } else {
        std::cout << "\n--- Some tests FAILED ---" << std::endl;
        return 1;
    }
}