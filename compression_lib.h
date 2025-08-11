#pragma once
#include <string>

void process_point_cloud(
    const std::string& input_path,
    const std::string& output_path,
    int l_max,
    int n,
    size_t total_batches_to_process
);