#include "compression_lib.h"
#include "utility.h"
#include <iostream>
#include <chrono>
#include <omp.h>

using T = float;
using PC_comp = PC_T<T>;
using Coeffs = Coeffs_T<T>;
using column_vector_d = dlib::matrix<double, 0, 1>;

class LossThetaPhi {
public:
    LossThetaPhi(const Coeffs& true_coeffs, const std::vector<T>& kappa, int l_max)
        : true_coeffs_(true_coeffs), kappa_values_(kappa), l_max_(l_max) {}

    double operator()(const column_vector_d& params) const {
        const size_t n = kappa_values_.size();
        std::vector<T> theta(n), phi(n);
        for (size_t i = 0; i < n; ++i) {
            theta[i] = static_cast<T>(params(i));
            phi[i] = static_cast<T>(params(i + n));
        }
        Coeffs predicted_coeffs = compress<T>(kappa_values_, theta, phi, l_max_);
        T loss = 0.0f;
        for (size_t i = 0; i < n; ++i) {
            for (int l = 0; l <= l_max_; ++l) {
                for (size_t m_idx = 0; m_idx < predicted_coeffs[i][l].size(); ++m_idx) {
                    loss += std::norm(predicted_coeffs[i][l][m_idx] - true_coeffs_[i][l][m_idx]);
                }
            }
        }
        return static_cast<double>(loss);
    }
private:
    const Coeffs& true_coeffs_;
    const std::vector<T>& kappa_values_;
    int l_max_;
};

class LossKappa {
public:
    LossKappa(const Coeffs& true_coeffs, const std::vector<T>& theta, const std::vector<T>& phi, int l_max)
        : true_coeffs_(true_coeffs), theta_values_(theta), phi_values_(phi), l_max_(l_max) {}

    double operator()(const column_vector_d& params) const {
        const size_t n = theta_values_.size();
        std::vector<T> kappa(n);
        for (size_t i = 0; i < n; ++i) {
            kappa[i] = static_cast<T>(params(i));
        }
        Coeffs predicted_coeffs = compress<T>(kappa, theta_values_, phi_values_, l_max_);
        T loss = 0.0f;
        for (size_t i = 0; i < n; ++i) {
            for (int l = 0; l <= l_max_; ++l) {
                for (size_t m_idx = 0; m_idx < predicted_coeffs[i][l].size(); ++m_idx) {
                    loss += std::norm(predicted_coeffs[i][l][m_idx] - true_coeffs_[i][l][m_idx]);
                }
            }
        }
        return static_cast<double>(loss);
    }
private:
    const Coeffs& true_coeffs_;
    const std::vector<T>& theta_values_;
    const std::vector<T>& phi_values_;
    int l_max_;
};

void process_point_cloud(
    const std::string& input_path,
    const std::string& output_path,
    int l_max,
    int n,
    size_t total_batches_to_process
) {
    constexpr double rho_begin_theta_phi = 1.0;
    constexpr double rho_end_theta_phi = 1e-6;
    constexpr long max_evals_theta_phi = 2000;

    constexpr double rho_begin_kappa = 0.2;
    constexpr double rho_end_kappa = 1e-6;
    constexpr long max_evals_kappa = 2000;

    PC point_cloud_double = PC::from_e57(input_path);
    PC_comp point_cloud;
    point_cloud.red = point_cloud_double.red;
    point_cloud.green = point_cloud_double.green;
    point_cloud.blue = point_cloud_double.blue;
    point_cloud.x_coords.assign(point_cloud_double.x_coords.begin(), point_cloud_double.x_coords.end());
    point_cloud.y_coords.assign(point_cloud_double.y_coords.begin(), point_cloud_double.y_coords.end());
    point_cloud.z_coords.assign(point_cloud_double.z_coords.begin(), point_cloud_double.z_coords.end());

    T radius_max = 0.0f;
    for (size_t i = 0; i < point_cloud.x_coords.size(); ++i) {
        T r = std::sqrt(point_cloud.x_coords[i]*point_cloud.x_coords[i] + point_cloud.y_coords[i]*point_cloud.y_coords[i] + point_cloud.z_coords[i]*point_cloud.z_coords[i]);
        if (r > radius_max) radius_max = r;
    }

    PC_comp reconstructed_full;
    reconstructed_full.red = point_cloud.red;
    reconstructed_full.green = point_cloud.green;
    reconstructed_full.blue = point_cloud.blue;
    reconstructed_full.x_coords.resize(point_cloud.x_coords.size());
    reconstructed_full.y_coords.resize(point_cloud.y_coords.size());
    reconstructed_full.z_coords.resize(point_cloud.z_coords.size());

    const size_t max_batches = std::min(total_batches_to_process, point_cloud.x_coords.size() / n);

    column_vector_d lower_b(2 * n);
    column_vector_d upper_b(2 * n);
    for (long j = 0; j < 2 * n; ++j) {
        lower_b(j) = -1e100;
        upper_b(j) = 1e100;
    }

    column_vector_d lower_b_k(n);
    column_vector_d upper_b_k(n);
    for(int k=0; k<n; ++k) {
        lower_b_k(k) = 0.0;
        upper_b_k(k) = 1.0;
    }

    #pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < max_batches; ++i) {
        PC_comp batch;
        batch.x_coords.resize(n);
        batch.y_coords.resize(n);
        batch.z_coords.resize(n);
        batch.red.resize(n);
        batch.green.resize(n);
        batch.blue.resize(n);

        size_t start_idx = i * n;
        for(size_t j = 0; j < n; ++j) {
            size_t src_idx = start_idx + j;
            batch.x_coords[j] = point_cloud.x_coords[src_idx] / radius_max;
            batch.y_coords[j] = point_cloud.y_coords[src_idx] / radius_max;
            batch.z_coords[j] = point_cloud.z_coords[src_idx] / radius_max;
            batch.red[j] = point_cloud.red[src_idx];
            batch.green[j] = point_cloud.green[src_idx];
            batch.blue[j] = point_cloud.blue[src_idx];
        }

        batch.to_spherical();
        Coeffs true_coefficients = compress<T>(batch.x_coords, batch.y_coords, batch.z_coords, l_max);

        column_vector_d kappa_params_d(n);
        for(int k=0; k<n; ++k) kappa_params_d(k) = 0.5;

        column_vector_d theta_phi_params_d(2 * n);
        for(int k=0; k<n; ++k) {
            theta_phi_params_d(k) = 2.0 * M_PI * k / (n-1);
            theta_phi_params_d(k+n) = M_PI * k / (n-1);
        }

        for (int iter = 0; iter < 1; ++iter) {
            std::vector<T> current_kappa_f(n);
            for(int k=0; k<n; ++k) current_kappa_f[k] = static_cast<T>(kappa_params_d(k));
            dlib::find_min_bobyqa(LossThetaPhi(true_coefficients, current_kappa_f, l_max), theta_phi_params_d, 2 * (2*n) + 1, lower_b, upper_b, rho_begin_theta_phi, rho_end_theta_phi, max_evals_theta_phi);

            std::vector<T> current_theta_f(n), current_phi_f(n);
            for(int k=0; k<n; ++k) {
                current_theta_f[k] = static_cast<T>(theta_phi_params_d(k));
                current_phi_f[k] = static_cast<T>(theta_phi_params_d(k+n));
            }
            dlib::find_min_bobyqa(LossKappa(true_coefficients, current_theta_f, current_phi_f, l_max), kappa_params_d, 2 * n + 1, lower_b_k, upper_b_k, rho_begin_kappa, rho_end_kappa, max_evals_kappa);
        }

        PC_comp reconstructed_batch;
        reconstructed_batch.red = batch.red;
        reconstructed_batch.green = batch.green;
        reconstructed_batch.blue = batch.blue;
        reconstructed_batch.x_coords.resize(n);
        reconstructed_batch.y_coords.resize(n);
        reconstructed_batch.z_coords.resize(n);
        for(int k=0; k<n; ++k) {
            reconstructed_batch.x_coords[k] = static_cast<T>(kappa_params_d(k));
            reconstructed_batch.y_coords[k] = static_cast<T>(theta_phi_params_d(k));
            reconstructed_batch.z_coords[k] = static_cast<T>(theta_phi_params_d(k+n));
        }
        reconstructed_batch.to_cartesian();

        #pragma omp critical
        {
            std::cout << "--- Finished Optimization for Batch " << i << " (Thread " << omp_get_thread_num() << ") ---" << std::endl;
            for(size_t j=0; j<n; ++j){
                size_t dest_idx = start_idx + j;
                reconstructed_full.x_coords[dest_idx] = reconstructed_batch.x_coords[j] * radius_max;
                reconstructed_full.y_coords[dest_idx] = reconstructed_batch.y_coords[j] * radius_max;
                reconstructed_full.z_coords[dest_idx] = reconstructed_batch.z_coords[j] * radius_max;
            }
        }
    }

    PC reconstructed_full_double;
    reconstructed_full_double.red = reconstructed_full.red;
    reconstructed_full_double.green = reconstructed_full.green;
    reconstructed_full_double.blue = reconstructed_full.blue;
    reconstructed_full_double.x_coords.assign(reconstructed_full.x_coords.begin(), reconstructed_full.x_coords.end());
    reconstructed_full_double.y_coords.assign(reconstructed_full.y_coords.begin(), reconstructed_full.y_coords.end());
    reconstructed_full_double.z_coords.assign(reconstructed_full.z_coords.begin(), reconstructed_full.z_coords.end());

    reconstructed_full_double.to_e57(output_path);
    std::cout << "Full reconstruction saved to " << output_path << std::endl;
}