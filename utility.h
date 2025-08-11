#pragma once

#include <vector>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <iomanip>
#include <numeric>
#include <complex>
#include <algorithm>

#include <E57SimpleReader.h>
#include "E57SimpleWriter.h"

#include <dlib/optimization.h>
#include <dlib/matrix.h>
#include "nanoflann.hpp"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

template<typename T>
using column_vector_t = dlib::matrix<T, 0, 1>;

template <typename T>
struct PC_T {
    std::vector<T> x_coords;
    std::vector<T> y_coords;
    std::vector<T> z_coords;
    std::vector<uint16_t> red;
    std::vector<uint16_t> green;
    std::vector<uint16_t> blue;

    void to_spherical() {
        if (x_coords.size() != y_coords.size() || x_coords.size() != z_coords.size()) {
            throw std::runtime_error("Coordinate vectors must have the same size.");
        }
        size_t num_points = x_coords.size();
        for (size_t i = 0; i < num_points; ++i) {
            T x = x_coords[i];
            T y = y_coords[i];
            T z = z_coords[i];

            T r = std::sqrt(x * x + y * y + z * z);
            T theta = 0.0;
            T phi = 0.0;

            if (r >= 1e-9) {
                theta = std::atan2(y, x);
                phi = std::acos(z / r);
            }
            x_coords[i] = r;
            y_coords[i] = theta;
            z_coords[i] = phi;
        }
    }

    void to_cartesian() {
        if (x_coords.size() != y_coords.size() || x_coords.size() != z_coords.size()) {
            throw std::runtime_error("Coordinate vectors must have the same size.");
        }
        size_t num_points = x_coords.size();
        for (size_t i = 0; i < num_points; ++i) {
            T r = x_coords[i];
            T theta = y_coords[i];
            T phi = z_coords[i];

            T x = r * std::cos(theta) * std::sin(phi);
            T y = r * std::sin(theta) * std::sin(phi);
            T z = r * std::cos(phi);

            x_coords[i] = x;
            y_coords[i] = y;
            z_coords[i] = z;
        }
    }
};

struct PC {
    std::vector<double> x_coords;
    std::vector<double> y_coords;
    std::vector<double> z_coords;
    std::vector<uint16_t> red;
    std::vector<uint16_t> green;
    std::vector<uint16_t> blue;

    static PC from_e57(const std::string& path) {
        e57::Reader reader(path);
        e57::Data3D scanHeader;
        reader.ReadData3D(0, scanHeader);
        const size_t totalPointCount = scanHeader.pointCount;
        PC result;
        result.x_coords.resize(totalPointCount);
        result.y_coords.resize(totalPointCount);
        result.z_coords.resize(totalPointCount);

        e57::Data3DPointsDouble buffers;
        buffers.cartesianX = result.x_coords.data();
        buffers.cartesianY = result.y_coords.data();
        buffers.cartesianZ = result.z_coords.data();
        buffers.colorRed = nullptr;
        buffers.colorGreen = nullptr;
        buffers.colorBlue = nullptr;

        if (scanHeader.pointFields.colorRedField) {
            result.red.resize(totalPointCount);
            result.green.resize(totalPointCount);
            result.blue.resize(totalPointCount);
            buffers.colorRed = result.red.data();
            buffers.colorGreen = result.green.data();
            buffers.colorBlue = result.blue.data();
        } else {
            const uint16_t white = 65535;
            result.red.assign(totalPointCount, white);
            result.green.assign(totalPointCount, white);
            result.blue.assign(totalPointCount, white);
        }

        e57::CompressedVectorReader dataReader = reader.SetUpData3DPointsData(0, totalPointCount, buffers);
        dataReader.read();
        dataReader.close();
        reader.Close();
        return result;
    }

    void to_e57(const std::string& path) {
        e57::Writer writer(path);
        e57::Data3D header;
        header.pointCount = x_coords.size();
        header.pointFields.cartesianXField = true;
        header.pointFields.cartesianYField = true;
        header.pointFields.cartesianZField = true;
        header.pointFields.colorRedField = true;
        header.pointFields.colorGreenField = true;
        header.pointFields.colorBlueField = true;
        header.colorLimits.colorRedMinimum = 0;
        header.colorLimits.colorRedMaximum = 65535;
        header.colorLimits.colorGreenMinimum = 0;
        header.colorLimits.colorGreenMaximum = 65535;
        header.colorLimits.colorBlueMinimum = 0;
        header.colorLimits.colorBlueMaximum = 65535;
        int scanIndex = writer.NewData3D(header);

        e57::Data3DPointsDouble buffers;
        buffers.cartesianX = x_coords.data();
        buffers.cartesianY = y_coords.data();
        buffers.cartesianZ = z_coords.data();
        buffers.colorRed = red.data();
        buffers.colorGreen = green.data();
        buffers.colorBlue = blue.data();
        e57::CompressedVectorWriter dataWriter = writer.SetUpData3DPointsData(scanIndex, x_coords.size(), buffers);
        dataWriter.write(x_coords.size());
        dataWriter.close();
        writer.Close();
    }
};


template <typename T>
T d_matrix(int l, int mp, int m, T beta) {
    if (l > 2) throw std::invalid_argument("l > 2 is not supported.");
    if (std::abs(mp) > l || std::abs(m) > l) return 0.0;
    const T c = std::cos(beta);
    const T s = std::sin(beta);
    if (l == 0) return 1.0;
    if (l == 1) {
        if (mp == 1 && m == 1) return (1 + c) / 2.0;
        if (mp == 1 && m == 0) return -s / std::sqrt(2.0);
        if (mp == 1 && m == -1) return (1 - c) / 2.0;
        if (mp == 0 && m == 1) return s / std::sqrt(2.0);
        if (mp == 0 && m == 0) return c;
        if (mp == 0 && m == -1) return -s / std::sqrt(2.0);
        if (mp == -1 && m == 1) return (1 - c) / 2.0;
        if (mp == -1 && m == 0) return s / std::sqrt(2.0);
        if (mp == -1 && m == -1) return (1 + c) / 2.0;
    }
    if (l == 2) {
        if (mp == 2 && m == 2) return std::pow((1 + c) / 2.0, 2);
        if (mp == 2 && m == 1) return -(1 + c) * s / 2.0;
        if (mp == 2 && m == 0) return std::sqrt(3.0 / 8.0) * s * s;
        if (mp == 1 && m == 1) return (2 * c * c + c - 1) / 2.0;
        if (mp == 1 && m == 0) return -std::sqrt(3.0 / 2.0) * s * c;
        if (mp == 0 && m == 0) return (3 * c * c - 1) / 2.0;
        if (m > mp) return d_matrix(l, m, mp, beta);
        T sign = ((mp - m) % 2 == 0) ? 1.0 : -1.0;
        return sign * d_matrix(l, -m, -mp, beta);
    }
    return 0.0;
}

template <typename T>
T B(int l) {
    return std::sqrt((T(2.0) * l + T(1.0)) / (T(4.0) * M_PI));
}

template <typename T>
T g_bessel(int l, T kappa) {
    if (kappa < 1e-9) return 0.0;
    return B<T>(l) * std::cyl_bessel_i(l + 0.5, kappa) / std::cyl_bessel_i(0.5, kappa);
}

template <typename T>
using Coeffs_T = std::vector<std::vector<std::vector<std::complex<T>>>>;

template <typename T>
Coeffs_T<T> compress(const std::vector<T>& kappa, const std::vector<T>& theta, const std::vector<T>& phi, int l_max) {
    size_t n = kappa.size();
    Coeffs_T<T> result(n, std::vector<std::vector<std::complex<T>>>(l_max + 1));
    for(size_t i = 0; i < n; ++i) {
        for(int l = 0; l <= l_max; ++l) {
            result[i][l].resize(2 * l + 1);
            T f_tilda = g_bessel<T>(l, kappa[i]);
            for(int m = -l; m <=l; ++m) {
                 std::complex<T> D_val = d_matrix<T>(l, 0, m, phi[i]) * std::exp(std::complex<T>(0, -m * theta[i]));
                 result[i][l][m+l] = f_tilda * D_val;
            }
        }
    }
    return result;
}

template <typename T>
struct PointCloudAdapter {
    const PC_T<T>& obj;

    PointCloudAdapter(const PC_T<T>& obj_) : obj(obj_) {}

    inline size_t kdtree_get_point_count() const { return obj.x_coords.size(); }

    inline T kdtree_get_pt(const size_t idx, const size_t dim) const {
        if (dim == 0) return obj.x_coords[idx];
        if (dim == 1) return obj.y_coords[idx];
        if (dim == 2) return obj.z_coords[idx];
        return 0;
    }

    template <class BBOX>
    bool kdtree_get_bbox(BBOX& /* bb */) const { return false; }
};

template <typename T>
void calculate_distances(const PC_T<T>& pc1, const PC_T<T>& pc2, double& chamfer_dist, double& hausdorff_dist) {
    PointCloudAdapter<T> pc1_adapter(pc1);
    PointCloudAdapter<T> pc2_adapter(pc2);

    using my_kd_tree_t = nanoflann::KDTreeSingleIndexAdaptor<
        nanoflann::L2_Simple_Adaptor<T, PointCloudAdapter<T>>,
        PointCloudAdapter<T>,
        3
    >;

    my_kd_tree_t index1(3, pc1_adapter, {10});
    my_kd_tree_t index2(3, pc2_adapter, {10});

    double sum_dist1 = 0.0;
    double max_dist1 = 0.0;
    #pragma omp parallel for reduction(+:sum_dist1) reduction(max:max_dist1)
    for (size_t i = 0; i < pc2.x_coords.size(); ++i) {
        T query_pt[3] = {pc2.x_coords[i], pc2.y_coords[i], pc2.z_coords[i]};
        size_t ret_index;
        T out_dist_sqr;
        nanoflann::KNNResultSet<T> resultSet(1);
        resultSet.init(&ret_index, &out_dist_sqr);
        index1.findNeighbors(resultSet, &query_pt[0]);
        sum_dist1 += out_dist_sqr;
        if (out_dist_sqr > max_dist1) {
            max_dist1 = out_dist_sqr;
        }
    }

    double sum_dist2 = 0.0;
    double max_dist2 = 0.0;
    #pragma omp parallel for reduction(+:sum_dist2) reduction(max:max_dist2)
    for (size_t i = 0; i < pc1.x_coords.size(); ++i) {
        T query_pt[3] = {pc1.x_coords[i], pc1.y_coords[i], pc1.z_coords[i]};
        size_t ret_index;
        T out_dist_sqr;
        nanoflann::KNNResultSet<T> resultSet(1);
        resultSet.init(&ret_index, &out_dist_sqr);
        index2.findNeighbors(resultSet, &query_pt[0]);
        sum_dist2 += out_dist_sqr;
        if (out_dist_sqr > max_dist2) {
            max_dist2 = out_dist_sqr;
        }
    }

    chamfer_dist = (sum_dist1 / pc2.x_coords.size()) + (sum_dist2 / pc1.x_coords.size());
    hausdorff_dist = std::sqrt(std::max(max_dist1, max_dist2));
}