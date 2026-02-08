#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <algorithm>
#include <cmath>

namespace py = pybind11;

namespace {

template <typename T>
py::array_t<T> copy_to_array(const std::vector<T>& data) {
    auto result = py::array_t<T>(data.size());
    auto buf = result.request();
    auto* ptr = static_cast<T*>(buf.ptr);
    std::copy(data.begin(), data.end(), ptr);
    return result;
}

template <typename T>
py::tuple arnoldi_iteration_impl(
    py::array_t<T, py::array::c_style | py::array::forcecast> Q_in,
    py::array_t<T, py::array::c_style | py::array::forcecast> w_in,
    double decay,
    int k
) {
    if (k == 0) {
        auto h = py::array_t<T>(1);
        auto grad = py::array_t<T>(0);
        return py::make_tuple(w_in, h, grad);
    }

    auto q_buf = Q_in.request();
    auto w_buf = w_in.request();

    if (q_buf.ndim != 2) {
        throw std::invalid_argument("Q must be a 2D array");
    }
    if (w_buf.ndim != 1) {
        throw std::invalid_argument("w must be a 1D array");
    }
    const auto rows = static_cast<std::size_t>(q_buf.shape[0]);
    const auto cols = static_cast<std::size_t>(q_buf.shape[1]);
    if (cols < static_cast<std::size_t>(k)) {
        throw std::invalid_argument("k larger than available columns in Q");
    }
    if (rows != static_cast<std::size_t>(w_buf.shape[0])) {
        throw std::invalid_argument("Q rows must match w length");
    }

    const auto* q_ptr = static_cast<const T*>(q_buf.ptr);
    std::vector<T> residual(rows);
    const auto* w_ptr = static_cast<const T*>(w_buf.ptr);
    std::copy(w_ptr, w_ptr + rows, residual.begin());

    std::vector<T> h_column(static_cast<std::size_t>(k) + 1, static_cast<T>(0));
    std::vector<T> gradients(static_cast<std::size_t>(k), static_cast<T>(0));

    for (int i = 0; i < k; ++i) {
        const auto col = static_cast<std::size_t>(i);
        T dot = static_cast<T>(0);
        const auto weight = static_cast<T>(std::pow(decay, k - 1 - i));

        for (std::size_t r = 0; r < rows; ++r) {
            dot += residual[r] * q_ptr[r + rows * col];
        }

        const auto h_mag = weight * static_cast<T>(std::abs(dot));
        T sign = static_cast<T>(0);
        if (dot > static_cast<T>(0)) {
            sign = static_cast<T>(1);
        } else if (dot < static_cast<T>(0)) {
            sign = static_cast<T>(-1);
        }
        const auto coeff = sign * h_mag;
        h_column[static_cast<std::size_t>(i)] = coeff;

        for (std::size_t r = 0; r < rows; ++r) {
            residual[r] -= coeff * q_ptr[r + rows * col];
        }

        const auto exponent = k - 1 - i;
        if (exponent > 0) {
            gradients[static_cast<std::size_t>(i)] =
                static_cast<T>(exponent * std::pow(decay, exponent - 1) * std::abs(dot));
        }
    }

    T squared_norm = static_cast<T>(0);
    for (std::size_t r = 0; r < rows; ++r) {
        squared_norm += residual[r] * residual[r];
    }
    h_column.back() = static_cast<T>(std::sqrt(static_cast<double>(squared_norm)));

    auto residual_arr = copy_to_array(residual);
    auto h_arr = copy_to_array(h_column);
    auto grad_arr = copy_to_array(gradients);
    return py::make_tuple(residual_arr, h_arr, grad_arr);
}

py::tuple dispatch_arnoldi(py::array Q, py::array w, double decay, int k) {
    if (Q.dtype().kind() == 'f' && Q.itemsize() == sizeof(float)) {
        return arnoldi_iteration_impl<float>(Q, w, decay, k);
    }
    return arnoldi_iteration_impl<double>(Q, w, decay, k);
}

}  // namespace

PYBIND11_MODULE(_kernels, m) {
    m.doc() = "Native kernels for OMEGA ACP";
    m.def(
        "arnoldi_iteration",
        &dispatch_arnoldi,
        py::arg("Q"),
        py::arg("w"),
        py::arg("decay"),
        py::arg("k"),
        "Compute the Hessenberg column and Arnoldi residual using native loops."
    );
}
