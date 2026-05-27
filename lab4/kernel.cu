constexpr int DEFAULT_BLOCK_X = 16;
constexpr int DEFAULT_BLOCK_Y = 16;
constexpr bool USE_CMD_ARGS = true;

#include <iostream>
#include <vector>
#include <fstream>
#include <chrono>
#include <string>
#include <algorithm>
#include <Eigen/Dense>
#include <cuda_runtime.h>

__global__ void matMulKernel(const double* A, const double* B, double* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N && col < N) {
        double sum = 0.0;
        for (int k = 0; k < N; ++k)
            sum += A[row * N + k] * B[k * N + col];
        C[row * N + col] = sum;
    }
}

std::vector<std::vector<double>> read_file(const std::string& fn) {
    std::ifstream in(fn);
    int n;
    if (!(in >> n)) return {};
    std::vector<std::vector<double>> m(n, std::vector<double>(n));
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            in >> m[i][j];
    return m;
}

void write_file(const std::string& fn, const std::vector<std::vector<double>>& m) {
    std::ofstream out(fn);
    int n = m.size();
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (j) out << ' ';
            out << m[i][j];
        }
        out << '\n';
    }
}

std::vector<double> flatten(const std::vector<std::vector<double>>& m) {
    if (m.empty()) return {};
    int r = m.size(), c = m[0].size();
    std::vector<double> f(r * c);
    for (int i = 0; i < r; ++i)
        std::copy(m[i].begin(), m[i].end(), f.begin() + i * c);
    return f;
}

std::vector<std::vector<double>> unflatten(const std::vector<double>& f, int r, int c) {
    std::vector<std::vector<double>> m(r, std::vector<double>(c));
    for (int i = 0; i < r; ++i)
        std::copy(f.begin() + i * c, f.begin() + (i + 1) * c, m[i].begin());
    return m;
}

void save_stats(double t, int N, int bx, int by) {
    std::ofstream out("result_statistic_cuda.csv", std::ios::app);
    if (out.tellp() == 0) out << "Size,Block_X,Block_Y,Time_Seconds\n";
    out << N << "," << bx << "," << by << "," << t << "\n";
}

bool verify(const std::vector<std::vector<double>>& A,
    const std::vector<std::vector<double>>& B,
    const std::vector<std::vector<double>>& C) {
    int n = A.size();
    Eigen::MatrixXd eA(n, n), eB(n, n), eC(n, n);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j) {
            eA(i, j) = A[i][j]; eB(i, j) = B[i][j]; eC(i, j) = C[i][j];
        }
    return (eA * eB - eC).cwiseAbs().maxCoeff() < 1e-8;
}

int main(int argc, char** argv) {
    int bx = DEFAULT_BLOCK_X;
    int by = DEFAULT_BLOCK_Y;

    if constexpr (USE_CMD_ARGS) {
        if (argc == 3) {
            bx = std::stoi(argv[1]);
            by = std::stoi(argv[2]);
        }
        else if (argc == 2) {
            bx = by = std::stoi(argv[1]);
        }
    }

    if (bx * by > 1024) {
        std::cerr << "Error: Block size product exceeds 1024.\n";
        return 1;
    }

    auto A = read_file("matrix_A.txt");
    auto B = read_file("matrix_B.txt");
    int N = A.size();
    if (N == 0 || N != (int)B.size()) {
        std::cerr << "Error: Invalid matrix size or mismatch.\n";
        return 1;
    }

    auto A_f = flatten(A);
    auto B_f = flatten(B);
    std::vector<double> C_f(N * N, 0.0);

    double* d_A, * d_B, * d_C;
    size_t bytes = N * N * sizeof(double);
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);

    cudaMemcpy(d_A, A_f.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B_f.data(), bytes, cudaMemcpyHostToDevice);

    dim3 block(bx, by);
    dim3 grid((N + bx - 1) / bx, (N + by - 1) / by);

    cudaDeviceSynchronize();
    auto t1 = std::chrono::steady_clock::now();

    matMulKernel << <grid, block >> > (d_A, d_B, d_C, N);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA Kernel Error: " << cudaGetErrorString(err) << "\n";
        return 1;
    }

    cudaDeviceSynchronize();
    auto t2 = std::chrono::steady_clock::now();
    double time = std::chrono::duration<double>(t2 - t1).count();

    cudaMemcpy(C_f.data(), d_C, bytes, cudaMemcpyDeviceToHost);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);

    auto C = unflatten(C_f, N, N);
    write_file("result_matrix.txt", C);
    save_stats(time, N, bx, by);

    bool ok = verify(A, B, C);
    std::cout << "Verification: " << (ok ? "PASSED" : "FAILED") << "\n";

    return 0;
}