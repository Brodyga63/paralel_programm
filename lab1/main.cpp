#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <random>
#include <iomanip>
#include <string>
#include <cmath>
#include <stdexcept>

using Matrix = std::vector<std::vector<double>>;

// Функция генерации с обработкой ошибок памяти
Matrix generateMatrix(int n) {
    try {
        Matrix mat(n, std::vector<double>(n));
        std::mt19937 gen(42); 
        std::uniform_real_distribution<> dis(-10.0, 10.0);
        
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                mat[i][j] = dis(gen);
            }
        }
        return mat;
    } catch (const std::bad_alloc& e) {
        std::cerr << "Memory allocation failed for N=" << n << ": " << e.what() << std::endl;
        throw; // Пробрасываем ошибку дальше, чтобы программа завершилась корректно
    }
}

void saveMatrixToFile(const std::string& filename, const Matrix& mat) {
    std::ofstream out(filename);
    if (!out.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return;
    }
    
    int n = mat.size();
    out << n << std::endl; 
    
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            out << std::fixed << std::setprecision(6) << mat[i][j];
            if (j < n - 1) out << " ";
        }
        out << std::endl;
    }
    out.close();
}

Matrix multiplyMatrices(const Matrix& A, const Matrix& B) {
    int n = A.size();
    // Инициализируем нулями
    Matrix C(n, std::vector<double>(n, 0.0));
    
    for (int i = 0; i < n; ++i) {
        for (int k = 0; k < n; ++k) {
            double a_ik = A[i][k]; 
            for (int j = 0; j < n; ++j) {
                C[i][j] += a_ik * B[k][j];
            }
        }
    }
    return C;
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <matrix_size>" << std::endl;
        return 1;
    }

    int N = 0;
    try {
        N = std::stoi(argv[1]);
    } catch (...) {
        std::cerr << "Invalid argument" << std::endl;
        return 1;
    }

    std::cout << "Starting experiment for N = " << N << std::endl;

    try {
        std::cout << "Generating matrices..." << std::endl;
        Matrix A = generateMatrix(N);
        std::cout << "Matrix A generated." << std::endl;
        
        Matrix B = generateMatrix(N);
        std::cout << "Matrix B generated." << std::endl;

        saveMatrixToFile("input_A.txt", A);
        saveMatrixToFile("input_B.txt", B);

        std::cout << "Multiplying matrices..." << std::endl;
        auto start = std::chrono::high_resolution_clock::now();
        
        Matrix C = multiplyMatrices(A, B);
        
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff = end - start;

        double time_seconds = diff.count();
        long long operations = 2LL * N * N * N; 
        double gflops = (operations / 1e9) / time_seconds;

        std::cout << "Time: " << time_seconds << " seconds" << std::endl;
        std::cout << "GFLOPS: " << gflops << std::endl;

        saveMatrixToFile("result_C.txt", C);

        std::ofstream metrics("metrics.csv", std::ios::app);
        if (metrics.is_open()) {
            metrics.seekp(0, std::ios::end);
            if (metrics.tellp() == 0) {
                metrics << "N,Time_Seconds,GFLOPS" << std::endl;
            }
            metrics << N << "," << time_seconds << "," << gflops << std::endl;
            metrics.close();
        }

        std::cout << "Done." << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Runtime Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}