#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <random>
#include <iomanip>
#include <string>
#include <cmath>
#include <omp.h>

using Matrix = std::vector<std::vector<double>>;

// Генерация с фиксированным seed
Matrix generateMatrix(int n) {
    Matrix mat(n, std::vector<double>(n));
    std::mt19937 gen(42); 
    std::uniform_real_distribution<> dis(-10.0, 10.0);
    
    #pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            mat[i][j] = dis(gen);
        }
    }
    return mat;
}

// Упрощенная и безопасная запись
void saveMatrixToFile(const std::string& filename, const Matrix& mat) {
    std::ofstream out(filename);
    if (!out.is_open()) {
        std::cerr << "Error: Cannot open file " << filename << std::endl;
        return;
    }
    
    int n = mat.size();
    out << n << "\n"; 
    
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            out << std::fixed << std::setprecision(6) << mat[i][j];
            if (j < n - 1) out << " ";
        }
        out << "\n";
    }
    out.flush(); // Принудительный сброс буфера
    out.close(); // Явное закрытие
}

Matrix multiplyMatricesParallel(const Matrix& A, const Matrix& B) {
    int n = A.size();
    Matrix C(n, std::vector<double>(n, 0.0));
    
    #pragma omp parallel for schedule(dynamic)
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
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <matrix_size> <num_threads>" << std::endl;
        return 1;
    }

    int N = 0;
    int numThreads = 0;
    try {
        N = std::stoi(argv[1]);
        numThreads = std::stoi(argv[2]);
    } catch (...) {
        std::cerr << "Invalid arguments." << std::endl;
        return 1;
    }

    if (N <= 0 || numThreads <= 0) {
        std::cerr << "Size and threads must be positive." << std::endl;
        return 1;
    }

    omp_set_num_threads(numThreads);

    std::cout << "Starting experiment: N=" << N << ", Threads=" << numThreads << std::endl;
    std::cout << "Max available threads: " << omp_get_max_threads() << std::endl;

    try {
        std::cout << "Generating matrices..." << std::endl;
        Matrix A = generateMatrix(N);
        Matrix B = generateMatrix(N);

        std::cout << "Multiplying (Parallel)..." << std::endl;
        auto start = std::chrono::high_resolution_clock::now();
        
        Matrix C = multiplyMatricesParallel(A, B);
        
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff = end - start;

        double time_seconds = diff.count();
        long long operations = 2LL * N * N * N; 
        double gflops = (operations / 1e9) / time_seconds;

        std::cout << "Time: " << time_seconds << " seconds" << std::endl;
        std::cout << "GFLOPS: " << gflops << std::endl;

        // Сохранение результата
        std::cout << "Saving result..." << std::endl;
        saveMatrixToFile("result_C_parallel.txt", C);
        std::cout << "Result saved." << std::endl;

                // Запись метрик 
        std::ofstream metrics("metrics_parallel.csv", std::ios::out | std::ios::app);
        if (!metrics.is_open()) {
            std::cerr << "CRITICAL: Cannot open metrics file!" << std::endl;
            return 1; 
        }
        
        // Проверяем размер файла простым способом
        metrics.seekp(0, std::ios::end);
        bool isFirst = (metrics.tellp() == 0);
        
        if (isFirst) {
            metrics << "N,Threads,Time_Seconds,GFLOPS" << std::endl;
        }
        
        metrics << N << "," << numThreads << "," << time_seconds << "," << gflops << std::endl;
        metrics.close(); // Явное закрытие
        
        std::cout << "Metrics saved." << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Runtime Exception: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}