#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <iomanip>
#include <string>
#include <fstream> 
#include <mpi.h>

using Matrix = std::vector<std::vector<double>>;

// Генерация случайной матрицы
Matrix generateMatrix(int n, int seed_offset = 0) {
    Matrix mat(n, std::vector<double>(n));
    std::mt19937 gen(42 + seed_offset); 
    std::uniform_real_distribution<> dis(-10.0, 10.0);
    
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            mat[i][j] = dis(gen);
        }
    }
    return mat;
}

// Запись матрицы в файл (только root процесс)
void saveMatrixToFile(const std::string& filename, const Matrix& mat, int rank) {
    if (rank != 0) return; // Только процесс 0 пишет файл

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

// Чтение матрицы из файла (только root процесс)
Matrix loadMatrixFromFile(const std::string& filename, int rank) {
    if (rank != 0) return Matrix(); // Только процесс 0 читает

    std::ifstream in(filename);
    if (!in.is_open()) {
        throw std::runtime_error("Cannot open file: " + filename);
    }
    
    int n;
    in >> n;
    Matrix mat(n, std::vector<double>(n));
    
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            in >> mat[i][j];
        }
    }
    in.close();
    return mat;
}

// Перемножение локальной части матрицы
Matrix multiplyLocalPart(const Matrix& localA, const Matrix& B) {
    int local_rows = localA.size();
    int n = B.size();
    Matrix localC(local_rows, std::vector<double>(n, 0.0));

    for (int i = 0; i < local_rows; ++i) {
        for (int k = 0; k < n; ++k) {
            double a_ik = localA[i][k];
            for (int j = 0; j < n; ++j) {
                localC[i][j] += a_ik * B[k][j];
            }
        }
    }
    return localC;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc != 2) {
        if (rank == 0) {
            std::cerr << "Usage: " << argv[0] << " <matrix_size>" << std::endl;
        }
        MPI_Finalize();
        return 1;
    }

    int N = std::stoi(argv[1]);

    // --- Шаг 1: Root генерирует матрицы ---
    Matrix A_full, B_full;
    if (rank == 0) {
        std::cout << "[Rank 0] Starting experiment: N=" << N << ", Processes=" << size << std::endl;
        A_full = generateMatrix(N, 0);
        B_full = generateMatrix(N, 1);

        // Сохраняем входные данные для верификации
        saveMatrixToFile("input_A_mpi.txt", A_full, rank);
        saveMatrixToFile("input_B_mpi.txt", B_full, rank);
    }

    // --- Шаг 2: Рассылаем размер матрицы всем ---
    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // --- Шаг 3: Рассылаем матрицу B всем процессам ---
    // Сначала узнаем количество элементов
    int total_elements = N * N;
    std::vector<double> B_flat(total_elements);
    
    if (rank == 0) {
        // Преобразуем B_full в плоский массив
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                B_flat[i * N + j] = B_full[i][j];
            }
        }
    }
    
    MPI_Bcast(B_flat.data(), total_elements, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Восстанавливаем B_local у всех процессов
    Matrix B_local(N, std::vector<double>(N));
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            B_local[i][j] = B_flat[i * N + j];
        }
    }

    // --- Шаг 4: Распределяем строки матрицы A ---
    int rows_per_proc = N / size;
    int remainder = N % size;
    int start_row = rank * rows_per_proc + std::min(rank, remainder);
    int end_row = start_row + rows_per_proc + (rank < remainder ? 1 : 0);
    int local_rows = end_row - start_row;

    // Создаем локальную часть A
    Matrix A_local(local_rows, std::vector<double>(N));

    // Root рассылает нужные строки каждому процессу
    if (rank == 0) {
        for (int p = 0; p < size; ++p) {
            int p_start = p * rows_per_proc + std::min(p, remainder);
            int p_end = p_start + rows_per_proc + (p < remainder ? 1 : 0);
            int p_rows = p_end - p_start;

            if (p == 0) {
                // Копируем себе
                for (int i = 0; i < p_rows; ++i) {
                    for (int j = 0; j < N; ++j) {
                        A_local[i][j] = A_full[p_start + i][j];
                    }
                }
            } else {
                // Отправляем другим
                std::vector<double> send_buffer(p_rows * N);
                for (int i = 0; i < p_rows; ++i) {
                    for (int j = 0; j < N; ++j) {
                        send_buffer[i * N + j] = A_full[p_start + i][j];
                    }
                }
                MPI_Send(send_buffer.data(), p_rows * N, MPI_DOUBLE, p, 0, MPI_COMM_WORLD);
            }
        }
    } else {
        // Получаем свои строки
        std::vector<double> recv_buffer(local_rows * N);
        MPI_Recv(recv_buffer.data(), local_rows * N, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        
        for (int i = 0; i < local_rows; ++i) {
            for (int j = 0; j < N; ++j) {
                A_local[i][j] = recv_buffer[i * N + j];
            }
        }
    }

    // --- Шаг 5: Локальное умножение ---
    auto start_time = MPI_Wtime();
    Matrix C_local = multiplyLocalPart(A_local, B_local);
    auto end_time = MPI_Wtime();

    double local_time = end_time - start_time;

    // --- Шаг 6: Сбор результатов в root ---
    std::vector<double> C_flat_result;
    if (rank == 0) {
        C_flat_result.resize(N * N);
        // Копируем свои строки
        for (int i = 0; i < local_rows; ++i) {
            for (int j = 0; j < N; ++j) {
                C_flat_result[(start_row + i) * N + j] = C_local[i][j];
            }
        }

        // Принимаем от других процессов
        for (int p = 1; p < size; ++p) {
            int p_start = p * rows_per_proc + std::min(p, remainder);
            int p_rows = (p < remainder ? rows_per_proc + 1 : rows_per_proc);
            
            std::vector<double> recv_buf(p_rows * N);
            MPI_Recv(recv_buf.data(), p_rows * N, MPI_DOUBLE, p, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            
            for (int i = 0; i < p_rows; ++i) {
                for (int j = 0; j < N; ++j) {
                    C_flat_result[(p_start + i) * N + j] = recv_buf[i * N + j];
                }
            }
        }
    } else {
        // Отправляем свои строки root'у
        std::vector<double> send_buf(local_rows * N);
        for (int i = 0; i < local_rows; ++i) {
            for (int j = 0; j < N; ++j) {
                send_buf[i * N + j] = C_local[i][j];
            }
        }
        MPI_Send(send_buf.data(), local_rows * N, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
    }

    // --- Шаг 7: Вывод результатов (только root) ---
    if (rank == 0) {
        // Преобразуем результат обратно в матрицу
        Matrix C_result(N, std::vector<double>(N));
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                C_result[i][j] = C_flat_result[i * N + j];
            }
        }

        // Сохраняем результат
        saveMatrixToFile("result_C_mpi.txt", C_result, rank);

        // Вычисляем GFLOPS
        long long operations = 2LL * N * N * N;
        double gflops = (operations / 1e9) / local_time; // Используем время локального вычисления

        std::cout << "[Rank 0] Time (local compute): " << local_time << " seconds" << std::endl;
        std::cout << "[Rank 0] GFLOPS: " << gflops << std::endl;

        // Запись метрик
        std::ofstream metrics("metrics_mpi.csv", std::ios::app);
        if (metrics.is_open()) {
            metrics.seekp(0, std::ios::end);
            if (metrics.tellp() == 0) {
                metrics << "N,Processes,Time_Seconds,GFLOPS" << std::endl;
            }
            metrics << N << "," << size << "," << local_time << "," << gflops << std::endl;
            metrics.close();
        }

        std::cout << "[Rank 0] Done." << std::endl;
    }

    MPI_Finalize();
    return 0;
}