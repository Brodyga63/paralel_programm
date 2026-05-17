import numpy as np
import sys
import os

def read_matrix(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
        n = int(lines[0].strip())
        matrix = []
        for line in lines[1:]:
            row = list(map(float, line.strip().split()))
            matrix.append(row)
        return np.array(matrix)

def main():
    if not os.path.exists("input_A.txt") or not os.path.exists("input_B.txt") or not os.path.exists("result_C.txt"):
        print("Error: Input or Result files not found. Run C++ program first.")
        sys.exit(1)

    print("Loading matrices...")
    A = read_matrix("input_A.txt")
    B = read_matrix("input_B.txt")
    C_cpp = read_matrix("result_C.txt")

    print("Calculating reference product using NumPy...")
    C_ref = np.dot(A, B)

    # Сравнение с учетом погрешности плавающей точки
    max_diff = np.max(np.abs(C_cpp - C_ref))
    mean_diff = np.mean(np.abs(C_cpp - C_ref))

    print(f"Max absolute difference: {max_diff:.10f}")
    print(f"Mean absolute difference: {mean_diff:.10f}")

    # Допустимая погрешность (зависит от размера матрицы и накопления ошибок)
    tolerance = 1e-6 * np.linalg.norm(C_ref) 
    
    if max_diff < 1e-3: # Жесткий порог для двойной точности обычно ок, но для больших N может расти
        print("VERIFICATION: PASSED")
    else:
        print("VERIFICATION: FAILED")
        sys.exit(1)

if __name__ == "__main__":
    main()