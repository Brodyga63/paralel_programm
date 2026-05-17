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
    
    if not os.path.exists("input_A.txt") or not os.path.exists("result_C_parallel.txt"):
        print("Warning: Input files not found. Skipping verification.")
        print("Please uncomment 'saveMatrixToFile' in main.cpp and re-run for N=200 to generate inputs.")
        return

    print("Loading matrices...")
    A = read_matrix("input_A.txt")
    C_parallel = read_matrix("result_C_parallel.txt")

    print("Calculating reference product using NumPy...")
    C_ref = np.dot(A, A)

    if os.path.exists("input_B.txt"):
        B = read_matrix("input_B.txt")
        C_ref = np.dot(A, B)
    else:
        print("Error: input_B.txt not found")
        return

    max_diff = np.max(np.abs(C_parallel - C_ref))
    print(f"Max absolute difference: {max_diff:.10f}")

    if max_diff < 1e-3:
        print("VERIFICATION: PASSED")
    else:
        print("VERIFICATION: FAILED")
        sys.exit(1)

if __name__ == "__main__":
    main()