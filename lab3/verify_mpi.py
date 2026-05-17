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
    if not os.path.exists("input_A_mpi.txt") or not os.path.exists("input_B_mpi.txt") or not os.path.exists("result_C_mpi.txt"):
        print("Error: Input or Result files not found. Run MPI program first.")
        sys.exit(1)

    print("Loading matrices...")
    A = read_matrix("input_A_mpi.txt")
    B = read_matrix("input_B_mpi.txt")
    C_mpi = read_matrix("result_C_mpi.txt")

    print("Calculating reference product using NumPy...")
    C_ref = np.dot(A, B)

    max_diff = np.max(np.abs(C_mpi - C_ref))
    print(f"Max absolute difference: {max_diff:.10f}")

    if max_diff < 1e-3:
        print("VERIFICATION: PASSED")
    else:
        print("VERIFICATION: FAILED")
        sys.exit(1)

if __name__ == "__main__":
    main()