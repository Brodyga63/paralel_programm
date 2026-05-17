import pandas as pd
import matplotlib.pyplot as plt

# Чтение данных
df = pd.read_csv('metrics.csv')

# График 1: Время от N
plt.figure(figsize=(10, 6))
plt.plot(df['N'], df['Time_Seconds'], marker='o', label='Execution Time')
plt.title('Matrix Multiplication Time vs Size')
plt.xlabel('Matrix Size (N)')
plt.ylabel('Time (seconds)')
plt.grid(True)
plt.legend()
plt.savefig('plot_time.png')
plt.show()

# График 2: GFLOPS от N
plt.figure(figsize=(10, 6))
plt.plot(df['N'], df['GFLOPS'], marker='s', color='orange', label='Performance')
plt.title('Performance vs Matrix Size')
plt.xlabel('Matrix Size (N)')
plt.ylabel('GFLOPS')
plt.grid(True)
plt.legend()
plt.savefig('plot_gflops.png')
plt.show()