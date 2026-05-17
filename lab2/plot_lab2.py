import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Чтение данных
try:
    df = pd.read_csv('metrics_parallel.csv')
except FileNotFoundError:
    print("Error: metrics_parallel.csv not found. Run the experiment first.")
    exit(1)

# Фильтрация данных для удобства (берем только полные наборы)
sizes = sorted(df['N'].unique())
threads = sorted(df['Threads'].unique())

print(f"Sizes: {sizes}")
print(f"Threads: {threads}")

# Подготовка данных для графиков
speedup_data = {}
efficiency_data = {}

for N in sizes:
    df_N = df[df['N'] == N].set_index('Threads').sort_index()
    
    # Базовое время - это время для 1 потока
    t1 = df_N.loc[1, 'Time_Seconds']
    
    speedups = []
    efficiencies = []
    
    for t in threads:
        if t in df_N.index:
            tp = df_N.loc[t, 'Time_Seconds']
            s = t1 / tp if tp > 0 else 0
            e = s / t if t > 0 else 0
            speedups.append(s)
            efficiencies.append(e)
        else:
            speedups.append(0)
            efficiencies.append(0)
            
    speedup_data[N] = speedups
    efficiency_data[N] = efficiencies

# --- График 1: Ускорение (Speedup) ---
plt.figure(figsize=(12, 6))
for N in sizes:
    plt.plot(threads, speedup_data[N], marker='o', label=f'N={N}')

plt.plot(threads, threads, 'k--', linewidth=1, label='Ideal Speedup (Linear)')
plt.title('Speedup vs Number of Threads', fontsize=14)
plt.xlabel('Number of Threads (P)')
plt.ylabel('Speedup (S = T1 / Tp)')
plt.xticks(threads)
plt.legend(loc='upper left')
plt.grid(True, linestyle='--', alpha=0.7)
plt.savefig('speedup_graph.png', dpi=300)
plt.show()

# --- График 2: Эффективность (Efficiency) ---
plt.figure(figsize=(12, 6))
for N in sizes:
    plt.plot(threads, efficiency_data[N], marker='s', label=f'N={N}')

plt.axhline(y=1.0, color='k', linestyle=':', linewidth=1, label='Ideal Efficiency (100%)')
plt.title('Parallel Efficiency vs Number of Threads', fontsize=14)
plt.xlabel('Number of Threads (P)')
plt.ylabel('Efficiency (E = S / P)')
plt.xticks(threads)
plt.legend(loc='best')
plt.grid(True, linestyle='--', alpha=0.7)
plt.ylim(0, 1.1) # Ограничим ось Y, чтобы было видно детали
plt.savefig('efficiency_graph.png', dpi=300)
plt.show()

print("Graphs saved as speedup_graph.png and efficiency_graph.png")