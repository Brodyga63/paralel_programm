import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Чтение данных
try:
    df = pd.read_csv('metrics_mpi.csv')
except FileNotFoundError:
    print("Error: metrics_mpi.csv not found.")
    exit(1)

sizes = sorted(df['N'].unique())
procs = sorted(df['Processes'].unique())

print(f"Sizes: {sizes}")
print(f"Processes: {procs}")

# Подготовка данных
speedup_data = {}
efficiency_data = {}

for N in sizes:
    df_N = df[df['N'] == N].set_index('Processes').sort_index()
    
    # Базовое время - 1 процесс
    if 1 not in df_N.index:
        continue
        
    t1 = df_N.loc[1, 'Time_Seconds']
    
    speedups = []
    efficiencies = []
    
    for p in procs:
        if p in df_N.index:
            tp = df_N.loc[p, 'Time_Seconds']
            s = t1 / tp if tp > 0 else 0
            e = s / p if p > 0 else 0
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
    plt.plot(procs, speedup_data[N], marker='o', label=f'N={N}')

plt.plot(procs, procs, 'k--', linewidth=1, label='Ideal Speedup (Linear)')
plt.title('MPI Speedup vs Number of Processes', fontsize=14)
plt.xlabel('Number of Processes (P)')
plt.ylabel('Speedup (S = T1 / Tp)')
plt.xticks(procs)
plt.legend(loc='upper left')
plt.grid(True, linestyle='--', alpha=0.7)
plt.savefig('speedup_mpi.png', dpi=300)
plt.show()

# --- График 2: Эффективность (Efficiency) ---
plt.figure(figsize=(12, 6))
for N in sizes:
    plt.plot(procs, efficiency_data[N], marker='s', label=f'N={N}')

plt.axhline(y=1.0, color='k', linestyle=':', linewidth=1, label='Ideal Efficiency (100%)')
plt.title('MPI Parallel Efficiency vs Number of Processes', fontsize=14)
plt.xlabel('Number of Processes (P)')
plt.ylabel('Efficiency (E = S / P)')
plt.xticks(procs)
plt.legend(loc='best')
plt.grid(True, linestyle='--', alpha=0.7)
plt.ylim(0, 1.1)
plt.savefig('efficiency_mpi.png', dpi=300)
plt.show()

print("Graphs saved as speedup_mpi.png and efficiency_mpi.png")