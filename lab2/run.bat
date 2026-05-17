@echo off
chcp 65001 > nul
echo === Compiling Lab 2 (OpenMP) ===
:: Важно: флаг -fopenmp обязателен для GCC/MinGW
g++ -O2 -std=c++17 -fopenmp -o matrix_mult_omp.exe main.cpp

if %errorlevel% neq 0 (
    echo ERROR: Compilation failed!
    pause
    exit /b 1
)
echo === Compilation OK ===

:: Очищаем старый файл метрик
del metrics_parallel.csv

echo === Starting Experiments ===

:: Размеры матриц
set SIZES=200 400 800 1200 1600 2000
:: Количество потоков
set THREADS=1 2 4 8

for %%N in (%SIZES%) do (
    for %%T in (%THREADS%) do (
        echo.
        echo --- Running N=%%N, Threads=%%T ---
        matrix_mult_omp.exe %%N %%T
        
        if %errorlevel% neq 0 (
            echo ERROR: Crash at N=%%N T=%%T
            pause
            exit /b 1
        )
    )
)

echo.
echo === All Experiments Finished ===
echo Check 'metrics_parallel.csv' for results.
pause