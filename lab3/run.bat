@echo off
chcp 65001 > nul
setlocal enabledelayedexpansion

:: === ПУТИ К ВАШЕМУ MS-MPI SDK ===
set MSMPI_INC="C:\Program Files (x86)\Microsoft SDKs\MPI\Include"
set MSMPI_LIB="C:\Program Files (x86)\Microsoft SDKs\MPI\Lib\x64"

echo === Compiling Lab 3 (MS-MPI + g++) ===

:: Проверка наличия main.cpp
if not exist "main.cpp" (
    echo ERROR: main.cpp not found!
    pause
    exit /b 1
)

:: Компиляция через g++ (стандартный компилятор MinGW)
g++ -O2 -std=c++17 -I%MSMPI_INC% main.cpp -L%MSMPI_LIB% -lmsmpi -o matrix_mult_mpi.exe

if %errorlevel% neq 0 (
    echo ERROR: Compilation failed! Check paths and code.
    pause
    exit /b 1
)
echo === Compilation OK ===

:: Очистка старых данных
if exist metrics_mpi.csv del metrics_mpi.csv

echo === Starting Experiments ===

set SIZES=200 400 800 1200 1600 2000
set PROCS=1 2 4 8

for %%N in (%SIZES%) do (
    for %%P in (%PROCS%) do (
        echo.
        echo --- Running N=%%N, Processes=%%P ---
        
        :: Запуск через mpiexec (стандарт для MS-MPI)
        mpiexec -n %%P matrix_mult_mpi.exe %%N
        
        if !errorlevel! neq 0 (
            echo ERROR: Crash at N=%%N P=%%P
            pause
            exit /b 1
        )
    )
)

echo.
echo === All Experiments Finished ===
pause