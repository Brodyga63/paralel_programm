@echo off
chcp 65001 > nul
echo === Start Compilation ===
g++ -O2 -std=c++17 -o matrix_mult.exe main.cpp

if %errorlevel% neq 0 (
    echo ERROR: Compilation failed! Check your code.
    pause
    exit /b 1
)
echo === Compilation OK ===

:: Удаляем старый CSV, чтобы начать чистый эксперимент
if exist metrics.csv del metrics.csv

echo === Start Experiments ===

:: Список размеров
for %%N in (200 400 800 1200 1600 2000) do (
    echo.
    echo --- Running N=%%N ---
    matrix_mult.exe %%N
    
    if %errorlevel% neq 0 (
        echo ERROR: Program crashed for N=%%N
        pause
        exit /b 1
    )
)

echo.
echo === All Experiments Finished ===
echo Check 'metrics.csv' for results.
echo.
pause