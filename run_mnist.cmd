@echo off
setlocal
cd /d "%~dp0mnist"
set OMP_NUM_THREADS=4
set MKL_NUM_THREADS=4
"..\.venv\Scripts\python.exe" -u main.py --no-accel --epochs 1 --save-model %*
exit /b %errorlevel%
