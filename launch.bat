@echo off
rem Vector launcher: sets itself up on first run, then starts Vector without a console.
rem Usage:  launch.bat [--debug] [--demo] [calibrate]
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo First run: creating the Python environment...
    py -3 -m venv .venv || python -m venv .venv
    ".venv\Scripts\python.exe" -m pip install --upgrade pip
    ".venv\Scripts\python.exe" -m pip install -e .[dev] || goto :fail
)
if not exist "models\hand_landmarker.task" (
    ".venv\Scripts\python.exe" scripts\fetch_model.py || goto :fail
)

start "" ".venv\Scripts\pythonw.exe" -m vector %*
echo Vector started. Esc Esc = emergency stop, Ctrl+Alt+Q = quit.
echo Log: %APPDATA%\Vector\vector.log
exit /b 0

:fail
echo Setup failed - see the messages above.
pause
exit /b 1
