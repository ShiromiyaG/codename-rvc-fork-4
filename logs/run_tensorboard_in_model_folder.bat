@echo off
title TensorBoard Launcher
color 0A

:: Define port
set PORT=25565
set ADDRESS=http://localhost:%PORT%

echo ===================================================
echo             TensorBoard Launcher
echo ===================================================
echo.
echo Please enter the path to your model's log directory.
echo (e.g. Logs^(Path^), C:\models\my_model)
echo.

:: Get user input
set /p UserInputPath="Model Directory: "

:: Remove outer quotes if the user dragged and dropped a folder
set UserInputPath=%UserInputPath:"=%

:: Check if path is empty or does not exist
if "%UserInputPath%"=="" (
    echo Error: Path cannot be empty.
    pause
    exit /b
)
if not exist "%UserInputPath%\" (
    echo Error: Directory "%UserInputPath%" does not exist.
    pause
    exit /b
)

echo.
echo Starting TensorBoard at %ADDRESS%
echo Reading logs from: "%UserInputPath%"
echo.
echo Press Ctrl + C to stop the server...
echo.

:: Open browser
explorer %ADDRESS%

:: Start TensorBoard
tensorboard --logdir="%UserInputPath%" --port=%PORT% --bind_all

pause
