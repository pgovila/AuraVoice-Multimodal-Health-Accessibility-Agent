@echo off
setlocal enabledelayedexpansion
title AuraVoice - Multimodal Health Accessibility Agent

echo ======================================================================
echo           AuraVoice - Multimodal Health Accessibility Agent
echo ======================================================================
echo.

:: Move to script directory
cd /d "%~dp0"

:: Check if virtual environment exists
if not exist "venv\Scripts\activate.bat" (
    echo [!] Virtual environment not found in .\venv
    echo [*] Checking Python installation...
    
    :: Prefer py -3.12 or python
    py -3.12 --version >nul 2>&1
    if !errorlevel! equ 0 (
        echo [*] Creating virtual environment using Python 3.12...
        py -3.12 -m venv venv
    ) else (
        python --version >nul 2>&1
        if !errorlevel! equ 0 (
            echo [*] Creating virtual environment using default Python...
            python -m venv venv
        ) else (
            echo [X] Error: Python was not found in PATH!
            echo     Please install Python 3.12 and ensure it is added to PATH.
            pause
            exit /b 1
        )
    )
    
    if not exist "venv\Scripts\activate.bat" (
        echo [X] Failed to create virtual environment.
        pause
        exit /b 1
    )

    echo [*] Activating virtual environment...
    call "venv\Scripts\activate.bat"

    echo [*] Installing dependencies from requirements.txt...
    python -m pip install --upgrade pip
    python -m pip install -r requirements.txt
    if !errorlevel! neq 0 (
        echo [X] Failed to install dependencies.
        pause
        exit /b 1
    )
) else (
    echo [*] Activating existing virtual environment...
    call "venv\Scripts\activate.bat"
)

:: Check for secrets.toml configuration
if not exist ".streamlit\secrets.toml" (
    echo.
    echo [!] Warning: .streamlit\secrets.toml not found!
    if exist ".streamlit\secrets.toml.example" (
        echo [*] Creating template from .streamlit\secrets.toml.example...
        copy ".streamlit\secrets.toml.example" ".streamlit\secrets.toml" >nul
        echo [!] Please configure your API keys in .streamlit\secrets.toml
    )
    echo.
)

:: Launch the Streamlit application
echo.
echo [*] Starting AuraVoice Streamlit application...
echo     Opening http://localhost:8501 in your browser...
echo.

python -m streamlit run app.py

if %errorlevel% neq 0 (
    echo.
    echo [X] AuraVoice exited with an error.
    pause
)
