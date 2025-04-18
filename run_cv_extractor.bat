@echo off
echo ========================================================
echo            CV EXTRACTION TOOL                           
echo ========================================================
echo.

REM Check for Python
python --version > nul 2>&1
if %errorlevel% neq 0 (
    echo Error: Python is not installed or not in PATH
    echo Please install Python 3.10 or later from https://www.python.org/downloads/
    goto end
)

REM Check virtual environment
if not exist venv (
    echo Creating virtual environment...
    python -m venv venv
    if %errorlevel% neq 0 (
        echo Error: Failed to create virtual environment
        goto end
    )
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Install dependencies
echo Installing dependencies...
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo Error: Failed to install dependencies
    goto end
)

REM Check for Ollama
ollama -v > nul 2>&1
if %errorlevel% neq 0 (
    echo WARNING: Ollama is not installed or not in PATH
    echo Please install Ollama from https://ollama.com/
    echo After installation, run 'ollama serve' in a separate terminal
    echo and pull the required models with:
    echo   ollama pull phi
    echo   ollama pull llama3
    echo   ollama pull mistral
) else (
    echo Ollama is installed!
)

REM Check if Ollama is running
echo Checking if Ollama is running...
curl -s http://localhost:11434/api/tags > nul 2>&1
if %errorlevel% equ 0 (
    echo Ollama is running!
    
    REM Check available models
    curl -s http://localhost:11434/api/tags > temp_models.json
    
    type temp_models.json | findstr /C:"phi" > nul
    if %errorlevel% equ 0 (
        echo - Phi model is available
    ) else (
        echo - Phi model is not available. Please run: ollama pull phi
    )
    
    type temp_models.json | findstr /C:"llama3" > nul
    if %errorlevel% equ 0 (
        echo - Llama3 model is available
    ) else (
        echo - Llama3 model is not available. Please run: ollama pull llama3
    )
    
    type temp_models.json | findstr /C:"mistral" > nul
    if %errorlevel% equ 0 (
        echo - Mistral model is available
    ) else (
        echo - Mistral model is not available. Please run: ollama pull mistral
    )
    
    del temp_models.json
) else (
    echo WARNING: Ollama is not running. 
    echo Please start Ollama with 'ollama serve' in a separate terminal.
)

echo.
echo --------------------------------------------------------
echo Starting CV Extractor...
echo --------------------------------------------------------
echo.
.\venv\Scripts\streamlit.exe run simple_cv_extractor.py --server.port=8501

:end
echo.
pause 