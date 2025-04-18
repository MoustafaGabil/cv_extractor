#!/bin/bash
# CV Extraction and Evaluation System

# ASCII art banner
echo "========================================================"
echo "            CV EXTRACTION TOOL                           "
echo "========================================================"
echo ""

# Function to check if a command exists
command_exists() {
    command -v "$1" &> /dev/null
}

# Check for Python
if ! command -v python3 &> /dev/null; then
    echo "Error: Python is not installed or not in PATH"
    echo "Please install Python 3.10 or later from https://www.python.org/downloads/"
    exit 1
fi

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo "Found Python version: $PYTHON_VERSION"

# Check for virtual environment
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    if [ $? -ne 0 ]; then
        echo "Error: Failed to create virtual environment"
        exit 1
    fi
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt
if [ $? -ne 0 ]; then
    echo "Error: Failed to install dependencies"
    exit 1
fi

# Check for Ollama
if ! command -v ollama &> /dev/null; then
    echo "WARNING: Ollama is not installed or not in PATH"
    echo "Please install Ollama from https://ollama.com/"
    echo "After installation, run 'ollama serve' in a separate terminal"
    echo "and pull the required models with:"
    echo "  ollama pull phi"
    echo "  ollama pull llama3"
    echo "  ollama pull mistral"
else
    echo "Ollama is installed!"
fi

# Check if Ollama is running
echo "Checking if Ollama is running..."
if curl -s http://localhost:11434/api/tags &> /dev/null; then
    echo "Ollama is running!"
    
    # Check available models
    curl -s http://localhost:11434/api/tags > temp_models.json
    
    if grep -q "phi" temp_models.json; then
        echo "- Phi model is available"
    else
        echo "- Phi model is not available. Please run: ollama pull phi"
    fi
    
    if grep -q "llama3" temp_models.json; then
        echo "- Llama3 model is available"
    else
        echo "- Llama3 model is not available. Please run: ollama pull llama3"
    fi
    
    if grep -q "mistral" temp_models.json; then
        echo "- Mistral model is available"
    else
        echo "- Mistral model is not available. Please run: ollama pull mistral"
    fi
    
    rm temp_models.json
else
    echo "WARNING: Ollama is not running."
    echo "Please start Ollama with 'ollama serve' in a separate terminal."
fi

echo ""
echo "--------------------------------------------------------"
echo "Starting CV Extractor..."
echo "--------------------------------------------------------"
echo ""
streamlit run simple_cv_extractor.py --server.port=8501

echo
echo "Done!" 