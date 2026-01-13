#!/bin/bash

# AutoEncoder Project Environment Setup Script

echo "Setting up AutoEncoder Project Environment..."
echo "============================================="

# Check if Python 3 is installed
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

# Check Python version
python_version=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
echo "Python version: $python_version"

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv autoencoder_env

# Activate virtual environment
echo "Activating virtual environment..."
source autoencoder_env/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "Installing required packages..."
pip install -r requirements.txt

echo ""
echo "Environment setup complete!"
echo "============================================="
echo "To activate the environment:"
echo "  source autoencoder_env/bin/activate"
echo ""
echo "To deactivate the environment:"
echo "  deactivate"
echo ""
echo "To run the test script:"
echo "  cd experiments"
echo "  python test_denoising_experiment.py"
echo ""
echo "To run the full experiment:"
echo "  cd experiments"
echo "  python autoencoder_denoising_experiment.py" 