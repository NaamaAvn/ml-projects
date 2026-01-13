#!/bin/bash
# Setup script for creating virtual environment and Jupyter kernel using venv

# Create virtual environment
python3 -m venv venv

# Activate the environment
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install requirements
pip install -r requirements.txt

# Install ipykernel to create Jupyter kernel
pip install ipykernel

# Create Jupyter kernel
python -m ipykernel install --user --name exercise1 --display-name "Python (Exercise1)"

echo "Environment setup complete!"
echo "To activate: source venv/bin/activate"
echo "The kernel 'Python (Exercise1)' should now be available in Jupyter notebooks"


