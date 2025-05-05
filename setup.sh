#!/bin/bash

# Ensure the script is run on Ubuntu
if [ "$(lsb_release -is)" != "Ubuntu" ]; then
    echo "This script is intended for Ubuntu systems only."
    exit 1
fi

# Update package list and install required system packages
echo "Updating package list and installing required system packages..."
sudo apt update
sudo apt install -y python3 python3-venv python3-pip
sudo apt install -y python3-virtualenv

# Check if the virtual environment exists, create it if not
if [ ! -d "venv" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv venv
else
    echo "Virtual environment already exists."
fi

# Activate the virtual environment
source venv/bin/activate

# Upgrade pip to the latest version
echo "Upgrading pip..."
pip install --upgrade pip

# Check and install required Python packages
REQUIRED_PACKAGES=(
    "numpy"
    "paddlepaddle>=2.0.0"
    "visualdl>=2.0.0"
)

for PACKAGE in "${REQUIRED_PACKAGES[@]}"; do
    pip show $(echo $PACKAGE | cut -d'=' -f1) > /dev/null 2>&1
    if [ $? -ne 0 ]; then
        echo "Installing $PACKAGE..."
        pip install "$PACKAGE"
    else
        echo "$PACKAGE is already installed."
    fi
done

echo "Setup complete."