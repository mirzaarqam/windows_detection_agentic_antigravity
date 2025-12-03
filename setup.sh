#!/bin/bash

# Install basic dependencies (torch, torchvision, opencv)
echo "Installing basic dependencies..."
pip install -r requirements.txt

# Install GroundingDINO and Segment Anything
# These need to be installed after torch is available
echo "Installing GroundingDINO..."
pip install git+https://github.com/IDEA-Research/GroundingDINO.git

echo "Installing Segment Anything..."
pip install git+https://github.com/facebookresearch/segment-anything.git

echo "Installation complete."
