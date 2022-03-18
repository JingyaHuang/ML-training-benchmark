#!/bin/bash
CMD=${1:-/bin/bash}

# Setup GPU to use
export CUDA_VISIBLE_DEVICES=4

# Install dependencies
pip install coloredlogs transformers>=4.15.0 datasets>=1.18.0
pip install scipy sklearn

# Run the example
# python -m unittest onnxruntime/test_examples.py
python -m unittest pytorch/test_pytorch_examples.py