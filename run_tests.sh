#!/bin/bash

echo "Activating virtual environment..."
source .venv/bin/activate

echo "Running tests..."
python3 -m pytest tests/

echo "Tests complete!"
