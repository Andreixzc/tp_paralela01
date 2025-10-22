#!/bin/bash
# download_mnist.sh - Download MNIST dataset in CSV format

echo "=========================================="
echo "  MNIST Dataset Download Script"
echo "=========================================="
echo ""

# Check if files already exist
if [ -f "mnist_train.csv" ] && [ -f "mnist_test.csv" ]; then
    echo "✓ MNIST files already exist!"
    echo "  - mnist_train.csv"
    echo "  - mnist_test.csv"
    echo ""
    read -p "Re-download? (y/N): " response
    if [[ ! $response =~ ^[Yy]$ ]]; then
        echo "Using existing files."
        exit 0
    fi
fi

echo "Downloading MNIST dataset..."
echo ""

# Option 1: Direct download from Kaggle mirror (if available)
echo "Trying direct download..."

# MNIST train set (60,000 samples)
if ! wget -q --show-progress -O mnist_train.csv \
    "https://pjreddie.com/media/files/mnist_train.csv" 2>/dev/null; then
    echo "Direct download failed. Trying alternative..."
    
    # Alternative: GitHub mirror
    if ! wget -q --show-progress -O mnist_train.csv \
        "https://raw.githubusercontent.com/oddrationale/mnist_csv/master/mnist_train.csv" 2>/dev/null; then
        echo ""
        echo "❌ Automatic download failed."
        echo ""
        echo "Please download manually from one of these sources:"
        echo ""
        echo "Option 1 (Recommended):"
        echo "  1. Go to: https://www.kaggle.com/datasets/oddrationale/mnist-in-csv"
        echo "  2. Download 'mnist_train.csv' and 'mnist_test.csv'"
        echo "  3. Place them in this directory: $(pwd)"
        echo ""
        echo "Option 2 (Alternative):"
        echo "  1. Go to: https://github.com/oddrationale/mnist_csv"
        echo "  2. Download the CSV files"
        echo ""
        echo "Option 3 (Generate from original):"
        echo "  1. Download original MNIST from: http://yann.lecun.com/exdb/mnist/"
        echo "  2. Convert to CSV using a Python script"
        echo ""
        exit 1
    fi
fi

# MNIST test set (10,000 samples)
if ! wget -q --show-progress -O mnist_test.csv \
    "https://pjreddie.com/media/files/mnist_test.csv" 2>/dev/null; then
    if ! wget -q --show-progress -O mnist_test.csv \
        "https://raw.githubusercontent.com/oddrationale/mnist_csv/master/mnist_test.csv" 2>/dev/null; then
        echo "Warning: Test set download failed, but train set is enough for the project."
    fi
fi

echo ""
echo "=========================================="
echo "  Download Complete!"
echo "=========================================="

if [ -f "mnist_train.csv" ]; then
    LINES=$(wc -l < mnist_train.csv)
    SIZE=$(du -h mnist_train.csv | cut -f1)
    echo "✓ mnist_train.csv: $LINES lines, $SIZE"
fi

if [ -f "mnist_test.csv" ]; then
    LINES=$(wc -l < mnist_test.csv)
    SIZE=$(du -h mnist_test.csv | cut -f1)
    echo "✓ mnist_test.csv: $LINES lines, $SIZE"
fi

echo ""
echo "You can now run: make all && ./run_tests.sh"
echo ""
