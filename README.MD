# K-Means Clustering - Parallel Implementation (CPU and GPU)

Parallel computing project implementing K-Means clustering algorithm with different parallelization strategies.

## Part 1: CPU Parallelization (OpenMP and MPI)

Implementations:
- **Sequential**: Baseline for comparison
- **OpenMP**: Thread-based parallelization
- **MPI + OpenMP**: Hybrid approach with distributed processes and shared memory

## Part 2: GPU Parallelization (CUDA)

Implementation:
- **CUDA**: GPU acceleration using NVIDIA CUDA

**Note**: OpenMP GPU offloading implementation is included but requires nvptx-none compiler support.

## Hardware Specifications

### CPU (Part 1)
- Processor: Intel i5
- Cores: 6 physical / 12 logical threads

### GPU (Part 2)
- Model: NVIDIA GeForce GTX 1080
- Compute Capability: 6.1
- Memory: 8GB
- CUDA Cores: 2560

## Dataset

**MNIST Training Set**
- 60,000 handwritten digit images
- 784 dimensions per image (28x28 pixels)
- Format: CSV

Download:
```bash
./download_mnist.sh
```

## Compilation

### CPU Implementations (Part 1)
```bash
make all        # Compile all CPU versions
make seq        # Sequential only
make omp        # OpenMP only
make hybrid     # MPI+OpenMP only
```

### GPU Implementation (Part 2)
```bash
make cuda       # Compile CUDA version
```

### Requirements

**CPU:**
- g++ with C++17 support
- OpenMP (included in g++)
- OpenMPI (for hybrid version)

**GPU:**
- NVIDIA CUDA Toolkit (>=12.0)
- Compatible NVIDIA driver
- g++ with GPU offloading support (for OpenMP GPU - optional)

Installation on Ubuntu:
```bash
sudo apt update
sudo apt install build-essential openmpi-bin libopenmpi-dev
sudo apt install nvidia-cuda-toolkit  # For GPU versions
```

## Execution

### CPU Versions

**Sequential:**
```bash
./kmeans_seq mnist_train.csv 10 20
```

**OpenMP (4 threads):**
```bash
export OMP_NUM_THREADS=4
./kmeans_omp mnist_train.csv 10 20
```

**Hybrid (2 processes x 2 threads):**
```bash
export OMP_NUM_THREADS=2
mpirun -np 2 ./kmeans_hybrid mnist_train.csv 10 20
```

### GPU Version

**CUDA:**
```bash
./kmeans_cuda mnist_train.csv 10 20
```

### Parameters
- **File**: `mnist_train.csv` - input dataset
- **K**: `10` - number of clusters
- **max_iter**: `20` - maximum iterations
- **seed** (optional): random seed for reproducibility

## Automated Testing

### CPU Tests
```bash
./run_tests_local.sh   # Test all CPU configurations
```

### GPU Tests
```bash
./run_gpu_tests.sh     # Test CUDA (multiple runs for average)
```

### Performance Comparison
```bash
python3 compare_results.py   # Compare CPU vs GPU results
```

## Results and Analysis

The test scripts generate:
- `RESUMO_PERFORMANCE.CSV` - CPU results (Part 1)
- `results_gpu_*.csv` - GPU results (Part 2)
- `comparison_summary.csv` - Complete CPU vs GPU comparison

The comparison script shows:
- Execution times
- Speedups vs sequential version
- GPU vs best CPU comparison
- Data transfer overhead
- Parallelization efficiency

## Project Structure

```
tp_paralela01/
├── kmeans_sequential.cpp    # Sequential version
├── kmeans_omp.cpp            # OpenMP CPU
├── kmeans_hybrid.cpp         # MPI+OpenMP
├── kmeans_cuda.cu            # CUDA GPU
├── kmeans_omp_gpu.cpp        # OpenMP GPU (requires nvptx)
├── Makefile                  # Build system
├── run_tests_local.sh        # CPU tests
├── run_gpu_tests.sh          # GPU tests
├── compare_results.py        # Analysis tool
├── download_mnist.sh         # Dataset download
└── README.MD                 # This file
```

## Implementation Details

### OpenMP GPU (`kmeans_omp_gpu.cpp`)
- Line ~120: `#pragma omp target enter data` - Transfer data to GPU
- Line ~135: `#pragma omp target teams distribute parallel for` - Parallel assignment
- Line ~145: Distance calculations on GPU
- Line ~160: `#pragma omp target exit data` - Retrieve results

**Note**: Requires compiler with nvptx offloading support. See OPENMP_GPU_NOTE.txt for details.

### CUDA (`kmeans_cuda.cu`)
- Line ~67: `assign_kernel` - Parallel assignment kernel  
- Line ~75: `reduce_kernel` - GPU-side reduction using atomics
- Line ~84: `update_kernel` - Centroid update on GPU
- All computation performed on GPU to minimize CPU-GPU transfers

## Performance Results

### Summary (MNIST: 60K points, 784 dimensions, K=10)

| Implementation | Time (s) | Speedup vs Sequential | Notes |
|----------------|----------|----------------------|-------|
| Sequential | 10.722 | 1.00x | Baseline |
| OpenMP 4T | 3.380 | 3.17x | Best single-node |
| Hybrid 2x2 | 2.964 | 3.62x | Best overall CPU |
| CUDA (GPU) | 6.408 | 1.67x | GPU implementation |

### Analysis

**CPU Performance:**
- OpenMP scales well up to 4 threads (3.17x speedup)
- Hybrid MPI+OpenMP achieves best performance (3.62x)
- Hyperthreading shows diminishing returns beyond 4 threads

**GPU Performance:**
- CUDA shows 1.67x speedup vs sequential
- Slower than optimized CPU implementations for this problem
- Atomic operations in reduction create contention bottleneck
- Small K (10 clusters) limits GPU parallelism

**Why CPU Outperforms GPU:**
- K-Means with small K has limited parallelism opportunities
- 10 clusters fit in CPU cache, enabling fast sequential access
- Atomic operations serialize GPU threads during reduction
- 785 dimensions too large for efficient shared memory usage

**When GPU Would Excel:**
- Larger K (50-1000 clusters): more parallelism, less contention
- Smaller dimensions (<100): fits in shared memory
- Different algorithms (DBSCAN, hierarchical) may be more GPU-friendly

## Troubleshooting

### OpenMP GPU compilation fails
```bash
# Check offloading support
g++ -fopenmp -foffload=nvptx-none --version
# If unavailable, use CUDA implementation instead
```

### CUDA: "no kernel image available"
```bash
# Recompile with correct architecture
nvcc -O3 -arch=sm_61 -o kmeans_cuda kmeans_cuda.cu
# Adjust sm_XX based on your GPU's compute capability
```

### GPU not detected
```bash
nvidia-smi  # Check if GPU is visible
# If fails, check NVIDIA driver installation
```

## References

- K-Means Algorithm: https://en.wikipedia.org/wiki/K-means_clustering
- MNIST Dataset: http://yann.lecun.com/exdb/mnist/
- OpenMP Specification: https://www.openmp.org/specifications/
- CUDA Programming Guide: https://docs.nvidia.com/cuda/

## Authors

Developed for Parallel Computing course.

## License

Educational code - Free for academic use.
