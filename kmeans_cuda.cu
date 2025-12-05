// kmeans_cuda_gpu_reduce.cu
// CUDA with GPU-side reduction - NO CPU involvement in update step

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cmath>
#include <random>
#include <chrono>
#include <unordered_set>
#include <iomanip>
#include <cuda_runtime.h>

using namespace std;

#define CUDA_CHECK(call) do { cudaError_t err = call; if (err != cudaSuccess) { \
    cerr << "CUDA error: " << cudaGetErrorString(err) << "\n"; exit(EXIT_FAILURE); } } while (0)

// Parse CSV
static bool parse_csv_line(const string &line, vector<float> &out) {
    out.clear();
    string cur;
    for (size_t i = 0; i <= line.size(); ++i) {
        if (i == line.size() || line[i] == ',') {
            if (!cur.empty()) {
                try { out.push_back(stof(cur)); } catch (...) { return false; }
                cur.clear();
            } else {
                out.push_back(0.0f);
            }
        } else if (!isspace((unsigned char)line[i])) {
            cur.push_back(line[i]);
        }
    }
    return !out.empty();
}

vector<vector<float>> load_csv(const string &path) {
    ifstream in(path);
    if (!in) throw runtime_error("Cannot open file: " + path);
    vector<vector<float>> data;
    string line;
    while (getline(in, line)) {
        if (line.empty()) continue;
        vector<float> row;
        if (parse_csv_line(line, row))
            data.push_back(move(row));
    }
    return data;
}

// Assignment kernel  
__global__ void assign_kernel(const float *data, const float *centroids, int *labels, 
                               size_t N, size_t D, int K) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x) {
        float best = 1e30f;
        int best_k = -1;
        for (int k = 0; k < K; ++k) {
            float d = 0.0f;
            for (size_t dim = 0; dim < D; ++dim) {
                float diff = data[i * D + dim] - centroids[k * D + dim];
                d += diff * diff;
            }
            if (d < best) {
                best = d;
                best_k = k;
            }
        }
        labels[i] = best_k;
    }
}

// GPU reduction - accumulate into cluster sums
__global__ void reduce_kernel(const float *data, const int *labels, float *sums, int *counts,
                               size_t N, size_t D, int K) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x) {
        int k = labels[i];
        for (size_t d = 0; d < D; ++d) {
            atomicAdd(&sums[k * D + d], data[i * D + d]);
        }
        atomicAdd(&counts[k], 1);
    }
}

// Update centroids kernel
__global__ void update_kernel(const float *sums, const int *counts, float *centroids,
                               size_t D, int K) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k < K && counts[k] > 0) {
        for (size_t d = 0; d < D; ++d) {
            centroids[k * D + d] = sums[k * D + d] / counts[k];
        }
    }
}

int main(int argc, char **argv) {
    if (argc < 4) {
        cerr << "Usage: " << argv[0] << " data.csv K max_iter [seed]\n";
        return 1;
    }
    
    string path = argv[1];
    int K = atoi(argv[2]);
    int max_iter = atoi(argv[3]);
    unsigned seed = (argc >= 5) ? (unsigned)atoi(argv[4]) 
                                 : (unsigned)chrono::high_resolution_clock::now().time_since_epoch().count();
    
    cerr << "Loading data from " << path << "...\n";
    auto data = load_csv(path);
    size_t N = data.size();
    size_t D = data[0].size();
    cerr << "Loaded " << N << " points with dimensionality " << D << "\n";
    
    // Initialize centroids
    mt19937 rng(seed);
    vector<vector<float>> centroids;
    unordered_set<int> chosen;
    uniform_int_distribution<int> uid(0, (int)N - 1);
    while ((int)centroids.size() < K) {
        int idx = uid(rng);
        if (chosen.insert(idx).second) {
            centroids.push_back(data[idx]);
        }
    }
    
    // Flatten data
    float *data_flat = new float[N * D];
    for (size_t i = 0; i < N; ++i)
        for (size_t d = 0; d < D; ++d)
            data_flat[i * D + d] = data[i][d];
    
    float *centroids_flat = new float[K * D];
    for (int k = 0; k < K; ++k)
        for (size_t d = 0; d < D; ++d)
            centroids_flat[k * D + d] = centroids[k][d];
    
    // GPU memory
    float *d_data, *d_centroids, *d_sums;
    int *d_labels, *d_counts;
    
    auto t0 = chrono::high_resolution_clock::now();
    CUDA_CHECK(cudaMalloc(&d_data, N * D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_centroids, K * D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_sums, K * D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_labels, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_counts, K * sizeof(int)));
    
    CUDA_CHECK(cudaMemcpy(d_data, data_flat, N * D * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_centroids, centroids_flat, K * D * sizeof(float), cudaMemcpyHostToDevice));
    auto t1 = chrono::high_resolution_clock::now();
    float transfer_time = chrono::duration<float>(t1 - t0).count();
    
    int threads = 256;
    int blocks = min((int)((N + threads - 1) / threads), 512);
    
    cerr << "Running K-Means on GPU...\n";
    auto gpu_start = chrono::high_resolution_clock::now();
    
    for (int iter = 0; iter < max_iter; ++iter) {
        // Assignment
        assign_kernel<<<blocks, threads>>>(d_data, d_centroids, d_labels, N, D, K);
        
        // Reset sums
        CUDA_CHECK(cudaMemset(d_sums, 0, K * D * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_counts, 0, K * sizeof(int)));
        
        // Reduction ON GPU
        reduce_kernel<<<blocks, threads>>>(d_data, d_labels, d_sums, d_counts, N, D, K);
        
        // Update centroids ON GPU
        int k_threads = min(K, 256);
        int k_blocks = (K + k_threads - 1) / k_threads;
        update_kernel<<<k_blocks, k_threads>>>(d_sums, d_counts, d_centroids, D, K);
        
        CUDA_CHECK(cudaGetLastError());
    }
    
    CUDA_CHECK(cudaDeviceSynchronize());
    auto gpu_end = chrono::high_resolution_clock::now();
    float gpu_time = chrono::duration<float>(gpu_end - gpu_start).count();
    
    // Copy results back
    auto t2 = chrono::high_resolution_clock::now();
    CUDA_CHECK(cudaMemcpy(centroids_flat, d_centroids, K * D * sizeof(float), cudaMemcpyDeviceToHost));
    auto t3 = chrono::high_resolution_clock::now();
    transfer_time += chrono::duration<float>(t3 - t2).count();
    
    float total_time = chrono::duration<float>(t3 - t0).count();
    
    cerr << "\nK-Means (GPU-ONLY REDUCTION) finished in " << max_iter << " iterations.\n";
    cerr << "Total time: " << fixed << setprecision(3) << total_time << " seconds\n";
    cerr << "GPU computation time: " << fixed << setprecision(3) << gpu_time << " seconds\n";
    cerr << "Data transfer time: " << fixed << setprecision(3) << transfer_time << " seconds\n";
    
    cout << "\nFinal centroids (first 5 dimensions):\n";
    for (int k = 0; k < K; ++k) {
        cout << "Cluster " << k << ": ";
        for (size_t d = 0; d < min((size_t)5, D); ++d) {
            cout << fixed << setprecision(4) << centroids_flat[k * D + d] << " ";
        }
        cout << "\n";
    }
    
    // Cleanup
    CUDA_CHECK(cudaFree(d_data));
    CUDA_CHECK(cudaFree(d_centroids));
    CUDA_CHECK(cudaFree(d_sums));
    CUDA_CHECK(cudaFree(d_labels));
    CUDA_CHECK(cudaFree(d_counts));
    delete[] data_flat;
    delete[] centroids_flat;
    
    return 0;
}
