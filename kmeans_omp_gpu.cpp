// kmeans_omp_gpu.cpp
// Versão OpenMP GPU OFFLOADING de K-Means
// 
// BASED ON: kmeans_sequential.cpp from this project
// MODIFICATIONS: Added OpenMP target directives for GPU execution
//
// HARDWARE: NVIDIA GeForce GTX 1080 (Compute 6.1, 8GB)
// TEMPOS DE EXECUÇÃO (GTX 1080, Local):
// Dataset: MNIST train (60000 pontos, 784 dimensões), K=10, max_iter=20
// 
// VERSÃO OpenMP GPU:
// Tempo total: [A PREENCHER] segundos
// Tempo computação GPU: [A PREENCHER] segundos
// Tempo transferência: [A PREENCHER] segundos
// Speedup vs Sequential: [A PREENCHER]x
//
// Compilação: g++ -O3 -fopenmp -foffload=nvptx-none -o kmeans_omp_gpu kmeans_omp_gpu.cpp -std=c++17
// Execução: ./kmeans_omp_gpu mnist_train.csv 10 20

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cmath>
#include <limits>
#include <random>
#include <chrono>
#include <unordered_set>
#include <iomanip>

using namespace std;
using vec = vector<double>;
using mat = vector<vec>;

// Parse CSV line
static bool parse_csv_line(const string &line, vec &out) {
    out.clear();
    string cur;
    
    for (size_t i = 0; i <= line.size(); ++i) {
        if (i == line.size() || line[i] == ',') {
            if (!cur.empty()) {
                try {
                    out.push_back(stod(cur));
                } catch (...) {
                    return false;
                }
                cur.clear();
            } else {
                out.push_back(0.0);
            }
        } else if (!isspace((unsigned char)line[i])) {
            cur.push_back(line[i]);
        }
    }
    return !out.empty();
}

// Load CSV
mat load_csv(const string &path) {
    ifstream in(path);
    if (!in) throw runtime_error("Cannot open file: " + path);
    
    mat data;
    string line;
    
    while (getline(in, line)) {
        if (line.empty()) continue;
        vec row;
        if (parse_csv_line(line, row))
            data.push_back(move(row));
    }
    
    return data;
}

// Euclidean squared distance
inline double dist2(const double *a, const double *b, size_t n) {
    double s = 0.0;
    for (size_t i = 0; i < n; ++i) {
        double d = a[i] - b[i];
        s += d * d;
    }
    return s;
}

struct KMeansResult {
    mat centroids;
    vector<int> labels;
    int iterations;
    double gpu_time;      // MUDANÇA: Tempo de computação na GPU
    double transfer_time; // MUDANÇA: Tempo de transferência de dados
};

// VERSÃO OpenMP GPU OFFLOADING
// MUDANÇA: Adicionado OpenMP target directives para execução na GPU
KMeansResult kmeans_omp_gpu(const mat &data, int K, int max_iter, mt19937 &rng) {
    size_t N = data.size();
    size_t D = data[0].size();
    
    if (N == 0) throw runtime_error("Empty dataset");
    
    // Inicialização dos centróides (CPU)
    mat centroids;
    centroids.reserve(K);
    unordered_set<int> chosen;
    uniform_int_distribution<int> uid(0, (int)N - 1);
    
    while ((int)centroids.size() < K) {
        int idx = uid(rng);
        if (chosen.insert(idx).second) {
            centroids.push_back(data[idx]);
        }
    }
    
    vector<int> labels(N, -1);
    
    // MUDANÇA: Converter data e centroids para arrays contíguos para GPU
    // Flatten data matrix para acesso eficiente na GPU
    double *data_flat = new double[N * D];
    for (size_t i = 0; i < N; ++i) {
        for (size_t d = 0; d < D; ++d) {
            data_flat[i * D + d] = data[i][d];
        }
    }
    
    double *centroids_flat = new double[K * D];
    int *labels_arr = new int[N];
    
    double gpu_time = 0.0;
    double transfer_time = 0.0;
    int iter;
    
    // MUDANÇA: Transferir dados para GPU uma única vez
    auto t_transfer_start = chrono::high_resolution_clock::now();
    #pragma omp target enter data map(to: data_flat[0:N*D])
    auto t_transfer_end = chrono::high_resolution_clock::now();
    transfer_time += chrono::duration<double>(t_transfer_end - t_transfer_start).count();
    
    for (iter = 0; iter < max_iter; ++iter) {
        // Converter centroids para flat array
        for (int k = 0; k < K; ++k) {
            for (size_t d = 0; d < D; ++d) {
                centroids_flat[k * D + d] = centroids[k][d];
            }
        }
        
        // MUDANÇA: Transferir centróides para GPU
        t_transfer_start = chrono::high_resolution_clock::now();
        #pragma omp target enter data map(to: centroids_flat[0:K*D])
        #pragma omp target enter data map(to: labels_arr[0:N])
        t_transfer_end = chrono::high_resolution_clock::now();
        transfer_time += chrono::duration<double>(t_transfer_end - t_transfer_start).count();
        
        // MUDANÇA: Assignment step na GPU
        // Cada ponto é atribuído ao centróide mais próximo em paralelo
        auto t_gpu_start = chrono::high_resolution_clock::now();
        
        #pragma omp target teams distribute parallel for map(tofrom: labels_arr[0:N])
        for (size_t i = 0; i < N; ++i) {
            double best = 1e100;  // Infinity
            int best_idx = -1;
            
            for (int k = 0; k < K; ++k) {
                // Calcular distância euclidiana quadrada
                double dist_sq = 0.0;
                for (size_t d = 0; d < D; ++d) {
                    double diff = data_flat[i * D + d] - centroids_flat[k * D + d];
                    dist_sq += diff * diff;
                }
                
                if (dist_sq < best) {
                    best = dist_sq;
                    best_idx = k;
                }
            }
            labels_arr[i] = best_idx;
        }
        
        auto t_gpu_end = chrono::high_resolution_clock::now();
        gpu_time += chrono::duration<double>(t_gpu_end - t_gpu_start).count();
        
        // MUDANÇA: Trazer labels de volta para CPU
        t_transfer_start = chrono::high_resolution_clock::now();
        #pragma omp target exit data map(from: labels_arr[0:N])
        #pragma omp target exit data map(delete: centroids_flat[0:K*D])
        t_transfer_end = chrono::high_resolution_clock::now();
        transfer_time += chrono::duration<double>(t_transfer_end - t_transfer_start).count();
        
        // Update step: CPU (mais eficiente para reduction neste caso)
        mat new_centroids(K, vec(D, 0.0));
        vector<int> counts(K, 0);
        
        for (size_t i = 0; i < N; ++i) {
            int cluster = labels_arr[i];
            for (size_t d = 0; d < D; ++d) {
                new_centroids[cluster][d] += data[i][d];
            }
            counts[cluster]++;
        }
        
        // Calcular médias e verificar convergência
        bool converged = true;
        for (int k = 0; k < K; ++k) {
            if (counts[k] == 0) continue;
            
            for (size_t d = 0; d < D; ++d) {
                new_centroids[k][d] /= counts[k];
            }
            
            double diff = 0.0;
            for (size_t d = 0; d < D; ++d) {
                double delta = new_centroids[k][d] - centroids[k][d];
                diff += delta * delta;
            }
            
            if (diff > 1e-12) {
                converged = false;
            }
        }
        
        centroids = move(new_centroids);
        
        if (converged) break;
    }
    
    // MUDANÇA: Limpar dados da GPU
    t_transfer_start = chrono::high_resolution_clock::now();
    #pragma omp target exit data map(delete: data_flat[0:N*D])
    t_transfer_end = chrono::high_resolution_clock::now();
    transfer_time += chrono::duration<double>(t_transfer_end - t_transfer_start).count();
    
    // Copiar labels finais
    for (size_t i = 0; i < N; ++i) {
        labels[i] = labels_arr[i];
    }
    
    delete[] data_flat;
    delete[] centroids_flat;
    delete[] labels_arr;
    
    return {centroids, labels, iter + 1, gpu_time, transfer_time};
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
                                 : (unsigned)chrono::high_resolution_clock::now()
                                       .time_since_epoch().count();
    
    cerr << "Loading data from " << path << "...\n";
    mat data = load_csv(path);
    cerr << "Loaded " << data.size() << " points with dimensionality " 
         << data[0].size() << "\n";
    
    mt19937 rng(seed);
    
    // Timing
    auto start = chrono::high_resolution_clock::now();
    KMeansResult res = kmeans_omp_gpu(data, K, max_iter, rng);
    auto end = chrono::high_resolution_clock::now();
    
    double elapsed = chrono::duration<double>(end - start).count();
    
    cerr << "K-Means finished in " << res.iterations << " iterations.\n";
    cerr << "Total time: " << fixed << setprecision(3) << elapsed << " seconds\n";
    cerr << "GPU computation time: " << fixed << setprecision(3) << res.gpu_time << " seconds\n";
    cerr << "Data transfer time: " << fixed << setprecision(3) << res.transfer_time << " seconds\n";
    
    // Imprimir centróides
    cout << "Final centroids (first 5 dimensions):\n";
    for (int k = 0; k < K; ++k) {
        cout << "Cluster " << k << ": ";
        for (size_t d = 0; d < min((size_t)5, res.centroids[k].size()); ++d) {
            cout << fixed << setprecision(4) << res.centroids[k][d] << " ";
        }
        cout << "\n";
    }
    
    return 0;
}
