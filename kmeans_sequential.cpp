// kmeans_sequential.cpp
// Versão SEQUENCIAL de K-Means para baseline
// 
// ORIGINAL CODE SOURCE: Custom implementation based on standard K-means algorithm
// https://en.wikipedia.org/wiki/K-means_clustering
//
// TEMPOS DE EXECUÇÃO NO SERVIDOR PARCODE (4 cores Intel):
// Dataset: MNIST train (60000 pontos, 784 dimensões), K=10, max_iter=20
// 
// VERSÃO SEQUENCIAL:
// Tempo: [A PREENCHER APÓS TESTES] segundos
//
// Compilação: g++ -O3 -o kmeans_seq kmeans_sequential.cpp -std=c++17
// Execução: ./kmeans_seq mnist_train.csv 10 20

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

// Parse CSV line (floats separated by comma)
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

// Load CSV into matrix
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
inline double dist2(const vec &a, const vec &b) {
    double s = 0.0;
    size_t n = a.size();
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
};

// Versão SEQUENCIAL (sem paralelização)
KMeansResult kmeans_sequential(const mat &data, int K, int max_iter, mt19937 &rng) {
    size_t N = data.size();
    size_t D = data[0].size();
    
    if (N == 0) throw runtime_error("Empty dataset");
    
    // Inicialização: escolher K pontos aleatórios como centróides iniciais
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
    int iter;
    
    for (iter = 0; iter < max_iter; ++iter) {
        // Assignment step: atribuir cada ponto ao centróide mais próximo
        for (size_t i = 0; i < N; ++i) {
            double best = numeric_limits<double>::infinity();
            int best_idx = -1;
            
            for (int k = 0; k < K; ++k) {
                double d = dist2(data[i], centroids[k]);
                if (d < best) {
                    best = d;
                    best_idx = k;
                }
            }
            labels[i] = best_idx;
        }
        
        // Update step: recalcular centróides como média dos pontos atribuídos
        mat new_centroids(K, vec(D, 0.0));
        vector<int> counts(K, 0);
        
        for (size_t i = 0; i < N; ++i) {
            int cluster = labels[i];
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
            
            if (dist2(new_centroids[k], centroids[k]) > 1e-12) {
                converged = false;
            }
        }
        
        centroids = move(new_centroids);
        
        if (converged) break;
    }
    
    return {centroids, labels, iter + 1};
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
    KMeansResult res = kmeans_sequential(data, K, max_iter, rng);
    auto end = chrono::high_resolution_clock::now();
    
    double elapsed = chrono::duration<double>(end - start).count();
    
    cerr << "K-Means finished in " << res.iterations << " iterations.\n";
    cerr << "Elapsed time: " << fixed << setprecision(3) << elapsed << " seconds\n";
    
    // Opcional: imprimir centróides
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
