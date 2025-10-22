// kmeans_hybrid.cpp
// Versão HÍBRIDA MPI + OpenMP de K-Means
//
// ORIGINAL CODE SOURCE: Custom implementation based on standard K-means
// algorithm https://en.wikipedia.org/wiki/K-means_clustering
//
// TEMPOS DE EXECUÇÃO - MÁQUINA LOCAL:
// Hardware: Intel Core i5-11300H (4 cores físicos, 8 threads lógicos)
// Sistema: Ubuntu 24.04
// Dataset: MNIST train (60000 pontos, 785 dimensões), K=10, max_iter=20
//
// 1 processo × 4 threads:  5.723 segundos  (speedup: 1.87x, eficiência: 47%)
// 2 processos × 2 threads: 2.964 segundos  (speedup: 3.62x, eficiência: 90%)
// 4 processos × 1 thread:  3.240 segundos  (speedup: 3.31x, eficiência: 83%)
//
// OBSERVAÇÕES:
// - Melhor desempenho: 2 processos × 2 threads (2.964s)
// - Abordagem híbrida superou OpenMP puro (3.380s com 4 threads)
// - Boa eficiência com 2×2 mostra benefício da combinação MPI+OpenMP
// - Testes realizados em máquina local devido à indisponibilidade do servidor
// PARCODE
//
// Compilação: mpicc -fopenmp -O3 -o kmeans_hybrid kmeans_hybrid.cpp -std=c++17
// -lstdc++ -lm
//         ou: mpic++ -fopenmp -O3 -o kmeans_hybrid kmeans_hybrid.cpp -std=c++17
//
// Execução exemplos:
//   export OMP_NUM_THREADS=4 && mpirun -np 1 ./kmeans_hybrid mnist_train.csv 10
//   20 export OMP_NUM_THREADS=2 && mpirun -np 2 ./kmeans_hybrid mnist_train.csv
//   10 20 export OMP_NUM_THREADS=1 && mpirun -np 4 ./kmeans_hybrid
//   mnist_train.csv 10 20

#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <mpi.h>
#include <omp.h>
#include <random>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>

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
  if (!in)
    throw runtime_error("Cannot open file: " + path);

  mat data;
  string line;

  while (getline(in, line)) {
    if (line.empty())
      continue;
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

// ====== VERSÃO HÍBRIDA MPI + OpenMP ======
// MPI: Distribuição de dados entre processos (paralelismo distribuído)
// OpenMP: Paralelização dentro de cada processo (paralelismo compartilhado)
KMeansResult kmeans_hybrid(const mat &local_data, int K, int max_iter,
                           mt19937 &rng, size_t global_N) {

  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  size_t local_N = local_data.size();
  size_t D = local_data[0].size();

  if (local_N == 0)
    throw runtime_error("Empty local dataset");

  // ====== MUDANÇA MPI: Inicialização dos centróides (somente rank 0) ======
  // Apenas o processo 0 escolhe os centróides iniciais
  mat centroids;
  if (rank == 0) {
    unordered_set<int> chosen;
    uniform_int_distribution<int> uid(0, (int)global_N - 1);
    centroids.reserve(K);
    while ((int)centroids.size() < K) {
      int idx = uid(rng);
      if (chosen.insert(idx).second && idx < (int)local_N) {
        centroids.push_back(local_data[idx]);
      }
    }
  }

  // ====== MUDANÇA MPI: Broadcast dos centróides para todos os processos ======
  // Todos os processos precisam ter os mesmos centróides
  vector<double> flat_centroids(K * D);
  if (rank == 0) {
    for (int k = 0; k < K; ++k) {
      copy(centroids[k].begin(), centroids[k].end(),
           flat_centroids.begin() + k * D);
    }
  }

  MPI_Bcast(flat_centroids.data(), K * D, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  // Reconstruir centróides em todos os processos
  centroids.resize(K, vec(D));
  for (int k = 0; k < K; ++k) {
    copy(flat_centroids.begin() + k * D, flat_centroids.begin() + (k + 1) * D,
         centroids[k].begin());
  }
  // ====== FIM DA MUDANÇA MPI ======

  vector<int> local_labels(local_N, -1);
  int iter;

  for (iter = 0; iter < max_iter; ++iter) {
// ====== MUDANÇA HÍBRIDA: Assignment step com MPI + OpenMP ======
// OpenMP: Paralelizar cálculo de distâncias dentro do processo
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < local_N; ++i) {
      double best = numeric_limits<double>::infinity();
      int best_idx = -1;

      for (int k = 0; k < K; ++k) {
        double d = dist2(local_data[i], centroids[k]);
        if (d < best) {
          best = d;
          best_idx = k;
        }
      }
      local_labels[i] = best_idx;
    }
    // ====== FIM DA MUDANÇA HÍBRIDA ======

    // ====== MUDANÇA HÍBRIDA: Update step com OpenMP + MPI ======
    // Cada processo calcula somas locais em paralelo
    mat local_sum(K, vec(D, 0.0));
    vector<int> local_count(K, 0);

#pragma omp parallel
    {
      // Arrays privados por thread
      mat thread_sum(K, vec(D, 0.0));
      vector<int> thread_count(K, 0);

// OpenMP: Cada thread acumula suas somas
#pragma omp for schedule(static)
      for (size_t i = 0; i < local_N; ++i) {
        int cluster = local_labels[i];
        for (size_t d = 0; d < D; ++d) {
          thread_sum[cluster][d] += local_data[i][d];
        }
        thread_count[cluster]++;
      }

// Redução entre threads do mesmo processo
#pragma omp critical
      {
        for (int k = 0; k < K; ++k) {
          for (size_t d = 0; d < D; ++d) {
            local_sum[k][d] += thread_sum[k][d];
          }
          local_count[k] += thread_count[k];
        }
      }
    }

    // ====== MUDANÇA MPI: Redução global entre processos ======
    // Agregar somas e contagens de todos os processos
    mat global_sum(K, vec(D, 0.0));
    vector<int> global_count(K, 0);

    for (int k = 0; k < K; ++k) {
      MPI_Allreduce(local_sum[k].data(), global_sum[k].data(), D, MPI_DOUBLE,
                    MPI_SUM, MPI_COMM_WORLD);
    }

    MPI_Allreduce(local_count.data(), global_count.data(), K, MPI_INT, MPI_SUM,
                  MPI_COMM_WORLD);
    // ====== FIM DA MUDANÇA MPI ======

    // Atualizar centróides e verificar convergência
    bool converged_local = true;
    for (int k = 0; k < K; ++k) {
      if (global_count[k] == 0)
        continue;

      vec updated(D);
      for (size_t d = 0; d < D; ++d) {
        updated[d] = global_sum[k][d] / global_count[k];
      }

      if (dist2(updated, centroids[k]) > 1e-12) {
        converged_local = false;
      }

      centroids[k] = move(updated);
    }

    // ====== MUDANÇA MPI: Sincronizar convergência entre processos ======
    int converged_int = converged_local ? 1 : 0;
    int converged_global;

    MPI_Allreduce(&converged_int, &converged_global, 1, MPI_INT, MPI_MIN,
                  MPI_COMM_WORLD);

    if (converged_global)
      break;
    // ====== FIM DA MUDANÇA MPI ======
  }

  return {centroids, local_labels, iter + 1};
}

int main(int argc, char **argv) {
  // ====== MUDANÇA MPI: Inicializar MPI ======
  MPI_Init(&argc, &argv);

  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  // ====== FIM DA MUDANÇA MPI ======

  if (argc < 4) {
    if (rank == 0) {
      cerr << "Usage: " << argv[0] << " data.csv K max_iter [seed]\n";
    }
    MPI_Finalize();
    return 1;
  }

  string path = argv[1];
  int K = atoi(argv[2]);
  int max_iter = atoi(argv[3]);
  unsigned seed = (argc >= 5) ? (unsigned)atoi(argv[4])
                              : (unsigned)chrono::high_resolution_clock::now()
                                    .time_since_epoch()
                                    .count();

  if (rank == 0) {
    int num_threads = omp_get_max_threads();
    cerr << "Hybrid MPI+OpenMP: " << size << " processes × " << num_threads
         << " threads\n";
  }

  // ====== MUDANÇA MPI: Rank 0 carrega dados completos ======
  mat data;
  size_t N = 0;
  size_t D = 0;

  if (rank == 0) {
    cerr << "Loading data from " << path << "...\n";
    data = load_csv(path);
    N = data.size();
    D = data[0].size();
    cerr << "Loaded " << N << " points with dimensionality " << D << "\n";
  }

  // Broadcast N e D para todos os processos
  MPI_Bcast(&N, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
  MPI_Bcast(&D, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
  // ====== FIM DA MUDANÇA MPI ======

  // ====== MUDANÇA MPI: Distribuir dados entre processos (Scatter) ======
  size_t chunk = N / size;
  size_t start = rank * chunk;
  size_t end = (rank == size - 1) ? N : start + chunk;
  size_t local_N = end - start;

  // Preparar buffers
  vector<double> sendbuf;
  if (rank == 0) {
    sendbuf.reserve(N * D);
    for (auto &row : data) {
      sendbuf.insert(sendbuf.end(), row.begin(), row.end());
    }
  }

  vector<double> localbuf(local_N * D);

  // Calcular contagens e deslocamentos para Scatterv
  vector<int> counts(size), displs(size);
  for (int r = 0; r < size; ++r) {
    size_t s = r * chunk;
    size_t e = (r == size - 1) ? N : s + chunk;
    counts[r] = (e - s) * D;
    displs[r] = s * D;
  }

  MPI_Scatterv(rank == 0 ? sendbuf.data() : nullptr, counts.data(),
               displs.data(), MPI_DOUBLE, localbuf.data(), local_N * D,
               MPI_DOUBLE, 0, MPI_COMM_WORLD);

  // Reconstruir local_data
  mat local_data(local_N, vec(D));
  for (size_t i = 0; i < local_N; ++i) {
    copy(localbuf.begin() + i * D, localbuf.begin() + (i + 1) * D,
         local_data[i].begin());
  }
  // ====== FIM DA MUDANÇA MPI ======

  mt19937 rng(seed + rank); // Seed diferente por processo

  // Timing
  MPI_Barrier(MPI_COMM_WORLD);
  auto start_time = chrono::high_resolution_clock::now();

  KMeansResult res = kmeans_hybrid(local_data, K, max_iter, rng, N);

  MPI_Barrier(MPI_COMM_WORLD);
  auto end_time = chrono::high_resolution_clock::now();

  double elapsed = chrono::duration<double>(end_time - start_time).count();

  if (rank == 0) {
    cerr << "K-Means finished in " << res.iterations << " iterations.\n";
    cerr << "Elapsed time: " << fixed << setprecision(3) << elapsed
         << " seconds\n";

    // Imprimir centróides
    cout << "Final centroids (first 5 dimensions):\n";
    for (int k = 0; k < K; ++k) {
      cout << "Cluster " << k << ": ";
      for (size_t d = 0; d < min((size_t)5, res.centroids[k].size()); ++d) {
        cout << fixed << setprecision(4) << res.centroids[k][d] << " ";
      }
      cout << "\n";
    }
  }

  // ====== MUDANÇA MPI: Finalizar MPI ======
  MPI_Finalize();
  // ====== FIM DA MUDANÇA MPI ======

  return 0;
}
