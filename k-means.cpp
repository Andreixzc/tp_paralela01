// kmeans.cpp
// Implementação sequencial de K-Means (C++17)
// Uso: ./kmeans data.csv K max_iter seed
// - data.csv: CSV onde cada linha é um exemplo, colunas separadas por vírgula
// (float)
// - K: número de clusters
// - max_iter: número máximo de iterações
// - seed: seed aleatória (opcional)

// ------------------------------
// NOTAS SOBRE PARALLELIZAÇÃO (comentários exigidos pelo enunciado):
// - Para OpenMP: paralelizar o loop que calcula a distância de cada ponto para
// cada centróide
//   e a acumulação de somas para novos centróides. Usar #pragma omp parallel
//   for com redução (ou arrays privativos + redução manual). Também paralelizar
//   a etapa de atribuição de pontos se necessário.
// - Para MPI: distribuir as linhas (pontos) entre os ranks (scatter or
// read-split).
//   Cada rank calcula soma local e contagem por centróide; em seguida usar
//   MPI_Allreduce para reduzir somas e contagens e atualizar centróides em
//   todos os ranks. A etapa de inicialização dos centróides pode ser feita pelo
//   rank 0 e depois broadcast com MPI_Bcast.
// - No código abaixo eu marco explicitamente (COMENTADO) onde as diretivas
// OpenMP e chamadas
//   MPI deveriam ser inseridas.
// ------------------------------

// clear && cmake -S . -B build && cmake --build build && ./build/main
// dataset/mnist_train.csv 2 5
/*
        clear
        cmake -S . -B build
        cmake --build build
        export OMP_NUM_THREADS=1
        mpirun -np 1 ./build/main dataset/mnist_train.csv 2 15
*/

#include "timer.hpp"
#include <bits/stdc++.h>

#include <mpi.h>
#include <omp.h>

using namespace std;

using vec = vector<double>;
using mat = vector<vec>;

// parse CSV line (floats separated by comma)
static bool parse_csv_line(const string &line, vec &out) {

  out.clear();
  string cur;

  for (size_t i = 0; i <= line.size(); ++i) {

    if (i == line.size() || line[i] == ',') {

      if (!cur.empty()) {

        try {
          out.push_back(stod(cur));
        }

        catch (...) {
          return false;
        }

        cur.clear();
      }

      // treat empty as 0
      else
        out.push_back(0.0);
    }

    else if (!isspace((unsigned char)line[i])) {
      cur.push_back(line[i]);
    }
  }

  return !out.empty();
}

// Load CSV into matrix (rows x cols)
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
      data.push_back(std::move(row));
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

// --- Versão híbrida MPI + OpenMP ---
KMeansResult kmeans_mpi_omp(const mat &local_data, int K, int max_iter,
                            mt19937 &rng) {

  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  size_t local_N = local_data.size();
  size_t D = local_data[0].size();

  if (local_N == 0)
    throw runtime_error("Empty dataset");

  // --- Inicialização dos centróides (somente rank 0) ---
  mat centroids;
  if (rank == 0) {
    unordered_set<int> chosen;
    uniform_int_distribution<int> uid(0, (int)local_N - 1);
    centroids.reserve(K);
    while ((int)centroids.size() < K) {
      int idx = uid(rng);
      if (chosen.insert(idx).second)
        centroids.push_back(local_data[idx]);
    }
  }

  // Broadcast dos centróides para todos os ranks
  vector<double> flat_centroids(K * D);
  if (rank == 0) {
    for (int k = 0; k < K; ++k)
      copy(centroids[k].begin(), centroids[k].end(),
           flat_centroids.begin() + k * D);
  }

  MPI_Bcast(flat_centroids.data(), K * D, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  centroids.resize(K, vec(D));
  for (int k = 0; k < K; ++k) {
    copy(flat_centroids.begin() + k * D, flat_centroids.begin() + (k + 1) * D,
         centroids[k].begin());
  }

  vector<int> local_labels(local_N, -1);
  mat local_sum(K, vec(D, 0.0));
  vector<int> local_count(K, 0);
  int iter;

  for (iter = 0; iter < max_iter; ++iter) {
    // Reset parciais
    for (int k = 0; k < K; ++k) {
      fill(local_sum[k].begin(), local_sum[k].end(), 0.0);
      local_count[k] = 0;
    }

// --- Paralelismo OpenMP (loop principal) ---
#pragma omp parallel
    {
      mat thread_sum(K, vec(D, 0.0));
      vector<int> thread_count(K, 0);

#pragma omp for schedule(static)
      for (size_t i = 0; i < local_N; ++i) {
        double best = numeric_limits<double>::infinity();
        int bi = -1;
        for (int k = 0; k < K; ++k) {
          double d = dist2(local_data[i], centroids[k]);
          if (d < best) {
            best = d;
            bi = k;
          }
        }
        local_labels[i] = bi;
        for (size_t d = 0; d < D; ++d)
          thread_sum[bi][d] += local_data[i][d];
        thread_count[bi]++;
      }

// Redução entre threads
#pragma omp critical
      {
        for (int k = 0; k < K; ++k) {
          for (size_t d = 0; d < D; ++d)
            local_sum[k][d] += thread_sum[k][d];
          local_count[k] += thread_count[k];
        }
      }
    }

    // --- Redução global MPI ---
    mat global_sum(K, vec(D, 0.0));
    vector<int> global_count(K, 0);

    for (int k = 0; k < K; ++k) {
      MPI_Allreduce(local_sum[k].data(), global_sum[k].data(), D, MPI_DOUBLE,
                    MPI_SUM, MPI_COMM_WORLD);
    }

    MPI_Allreduce(local_count.data(), global_count.data(), K, MPI_INT, MPI_SUM,
                  MPI_COMM_WORLD);

    // --- Atualiza centróides ---
    bool converged_local = true;
    for (int k = 0; k < K; ++k) {
      if (global_count[k] == 0)
        continue;
      vec updated(D);
      for (size_t d = 0; d < D; ++d)
        updated[d] = global_sum[k][d] / global_count[k];
      if (dist2(updated, centroids[k]) > 1e-12)
        converged_local = false;
      centroids[k].swap(updated);
    }

    // --- Checa convergência global ---
    int converged_int = converged_local ? 1 : 0;
    int converged_global;

    MPI_Allreduce(&converged_int, &converged_global, 1, MPI_INT, MPI_MIN,
                  MPI_COMM_WORLD);

    if (converged_global)
      break;
  }

  // Coleta de resultados (apenas labels, se necessário)
  vector<int> global_labels;
  if (rank == 0)
    global_labels.resize(local_N);
  vector<int> recv_counts(size), displs(size);

  for (int r = 0; r < size; ++r) {
    recv_counts[r] = (r == size - 1) ? N - r * chunk : chunk;
    displs[r] = r * chunk;
  }

  MPI_Gatherv(local_labels.data(), local_N, MPI_INT, global_labels.data(),
              recv_counts.data(), displs.data(), MPI_INT, 0, MPI_COMM_WORLD);

  return {centroids, global_labels, iter + 1};
}

int main(int argc, char **argv) {

  // NOTE: WTF?
  // ios::sync_with_stdio(false);
  // cin.tie(nullptr);

  // WARNING: Corrigir a main para nao carregar n vezes

  // Inicializa o MPI
  MPI_Init(&argc, &argv);

  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD,
                &rank); // rank = ID do processo atual (0, 1, 2, …)
  MPI_Comm_size(MPI_COMM_WORLD, &size); // size = número total de processos

  if (argc < 4) {
    cerr << "Usage: " << argv[0] << " data.csv K max_iter [seed]\n";
    return 1;
  }

  string path = argv[1];

  int K = atoi(argv[2]);
  int max_iter = atoi(argv[3]);
  unsigned seed = (argc >= 5) ? (unsigned)atoi(argv[4])
                              : (unsigned)chrono::high_resolution_clock::now()
                                    .time_since_epoch()
                                    .count();

  mat data;
  mat local_data;

  cerr << "Loading data...\n";
  if (rank == 0)
    data = load_csv(path);
  cerr << "Loaded " << data.size() << " points with dimensionality "
       << data[0].size() << "\n";

  size_t N = (rank == 0) ? data.size() : 0;
  MPI_Bcast(&N, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);

  // Calcula tamanhos locais
  size_t chunk = N / size;
  size_t start = rank * chunk;
  size_t end = (rank == size - 1) ? N : start + chunk;
  size_t local_N = end - start;

  // Prepara buffers
  size_t D;
  if (rank == 0)
    D = data[0].size();
  MPI_Bcast(&D, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);

  vector<double> sendbuf;
  if (rank == 0) {
    sendbuf.reserve(N * D);
    for (auto &row : data) {
      sendbuf.insert(sendbuf.end(), row.begin(), row.end());
    }
  }

  vector<double> localbuf(local_N * D);

  // Scatter real
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

  // Reconstrói local_data
  local_data.resize(local_N, vec(D));
  for (size_t i = 0; i < local_N; ++i) {
    copy(localbuf.begin() + i * D, localbuf.begin() + (i + 1) * D,
         local_data[i].begin());
  }

  mt19937 rng(seed);

  Timer timer;

  // KMeansResult res = kmeans_sequential(data, K, max_iter, rng);
  KMeansResult res;

  timer.start();
  for (int i = 0; i < 5; i++) {
    res = kmeans_mpi_omp(local_data, K, max_iter, rng);
  }
  timer.stop();

  cerr << "KMeans finished in " << res.iterations << " iterations.\n";
  cerr << "Elapsed time (seq): " << timer.result() / 5 << "s\n";

  // // print centroids (first few elements) to stdout
  // cout.setf(std::ios::fixed); cout << setprecision(6);

  // for (int k = 0; k < K; ++k) {
  //     for (size_t d = 0; d < res.centroids[k].size(); ++d) {
  //         if (d) cout << ',';
  //         cout << res.centroids[k][d];
  //     }
  //     cout << '\n';
  // }

  // Finaliza
  MPI_Finalize();

  return 0;
}
