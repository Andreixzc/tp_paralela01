===============================================================================
  TRABALHO PRÁTICO - PROGRAMAÇÃO PARALELA HÍBRIDA MPI/OPENMP
  K-MEANS CLUSTERING - MNIST DATASET
===============================================================================

AUTORES:
--------
[PREENCHER COM NOMES DOS MEMBROS DO GRUPO]

DATA: [PREENCHER]

===============================================================================
DESCRIÇÃO DA APLICAÇÃO
===============================================================================

K-Means é um algoritmo de agrupamento (clustering) não supervisionado usado
para particionar N pontos de dados em K grupos (clusters) baseado em 
similaridade. O algoritmo funciona iterativamente:

1. INICIALIZAÇÃO: Escolher K centróides aleatórios
2. ASSIGNMENT STEP: Atribuir cada ponto ao centróide mais próximo
3. UPDATE STEP: Recalcular centróides como média dos pontos atribuídos
4. REPETIR passos 2-3 até convergência ou máximo de iterações

APLICAÇÃO: Classificação de dígitos manuscritos do dataset MNIST
- 60.000 imagens de treino, 10.000 de teste
- Cada imagem: 28×28 pixels (784 dimensões)
- 10 classes (dígitos 0-9)

PARALELIZAÇÃO:
- Assignment step: Altamente paralelizável (cálculo independente por ponto)
- Update step: Requer sincronização para agregar somas

===============================================================================
CÓDIGO FONTE ORIGINAL
===============================================================================

Implementação baseada no algoritmo padrão de K-Means:
https://en.wikipedia.org/wiki/K-means_clustering

Implementação original desenvolvida pelo grupo, inspirada em:
- Scikit-learn K-Means: https://scikit-learn.org/stable/modules/clustering.html#k-means
- Tutorial K-Means em C++: https://github.com/topics/kmeans-clustering-algorithm

===============================================================================
ARQUIVOS DO PROJETO
===============================================================================

kmeans_sequential.cpp  - Versão sequencial (baseline)
kmeans_omp.cpp         - Versão OpenMP (paralelismo compartilhado)
kmeans_hybrid.cpp      - Versão MPI+OpenMP (paralelismo híbrido)
mnist_train.csv        - Dataset de treino MNIST
mnist_test.csv         - Dataset de teste MNIST (opcional)
README.txt             - Este arquivo

===============================================================================
REQUISITOS DO SISTEMA
===============================================================================

- Sistema operacional: Linux (testado em Ubuntu)
- Compilador: g++ 8.0+ ou mpic++
- OpenMP 4.5+
- MPI (OpenMPI ou MPICH)
- C++17

Servidor PARCODE (PUC Minas):
- CPU: Intel 4 núcleos
- GPU: Nvidia GT 1030 (não utilizada neste trabalho)
- Acesso via SSH: ssh a<codigo>@parcode.icei.pucminas.br

===============================================================================
INSTRUÇÕES DE COMPILAÇÃO
===============================================================================

1. VERSÃO SEQUENCIAL
--------------------
g++ -O3 -o kmeans_seq kmeans_sequential.cpp -std=c++17

2. VERSÃO OPENMP
----------------
g++ -fopenmp -O3 -o kmeans_omp kmeans_omp.cpp -std=c++17

3. VERSÃO HÍBRIDA MPI+OPENMP
-----------------------------
# Opção 1: Usando mpicc
mpicc -fopenmp -O3 -o kmeans_hybrid kmeans_hybrid.cpp -std=c++17 -lstdc++ -lm

# Opção 2: Usando mpic++ (recomendado)
mpic++ -fopenmp -O3 -o kmeans_hybrid kmeans_hybrid.cpp -std=c++17

===============================================================================
INSTRUÇÕES DE EXECUÇÃO
===============================================================================

FORMATO GERAL:
./programa <arquivo_csv> <K> <max_iter> [seed]

Parâmetros:
  - arquivo_csv: Caminho para o dataset (ex: mnist_train.csv)
  - K: Número de clusters (ex: 10 para MNIST)
  - max_iter: Número máximo de iterações (ex: 20)
  - seed: Seed aleatória opcional (para reprodutibilidade)

-------------------------------------------------------------------------------
1. VERSÃO SEQUENCIAL
--------------------
./kmeans_seq mnist_train.csv 10 20

-------------------------------------------------------------------------------
2. VERSÃO OPENMP - Testes com diferentes números de threads
------------------------------------------------------------

# 1 thread
export OMP_NUM_THREADS=1
./kmeans_omp mnist_train.csv 10 20

# 2 threads
export OMP_NUM_THREADS=2
./kmeans_omp mnist_train.csv 10 20

# 4 threads
export OMP_NUM_THREADS=4
./kmeans_omp mnist_train.csv 10 20

# 8 threads
export OMP_NUM_THREADS=8
./kmeans_omp mnist_train.csv 10 20

-------------------------------------------------------------------------------
3. VERSÃO HÍBRIDA MPI+OPENMP
-----------------------------

# 1 processo × 4 threads = 4 workers totais
export OMP_NUM_THREADS=4
mpirun -np 1 ./kmeans_hybrid mnist_train.csv 10 20

# 2 processos × 2 threads = 4 workers totais
export OMP_NUM_THREADS=2
mpirun -np 2 ./kmeans_hybrid mnist_train.csv 10 20

# 4 processos × 1 thread = 4 workers totais (MPI puro)
export OMP_NUM_THREADS=1
mpirun -np 4 ./kmeans_hybrid mnist_train.csv 10 20

===============================================================================
SCRIPT DE TESTE AUTOMATIZADO (SERVIDOR PARCODE)
===============================================================================

Criar arquivo: run_tests.sh

#!/bin/bash

echo "========================================="
echo "K-MEANS PARALELO - TESTES NO PARCODE"
echo "========================================="

DATASET="mnist_train.csv"
K=10
MAX_ITER=20
SEED=42

echo ""
echo "=== VERSÃO SEQUENCIAL ==="
./kmeans_seq $DATASET $K $MAX_ITER $SEED

echo ""
echo "=== VERSÃO OPENMP - 1 THREAD ==="
export OMP_NUM_THREADS=1
./kmeans_omp $DATASET $K $MAX_ITER $SEED

echo ""
echo "=== VERSÃO OPENMP - 2 THREADS ==="
export OMP_NUM_THREADS=2
./kmeans_omp $DATASET $K $MAX_ITER $SEED

echo ""
echo "=== VERSÃO OPENMP - 4 THREADS ==="
export OMP_NUM_THREADS=4
./kmeans_omp $DATASET $K $MAX_ITER $SEED

echo ""
echo "=== VERSÃO OPENMP - 8 THREADS ==="
export OMP_NUM_THREADS=8
./kmeans_omp $DATASET $K $MAX_ITER $SEED

echo ""
echo "=== VERSÃO HÍBRIDA - 1 PROC × 4 THREADS ==="
export OMP_NUM_THREADS=4
mpirun -np 1 ./kmeans_hybrid $DATASET $K $MAX_ITER $SEED

echo ""
echo "=== VERSÃO HÍBRIDA - 2 PROCS × 2 THREADS ==="
export OMP_NUM_THREADS=2
mpirun -np 2 ./kmeans_hybrid $DATASET $K $MAX_ITER $SEED

echo ""
echo "=== VERSÃO HÍBRIDA - 4 PROCS × 1 THREAD ==="
export OMP_NUM_THREADS=1
mpirun -np 4 ./kmeans_hybrid $DATASET $K $MAX_ITER $SEED

echo ""
echo "========================================="
echo "TESTES CONCLUÍDOS"
echo "========================================="

Para executar:
chmod +x run_tests.sh
./run_tests.sh

===============================================================================
DATASET MNIST
===============================================================================

FORMATO: CSV (Comma-Separated Values)
- Primeira coluna: Label (0-9)
- Colunas 2-785: Valores dos pixels (0-255)

O arquivo mnist_train.csv pode ser obtido em:
https://www.kaggle.com/datasets/oddrationale/mnist-in-csv

Ou gerar a partir do formato original:
https://yann.lecun.com/exdb/mnist/

NOTA: Para garantir tempo de execução > 10 segundos no parcode, usar:
- Dataset completo: mnist_train.csv (60.000 pontos)
- K = 10 clusters
- max_iter = 20 iterações
- Se ainda for muito rápido, aumentar max_iter ou usar K maior

===============================================================================
ANÁLISE DE DESEMPENHO
===============================================================================

MÉTRICAS A AVALIAR:
1. Tempo de execução (segundos)
2. Speedup = Tempo_sequencial / Tempo_paralelo
3. Eficiência = Speedup / Número_de_workers

RESULTADOS ESPERADOS:
- OpenMP: Speedup quase linear até 4 threads (número de cores físicos)
- OpenMP com 8 threads: Menor eficiência devido a hyperthreading
- Híbrido: Melhor balanceamento entre comunicação MPI e paralelismo OpenMP

===============================================================================
PROBLEMAS COMUNS E SOLUÇÕES
===============================================================================

PROBLEMA: "Cannot open file"
SOLUÇÃO: Verificar caminho do dataset. Usar caminho absoluto se necessário.

PROBLEMA: Tempo de execução < 10 segundos
SOLUÇÃO: Aumentar max_iter ou usar dataset maior

PROBLEMA: MPI não encontrado
SOLUÇÃO: Instalar OpenMPI: sudo apt install openmpi-bin libopenmpi-dev

PROBLEMA: OpenMP não funciona
SOLUÇÃO: Verificar flag -fopenmp na compilação

PROBLEMA: Resultados diferentes entre execuções
SOLUÇÃO: Usar mesma seed: ./kmeans_seq mnist_train.csv 10 20 42

===============================================================================
REFERÊNCIAS
===============================================================================

[1] MacQueen, J. B. (1967). "Some Methods for classification and Analysis 
    of Multivariate Observations". Proceedings of 5th Berkeley Symposium 
    on Mathematical Statistics and Probability. pp. 281–297.

[2] OpenMP Architecture Review Board. "OpenMP Application Programming 
    Interface Version 4.5". November 2015.

[3] MPI Forum. "MPI: A Message-Passing Interface Standard Version 3.1". 
    June 2015.

[4] LeCun, Y., Cortes, C., and Burges, C. J. C. "MNIST handwritten digit 
    database". http://yann.lecun.com/exdb/mnist/

===============================================================================
CONTATO
===============================================================================

Para dúvidas sobre o código:
[PREENCHER COM E-MAILS DO GRUPO]

Professor:
[PREENCHER COM NOME E E-MAIL DO PROFESSOR]

===============================================================================
