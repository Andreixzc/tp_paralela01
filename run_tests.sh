#!/bin/bash
# run_tests.sh - Script de teste automatizado para servidor PARCODE
# Executa todas as configurações requeridas pelo trabalho

# Configurações
DATASET="mnist_train.csv"
K=10
MAX_ITER=20
SEED=42

# Verificar se dataset existe
if [ ! -f "$DATASET" ]; then
    echo "ERRO: Dataset $DATASET não encontrado!"
    echo "Baixe o MNIST dataset de: https://www.kaggle.com/datasets/oddrationale/mnist-in-csv"
    exit 1
fi

# Verificar se executáveis existem
if [ ! -f "kmeans_seq" ] || [ ! -f "kmeans_omp" ] || [ ! -f "kmeans_hybrid" ]; then
    echo "ERRO: Executáveis não encontrados!"
    echo "Execute 'make all' primeiro para compilar."
    exit 1
fi

echo "==============================================================================="
echo "  K-MEANS PARALELO - TESTES AUTOMATIZADOS NO SERVIDOR PARCODE"
echo "==============================================================================="
echo "Dataset: $DATASET"
echo "K clusters: $K"
echo "Max iterações: $MAX_ITER"
echo "Seed: $SEED"
echo "==============================================================================="
echo ""

# Criar arquivo de saída com timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT="resultados_${TIMESTAMP}.txt"

echo "Resultados serão salvos em: $OUTPUT"
echo ""

# Função para executar e registrar
run_test() {
    local name=$1
    local cmd=$2
    
    echo "==================================="
    echo "  $name"
    echo "==================================="
    echo ""
    
    # Executar e mostrar na tela
    eval $cmd
    
    # Também salvar no arquivo
    echo "=== $name ===" >> $OUTPUT
    eval $cmd 2>> $OUTPUT
    echo "" >> $OUTPUT
    
    echo ""
}

# ===== VERSÃO SEQUENCIAL =====
run_test "VERSÃO SEQUENCIAL (BASELINE)" \
         "./kmeans_seq $DATASET $K $MAX_ITER $SEED"

# ===== VERSÃO OPENMP =====
run_test "VERSÃO OPENMP - 1 THREAD" \
         "export OMP_NUM_THREADS=1 && ./kmeans_omp $DATASET $K $MAX_ITER $SEED"

run_test "VERSÃO OPENMP - 2 THREADS" \
         "export OMP_NUM_THREADS=2 && ./kmeans_omp $DATASET $K $MAX_ITER $SEED"

run_test "VERSÃO OPENMP - 4 THREADS" \
         "export OMP_NUM_THREADS=4 && ./kmeans_omp $DATASET $K $MAX_ITER $SEED"

run_test "VERSÃO OPENMP - 8 THREADS" \
         "export OMP_NUM_THREADS=8 && ./kmeans_omp $DATASET $K $MAX_ITER $SEED"

# ===== VERSÃO HÍBRIDA MPI+OPENMP =====
run_test "VERSÃO HÍBRIDA - 1 PROCESSO × 4 THREADS" \
         "export OMP_NUM_THREADS=4 && mpirun -np 1 ./kmeans_hybrid $DATASET $K $MAX_ITER $SEED"

run_test "VERSÃO HÍBRIDA - 2 PROCESSOS × 2 THREADS" \
         "export OMP_NUM_THREADS=2 && mpirun -np 2 ./kmeans_hybrid $DATASET $K $MAX_ITER $SEED"

run_test "VERSÃO HÍBRIDA - 4 PROCESSOS × 1 THREAD (MPI PURO)" \
         "export OMP_NUM_THREADS=1 && mpirun -np 4 ./kmeans_hybrid $DATASET $K $MAX_ITER $SEED"

echo "==============================================================================="
echo "  TESTES CONCLUÍDOS"
echo "==============================================================================="
echo ""
echo "Resultados salvos em: $OUTPUT"
echo ""
echo "Para atualizar os tempos nos arquivos .cpp, copie os valores de $OUTPUT"
echo "e cole nos comentários no início de cada arquivo."
echo ""
echo "Exemplo de análise:"
echo "  - Calcule speedup: Tempo_seq / Tempo_paralelo"
echo "  - Calcule eficiência: Speedup / Número_de_workers"
echo ""
