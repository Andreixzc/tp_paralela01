#!/bin/bash
# run_gpu_tests.sh
# Script automatizado para testar versões GPU do K-Means
# Uso: ./run_gpu_tests.sh

set -e

echo "=========================================="
echo " Benchmark GPU - K-Means Clustering"
echo "=========================================="
echo ""

# Configurações
DATASET="mnist_train.csv"
K=10
MAX_ITER=20
SEED=42
NUM_RUNS=3 # Número de execuções para média

# Arquivo de saída
OUTPUT_FILE="results_gpu_$(date +%Y%m%d_%H%M%S).csv"
LOG_FILE="results_gpu_$(date +%Y%m%d_%H%M%S).txt"

# Verificar dataset
if [ ! -f "$DATASET" ]; then
echo "ERRO: Dataset $DATASET não encontrado!"
echo "Execute: ./download_mnist.sh"
exit 1
fi

# Verificar se executáveis existem
echo "Compilando CUDA..."
if ! make cuda; then
echo "ERRO: Falha ao compilar CUDA!"
exit 1
fi
echo ""

# Tentar compilar OpenMP GPU (pode falhar por falta de suporte nvptx)
echo "Tentando compilar OpenMP GPU..."
HAS_OMP_GPU=0
if make omp_gpu 2>/dev/null; then
HAS_OMP_GPU=1
echo " OpenMP GPU compilado com sucesso"
else
echo "⚠ OpenMP GPU não disponível (requer nvptx-tools)"
echo " Continuando apenas com CUDA..."
echo " Ver OPENMP_GPU_NOTE.txt para detalhes"
fi
echo ""

# Criar header do CSV
echo "Version,Run,Total_Time,GPU_Time,Transfer_Time,Iterations" > "$OUTPUT_FILE"

echo "Iniciando testes..." | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Função para executar e extrair tempos
run_benchmark() {
local version=$1
local executable=$2

echo "========================================" | tee -a "$LOG_FILE"
echo " Testando: $version" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

for run in $(seq 1 $NUM_RUNS); do
echo " Run $run/$NUM_RUNS..." | tee -a "$LOG_FILE"

# Executar e capturar saída
output=$(./$executable $DATASET $K $MAX_ITER $SEED 2>&1)

# Extrair tempos usando grep e awk
total_time=$(echo "$output" | grep "Total time:" | awk '{print $3}')
gpu_time=$(echo "$output" | grep "GPU computation time:" | awk '{print $4}')
transfer_time=$(echo "$output" | grep "Data transfer time:" | awk '{print $4}')
iterations=$(echo "$output" | grep "finished in" | awk '{print $4}')

# Salvar no CSV
echo "$version,$run,$total_time,$gpu_time,$transfer_time,$iterations" >> "$OUTPUT_FILE"

# Mostrar resumo
echo " Total: ${total_time}s | GPU: ${gpu_time}s | Transfer: ${transfer_time}s" | tee -a "$LOG_FILE"

# Pequena pausa entre runs
sleep 1
done

echo "" | tee -a "$LOG_FILE"
}

# Executar benchmarks
if [ $HAS_OMP_GPU -eq 1 ]; then
run_benchmark "OpenMP_GPU" "kmeans_omp_gpu"
fi
run_benchmark "CUDA" "kmeans_cuda"

# Calcular médias
echo "========================================" | tee -a "$LOG_FILE"
echo " Calculando médias..." | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Usar awk para calcular médias
VERSIONS="CUDA"
if [ $HAS_OMP_GPU -eq 1 ]; then
VERSIONS="OpenMP_GPU CUDA"
fi

for version in $VERSIONS; do
avg_total=$(grep "^$version," "$OUTPUT_FILE" | awk -F, '{sum+=$3; count++} END {printf "%.3f", sum/count}')
avg_gpu=$(grep "^$version," "$OUTPUT_FILE" | awk -F, '{sum+=$4; count++} END {printf "%.3f", sum/count}')
avg_transfer=$(grep "^$version," "$OUTPUT_FILE" | awk -F, '{sum+=$5; count++} END {printf "%.3f", sum/count}')

echo "$version:" | tee -a "$LOG_FILE"
echo " Tempo médio total: ${avg_total}s" | tee -a "$LOG_FILE"
echo " Tempo médio GPU: ${avg_gpu}s" | tee -a "$LOG_FILE"
echo " Tempo médio transferência: ${avg_transfer}s" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
done

echo "========================================" | tee -a "$LOG_FILE"
echo " Testes concluídos!" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "Resultados salvos em:" | tee -a "$LOG_FILE"
echo " - $OUTPUT_FILE (CSV)" | tee -a "$LOG_FILE"
echo " - $LOG_FILE (log)" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Comparação com resultados CPU (se existir)
if [ -f "RESUMO_PERFORMANCE.CSV" ]; then
echo "========================================" | tee -a "$LOG_FILE"
echo " Comparação CPU vs GPU" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Tempo sequencial CPU
seq_time=$(grep "^Sequential," RESUMO_PERFORMANCE.CSV | awk -F, '{print $5}')

if [ ! -z "$seq_time" ]; then
echo "Tempo CPU Sequential: ${seq_time}s" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

for version in $VERSIONS; do
avg_total=$(grep "^$version," "$OUTPUT_FILE" | awk -F, '{sum+=$3; count++} END {printf "%.3f", sum/count}')
speedup=$(echo "$seq_time / $avg_total" | bc -l | xargs printf "%.2f")

echo "$version speedup vs CPU Sequential: ${speedup}x" | tee -a "$LOG_FILE"
done
echo "" | tee -a "$LOG_FILE"
fi
fi

echo "Use Python para análise mais detalhada:" | tee -a "$LOG_FILE"
echo " python3 compare_results.py" | tee -a "$LOG_FILE"
