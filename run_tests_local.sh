#!/bin/bash
# run_tests_local.sh - Script de teste para máquina local
# Adaptado para hardware diferente do servidor PARCODE

# Detectar número de cores
NUM_CORES=$(nproc)
NUM_PHYSICAL_CORES=$(lscpu | grep "^Core(s) per socket:" | awk '{print $4}')
NUM_SOCKETS=$(lscpu | grep "^Socket(s):" | awk '{print $2}')
TOTAL_PHYSICAL=$((NUM_PHYSICAL_CORES * NUM_SOCKETS))

echo "==============================================================================="
echo "  K-MEANS PARALELO - TESTES LOCAIS"
echo "==============================================================================="
echo "Hardware detectado:"
echo "  - Total de threads (lógicos): $NUM_CORES"
echo "  - Cores físicos: $TOTAL_PHYSICAL"
echo "  - CPU: $(lscpu | grep "Model name:" | cut -d: -f2 | xargs)"
echo "==============================================================================="
echo ""

# Configurações
DATASET="mnist_train.csv"
K=10
MAX_ITER=20
SEED=42

# Verificar dataset
if [ ! -f "$DATASET" ]; then
    echo "❌ ERRO: Dataset $DATASET não encontrado!"
    echo ""
    echo "Execute primeiro: ./download_mnist.sh"
    exit 1
fi

# Verificar executáveis
if [ ! -f "kmeans_seq" ] || [ ! -f "kmeans_omp" ] || [ ! -f "kmeans_hybrid" ]; then
    echo "❌ ERRO: Executáveis não encontrados!"
    echo ""
    echo "Execute primeiro: make all"
    exit 1
fi

# Criar arquivo de resultados
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT="resultados_local_${TIMESTAMP}.txt"

echo "Dataset: $DATASET (K=$K, max_iter=$MAX_ITER)"
echo "Resultados serão salvos em: $OUTPUT"
echo ""

# Função para executar e registrar
run_test() {
    local name=$1
    local cmd=$2
    
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  $name"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    # Executar
    echo "$ $cmd"
    echo ""
    eval $cmd 2>&1 | tee -a $OUTPUT
    
    echo ""
}

# Salvar info do sistema
{
    echo "==============================================================================="
    echo "INFORMAÇÕES DO SISTEMA"
    echo "==============================================================================="
    echo "Data/Hora: $(date)"
    echo "Hostname: $(hostname)"
    echo ""
    lscpu | grep -E "Model name|Architecture|CPU\(s\)|Thread|Core|Socket"
    echo ""
    echo "==============================================================================="
    echo ""
} | tee $OUTPUT

# ===== VERSÃO SEQUENCIAL =====
run_test "1. VERSÃO SEQUENCIAL (BASELINE)" \
         "./kmeans_seq $DATASET $K $MAX_ITER $SEED"

# ===== VERSÃO OPENMP =====
# Testar com: 1, 2, cores_físicos, threads_totais

run_test "2. VERSÃO OPENMP - 1 THREAD" \
         "export OMP_NUM_THREADS=1 && ./kmeans_omp $DATASET $K $MAX_ITER $SEED"

run_test "3. VERSÃO OPENMP - 2 THREADS" \
         "export OMP_NUM_THREADS=2 && ./kmeans_omp $DATASET $K $MAX_ITER $SEED"

run_test "4. VERSÃO OPENMP - 4 THREADS" \
         "export OMP_NUM_THREADS=4 && ./kmeans_omp $DATASET $K $MAX_ITER $SEED"

if [ $TOTAL_PHYSICAL -ge 6 ]; then
    run_test "5. VERSÃO OPENMP - 6 THREADS" \
             "export OMP_NUM_THREADS=6 && ./kmeans_omp $DATASET $K $MAX_ITER $SEED"
fi

run_test "6. VERSÃO OPENMP - 8 THREADS" \
         "export OMP_NUM_THREADS=8 && ./kmeans_omp $DATASET $K $MAX_ITER $SEED"

if [ $NUM_CORES -ge 12 ]; then
    run_test "7. VERSÃO OPENMP - 12 THREADS (todos os lógicos)" \
             "export OMP_NUM_THREADS=12 && ./kmeans_omp $DATASET $K $MAX_ITER $SEED"
fi

# ===== VERSÃO HÍBRIDA MPI+OPENMP =====
# Configurações mantendo ~4 workers totais para comparar com PARCODE

run_test "8. VERSÃO HÍBRIDA - 1 PROCESSO × 4 THREADS" \
         "export OMP_NUM_THREADS=4 && mpirun -np 1 ./kmeans_hybrid $DATASET $K $MAX_ITER $SEED"

run_test "9. VERSÃO HÍBRIDA - 2 PROCESSOS × 2 THREADS" \
         "export OMP_NUM_THREADS=2 && mpirun -np 2 ./kmeans_hybrid $DATASET $K $MAX_ITER $SEED"

run_test "10. VERSÃO HÍBRIDA - 4 PROCESSOS × 1 THREAD" \
          "export OMP_NUM_THREADS=1 && mpirun -np 4 ./kmeans_hybrid $DATASET $K $MAX_ITER $SEED"

# Testes extras aproveitando mais cores (se disponível)
if [ $TOTAL_PHYSICAL -ge 6 ]; then
    run_test "11. VERSÃO HÍBRIDA - 2 PROCESSOS × 3 THREADS" \
             "export OMP_NUM_THREADS=3 && mpirun -np 2 ./kmeans_hybrid $DATASET $K $MAX_ITER $SEED"
    
    run_test "12. VERSÃO HÍBRIDA - 3 PROCESSOS × 2 THREADS" \
             "export OMP_NUM_THREADS=2 && mpirun -np 3 ./kmeans_hybrid $DATASET $K $MAX_ITER $SEED"
fi

echo "==============================================================================="
echo "  ✓ TESTES CONCLUÍDOS"
echo "==============================================================================="
echo ""
echo "Resultados salvos em: $OUTPUT"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  PRÓXIMOS PASSOS:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "1. Analise os resultados em: $OUTPUT"
echo ""
echo "2. Calcule speedups:"
echo "   Speedup = Tempo_Sequencial / Tempo_Paralelo"
echo ""
echo "3. Atualize os tempos nos arquivos .cpp (cabeçalhos)"
echo "   - kmeans_omp.cpp: Tempos OpenMP"
echo "   - kmeans_hybrid.cpp: Tempos híbridos"
echo ""
echo "4. IMPORTANTE: Quando tiver acesso ao PARCODE:"
echo "   - Execute os mesmos testes lá"
echo "   - Use os tempos do PARCODE na versão final"
echo "   - Mencione no relatório que testou localmente primeiro"
echo ""
echo "5. Para o trabalho final, use:"
echo "   - Tempos do PARCODE (servidor oficial)"
echo "   - Se PARCODE não estiver disponível, use seus tempos locais"
echo "   - Documente qual hardware foi usado"
echo ""
echo "==============================================================================="
