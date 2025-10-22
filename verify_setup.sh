#!/bin/bash
# verify_setup.sh - Verificação rápida do ambiente

echo "==============================================================================="
echo "  VERIFICAÇÃO DO AMBIENTE DE DESENVOLVIMENTO"
echo "==============================================================================="
echo ""

ERRORS=0
WARNINGS=0

# Função para verificar comando
check_cmd() {
    local cmd=$1
    local name=$2
    
    if command -v $cmd &> /dev/null; then
        local version=$($cmd --version 2>&1 | head -n1)
        echo "✓ $name: $version"
        return 0
    else
        echo "✗ $name: NÃO ENCONTRADO"
        ((ERRORS++))
        return 1
    fi
}

# Verificar ferramentas básicas
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Compiladores e Ferramentas"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
check_cmd "g++" "GNU C++ Compiler"
check_cmd "mpicc" "MPI C Compiler"
check_cmd "mpirun" "MPI Runner"
check_cmd "make" "GNU Make"
echo ""

# Verificar OpenMP
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  OpenMP Support"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Testar compilação com OpenMP
cat > /tmp/test_omp.cpp << 'EOF'
#include <omp.h>
#include <iostream>
int main() {
    std::cout << "Max threads: " << omp_get_max_threads() << std::endl;
    return 0;
}
EOF

if g++ -fopenmp /tmp/test_omp.cpp -o /tmp/test_omp 2>/dev/null; then
    echo "✓ OpenMP: Suportado"
    THREADS=$(/tmp/test_omp)
    echo "  $THREADS"
    rm -f /tmp/test_omp /tmp/test_omp.cpp
else
    echo "✗ OpenMP: ERRO ao compilar"
    ((ERRORS++))
fi
echo ""

# Informações de hardware
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Hardware"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

CPU_MODEL=$(lscpu | grep "Model name:" | cut -d: -f2 | xargs)
CORES=$(nproc)
PHYSICAL=$(lscpu | grep "^Core(s) per socket:" | awk '{print $4}')
SOCKETS=$(lscpu | grep "^Socket(s):" | awk '{print $2}')
TOTAL_PHYSICAL=$((PHYSICAL * SOCKETS))

echo "CPU: $CPU_MODEL"
echo "Cores físicos: $TOTAL_PHYSICAL"
echo "Threads lógicos: $CORES"
echo ""

# Verificar arquivos do projeto
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Arquivos do Projeto"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

FILES=(
    "kmeans_sequential.cpp"
    "kmeans_omp.cpp"
    "kmeans_hybrid.cpp"
    "Makefile"
    "README.txt"
)

for file in "${FILES[@]}"; do
    if [ -f "$file" ]; then
        echo "✓ $file"
    else
        echo "✗ $file: NÃO ENCONTRADO"
        ((ERRORS++))
    fi
done
echo ""

# Verificar dataset
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Dataset MNIST"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if [ -f "mnist_train.csv" ]; then
    SIZE=$(du -h mnist_train.csv | cut -f1)
    LINES=$(wc -l < mnist_train.csv)
    echo "✓ mnist_train.csv: $SIZE, $LINES linhas"
    
    if [ $LINES -lt 60000 ]; then
        echo "⚠ Warning: Arquivo pode estar incompleto (esperado: 60001 linhas)"
        ((WARNINGS++))
    fi
else
    echo "✗ mnist_train.csv: NÃO ENCONTRADO"
    echo "  Execute: ./download_mnist.sh"
    ((ERRORS++))
fi
echo ""

# Verificar executáveis
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Executáveis Compilados"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

EXECS=("kmeans_seq" "kmeans_omp" "kmeans_hybrid")
EXEC_FOUND=0

for exec in "${EXECS[@]}"; do
    if [ -f "$exec" ]; then
        echo "✓ $exec"
        ((EXEC_FOUND++))
    else
        echo "⚠ $exec: Não compilado"
    fi
done

if [ $EXEC_FOUND -eq 0 ]; then
    echo ""
    echo "Nenhum executável encontrado. Execute: make all"
    ((WARNINGS++))
fi
echo ""

# Resumo
echo "==============================================================================="
echo "  RESUMO"
echo "==============================================================================="

if [ $ERRORS -eq 0 ] && [ $WARNINGS -eq 0 ]; then
    echo "✓ Ambiente configurado corretamente!"
    echo ""
    echo "Próximos passos:"
    echo "  1. Se ainda não compilou: make all"
    echo "  2. Se não tem o dataset: ./download_mnist.sh"
    echo "  3. Executar testes: ./run_tests_local.sh"
elif [ $ERRORS -eq 0 ]; then
    echo "⚠ Ambiente OK, mas há $WARNINGS aviso(s)"
    echo ""
    echo "Revise os avisos acima antes de continuar."
else
    echo "✗ Encontrados $ERRORS erro(s) e $WARNINGS aviso(s)"
    echo ""
    echo "AÇÕES NECESSÁRIAS:"
    
    if ! command -v g++ &> /dev/null || ! command -v mpicc &> /dev/null; then
        echo "  - Instalar dependências: sudo apt install build-essential openmpi-bin libopenmpi-dev"
    fi
    
    if [ ! -f "mnist_train.csv" ]; then
        echo "  - Baixar dataset: ./download_mnist.sh"
    fi
    
    if [ ! -f "kmeans_sequential.cpp" ]; then
        echo "  - Copiar arquivos .cpp para este diretório"
    fi
fi

echo ""
echo "==============================================================================="

exit $ERRORS
