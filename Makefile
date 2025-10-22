# Makefile para K-Means Paralelo
# Uso: make all      - Compila todas as versões
#      make seq      - Compila versão sequencial
#      make omp      - Compila versão OpenMP
#      make hybrid   - Compila versão híbrida
#      make clean    - Remove executáveis

CXX = g++
MPICXX = mpic++
CXXFLAGS = -O3 -std=c++17 -Wall
OMPFLAGS = -fopenmp

# Executáveis
SEQ = kmeans_seq
OMP = kmeans_omp
HYBRID = kmeans_hybrid

# Arquivos fonte
SRC_SEQ = kmeans_sequential.cpp
SRC_OMP = kmeans_omp.cpp
SRC_HYBRID = kmeans_hybrid.cpp

.PHONY: all seq omp hybrid clean test

# Compilar todas as versões
all: seq omp hybrid

# Versão sequencial
seq: $(SEQ)

$(SEQ): $(SRC_SEQ)
	$(CXX) $(CXXFLAGS) -o $@ $<
	@echo "✓ Versão sequencial compilada: ./$(SEQ)"

# Versão OpenMP
omp: $(OMP)

$(OMP): $(SRC_OMP)
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) -o $@ $<
	@echo "✓ Versão OpenMP compilada: ./$(OMP)"

# Versão híbrida MPI+OpenMP
hybrid: $(HYBRID)

$(HYBRID): $(SRC_HYBRID)
	$(MPICXX) $(CXXFLAGS) $(OMPFLAGS) -o $@ $<
	@echo "✓ Versão híbrida compilada: ./$(HYBRID)"

# Limpar executáveis
clean:
	rm -f $(SEQ) $(OMP) $(HYBRID)
	@echo "✓ Executáveis removidos"

# Teste rápido (requer mnist_train.csv)
test: all
	@echo "=== Teste rápido ==="
	@echo "Sequencial:"
	@./$(SEQ) mnist_train.csv 3 5 42 || echo "ERRO: Verifique se mnist_train.csv existe"
	@echo ""
	@echo "OpenMP (2 threads):"
	@export OMP_NUM_THREADS=2 && ./$(OMP) mnist_train.csv 3 5 42 || echo "ERRO"
	@echo ""
	@echo "Híbrido (2 procs × 1 thread):"
	@export OMP_NUM_THREADS=1 && mpirun -np 2 ./$(HYBRID) mnist_train.csv 3 5 42 || echo "ERRO: MPI não disponível?"

# Mostrar ajuda
help:
	@echo "Makefile para K-Means Paralelo"
	@echo ""
	@echo "Targets disponíveis:"
	@echo "  make all      - Compila todas as versões"
	@echo "  make seq      - Compila versão sequencial"
	@echo "  make omp      - Compila versão OpenMP"
	@echo "  make hybrid   - Compila versão híbrida MPI+OpenMP"
	@echo "  make clean    - Remove executáveis"
	@echo "  make test     - Executa teste rápido"
	@echo "  make help     - Mostra esta mensagem"
