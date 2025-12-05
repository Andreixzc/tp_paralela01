# Makefile para K-Means Paralelo
# Uso: make all - Compila todas as versões
# make seq - Compila versão sequencial
# make omp - Compila versão OpenMP
# make hybrid - Compila versão híbrida
# make gpu - Compila versões GPU (OpenMP + CUDA)
# make clean - Remove executáveis

CXX = g++
MPICXX = mpic++
NVCC = nvcc
CXXFLAGS = -O3 -std=c++17 -Wall
OMPFLAGS = -fopenmp
OMPGPUFLAGS = -fopenmp -foffload=nvptx-none
CUDAFLAGS = -O3 -arch=sm_61

# Executáveis
SEQ = kmeans_seq
OMP = kmeans_omp
HYBRID = kmeans_hybrid
OMP_GPU = kmeans_omp_gpu
CUDA = kmeans_cuda

# Arquivos fonte
SRC_SEQ = kmeans_sequential.cpp
SRC_OMP = kmeans_omp.cpp
SRC_HYBRID = kmeans_hybrid.cpp
SRC_OMP_GPU = kmeans_omp_gpu.cpp
SRC_CUDA = kmeans_cuda.cu

.PHONY: all seq omp hybrid gpu omp_gpu cuda clean test test_gpu

# Compilar todas as versões CPU
all: seq omp hybrid

# Compilar versões GPU
gpu: omp_gpu cuda

# Versão sequencial
seq: $(SEQ)

$(SEQ): $(SRC_SEQ)
	$(CXX) $(CXXFLAGS) -o $@ $<
	@echo " Versão sequencial compilada: ./$(SEQ)"

# Versão OpenMP
omp: $(OMP)

$(OMP): $(SRC_OMP)
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) -o $@ $<
	@echo " Versão OpenMP compilada: ./$(OMP)"

# Versão híbrida MPI+OpenMP
hybrid: $(HYBRID)

$(HYBRID): $(SRC_HYBRID)
	$(MPICXX) $(CXXFLAGS) $(OMPFLAGS) -o $@ $<
	@echo " Versão híbrida compilada: ./$(HYBRID)"

# Versão OpenMP GPU offloading
omp_gpu: $(OMP_GPU)

$(OMP_GPU): $(SRC_OMP_GPU)
	$(CXX) $(CXXFLAGS) $(OMPGPUFLAGS) -o $@ $<
	@echo " Versão OpenMP GPU compilada: ./$(OMP_GPU)"

# Versão CUDA
cuda: $(CUDA)

$(CUDA): $(SRC_CUDA)
	$(NVCC) $(CUDAFLAGS) -o $@ $<
	@echo " Versão CUDA compilada: ./$(CUDA)"

# Limpar executáveis
clean:
	rm -f $(SEQ) $(OMP) $(HYBRID) $(OMP_GPU) $(CUDA)
	@echo " Executáveis removidos"

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

# Teste GPU
test_gpu: gpu
	@echo "=== Teste GPU ==="
	@echo "OpenMP GPU:"
	@./$(OMP_GPU) mnist_train.csv 3 5 42 || echo "ERRO: Verifique suporte OpenMP GPU"
	@echo ""
	@echo "CUDA:"
	@./$(CUDA) mnist_train.csv 3 5 42 || echo "ERRO: Verifique instalação CUDA"

# Mostrar ajuda
help:
	@echo "Makefile para K-Means Paralelo"
	@echo ""
	@echo "Targets disponíveis:"
	@echo " make all - Compila todas as versões CPU"
	@echo " make gpu - Compila todas as versões GPU"
	@echo " make seq - Compila versão sequencial"
	@echo " make omp - Compila versão OpenMP"
	@echo " make hybrid - Compila versão híbrida MPI+OpenMP"
	@echo " make omp_gpu - Compila versão OpenMP GPU"
	@echo " make cuda - Compila versão CUDA"
	@echo " make clean - Remove executáveis"
	@echo " make test - Executa teste rápido (CPU)"
	@echo " make test_gpu - Executa teste rápido (GPU)"
	@echo " make help - Mostra esta mensagem"

