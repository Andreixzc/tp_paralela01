#!/usr/bin/env python3
"""
compare_results.py
Script para comparar resultados CPU (Parte 1) vs GPU (Parte 2)
Uso: python3 compare_results.py [cpu_csv] [gpu_csv]
"""

import sys
import csv
from typing import Dict, List, Tuple
import os

def load_cpu_results(filename: str = "RESUMO_PERFORMANCE.CSV") -> Dict[str, Dict]:
    """Carrega resultados CPU do Projeto 1"""
    results = {}
    
    if not os.path.exists(filename):
        print(f"Aviso: {filename} não encontrado")
        return results
    
    with open(filename, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            config = row['Configuration']
            results[config] = {
                'time': float(row['Time_seconds']),
                'speedup': float(row['Speedup']),
                'efficiency': float(row['Efficiency_percent']),
                'workers': int(row['Total_Workers']),
                'type': row['Type']
            }
    
    return results

def load_gpu_results(filename: str) -> Dict[str, Dict]:
    """Carrega resultados GPU do Projeto 2"""
    results = {}
    
    if not os.path.exists(filename):
        print(f"Aviso: {filename} não encontrado")
        return results
    
    # Agrupar por versão e calcular médias
    version_data = {}
    
    with open(filename, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            version = row['Version']
            if version not in version_data:
                version_data[version] = {
                    'total_times': [],
                    'gpu_times': [],
                    'transfer_times': [],
                    'iterations': []
                }
            
            version_data[version]['total_times'].append(float(row['Total_Time']))
            version_data[version]['gpu_times'].append(float(row['GPU_Time']))
            version_data[version]['transfer_times'].append(float(row['Transfer_Time']))
            version_data[version]['iterations'].append(int(row['Iterations']))
    
    # Calcular médias
    for version, data in version_data.items():
        results[version] = {
            'time': sum(data['total_times']) / len(data['total_times']),
            'gpu_time': sum(data['gpu_times']) / len(data['gpu_times']),
            'transfer_time': sum(data['transfer_times']) / len(data['transfer_times']),
            'iterations': int(sum(data['iterations']) / len(data['iterations'])),
            'runs': len(data['total_times'])
        }
    
    return results

def print_comparison_table(cpu_results: Dict, gpu_results: Dict, baseline_time: float):
    """Imprime tabela de comparação formatada"""
    
    print("\n" + "="*80)
    print("  COMPARAÇÃO DE DESEMPENHO: CPU (Parte 1) vs GPU (Parte 2)")
    print("="*80)
    print()
    
    # Resultados CPU
    print("PARTE 1 - IMPLEMENTAÇÕES CPU:")
    print("-" * 80)
    print(f"{'Configuração':<25} {'Tempo (s)':<12} {'Speedup':<10} {'Eficiência':<12}")
    print("-" * 80)
    
    for config, data in sorted(cpu_results.items()):
        print(f"{config:<25} {data['time']:>10.3f}  {data['speedup']:>8.2f}x  {data['efficiency']:>10.1f}%")
    
    print()
    
    # Resultados GPU
    print("PARTE 2 - IMPLEMENTAÇÕES GPU:")
    print("-" * 80)
    print(f"{'Versão':<25} {'Total (s)':<12} {'GPU (s)':<12} {'Transfer (s)':<15} {'Speedup':<10}")
    print("-" * 80)
    
    for version, data in sorted(gpu_results.items()):
        speedup = baseline_time / data['time']
        print(f"{version:<25} {data['time']:>10.3f}  {data['gpu_time']:>10.3f}  "
              f"{data['transfer_time']:>13.3f}  {speedup:>8.2f}x")
    
    print()
    
    # Análise comparativa
    print("ANÁLISE COMPARATIVA:")
    print("-" * 80)
    
    # Melhor CPU
    best_cpu = min(cpu_results.items(), key=lambda x: x[1]['time'])
    best_cpu_time = best_cpu[1]['time']
    print(f"Melhor CPU: {best_cpu[0]} ({best_cpu_time:.3f}s)")
    
    # Melhor GPU
    if gpu_results:
        best_gpu = min(gpu_results.items(), key=lambda x: x[1]['time'])
        best_gpu_time = best_gpu[1]['time']
        print(f"Melhor GPU: {best_gpu[0]} ({best_gpu_time:.3f}s)")
        
        # GPU vs melhor CPU
        speedup_vs_best_cpu = best_cpu_time / best_gpu_time
        print(f"\nSpeedup GPU vs Melhor CPU: {speedup_vs_best_cpu:.2f}x")
        
        # GPU vs sequential
        speedup_vs_seq = baseline_time / best_gpu_time
        print(f"Speedup GPU vs Sequential: {speedup_vs_seq:.2f}x")
        
        # Overhead de transferência
        for version, data in gpu_results.items():
            overhead = (data['transfer_time'] / data['time']) * 100
            print(f"\n{version}:")
            print(f"  - Overhead de transferência: {overhead:.1f}%")
            print(f"  - Tempo efetivo de computação GPU: {data['gpu_time']:.3f}s")
    
    print()
    print("="*80)

def save_comparison_csv(cpu_results: Dict, gpu_results: Dict, 
                        baseline_time: float, output_file: str = "comparison_summary.csv"):
    """Salva comparação em CSV"""
    
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Header
        writer.writerow(['Category', 'Configuration', 'Time_seconds', 'Speedup_vs_Sequential', 
                        'GPU_Time', 'Transfer_Time', 'Transfer_Overhead_%'])
        
        # CPU results
        for config, data in sorted(cpu_results.items()):
            writer.writerow(['CPU', config, f"{data['time']:.3f}", f"{data['speedup']:.2f}", 
                           '', '', ''])
        
        # GPU results
        for version, data in sorted(gpu_results.items()):
            speedup = baseline_time / data['time']
            overhead = (data['transfer_time'] / data['time']) * 100
            writer.writerow(['GPU', version, f"{data['time']:.3f}", f"{speedup:.2f}",
                           f"{data['gpu_time']:.3f}", f"{data['transfer_time']:.3f}", 
                           f"{overhead:.1f}"])
    
    print(f"Comparação salva em: {output_file}")

def main():
    # Arquivos de entrada
    cpu_file = sys.argv[1] if len(sys.argv) > 1 else "RESUMO_PERFORMANCE.CSV"
    
    # Encontrar arquivo GPU mais recente
    gpu_file = None
    if len(sys.argv) > 2:
        gpu_file = sys.argv[2]
    else:
        # Procurar por results_gpu_*.csv
        import glob
        gpu_files = sorted(glob.glob("results_gpu_*.csv"), reverse=True)
        if gpu_files:
            gpu_file = gpu_files[0]
            print(f"Usando arquivo GPU: {gpu_file}")
    
    # Carregar resultados
    cpu_results = load_cpu_results(cpu_file)
    gpu_results = load_gpu_results(gpu_file) if gpu_file else {}
    
    if not cpu_results and not gpu_results:
        print("ERRO: Nenhum resultado encontrado!")
        print("Execute:")
        print("  - Parte 1: ./run_tests_local.sh")
        print("  - Parte 2: ./run_gpu_tests.sh")
        sys.exit(1)
    
    # Baseline (sequential)
    baseline_time = cpu_results.get('Sequential', {}).get('time', 10.0)
    
    # Imprimir comparação
    print_comparison_table(cpu_results, gpu_results, baseline_time)
    
    # Salvar CSV
    if cpu_results or gpu_results:
        save_comparison_csv(cpu_results, gpu_results, baseline_time)
    
    print("\nPara visualização gráfica, considere usar ferramentas como:")
    print("  - LibreOffice Calc / Excel (abrir comparison_summary.csv)")
    print("  - matplotlib (criar gráficos em Python)")

if __name__ == "__main__":
    main()
