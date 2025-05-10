import matplotlib.pyplot as plt
import numpy as np

def plot_benchmark_results(filename = "benchmark_data.txt"):
    data = np.loadtxt(filename)
    matrix_sizes = data[:, 0]
    neon_times = data[:, 1] / 1000000 # to seconds
    classic_times = data[:, 2] / 1000000 # to seconds
    
    plt.figure(figsize = (10, 6))
    
    plt.plot(matrix_sizes, neon_times, 'b-o', label = 'NEON optimized')
    plt.plot(matrix_sizes, classic_times, 'r--s', label = 'Classic implementation')
    
    plt.title('Performance Comparison: NEON vs Classic Implementation')
    plt.xlabel('Matrix Size')
    plt.ylabel('Execution Time (seconds)')
    plt.grid(True, which = "both", ls = "--")
    plt.legend()
    
    plt.savefig('data/performance_comparison.png', dpi = 300, bbox_inches = 'tight')
    plt.show()

def plot_benchmark_X_log_scale(filename = "benchmark_data.txt"):
    data = np.loadtxt(filename)
    matrix_sizes = data[:, 0]
    neon_times = data[:, 1] / 1000000 # to seconds
    classic_times = data[:, 2] / 1000000 # to seconds
    
    plt.figure(figsize = (10, 6))
    
    plt.semilogx(matrix_sizes, neon_times, 'b-o', label = 'NEON optimized')
    plt.semilogx(matrix_sizes, classic_times, 'r--s', label = 'Classic implementation')
    
    plt.title('Performance Comparison (X-Log Scale)')
    plt.xlabel('Matrix Size (log scale)')
    plt.ylabel('Execution Time (seconds)')
    plt.grid(True, which = "both", ls = "--")
    plt.legend()
    
    plt.savefig('data/performance_comparison_X_log.png', dpi = 300, bbox_inches='tight')
    plt.show()

def plot_benchmark_log_scale(filename = "benchmark_data.txt"):
    data = np.loadtxt(filename)
    matrix_sizes = data[:, 0]
    neon_times = data[:, 1] / 1000000 # to seconds
    classic_times = data[:, 2] / 1000000 # to seconds
    
    plt.figure(figsize = (10, 6))
    
    plt.loglog(matrix_sizes, neon_times, 'b-o', label = 'NEON optimized')
    plt.loglog(matrix_sizes, classic_times, 'r--s', label = 'Classic implementation')
    
    plt.title('Performance Comparison (Log-Log Scale)')
    plt.xlabel('Matrix Size (log scale)')
    plt.ylabel('Execution Time (log scale)')
    plt.grid(True, which = "both", ls="--")
    plt.legend()
    
    plt.savefig('data/performance_comparison_log.png', dpi = 300, bbox_inches = 'tight')
    plt.show()

if __name__ == "__main__":
    plot_benchmark_results("build/benchmark_data.txt")
    plot_benchmark_X_log_scale("build/benchmark_data.txt")
    plot_benchmark_log_scale("build/benchmark_data.txt")