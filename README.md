# Pulsar-backend-data-processing
A high-performance multi-GPU system for real-time pulsar radio signal processing. Captures UDP data via optimized multi-threaded reception, processes with GPU-accelerated FFT, stores in FITS format.Features high-concurrency lock-free buffers, multi-channel accumulation, parallel computing, batch storage, and performance monitoring. 

编译命令

```bash
# 编译CUDA文件
nvcc -std=c++17 -O3 -c cuda_fft.cu -o cuda_fft.o

# 编译主程序
g++ -std=c++17 -O3 -I/usr/local/cuda/include -c main.cpp -o main.o

# 链接
g++ main.o cuda_fft.o -o udp_processor \
    -L/usr/local/cuda/lib64 -lcudart -lcufft \
    -lboost_system -lcfitsio -lpthread
    


```
//先编译源文件生成目标文件（.o）
//使用 nvcc 链接目标文件和库
//生成最终的可执行文件
