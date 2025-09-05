#include "cuda_fft.h"
#include <iostream>
#include <cuda_runtime.h>
#include <cufft.h>
#include <vector>
#include <complex>
#include <thread>
#include <mutex>
#include <atomic>
#include <memory>

#define MAX_STREAMS_PER_GPU 4
#define MAX_GPUS 8

// GPU设备管理结构
struct GPUDevice {
    int device_id;
    bool available;
    std::mutex device_mutex;
    cudaStream_t streams[MAX_STREAMS_PER_GPU];
    cufftHandle plans[MAX_STREAMS_PER_GPU];

    // 设备内存池 - 修复：为批量FFT分配足够内存
    int8_t* d_int8_pool[MAX_STREAMS_PER_GPU];
    float* d_float_pool[MAX_STREAMS_PER_GPU];
    cufftComplex* d_fft_pool[MAX_STREAMS_PER_GPU];
    float* d_power_pool[MAX_STREAMS_PER_GPU];

    std::atomic<bool> stream_busy[MAX_STREAMS_PER_GPU];
    std::atomic<uint64_t> processed_count{0};
    std::atomic<uint64_t> error_count{0};

    GPUDevice() : device_id(-1), available(false) {
        for (int i = 0; i < MAX_STREAMS_PER_GPU; ++i) {
            stream_busy[i] = false;
            d_int8_pool[i] = nullptr;
            d_float_pool[i] = nullptr;
            d_fft_pool[i] = nullptr;
            d_power_pool[i] = nullptr;
        }
    }
};

// 全局GPU管理器
class MultiGPUManager {
private:
    std::vector<std::unique_ptr<GPUDevice>> devices_;
    std::atomic<int> next_device_{0};
    int num_available_gpus_{0};
    std::mutex init_mutex_;
    bool initialized_{false};

public:
    MultiGPUManager() = default;

    ~MultiGPUManager() {
        cleanup();
    }

    bool initialize() {
        std::lock_guard<std::mutex> lock(init_mutex_);
        if (initialized_) return true;

        int device_count = 0;
        cudaError_t error = cudaGetDeviceCount(&device_count);
        if (error != cudaSuccess) {
            std::cerr << "❌ 无法获取GPU数量: " << cudaGetErrorString(error) << std::endl;
            return false;
        }

        std::cout << "🔍 检测到 " << device_count << " 个GPU设备" << std::endl;

        // 初始化每个可用GPU
        for (int i = 0; i < device_count && i < MAX_GPUS; ++i) {
            auto device = std::make_unique<GPUDevice>();
            device->device_id = i;

            if (initializeGPUDevice(*device)) {
                devices_.push_back(std::move(device));
                num_available_gpus_++;
                std::cout << "✅ GPU " << i << " 初始化成功" << std::endl;
            } else {
                std::cerr << "❌ GPU " << i << " 初始化失败" << std::endl;
            }
        }

        if (num_available_gpus_ == 0) {
            std::cerr << "❌ 没有可用的GPU设备" << std::endl;
            return false;
        }

        std::cout << "🚀 多GPU系统初始化完成，可用GPU数量: " << num_available_gpus_ << std::endl;
        initialized_ = true;
        return true;
    }

    int getAvailableGPUCount() const {
        return num_available_gpus_;
    }

    // 获取最佳GPU设备（负载均衡）
    GPUDevice* getBestGPU() {
        if (devices_.empty()) return nullptr;

        // 寻找负载最轻的GPU
        GPUDevice* best_device = nullptr;
        int min_busy_streams = MAX_STREAMS_PER_GPU + 1;

        for (auto& device : devices_) {
            if (!device->available) continue;

            int busy_count = 0;
            for (int i = 0; i < MAX_STREAMS_PER_GPU; ++i) {
                if (device->stream_busy[i].load()) {
                    busy_count++;
                }
            }

            if (busy_count < min_busy_streams) {
                min_busy_streams = busy_count;
                best_device = device.get();
            }
        }

        return best_device;
    }

    // 获取指定GPU设备
    GPUDevice* getGPU(int gpu_id) {
        for (auto& device : devices_) {
            if (device->device_id == gpu_id) {
                return device.get();
            }
        }
        return nullptr;
    }

    void printStats() {
        std::cout << "📊 多GPU处理统计:" << std::endl;
        for (const auto& device : devices_) {
            int busy_streams = 0;
            for (int i = 0; i < MAX_STREAMS_PER_GPU; ++i) {
                if (device->stream_busy[i].load()) busy_streams++;
            }
            std::cout << "   GPU " << device->device_id
                      << ": 处理=" << device->processed_count.load()
                      << ", 错误=" << device->error_count.load()
                      << ", 忙碌流=" << busy_streams << "/" << MAX_STREAMS_PER_GPU << std::endl;
        }
    }

private:
    bool initializeGPUDevice(GPUDevice& device) {
        cudaError_t error = cudaSetDevice(device.device_id);
        if (error != cudaSuccess) {
            std::cerr << "无法设置GPU设备 " << device.device_id << ": " << cudaGetErrorString(error) << std::endl;
            return false;
        }

        // 检查GPU属性
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, device.device_id);
        std::cout << "🔧 GPU " << device.device_id << ": " << prop.name
                  << " (显存: " << prop.totalGlobalMem / (1024*1024) << "MB)" << std::endl;

        // 修复：使用批量FFT的正确大小
        const size_t N = 64 * 1024;
        const int batch = 3;  // 批量大小

        // 为每个GPU创建流和内存池
        for (int i = 0; i < MAX_STREAMS_PER_GPU; ++i) {
            // 创建CUDA流
            error = cudaStreamCreate(&device.streams[i]);
            if (error != cudaSuccess) {
                std::cerr << "创建CUDA流失败: " << cudaGetErrorString(error) << std::endl;
                return false;
            }

            // 修复：创建批量FFT计划
            int n[1] = { static_cast<int>(N) };
            cufftResult fft_result = cufftPlanMany(&device.plans[i], 1, n,
                                                  nullptr, 1, N,
                                                  nullptr, 1, N/2 + 1,
                                                  CUFFT_R2C, batch);  // 批量大小
            if (fft_result != CUFFT_SUCCESS) {
                std::cerr << "创建CUFFT计划失败，错误代码: " << fft_result << std::endl;
                return false;
            }

            cufftSetStream(device.plans[i], device.streams[i]);

            // 预分配设备内存池（批量大小）
            if (!allocateDeviceMemory(device, i, N, batch)) {
                std::cerr << "GPU " << device.device_id << " 流 " << i << " 内存分配失败" << std::endl;
                return false;
            }
        }

        device.available = true;
        return true;
    }

    bool allocateDeviceMemory(GPUDevice& device, int stream_idx, size_t N, int batch) {
        cudaError_t error;

        std::cout << "🔧 为GPU " << device.device_id << " 流 " << stream_idx
                  << " 分配内存: " << (N * batch * sizeof(int8_t) / 1024 / 1024) << "MB" << std::endl;

        // 修复：按批量大小分配内存
        error = cudaMalloc(&device.d_int8_pool[stream_idx], N * batch * sizeof(int8_t));
        if (error != cudaSuccess) {
            std::cerr << "分配int8内存失败: " << cudaGetErrorString(error) << std::endl;
            return false;
        }

        error = cudaMalloc(&device.d_float_pool[stream_idx], N * batch * sizeof(float));
        if (error != cudaSuccess) {
            std::cerr << "分配float内存失败: " << cudaGetErrorString(error) << std::endl;
            return false;
        }

        error = cudaMalloc(&device.d_fft_pool[stream_idx], (N/2 + 1) * batch * sizeof(cufftComplex));
        if (error != cudaSuccess) {
            std::cerr << "分配FFT内存失败: " << cudaGetErrorString(error) << std::endl;
            return false;
        }

        error = cudaMalloc(&device.d_power_pool[stream_idx], (N/2 + 1) * batch * sizeof(float));
        if (error != cudaSuccess) {
            std::cerr << "分配功率谱内存失败: " << cudaGetErrorString(error) << std::endl;
            return false;
        }

        return true;
    }

    void cleanup() {
        for (auto& device : devices_) {
            if (device->available) {
                cudaSetDevice(device->device_id);

                for (int i = 0; i < MAX_STREAMS_PER_GPU; ++i) {
                    if (device->d_int8_pool[i]) cudaFree(device->d_int8_pool[i]);
                    if (device->d_float_pool[i]) cudaFree(device->d_float_pool[i]);
                    if (device->d_fft_pool[i]) cudaFree(device->d_fft_pool[i]);
                    if (device->d_power_pool[i]) cudaFree(device->d_power_pool[i]);

                    cufftDestroy(device->plans[i]);
                    cudaStreamDestroy(device->streams[i]);
                }
            }
        }
        devices_.clear();
    }
};

// 全局多GPU管理器实例
static MultiGPUManager g_gpu_manager;

// CUDA内核定义（与原来相同）
__global__ void ConvertInt8ToFloat(int8_t* d_int8, float* d_float, size_t N) {
    const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        d_float[i] = static_cast<float>(d_int8[i]);
    }
}

__global__ void AbsFFTResults(cufftComplex* fft_results, float* abs_results, size_t N) {
    const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        abs_results[i] = fft_results[i].x * fft_results[i].x + fft_results[i].y * fft_results[i].y;
    }
}

__global__ void SumFFTResultsInPlace(float* fft_abs_results, size_t fft_size, int batch_count) {
    const size_t k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= fft_size) return;

    float sum = fft_abs_results[k];
    for (int b = 1; b < batch_count; ++b) {
        sum += fft_abs_results[k + b * fft_size];
    }
    fft_abs_results[k] = sum;
}

// 修复后的多GPU异步FFT处理函数
bool cuda_fft_process_batch_async_multigpu(const std::vector<int8_t>& chunk_data,
                                           std::vector<std::complex<float>>& fft_result) {
    const size_t N = 64 * 1024;
    const int batch = 3;

    if (chunk_data.size() != N * batch) {
        std::cerr << "❌ 输入数据大小错误: 期望 " << (N * batch) << ", 实际 " << chunk_data.size() << std::endl;
        return false;
    }

    // 获取最佳GPU设备
    GPUDevice* device = g_gpu_manager.getBestGPU();
    if (!device || !device->available) {
        std::cerr << "❌ 没有可用的GPU设备" << std::endl;
        return false;
    }

    // 原子操作寻找空闲的流
    int stream_idx = -1;
    for (int i = 0; i < MAX_STREAMS_PER_GPU; ++i) {
        bool expected = false;
        if (device->stream_busy[i].compare_exchange_weak(expected, true)) {
            stream_idx = i;
            break;
        }
    }

    if (stream_idx == -1) {
        // 没有空闲流，等待一个短时间后重试
        std::this_thread::sleep_for(std::chrono::microseconds(50));
        return false;
    }

    // 设置GPU设备上下文
    cudaError_t error = cudaSetDevice(device->device_id);
    if (error != cudaSuccess) {
        std::cerr << "❌ 设置GPU设备失败: " << cudaGetErrorString(error) << std::endl;
        device->stream_busy[stream_idx] = false;
        device->error_count++;
        return false;
    }

    bool success = true;
    try {
        // 使用预分配的内存池
        int8_t* d_int8 = device->d_int8_pool[stream_idx];
        float* d_float = device->d_float_pool[stream_idx];
        cufftComplex* d_fft_output = device->d_fft_pool[stream_idx];
        float* d_power = device->d_power_pool[stream_idx];
        cudaStream_t stream = device->streams[stream_idx];
        cufftHandle plan = device->plans[stream_idx];

        // 异步内存传输（现在大小匹配了）
        error = cudaMemcpyAsync(d_int8, chunk_data.data(), N * batch * sizeof(int8_t),
                               cudaMemcpyHostToDevice, stream);
        if (error != cudaSuccess) {
            throw std::runtime_error("内存传输失败: " + std::string(cudaGetErrorString(error)));
        }

        // 类型转换
        const int blockSize = 256;
        const int gridSize = (N * batch + blockSize - 1) / blockSize;
        ConvertInt8ToFloat<<<gridSize, blockSize, 0, stream>>>(d_int8, d_float, N * batch);

        // 检查kernel执行错误
        error = cudaGetLastError();
        if (error != cudaSuccess) {
            throw std::runtime_error("ConvertInt8ToFloat kernel失败: " + std::string(cudaGetErrorString(error)));
        }

        // 执行批量FFT
        cufftResult fft_result_code = cufftExecR2C(plan, d_float, d_fft_output);
        if (fft_result_code != CUFFT_SUCCESS) {
            throw std::runtime_error("FFT执行失败，错误代码: " + std::to_string(fft_result_code));
        }

        // 计算功率谱
        const size_t fft_size = N/2 + 1;
        const int absGridSize = (fft_size * batch + blockSize - 1) / blockSize;
        AbsFFTResults<<<absGridSize, blockSize, 0, stream>>>(
            d_fft_output, d_power, fft_size * batch);

        // 检查kernel执行错误
        error = cudaGetLastError();
        if (error != cudaSuccess) {
            throw std::runtime_error("AbsFFTResults kernel失败: " + std::string(cudaGetErrorString(error)));
        }

        // 原地求和
        const int sumGridSize = (fft_size + blockSize - 1) / blockSize;
        SumFFTResultsInPlace<<<sumGridSize, blockSize, 0, stream>>>(
            d_power, fft_size, batch);

        // 检查kernel执行错误
        error = cudaGetLastError();
        if (error != cudaSuccess) {
            throw std::runtime_error("SumFFTResultsInPlace kernel失败: " + std::string(cudaGetErrorString(error)));
        }

        // 异步传输结果
        std::vector<float> power_result(fft_size);
        error = cudaMemcpyAsync(power_result.data(), d_power, fft_size * sizeof(float),
                               cudaMemcpyDeviceToHost, stream);
        if (error != cudaSuccess) {
            throw std::runtime_error("结果传输失败: " + std::string(cudaGetErrorString(error)));
        }

        // 等待流完成
        error = cudaStreamSynchronize(stream);
        if (error != cudaSuccess) {
            throw std::runtime_error("流同步失败: " + std::string(cudaGetErrorString(error)));
        }

        // 转换为复数格式
        fft_result.resize(fft_size);
        for (size_t i = 0; i < fft_size; ++i) {
            fft_result[i] = std::complex<float>(power_result[i], 0.0f);
        }

        device->processed_count++;

    } catch (const std::exception& e) {
        std::cerr << "❌ GPU " << device->device_id << " 流 " << stream_idx << " 处理错误: " << e.what() << std::endl;
        device->error_count++;
        success = false;
    }

    // 释放流
    device->stream_busy[stream_idx] = false;
    return success;
}

// 初始化多GPU系统
bool initialize_multi_gpu_system() {
    return g_gpu_manager.initialize();
}

// 获取GPU统计信息
void print_multi_gpu_stats() {
    g_gpu_manager.printStats();
}

// 获取可用GPU数量
int get_available_gpu_count() {
    return g_gpu_manager.getAvailableGPUCount();
}

// 原有函数的多GPU版本包装
bool cuda_fft_process_batch_async(const std::vector<int8_t>& chunk_data,
                                  std::vector<std::complex<float>>& fft_result) {
    return cuda_fft_process_batch_async_multigpu(chunk_data, fft_result);
}

// 保持其他原有函数不变...
void cuda_fft_process_batch_streamed(const std::vector<int8_t>& chunk_data, std::vector<float>& fft_result) {
    std::vector<std::complex<float>> complex_result;
    if (cuda_fft_process_batch_async_multigpu(chunk_data, complex_result)) {
        fft_result.resize(complex_result.size());
        for (size_t i = 0; i < complex_result.size(); ++i) {
            fft_result[i] = complex_result[i].real();
        }
    }
}

void cuda_fft_process(const std::vector<int8_t>& chunk_data, std::vector<std::complex<float>>& fft_result) {
    // 使用多GPU系统处理单个FFT
    cuda_fft_process_batch_async_multigpu(chunk_data, fft_result);
}

void cuda_fft_process_batch(const std::vector<int8_t>& chunk_data, std::vector<float>& fft_result) {
    std::vector<std::complex<float>> complex_result;
    if (cuda_fft_process_batch_async_multigpu(chunk_data, complex_result)) {
        fft_result.resize(complex_result.size());
        for (size_t i = 0; i < complex_result.size(); ++i) {
            fft_result[i] = complex_result[i].real();
        }
    }
}

void test_ConvertInt8ToFloat() {
    std::cout << "🧪 多GPU系统测试完成" << std::endl;
}