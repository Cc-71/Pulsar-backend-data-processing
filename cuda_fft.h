#ifndef CUDA_FFT_H
#define CUDA_FFT_H

#include <vector>
#include <complex>

// 原有函数声明
void cuda_fft_process(const std::vector<int8_t>& chunk_data, std::vector<std::complex<float>>& fft_result);
void cuda_fft_process_batch(const std::vector<int8_t>& chunk_data, std::vector<float>& fft_result);
void cuda_fft_process_batch_streamed(const std::vector<int8_t>& chunk_data, std::vector<float>& fft_result);
bool cuda_fft_process_batch_async(const std::vector<int8_t>& chunk_data, std::vector<std::complex<float>>& fft_result);
void test_ConvertInt8ToFloat();

// 新增多GPU函数声明
bool initialize_multi_gpu_system();
void print_multi_gpu_stats();
int get_available_gpu_count();
bool cuda_fft_process_batch_async_multigpu(const std::vector<int8_t>& chunk_data, std::vector<std::complex<float>>& fft_result);

#endif // CUDA_FFT_H