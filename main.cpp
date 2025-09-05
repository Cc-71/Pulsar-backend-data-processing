#include <iostream>
#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <cstring>
#include <boost/asio.hpp>
#include <fitsio.h>
#include <chrono>
#include <iomanip>
#include <filesystem>
#include <complex>
#include <memory>
#include <atomic>
#include <vector>
#include <map>
#include <sys/socket.h>
#include <netinet/in.h>
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>
#include <cuda_runtime.h>
#include "cuda_fft.h"
#include <string>
#include <sstream>
#include <fstream>
#include <regex>
#include <cstdlib>
#include <deque>
#include <array>

namespace fs = std::filesystem;
using namespace boost::asio;
using namespace std::chrono;

extern bool initialize_multi_gpu_system();
extern void print_multi_gpu_stats();
extern int get_available_gpu_count();

const int PORT = 60524;
const int BUFFER_SIZE = 4104;  //UDP数据包缓冲区大小
static const size_t PACKET_PAYLOAD_SIZE = 4096;
static const size_t PACKET_HEADER_SIZE = 8;
const int PACKETS_PER_CHUNK = 16 * 3;  // 每个数据块包含的数据包数量 (48个)
const int CHUNK_SIZE = 64 * 1024 * 3; // 每个数据块大小：192KB

// 网络接收参数
const int NUM_RECV_THREADS = 2;  // 接收线程数量
const int BATCH_SIZE = 64;  // 每次批量接收的数据包数量

//设置数据包缓存大小为4MB
const size_t MAX_QUEUE_SIZE = 200000;  // 控制内存使用

//设置CHUNK缓存大小为4MB
static const size_t MAX_CHUNK_COUNT = 6400; //// 数据块缓存最大数量
static const size_t ACC_BUFFER_SIZE = MAX_CHUNK_COUNT * CHUNK_SIZE; // 累积缓冲区总大小

const int NUM_THREADS = 4;   // 数据包解析线程数量
int NUM_FFT_THREADS = 16;   // FFT处理线程数量（必须>4）
int OPTIMAL_FFT_THREADS_PER_GPU = 4;  // 每个GPU的最佳线程数

const int FFT_RESULTS_PER_SAVE = 3;  // 每个文件保存的FFT结果数量

// 批量保存配置
const int BATCH_SAVE_SIZE = 10;    // 批量保存的FFT结果数量
const int SAVE_TIMEOUT_MS = 1000;  // 保存超时时间

// 性能监控相关
const int SAVE_STATS_INTERVAL = 5;  // 性能统计输出间隔
std::atomic<uint64_t> saved_count{0};

// ==================== 性能测量结构 ====================
struct PerformanceMetrics {
    // 网络接收性能
    std::atomic<uint64_t> total_received_packets{0};
    std::atomic<uint64_t> total_received_bytes{0};
    std::atomic<uint64_t> network_errors{0};
    std::atomic<uint64_t> network_timeouts{0};

    // 延迟测量
    std::atomic<double> avg_packet_latency_us{0.0};
    std::atomic<double> avg_processing_latency_us{0.0};
    std::atomic<double> avg_fft_latency_us{0.0};
    std::atomic<double> avg_save_latency_us{0.0};

    // 吞吐量测量
    std::atomic<double> current_receive_rate{0.0};
    std::atomic<double> peak_receive_rate{0.0};
    std::atomic<double> current_process_rate{0.0};
    std::atomic<double> peak_process_rate{0.0};

    // 瓶颈指标
    std::atomic<uint64_t> queue_overflows{0};
    std::atomic<uint64_t> cuda_resource_waits{0};
    std::atomic<uint64_t> file_write_errors{0};
    std::atomic<uint64_t> memory_allocation_failures{0};

     // 新增多GPU相关指标
    std::atomic<int> active_gpu_count{0};
    std::atomic<uint64_t> total_gpu_processed{0};
    std::atomic<uint64_t> gpu_errors{0};
    std::atomic<double> gpu_utilization_avg{0.0};
    std::atomic<double> multi_gpu_speedup{1.0};

    // 时间戳
    std::chrono::steady_clock::time_point start_time;
    std::chrono::steady_clock::time_point last_update_time;

    PerformanceMetrics() {
        start_time = std::chrono::steady_clock::now();
        last_update_time = start_time;
    }

    void updateReceiveRate(double rate) {
        current_receive_rate.store(rate);
        if (rate > peak_receive_rate.load()) {
            peak_receive_rate.store(rate);
        }
    }

    void updateProcessRate(double rate) {
        current_process_rate.store(rate);
        if (rate > peak_process_rate.load()) {
            peak_process_rate.store(rate);
        }
    }

    void printSummary() {
        auto now = std::chrono::steady_clock::now();
        auto total_duration = std::chrono::duration_cast<std::chrono::seconds>(now - start_time);

        std::cout << "\n🔍 ========== 性能分析报告 ========== 🔍" << std::endl;
        std::cout << "⏱️  总运行时间: " << total_duration.count() << "秒" << std::endl;
        std::cout << "🎮 活跃GPU数量: " << active_gpu_count.load() << std::endl;
        std::cout << "📦 总接收数据包: " << total_received_packets.load() << std::endl;
        std::cout << "📊 总接收数据量: " << (total_received_bytes.load() / 1024.0 / 1024.0) << "MB" << std::endl;
        std::cout << "🚀 峰值接收速率: " << peak_receive_rate.load() << " pkt/s" << std::endl;
        std::cout << "⚡ 峰值处理速率: " << peak_process_rate.load() << " pkt/s" << std::endl;
        std::cout << "📈 多GPU加速比: " << multi_gpu_speedup.load() << "x" << std::endl;
        std::cout << "🎯 GPU平均利用率: " << gpu_utilization_avg.load() << "%" << std::endl;
        std::cout << "🔧 GPU总处理量: " << total_gpu_processed.load() << std::endl;
        std::cout << "⏰ 平均数据包延迟: " << avg_packet_latency_us.load() << "μs" << std::endl;
        std::cout << "🔧 平均处理延迟: " << avg_processing_latency_us.load() << "μs" << std::endl;
        std::cout << "📈 平均FFT延迟: " << avg_fft_latency_us.load() << "μs" << std::endl;
        std::cout << "💾 平均保存延迟: " << avg_save_latency_us.load() << "μs" << std::endl;
        std::cout << "❌ 网络错误次数: " << network_errors.load() << std::endl;
        std::cout << "❌ GPU错误次数: " << gpu_errors.load() << std::endl;
        std::cout << "⏳ 网络超时次数: " << network_timeouts.load() << std::endl;
        std::cout << "🚫 队列溢出次数: " << queue_overflows.load() << std::endl;
        std::cout << "🎯 CUDA资源等待次数: " << cuda_resource_waits.load() << std::endl;
        std::cout << "💿 文件写入错误次数: " << file_write_errors.load() << std::endl;
        std::cout << "=============================================\n" << std::endl;
    }
};

// 全局性能指标
static PerformanceMetrics g_metrics;

// ==================== CUDA利用率估算器 ====================
class CUDAUtilizationEstimator {
private:
    std::atomic<size_t> active_kernels_{0};
    std::atomic<size_t> total_kernel_time_us_{0};
    std::chrono::steady_clock::time_point start_time_;
    std::mutex stats_mutex_;

public:
    CUDAUtilizationEstimator() : start_time_(std::chrono::steady_clock::now()) {}

    void kernelStart() {
        active_kernels_++;
    }

    void kernelEnd(size_t duration_us) {
        active_kernels_--;
        total_kernel_time_us_ += duration_us;
    }

    float getEstimatedUtilization() {
        auto now = std::chrono::steady_clock::now();
        auto wall_elapsed = std::chrono::duration_cast<std::chrono::microseconds>(now - start_time_);

        if (wall_elapsed.count() > 0) {
            float util = std::min(100.0f,
                (float)total_kernel_time_us_.load() * 100.0f / wall_elapsed.count());
            return util;
        }
        return 0.0f;
    }

    size_t getActiveKernels() const {
        return active_kernels_.load();
    }

    void reset() {
        total_kernel_time_us_ = 0;
        start_time_ = std::chrono::steady_clock::now();
    }
};

static CUDAUtilizationEstimator g_cuda_estimator;

// ==================== 延迟测量器 ====================
class LatencyTracker {
private:
    std::deque<double> samples_;
    std::mutex mutex_;
    size_t max_samples_;

public:
    LatencyTracker(size_t max_samples = 1000) : max_samples_(max_samples) {}

    void addSample(double latency_us) {
        std::lock_guard<std::mutex> lock(mutex_);
        samples_.push_back(latency_us);
        if (samples_.size() > max_samples_) {
            samples_.pop_front();
        }
    }

    double getAverageLatency() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (samples_.empty()) return 0.0;
        double sum = 0.0;
        for (double sample : samples_) {
            sum += sample;
        }
        return sum / samples_.size();
    }

    double getMaxLatency() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (samples_.empty()) return 0.0;
        return *std::max_element(samples_.begin(), samples_.end());
    }

    double getMinLatency() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (samples_.empty()) return 0.0;
        return *std::min_element(samples_.begin(), samples_.end());
    }
};

// 全局延迟追踪器
static LatencyTracker g_network_latency;
static LatencyTracker g_processing_latency;
static LatencyTracker g_fft_latency;
static LatencyTracker g_save_latency;

// ==================== 数据包数据结构 ====================
struct PacketData {
    std::vector<uint8_t> data;
    std::chrono::steady_clock::time_point timestamp;

    PacketData() = default;
    PacketData(const std::vector<uint8_t>& d)
        : data(d), timestamp(std::chrono::steady_clock::now()) {}
};

// ==================== 带时间戳的数据包 ====================
struct TimestampedPacket {
    std::vector<uint8_t> data;
    ip::udp::endpoint endpoint;
    std::chrono::steady_clock::time_point receive_time;
    std::chrono::steady_clock::time_point process_start_time;

    TimestampedPacket(const std::vector<uint8_t>& d, const ip::udp::endpoint& ep)
        : data(d), endpoint(ep), receive_time(std::chrono::steady_clock::now()) {}
};

// ==================== 无锁环形缓冲区 ====================
template<typename T, size_t Size>
class LockFreeRingBuffer {
private:
    static_assert((Size & (Size - 1)) == 0, "Size must be power of 2");

    std::array<T, Size> buffer_;
    std::atomic<size_t> write_pos_{0};
    std::atomic<size_t> read_pos_{0};

public:
    bool push(const T& item) {
        const size_t current_write = write_pos_.load(std::memory_order_relaxed);
        const size_t next_write = (current_write + 1) & (Size - 1);

        if (next_write == read_pos_.load(std::memory_order_acquire)) {
            return false; // 队列满
        }

        buffer_[current_write] = item;
        write_pos_.store(next_write, std::memory_order_release);
        return true;
    }

    bool pop(T& item) {
        const size_t current_read = read_pos_.load(std::memory_order_relaxed);

        if (current_read == write_pos_.load(std::memory_order_acquire)) {
            return false; // 队列空
        }

        item = buffer_[current_read];
        read_pos_.store((current_read + 1) & (Size - 1), std::memory_order_release);
        return true;
    }

    size_t size() const {
        const size_t write = write_pos_.load(std::memory_order_acquire);
        const size_t read = read_pos_.load(std::memory_order_acquire);
        return (write - read) & (Size - 1);
    }

    bool empty() const {
        return read_pos_.load(std::memory_order_acquire) ==
               write_pos_.load(std::memory_order_acquire);
    }

    bool full() const {
        const size_t write = write_pos_.load(std::memory_order_relaxed);
        const size_t next_write = (write + 1) & (Size - 1);
        return next_write == read_pos_.load(std::memory_order_acquire);
    }
};

// ==================== 优化的数据包队列 ====================
class OptimizedPacketQueue {
private:
    static const size_t RING_BUFFER_SIZE = 16384; // 必须是2的幂
    std::vector<LockFreeRingBuffer<std::shared_ptr<PacketData>, RING_BUFFER_SIZE>> buffers_;
    std::atomic<size_t> next_buffer_{0};
    std::atomic<uint64_t> total_packets_{0};

public:
    OptimizedPacketQueue(size_t num_buffers = 4) : buffers_(num_buffers) {}

    bool push(const std::vector<uint8_t>& data) {
        auto packet_data = std::make_shared<PacketData>(data);
        size_t buffer_idx = next_buffer_.fetch_add(1) % buffers_.size();

        if (buffers_[buffer_idx].push(packet_data)) {
            total_packets_.fetch_add(1);
            return true;
        }
        return false;
    }

    bool pop(std::shared_ptr<PacketData>& packet_data, size_t buffer_idx = 0) {
        if (buffer_idx >= buffers_.size()) {
            buffer_idx = 0;
        }
        return buffers_[buffer_idx].pop(packet_data);
    }

    size_t get_buffer_count() const {
        return buffers_.size();
    }

    uint64_t get_total_packets() const {
        return total_packets_.load();
    }

    size_t total_size() const {
        size_t total = 0;
        for (const auto& buffer : buffers_) {
            total += buffer.size();
        }
        return total;
    }
};

// ==================== 增强的数据包队列 ====================
class EnhancedPacketQueue {
private:
    OptimizedPacketQueue optimized_queue_;
    std::atomic<bool> stopped_{false};
    std::queue<TimestampedPacket> queue_;
    mutable std::mutex mutex_;
    std::condition_variable cond_;
    std::atomic<uint64_t> total_packets_{0};
    std::atomic<uint64_t> processed_packets_{0};

public:
    EnhancedPacketQueue() : optimized_queue_(4) {} // 4个环形缓冲区

    // 保持原有接口兼容性
    bool push(const std::vector<uint8_t>& data, const ip::udp::endpoint& ep) {
        total_packets_.fetch_add(1);
        return optimized_queue_.push(data);
    }

    bool pop(TimestampedPacket& packet) {
        std::shared_ptr<PacketData> packet_data;

        // 尝试从不同的缓冲区获取数据
        for (size_t i = 0; i < optimized_queue_.get_buffer_count(); ++i) {
            if (optimized_queue_.pop(packet_data, i)) {
                packet.data = packet_data->data;
                packet.receive_time = packet_data->timestamp;
                packet.process_start_time = std::chrono::steady_clock::now();
                processed_packets_.fetch_add(1);
                return true;
            }
        }

        if (stopped_) return false;

        // 短暂等待
        std::this_thread::sleep_for(std::chrono::microseconds(50));
        return false;
    }

    void stop() {
        stopped_ = true;
    }

    uint64_t getTotalPackets() const {
        return optimized_queue_.get_total_packets();
    }

    uint64_t getProcessedPackets() const {
        return processed_packets_.load();
    }

    size_t getQueueSize() const {
        return optimized_queue_.total_size();
    }
};

// ==================== 数据包头部信息结构 ====================
struct HeaderInfo {
    bool power_spectra;
    bool multi_beam;
    int beam_id;
    int polarization;
    uint64_t sequence;
};

class MyPacket {
public:
    // 构造函数：接收原始数据包和网络端点信息
    MyPacket(const std::vector<uint8_t>& buffer, const ip::udp::endpoint& endpoint)
        : buffer_(buffer), endpoint_(endpoint) {
        parseHeader();  // 自动解析头部信息
    }

     // 获取解析后的头部信息
    HeaderInfo getHeader() const {
        return header_;
    }

    // 获取有效载荷数据指针(4096字节)
    const uint8_t* getPayload() const {
        if (buffer_.size() < PACKET_HEADER_SIZE + PACKET_PAYLOAD_SIZE) {
            throw std::runtime_error("Buffer too small to contain payload");
        }
        return buffer_.data() + PACKET_HEADER_SIZE;
    }

    // 获取有效载荷大小
    size_t getPayloadSize() const {
        return PACKET_PAYLOAD_SIZE;
    }

private:
    std::vector<uint8_t> buffer_;    // 原始数据包缓冲区
    ip::udp::endpoint endpoint_;     // 网络端点信息
    HeaderInfo header_;              // 解析后的头部信息

    void parseHeader() {
        if (buffer_.size() < PACKET_HEADER_SIZE) {
            throw std::runtime_error("Buffer too small to contain header");
        }
        // 将前8字节解释为64位整数
        uint64_t bitmask = *reinterpret_cast<const uint64_t*>(buffer_.data());
        // 位字段解析：
        uint64_t seq_number = bitmask & ((1ULL << 56) - 1);
        bool bit7 = (bitmask >> 63) & 1;
        bool bit6 = (bitmask >> 62) & 1;
        int beam_id = (bitmask >> 57) & 0b11111;
        // 极化信息从IP地址最后一位推导
        int polarization = (endpoint_.address().to_string().back() - '0') % 2;
          // 构造头部信息结构
        header_ = {
            bit7 == 0,  // power_spectra
            bit6 == 1,  // multi_beam
            beam_id,
            polarization,
            seq_number
        };
    }
};

class OptimizedUDPServer {
public:
    OptimizedUDPServer(unsigned short port) : port_(port), running_(true) {
        for (int i = 0; i < NUM_RECV_THREADS; ++i) {
            int sockfd = create_socket(port);
            sockets_.push_back(sockfd);
        }

        std::cout << "🚀 优化UDP服务器启动，端口: " << port
                  << "，接收线程数: " << NUM_RECV_THREADS << std::endl;

        for (int i = 0; i < NUM_RECV_THREADS; ++i) {
            recv_threads_.emplace_back([this, i]() {
                receive_thread_func(sockets_[i]);
            });
        }
    }

    ~OptimizedUDPServer() {
        stop();
    }

    void stop() {
        running_ = false;
        for (int sockfd : sockets_) {
            if (sockfd >= 0) {
                close(sockfd);
            }
        }
        for (auto& t : recv_threads_) {
            if (t.joinable()) {
                t.join();
            }
        }
    }

    EnhancedPacketQueue& get_queue() { return packet_queue_; }

    // 新增：网络性能测试
    void performNetworkTest() {
        std::cout << "🔍 开始网络性能测试..." << std::endl;

        auto start_time = std::chrono::steady_clock::now();
        uint64_t start_packets = packet_queue_.getTotalPackets();

        std::this_thread::sleep_for(std::chrono::seconds(10));

        auto end_time = std::chrono::steady_clock::now();
        uint64_t end_packets = packet_queue_.getTotalPackets();

        auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);
        uint64_t packets_received = end_packets - start_packets;

        double packets_per_second = static_cast<double>(packets_received) / duration.count();
        double mbps = (packets_per_second * BUFFER_SIZE * 8) / (1024 * 1024);

        std::cout << "📊 网络性能测试结果:" << std::endl;
        std::cout << "   接收速率: " << packets_per_second << " packets/s" << std::endl;
        std::cout << "   带宽利用率: " << mbps << " Mbps" << std::endl;
        std::cout << "   理论最大吞吐: " << (packets_per_second * 1.2) << " packets/s" << std::endl;

        g_metrics.updateReceiveRate(packets_per_second);
    }

    uint64_t getDroppedPackets() const {
        return dropped_packets_.load();
    }

private:
    int create_socket(unsigned short port) {
        int sockfd = socket(AF_INET, SOCK_DGRAM, 0);
        if (sockfd < 0) {
            throw std::runtime_error("Failed to create socket");
        }

        int opt = 1;
        if (setsockopt(sockfd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt)) < 0) {
            close(sockfd);
            throw std::runtime_error("Failed to set SO_REUSEADDR");
        }

        if (setsockopt(sockfd, SOL_SOCKET, SO_REUSEPORT, &opt, sizeof(opt)) < 0) {
            std::cerr << "Warning: SO_REUSEPORT not supported" << std::endl;
        }

        int buffer_size = 64 * 1024 * 1024;
        if (setsockopt(sockfd, SOL_SOCKET, SO_RCVBUF, &buffer_size, sizeof(buffer_size)) < 0) {
            std::cerr << "Warning: Failed to set receive buffer size" << std::endl;
        }

        int flags = fcntl(sockfd, F_GETFL, 0);
        if (flags < 0 || fcntl(sockfd, F_SETFL, flags | O_NONBLOCK) < 0) {
            close(sockfd);
            throw std::runtime_error("Failed to set non-blocking mode");
        }

        struct sockaddr_in addr;
        memset(&addr, 0, sizeof(addr));
        addr.sin_family = AF_INET;
        addr.sin_addr.s_addr = INADDR_ANY;
        addr.sin_port = htons(port);

        if (bind(sockfd, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
            close(sockfd);
            throw std::runtime_error("Failed to bind socket to port " + std::to_string(port));
        }

        return sockfd;
    }

    void receive_thread_func(int sockfd) {
        std::vector<struct mmsghdr> msgs(BATCH_SIZE);
        std::vector<struct sockaddr_in> addrs(BATCH_SIZE);
        std::vector<std::vector<uint8_t>> buffers(BATCH_SIZE);
        std::vector<struct iovec> iovs(BATCH_SIZE);

        for (int i = 0; i < BATCH_SIZE; ++i) {
            buffers[i].resize(BUFFER_SIZE);
            iovs[i].iov_base = buffers[i].data();
            iovs[i].iov_len = BUFFER_SIZE;
            msgs[i].msg_hdr.msg_iov = &iovs[i];
            msgs[i].msg_hdr.msg_iovlen = 1;
            msgs[i].msg_hdr.msg_name = &addrs[i];
            msgs[i].msg_hdr.msg_namelen = sizeof(struct sockaddr_in);
            msgs[i].msg_len = 0;
        }

        while (running_) {
            int count = recvmmsg(sockfd, msgs.data(), BATCH_SIZE, MSG_DONTWAIT, nullptr);

            if (count > 0) {
                for (int i = 0; i < count; ++i) {
                    if (msgs[i].msg_len == BUFFER_SIZE) {
                        std::vector<uint8_t> data(buffers[i].begin(),
                                                buffers[i].begin() + msgs[i].msg_len);

                        ip::udp::endpoint ep(
                            ip::address_v4(ntohl(addrs[i].sin_addr.s_addr)),
                            ntohs(addrs[i].sin_port)
                        );

                        if (!packet_queue_.push(data, ep)) {
                            dropped_packets_++;
                        }
                    }
                }

            } else if (count < 0) {
                if (errno == EAGAIN || errno == EWOULDBLOCK) {
                    g_metrics.network_timeouts++;
                    std::this_thread::sleep_for(std::chrono::microseconds(100));
                } else {
                    g_metrics.network_errors++;
                    std::cerr << "recvmmsg error: " << strerror(errno) << std::endl;
                    std::this_thread::sleep_for(std::chrono::milliseconds(10));
                }
            } else {
                std::this_thread::sleep_for(std::chrono::microseconds(100));
            }
        }
    }

    unsigned short port_;
    std::atomic<bool> running_;
    std::vector<int> sockets_;
    std::vector<std::thread> recv_threads_;
    EnhancedPacketQueue packet_queue_;
    std::atomic<uint64_t> dropped_packets_{0};
};

class ThreadSafeQueue {
public:
        // 默认构造函数
        ThreadSafeQueue() = default;
        // 禁止拷贝构造和赋值
        ThreadSafeQueue(const ThreadSafeQueue&) = delete;
        ThreadSafeQueue& operator=(const ThreadSafeQueue&) = delete;

        // 将元素推入队列
        void push(const std::vector<float>& item) {
            std::lock_guard<std::mutex> lock(m_mutex);
            m_queue.push(item);
            m_cond.notify_one();
        }

        // 尝试从队列中弹出元素
        bool try_pop(std::vector<float>& item) {
            std::lock_guard<std::mutex> lock(m_mutex);
            if (m_queue.empty()) {
                return false;
            }
            item = m_queue.front();
            m_queue.pop();
            return true;
        }

        // 等待并从队列中弹出元素
        void wait_and_pop(std::vector<float>& item) {
            std::unique_lock<std::mutex> lock(m_mutex);
            m_cond.wait(lock, [this] { return !m_queue.empty(); });
            item = m_queue.front();
            m_queue.pop();
        }

        // 检查队列是否为空
        bool empty() const {
            std::lock_guard<std::mutex> lock(m_mutex);
            return m_queue.empty();
        }

        // 获取队列大小
        size_t size() const {
            std::lock_guard<std::mutex> lock(m_mutex);
            return m_queue.size();
        }

private:
        mutable std::mutex m_mutex;               // 保护队列的互斥锁
        std::queue<std::vector<float>> m_queue;   // 实际的队列容器
        std::condition_variable m_cond;           // 条件变量，用于线程间同步
};

// ==================== 增强的FITS保存函数 ====================
void saveToFitsFileBatch(const std::vector<std::vector<float>>& data_batch, const std::string& filename) {
    auto start_time = std::chrono::steady_clock::now();

    if (data_batch.empty()) return;

    fitsfile* fptr;
    int status = 0;

    // 创建FITS文件
    fits_create_file(&fptr, filename.c_str(), &status);
    if (status != 0) {
        g_metrics.file_write_errors++;
        char error_text[30];
        fits_get_errstatus(status, error_text);
        std::cerr << "FITS创建错误: " << error_text << " (文件: " << filename << ")" << std::endl;
        return;
    }

    // 创建主HDU
    fits_create_img(fptr, FLOAT_IMG, 0, nullptr, &status);

    // 添加头部信息
    bool simple = true;
    short bitpix = 32;
    int naxis = 0;
    bool extend = true;
    const char* hdrver = "3.4";
    const char* fitstype = "PSRFITS";

    fits_update_key(fptr, TLOGICAL, "SIMPLE", &simple, nullptr, &status);
    fits_update_key(fptr, TSHORT, "BITPIX", &bitpix, nullptr, &status);
    fits_update_key(fptr, TINT, "NAXIS", &naxis, nullptr, &status);
    fits_update_key(fptr, TLOGICAL, "EXTEND", &extend, nullptr, &status);
    fits_update_key(fptr, TSTRING, "HDRVER", const_cast<char*>(hdrver), nullptr, &status);
    fits_update_key(fptr, TSTRING, "FITSTYPE", const_cast<char*>(fitstype), nullptr, &status);

    // 创建2D扩展HDU
    long naxes[2] = {
        static_cast<long>(data_batch[0].size()),
        static_cast<long>(data_batch.size())
    };
    fits_create_img(fptr, FLOAT_IMG, 2, naxes, &status);

    int num_spectra = static_cast<int>(data_batch.size());
    int spec_length = static_cast<int>(data_batch[0].size());
    fits_update_key(fptr, TINT, "NSPEC", &num_spectra, "Number of spectra in this file", &status);
    fits_update_key(fptr, TINT, "SPECLEN", &spec_length, "Length of each spectrum", &status);

    // 展平数据
    std::vector<float> flattened_data;
    flattened_data.reserve(data_batch.size() * data_batch[0].size());

    for (const auto& spectrum : data_batch) {
        flattened_data.insert(flattened_data.end(), spectrum.begin(), spectrum.end());
    }

    // 写入数据
    fits_write_img(fptr, TFLOAT, 1, flattened_data.size(),
                   const_cast<void*>(static_cast<const void*>(flattened_data.data())), &status);

    if (status != 0) {
        g_metrics.file_write_errors++;
        char error_text[30];
        fits_get_errstatus(status, error_text);
        std::cerr << "FITS写入错误: " << error_text << " (文件: " << filename << ")" << std::endl;
    }

    fits_close_file(fptr, &status);

    auto end_time = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    g_save_latency.addSample(duration.count());
    g_metrics.avg_save_latency_us.store(g_save_latency.getAverageLatency());
}

// ==================== 增强的DataAccumulator ====================
class DataAccumulator {
public:
    ThreadSafeQueue fftQueue;
    static const int BATCH_SAVE_SIZE = 10;
    std::vector<std::vector<float>> save_batch_;
    std::mutex save_batch_mutex_;

    DataAccumulator() = default;
    DataAccumulator(bool multi_beam, int polarization)
        : multi_beam_(multi_beam), polarization_(polarization), packet_count_(0), file_counter_(0), get_chunk_idx(0) {
        try {
            acc_buffer_ = new int8_t[ACC_BUFFER_SIZE];
            save_batch_.reserve(BATCH_SAVE_SIZE);
        } catch (const std::bad_alloc& e) {
            g_metrics.memory_allocation_failures++;
            std::cerr << "内存分配失败: " << e.what() << std::endl;
            throw;
        }
    }

    ~DataAccumulator() {
        delete[] acc_buffer_;
    }

    void addPacket(const MyPacket& packet) {
        std::unique_lock<std::mutex> lock(mutex_);

        bool has_space = cond_.wait_for(lock, std::chrono::milliseconds(10),
            [this]{ return (packet_count_*PACKET_PAYLOAD_SIZE-get_chunk_idx * CHUNK_SIZE < ACC_BUFFER_SIZE)||stopped_; });

        if (!has_space) {
            dropped_packets_++;
            if (dropped_packets_ % 1000 == 0) {
                std::cout << "⚠️  警告: DataAccumulator(" << multi_beam_ << "," << polarization_
                          << ") 缓冲区满，已丢弃 " << dropped_packets_ << " 个数据包" << std::endl;
            }
            return;
        }

        if(stopped_) {
            return;
        }

        const int8_t* payload = reinterpret_cast<const int8_t*>(packet.getPayload());
        size_t write_pos = (packet_count_ * PACKET_PAYLOAD_SIZE) % ACC_BUFFER_SIZE;
        memcpy(acc_buffer_ + write_pos, payload, PACKET_PAYLOAD_SIZE);
        packet_count_++;
        cond_.notify_one();
    }

    void printState() {
        size_t remain_packet_count = packet_count_ - get_chunk_idx*PACKETS_PER_CHUNK;
        size_t process_rate = get_chunk_idx - last_chunk_idx;
        last_chunk_idx = get_chunk_idx;
        if(process_rate > max_rate){
            max_rate = process_rate;
        }
        double rate = max_rate*0.1875;
        double last_rate = process_rate*0.1875;
        size_t fft_queue_size = fftQueue.size();

        std::cout << "📊 Accumulator(" << multi_beam_ << "," << polarization_ << "): "
                  << "remain_pkts=" << remain_packet_count
                  << ", chunks=" << get_chunk_idx
                  << ", total_pkts=" << packet_count_
                  << ", fft_queue=" << fft_queue_size
                  << ", max_rate=" << rate << "MB/s"
                  << ", last_rate=" << last_rate << "MB/s"
                  << ", files=" << file_counter_
                  << ", dropped=" << dropped_packets_ << std::endl;
    }

    bool popFromChunkQueue(std::vector<int8_t>& data){
        std::unique_lock<std::mutex> chunk_lock(chunk_queue_mutex_);
        if(stopped_) return false;
        if((packet_count_ < (get_chunk_idx + 1) * PACKETS_PER_CHUNK)) return false;

        data.clear();
        size_t read_pos = (get_chunk_idx%MAX_CHUNK_COUNT)*CHUNK_SIZE;
        data.insert(data.end(), acc_buffer_+read_pos, acc_buffer_+read_pos+CHUNK_SIZE);
        get_chunk_idx++;
        return true;
    }

    void stop() {
        std::lock_guard<std::mutex> lock(mutex_);
        stopped_ = true;
        cond_.notify_all();
    }

    std::string generateFitsFileName() {
        std::lock_guard<std::mutex> lock(file_counter_mutex_);
        return (multi_beam_ ? "multi" : "single") + std::string("_") +
               std::to_string(polarization_) + std::string("_") +
               std::to_string(file_counter_++) + std::string(".fits");
    }

    bool addToBatch(const std::vector<float>& fft_result) {
        std::lock_guard<std::mutex> lock(save_batch_mutex_);
        save_batch_.push_back(fft_result);
        return save_batch_.size() >= BATCH_SAVE_SIZE;
    }

    std::vector<std::vector<float>> getBatchAndClear() {
        std::lock_guard<std::mutex> lock(save_batch_mutex_);
        std::vector<std::vector<float>> result = std::move(save_batch_);
        save_batch_.clear();
        save_batch_.reserve(BATCH_SAVE_SIZE);
        return result;
    }

    std::vector<std::vector<float>> getRemainderBatch() {
        std::lock_guard<std::mutex> lock(save_batch_mutex_);
        if (save_batch_.empty()) {
            return {};
        }
        std::vector<std::vector<float>> result = std::move(save_batch_);
        save_batch_.clear();
        return result;
    }

    size_t getRemainPackets() const {
        return packet_count_ - get_chunk_idx * PACKETS_PER_CHUNK;
    }

    size_t getFileCount() const {
        std::lock_guard<std::mutex> lock(file_counter_mutex_);
        return file_counter_;
    }

    size_t getChunkCount() const {
        return get_chunk_idx;
    }

    size_t getTotalPackets() const {
        return packet_count_;
    }

    size_t getDroppedPackets() const {
        return dropped_packets_;
    }

private:
    bool multi_beam_;
    int polarization_;
    int8_t *acc_buffer_;
    int packet_count_;
    int file_counter_;
    size_t get_chunk_idx = 0;
    mutable size_t last_chunk_idx = 0;
    size_t max_rate = 0;
    std::mutex mutex_;
    mutable std::mutex file_counter_mutex_;
    std::mutex chunk_queue_mutex_;
    std::condition_variable cond_;
    bool stopped_ = false;
    std::atomic<size_t> dropped_packets_{0};
};

// GPU监控类
struct GPUUtilization {
    int gpu_util_percent = 0;
    int memory_util_percent = 0;
    int temperature = 0;
    int power_usage = 0;
    int memory_used_mb = 0;
    int memory_total_mb = 0;
    std::string gpu_name;
    bool valid = false;
};

class GPUMonitor {
public:
    GPUMonitor() = default;

    GPUUtilization getGPUUtilizationViaCUDA(int device_id = 0) {
        GPUUtilization util;
        int device_count = 0;
        cudaError_t error = cudaGetDeviceCount(&device_count);
        if (error != cudaSuccess || device_count == 0 || device_id >= device_count) {
            return util;
        }

        cudaSetDevice(device_id);
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, device_id);
        util.gpu_name = prop.name;

        size_t free_mem, total_mem;
        if (cudaMemGetInfo(&free_mem, &total_mem) == cudaSuccess) {
            util.memory_total_mb = total_mem / (1024 * 1024);
            util.memory_used_mb = (total_mem - free_mem) / (1024 * 1024);
            util.memory_util_percent = (int)((total_mem - free_mem) * 100 / total_mem);
        }

        util.valid = true;
        return util;
    }
};

// 修改GPU监控类以支持多GPU
class MultiGPUMonitor {
public:
    MultiGPUMonitor() = default;

    void printMultiGPUStatus() {
        int gpu_count = get_available_gpu_count();

        std::cout << "🎮 多GPU系统状态 (共" << gpu_count << "个GPU):" << std::endl;

        // 打印每个GPU的状态
        for (int i = 0; i < gpu_count; ++i) {
            printSingleGPUStatus(i);
        }

        // 打印多GPU统计
        print_multi_gpu_stats();

        // 计算整体利用率
        float total_util = 0.0f;
        for (int i = 0; i < gpu_count; ++i) {
            total_util += getGPUUtilization(i);
        }
        float avg_util = gpu_count > 0 ? total_util / gpu_count : 0.0f;
        g_metrics.gpu_utilization_avg.store(avg_util);

        std::cout << "📊 多GPU平均利用率: " << std::fixed << std::setprecision(1)
                  << avg_util << "%" << std::endl;
    }

private:
    void printSingleGPUStatus(int gpu_id) {
        cudaSetDevice(gpu_id);

        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, gpu_id);

        size_t free_mem, total_mem;
        if (cudaMemGetInfo(&free_mem, &total_mem) == cudaSuccess) {
            int memory_util_percent = (int)((total_mem - free_mem) * 100 / total_mem);
            int memory_used_mb = (total_mem - free_mem) / (1024 * 1024);
            int memory_total_mb = total_mem / (1024 * 1024);

            std::string mem_color = memory_util_percent >= 80 ? "🔴" :
                                   memory_util_percent >= 50 ? "🟡" : "🟢";

            std::cout << "   " << mem_color << " GPU " << gpu_id << " (" << prop.name << "): "
                      << "显存=" << memory_util_percent << "% "
                      << "(" << memory_used_mb << "MB/" << memory_total_mb << "MB)" << std::endl;
        }
    }

    float getGPUUtilization(int gpu_id) {
        // 简化的GPU利用率估算
        // 实际项目中可以使用NVML库获取更精确的利用率
        return g_cuda_estimator.getEstimatedUtilization();
    }
};

static MultiGPUMonitor g_multi_gpu_monitor;

// ==================== 增强的GPU状态打印 ====================
void print_gpu_resource_status() {
    static GPUMonitor monitor;
    GPUUtilization util = monitor.getGPUUtilizationViaCUDA(0);

    if (util.valid) {
        float estimated_util = g_cuda_estimator.getEstimatedUtilization();
        size_t active_kernels = g_cuda_estimator.getActiveKernels();

        std::string util_color = estimated_util >= 70 ? "🔴" :
                               estimated_util >= 30 ? "🟡" : "🟢";
        std::string mem_color = util.memory_util_percent >= 80 ? "🔴" :
                               util.memory_util_percent >= 50 ? "🟡" : "🟢";

        std::cout << util_color << " GPU状态: " << util.gpu_name << std::endl;
        std::cout << "   📊 估算GPU利用率: " << std::fixed << std::setprecision(1)
                  << estimated_util << "% (活跃内核: " << active_kernels << ")" << std::endl;
        std::cout << "   " << mem_color << " 显存使用: " << util.memory_util_percent << "% "
                  << "(" << util.memory_used_mb << "MB/" << util.memory_total_mb << "MB)" << std::endl;

        // 添加延迟统计
        std::cout << "   ⏱️  FFT平均延迟: " << g_fft_latency.getAverageLatency() << "μs" << std::endl;
        std::cout << "   💾 保存平均延迟: " << g_save_latency.getAverageLatency() << "μs" << std::endl;
    } else {
        std::cout << "🔴 GPU状态: 无法获取GPU信息" << std::endl;
    }
}

// 修改增强的监控线程以支持多GPU
void monitor_thread_enhanced_multigpu(OptimizedUDPServer& server, DataAccumulator accumulators[4], std::atomic<bool>& running) {
    auto last_time = steady_clock::now();
    uint64_t last_total = 0;
    uint64_t last_processed = 0;
    size_t monitor_count = 0;
    auto baseline_start_time = steady_clock::now();
    uint64_t baseline_processed = 0;
    bool baseline_recorded = false;

    while (running) {
        std::this_thread::sleep_for(seconds(1));
        monitor_count++;

        auto now = steady_clock::now();
        auto elapsed = duration_cast<duration<double>>(now - last_time).count();
        last_time = now;

        uint64_t current_total = server.get_queue().getTotalPackets();
        uint64_t current_processed = server.get_queue().getProcessedPackets();
        uint64_t current_dropped = server.getDroppedPackets();
        size_t queue_size = server.get_queue().getQueueSize();

        uint64_t total_diff = current_total - last_total;
        uint64_t processed_diff = current_processed - last_processed;

        last_total = current_total;
        last_processed = current_processed;

        double receive_rate = total_diff / elapsed;
        double process_rate = processed_diff / elapsed;

        // 计算多GPU加速比（相对于单GPU理论性能）
        if (!baseline_recorded && monitor_count > 10) {
            baseline_start_time = now;
            baseline_processed = current_processed;
            baseline_recorded = true;
        }

        if (baseline_recorded && monitor_count > 20) {
            auto baseline_elapsed = duration_cast<duration<double>>(now - baseline_start_time).count();
            double actual_rate = (current_processed - baseline_processed) / baseline_elapsed;
            double theoretical_single_gpu_rate = actual_rate / get_available_gpu_count();
            double speedup = actual_rate / theoretical_single_gpu_rate;
            g_metrics.multi_gpu_speedup.store(speedup);
        }

        // 更新全局性能指标
        g_metrics.updateReceiveRate(receive_rate);
        g_metrics.updateProcessRate(process_rate);
        g_metrics.active_gpu_count.store(get_available_gpu_count());

        size_t total_remain_packets = 0;
        size_t total_fft_queue_size = 0;
        size_t total_files = 0;
        size_t total_acc_dropped = 0;

        for (size_t i = 0; i < 4; ++i) {
            total_remain_packets += accumulators[i].getRemainPackets();
            total_fft_queue_size += accumulators[i].fftQueue.size();
            total_files += accumulators[i].getFileCount();
            total_acc_dropped += accumulators[i].getDroppedPackets();
        }

        // 瓶颈检测（多GPU版本）
        std::string bottleneck_warning = "";
        if (queue_size > MAX_QUEUE_SIZE * 0.8) {
            bottleneck_warning += "⚠️ 网络队列接近满载! ";
        }
        if (total_fft_queue_size > 100 * get_available_gpu_count()) {
            bottleneck_warning += "⚠️ FFT队列积压(多GPU)! ";
        }
        if (total_remain_packets > 10000) {
            bottleneck_warning += "⚠️ 数据包积压严重! ";
        }
        if (g_metrics.cuda_resource_waits.load() > 0 && monitor_count % 5 == 0) {
            bottleneck_warning += "⚠️ GPU资源竞争! ";
        }

        std::cout << "📊 多GPU监控统计: "
                  << "GPU数=" << get_available_gpu_count() << ", "
                  << "接收=" << std::fixed << std::setprecision(0) << receive_rate << "pkt/s, "
                  << "处理=" << process_rate << "pkt/s, "
                  << "加速比=" << std::setprecision(2) << g_metrics.multi_gpu_speedup.load() << "x, "
                  << "队列=" << queue_size << ", "
                  << "积压=" << total_remain_packets << "pkts, "
                  << "FFT队列=" << total_fft_queue_size << ", "
                  << "文件=" << total_files << ", "
                  << "丢包=" << total_acc_dropped << " " << bottleneck_warning << std::endl;

        // 显示多GPU状态
        g_multi_gpu_monitor.printMultiGPUStatus();

        // 显示每个accumulator的状态
        for (size_t i = 0; i < 4; ++i) {
            accumulators[i].printState();
        }

        // 每30秒打印详细性能报告
        if (monitor_count % 30 == 0) {
            std::cout << "\n🔍 ========== 多GPU详细性能报告 ========== 🔍" << std::endl;
            std::cout << "🎮 激活GPU数量: " << get_available_gpu_count() << std::endl;
            std::cout << "📈 峰值接收速率: " << g_metrics.peak_receive_rate.load() << " pkt/s" << std::endl;
            std::cout << "📈 峰值处理速率: " << g_metrics.peak_process_rate.load() << " pkt/s" << std::endl;
            std::cout << "🚀 多GPU加速比: " << g_metrics.multi_gpu_speedup.load() << "x" << std::endl;
            std::cout << "⏱️  平均网络延迟: " << g_network_latency.getAverageLatency() << "μs (最大: " << g_network_latency.getMaxLatency() << "μs)" << std::endl;
            std::cout << "⏱️  平均FFT延迟: " << g_fft_latency.getAverageLatency() << "μs (最大: " << g_fft_latency.getMaxLatency() << "μs)" << std::endl;
            std::cout << "⏱️  平均保存延迟: " << g_save_latency.getAverageLatency() << "μs (最大: " << g_save_latency.getMaxLatency() << "μs)" << std::endl;
            std::cout << "❌ 网络错误: " << g_metrics.network_errors.load() << std::endl;
            std::cout << "❌ 文件错误: " << g_metrics.file_write_errors.load() << std::endl;
            std::cout << "❌ 队列溢出: " << g_metrics.queue_overflows.load() << std::endl;
            std::cout << "❌ CUDA等待: " << g_metrics.cuda_resource_waits.load() << std::endl;
            std::cout << "==========================================\n" << std::endl;
        }
    }
}

// ==================== 增强的监控线程 ====================
void monitor_thread_enhanced(OptimizedUDPServer& server, DataAccumulator accumulators[4], std::atomic<bool>& running) {
    auto last_time = steady_clock::now();
    uint64_t last_total = 0;
    uint64_t last_processed = 0;
    size_t monitor_count = 0;

    while (running) {
        std::this_thread::sleep_for(seconds(1));
        monitor_count++;

        auto now = steady_clock::now();
        auto elapsed = duration_cast<duration<double>>(now - last_time).count();
        last_time = now;

        uint64_t current_total = server.get_queue().getTotalPackets();
        uint64_t current_processed = server.get_queue().getProcessedPackets();
        uint64_t current_dropped = server.getDroppedPackets();
        size_t queue_size = server.get_queue().getQueueSize();

        uint64_t total_diff = current_total - last_total;
        uint64_t processed_diff = current_processed - last_processed;

        last_total = current_total;
        last_processed = current_processed;

        double receive_rate = total_diff / elapsed;
        double process_rate = processed_diff / elapsed;

        // 更新全局性能指标
        g_metrics.updateReceiveRate(receive_rate);
        g_metrics.updateProcessRate(process_rate);

        size_t total_remain_packets = 0;
        size_t total_fft_queue_size = 0;
        size_t total_files = 0;
        size_t total_acc_dropped = 0;

        for (size_t i = 0; i < 4; ++i) {
            total_remain_packets += accumulators[i].getRemainPackets();
            total_fft_queue_size += accumulators[i].fftQueue.size();
            total_files += accumulators[i].getFileCount();
            total_acc_dropped += accumulators[i].getDroppedPackets();
        }

        // 瓶颈检测
        std::string bottleneck_warning = "";
        if (queue_size > MAX_QUEUE_SIZE * 0.8) {
            bottleneck_warning += "⚠️ 网络队列接近满载! ";
        }
        if (total_fft_queue_size > 100) {
            bottleneck_warning += "⚠️ FFT队列积压! ";
        }
        if (total_remain_packets > 10000) {
            bottleneck_warning += "⚠️ 数据包积压严重! ";
        }

        std::cout << "📊 监控统计: "
                  << "接收=" << std::fixed << std::setprecision(0) << receive_rate << "pkt/s, "
                  << "处理=" << process_rate << "pkt/s, "
                  << "队列=" << queue_size << ", "
                  << "积压=" << total_remain_packets << "pkts, "
                  << "FFT队列=" << total_fft_queue_size << ", "
                  << "文件=" << total_files << ", "
                  << "丢包=" << total_acc_dropped << " " << bottleneck_warning << std::endl;

        print_gpu_resource_status();

        for (size_t i = 0; i < 4; ++i) {
            accumulators[i].printState();
        }

        // 每30秒打印详细性能报告
        if (monitor_count % 30 == 0) {
            std::cout << "\n🔍 ========== 详细性能报告 ========== 🔍" << std::endl;
            std::cout << "📈 峰值接收速率: " << g_metrics.peak_receive_rate.load() << " pkt/s" << std::endl;
            std::cout << "📈 峰值处理速率: " << g_metrics.peak_process_rate.load() << " pkt/s" << std::endl;
            std::cout << "⏱️  平均网络延迟: " << g_network_latency.getAverageLatency() << "μs (最大: " << g_network_latency.getMaxLatency() << "μs)" << std::endl;
            std::cout << "⏱️  平均FFT延迟: " << g_fft_latency.getAverageLatency() << "μs (最大: " << g_fft_latency.getMaxLatency() << "μs)" << std::endl;
            std::cout << "⏱️  平均保存延迟: " << g_save_latency.getAverageLatency() << "μs (最大: " << g_save_latency.getMaxLatency() << "μs)" << std::endl;
            std::cout << "❌ 网络错误: " << g_metrics.network_errors.load() << std::endl;
            std::cout << "❌ 文件错误: " << g_metrics.file_write_errors.load() << std::endl;
            std::cout << "❌ 队列溢出: " << g_metrics.queue_overflows.load() << std::endl;
            std::cout << "❌ CUDA等待: " << g_metrics.cuda_resource_waits.load() << std::endl;
            std::cout << "==========================================\n" << std::endl;
        }
    }
}

// ==================== 增强的FFT线程 ====================
void fft_thread_optimized(DataAccumulator accumulators[4], std::atomic<bool>& running, size_t thread_id) {
    std::vector<int8_t> chunkData;
    std::vector<std::complex<float>> fft_result_complex;
    std::vector<float> fft_result_real;
    size_t acc_id = thread_id % 4;
    DataAccumulator* acc = &accumulators[acc_id];

    while (running) {
        if (acc->popFromChunkQueue(chunkData)) {
            try {
                g_cuda_estimator.kernelStart();
                auto fft_start_time = std::chrono::high_resolution_clock::now();

                bool success = false;
                int retry_count = 0;
                const int max_retries = 10;

                while (!success && retry_count < max_retries && running) {
                    success = cuda_fft_process_batch_async(chunkData, fft_result_complex);
                    if (!success) {
                        retry_count++;
                        g_metrics.cuda_resource_waits++;
                        std::this_thread::sleep_for(std::chrono::microseconds(50 * retry_count));
                    }
                }

                auto fft_end_time = std::chrono::high_resolution_clock::now();
                auto fft_duration = std::chrono::duration_cast<std::chrono::microseconds>(fft_end_time - fft_start_time);

                g_cuda_estimator.kernelEnd(fft_duration.count());
                g_fft_latency.addSample(fft_duration.count());
                g_metrics.avg_fft_latency_us.store(g_fft_latency.getAverageLatency());

                if (!success) {
                    std::cerr << "⚠️ FFT线程" << thread_id << ": CUDA资源持续不可用，跳过处理" << std::endl;
                    continue;
                }

                fft_result_real.clear();
                fft_result_real.reserve(fft_result_complex.size());

                for (const auto& complex_val : fft_result_complex) {
                    float magnitude_squared = complex_val.real() * complex_val.real() +
                                            complex_val.imag() * complex_val.imag();
                    fft_result_real.push_back(magnitude_squared);
                }

                acc->fftQueue.push(fft_result_real);

            } catch (const std::exception& e) {
                std::cerr << "❌ CUDA FFT处理错误: " << e.what() << std::endl;
            }
        } else {
            std::this_thread::sleep_for(std::chrono::microseconds(200));
        }
    }
}

// ==================== 增强的保存线程 ====================
void save_thread_optimized_batch(DataAccumulator accumulators[4], std::atomic<bool>& running, size_t thread_id) {
    size_t acc_id = thread_id % 4;
    DataAccumulator* acc = &accumulators[acc_id];
    std::vector<float> fft_result;

    try {
        fs::create_directories("./output_fits");
    } catch (const std::exception& e) {
        std::cerr << "❌ 创建输出目录失败: " << e.what() << std::endl;
        g_metrics.file_write_errors++;
    }

    while (running) {
        if (acc->fftQueue.try_pop(fft_result)) {
            try {
                auto processing_start = std::chrono::steady_clock::now();
                bool batch_ready = acc->addToBatch(fft_result);

                if (batch_ready) {
                    auto start_time = std::chrono::steady_clock::now();

                    auto batch = acc->getBatchAndClear();
                    std::string filename = acc->generateFitsFileName();
                    saveToFitsFileBatch(batch, "./output_fits/" + filename);
                    saved_count++;

                    auto end_time = std::chrono::steady_clock::now();
                    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);

                    // 记录处理延迟
                    auto processing_duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - processing_start);
                    g_processing_latency.addSample(processing_duration.count());
                    g_metrics.avg_processing_latency_us.store(g_processing_latency.getAverageLatency());
                }

            } catch (const std::exception& e) {
                std::cerr << "❌ 保存FITS文件错误: " << e.what() << std::endl;
                g_metrics.file_write_errors++;
            }
        } else {
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
    }

    // 保存剩余数据
    auto remainder = acc->getRemainderBatch();
    if (!remainder.empty()) {
        try {
            std::string filename = acc->generateFitsFileName();
            saveToFitsFileBatch(remainder, "./output_fits/" + filename);
        } catch (const std::exception& e) {
            std::cerr << "❌ 保存最终FITS文件错误: " << e.what() << std::endl;
            g_metrics.file_write_errors++;
        }
    }
}

// ==================== 增强的数据包解析线程 ====================
void parse_packet_header_thread(EnhancedPacketQueue& queue, std::atomic<bool>& running, DataAccumulator accumulators[4]) {
    TimestampedPacket packet({}, ip::udp::endpoint());

    while(running) {
        if(queue.pop(packet)) {
            try {
                auto processing_start = std::chrono::steady_clock::now();
                MyPacket myPacket(packet.data, packet.endpoint);
                HeaderInfo header = myPacket.getHeader();
                int index = header.multi_beam * 2 + header.polarization;
                accumulators[index].addPacket(myPacket);

                auto processing_end = std::chrono::steady_clock::now();
                auto processing_duration = std::chrono::duration_cast<std::chrono::microseconds>(processing_end - processing_start);
                g_processing_latency.addSample(processing_duration.count());

            } catch(const std::exception& e) {
                std::cerr << "❌ 数据包处理错误: " << e.what() << std::endl;
            }
        } else {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }
}

// ==================== 主函数 ====================
// 修改后的main函数
int main() {
    try {
        std::cout << "🚀 启动多GPU优化的实时数据处理系统..." << std::endl;

        // 首先初始化多GPU系统
        std::cout << "🔧 正在初始化多GPU系统..." << std::endl;
        if (!initialize_multi_gpu_system()) {
            std::cerr << "❌ 多GPU系统初始化失败，退出程序" << std::endl;
            return 1;
        }

        // 根据GPU数量动态调整FFT线程数
        int available_gpus = get_available_gpu_count();
        NUM_FFT_THREADS = available_gpus * OPTIMAL_FFT_THREADS_PER_GPU;

        std::cout << "✅ 多GPU系统初始化完成" << std::endl;
        std::cout << "🎮 检测到 " << available_gpus << " 个可用GPU" << std::endl;
        std::cout << "⚙️  FFT线程数调整为: " << NUM_FFT_THREADS << " (每GPU " << OPTIMAL_FFT_THREADS_PER_GPU << " 线程)" << std::endl;

        // 计算理论性能提升
        double theoretical_speedup = available_gpus * 0.85; // 考虑85%的并行效率
        std::cout << "📈 理论性能提升: " << std::fixed << std::setprecision(2) << theoretical_speedup << "x" << std::endl;

        OptimizedUDPServer server(PORT);

        DataAccumulator accumulators[4] = {
            DataAccumulator(false, 0),
            DataAccumulator(false, 1),
            DataAccumulator(true, 0),
            DataAccumulator(true, 1)
        };

        std::atomic<bool> running{true};
        std::vector<std::thread> workers;

        // 数据包解析线程数量保持不变
        for(int i = 0; i < NUM_THREADS; ++i) {
            workers.emplace_back(parse_packet_header_thread, std::ref(server.get_queue()), std::ref(running), accumulators);
        }

        // 使用动态计算的FFT线程数量
        std::vector<std::thread> fft_threads;
        for(int i = 0; i < NUM_FFT_THREADS; ++i) {
            fft_threads.emplace_back(fft_thread_optimized, accumulators, std::ref(running), i);
        }

        // 保存线程数量也相应调整
        std::vector<std::thread> save_threads;
        for(int i = 0; i < NUM_FFT_THREADS; ++i) {
            save_threads.emplace_back(save_thread_optimized_batch, accumulators, std::ref(running), i);
        }

        // 使用多GPU版本的监控线程
        std::thread monitor(monitor_thread_enhanced_multigpu, std::ref(server), accumulators, std::ref(running));

        std::cout << "✅ 所有线程已启动" << std::endl;
        std::cout << "   - 网络接收线程: " << NUM_RECV_THREADS << std::endl;
        std::cout << "   - 数据包解析线程: " << NUM_THREADS << std::endl;
        std::cout << "   - FFT处理线程: " << NUM_FFT_THREADS << " (分布在 " << available_gpus << " 个GPU上)" << std::endl;
        std::cout << "   - 保存线程: " << NUM_FFT_THREADS << std::endl;

        std::cout << "📊 开始5秒后进行网络性能测试..." << std::endl;

        // 等待5秒让系统稳定
        std::this_thread::sleep_for(std::chrono::seconds(5));

        // 执行网络性能测试
        server.performNetworkTest();

        std::cout << "🎯 多GPU系统运行中，按回车键退出..." << std::endl;
        std::cout << "📈 预期性能提升: " << theoretical_speedup << "x (基于 " << available_gpus << " GPU)" << std::endl;
        std::cin.get();

        std::cout << "🔄 正在停止多GPU系统..." << std::endl;
        running = false;
        server.get_queue().stop();

        for(size_t i = 0; i < 4; ++i){
            accumulators[i].stop();
        }

        server.stop();

        for(auto& t : fft_threads) t.join();
        for(auto& t : save_threads) t.join();
        for(auto& t : workers) t.join();
        monitor.join();

        std::cout << "📊 最终多GPU性能统计:" << std::endl;
        g_metrics.printSummary();

        // 打印最终多GPU统计
        print_multi_gpu_stats();

        std::cout << "✅ 多GPU系统已安全停止" << std::endl;

    } catch (std::exception& e) {
        std::cerr << "❌ 多GPU系统异常: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}