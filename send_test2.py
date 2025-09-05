import os
import socket
import time
from threading import Lock, Thread
import math

# 配置参数
receiver_ip = '0.0.0.0'  # 接收端IP地址
receiver_port = 60524  # 接收端端口
input_dir = "./udp_packet"  # 数据包目录

# 全局变量
packet_list = []  # 存储所有预加载的数据包
packet_count = 0  # 总发送包数
total_bytes_sent = 0  # 总发送字节数
current_rate = 20000  # 初始速率（包/秒）
rate_lock = Lock()  # 用于速率修改的锁
should_exit = False  # 退出标志

# 监控统计变量
last_packet_count = 0  # 上一秒的包计数
last_bytes_sent = 0  # 上一秒的字节计数
stats_lock = Lock()  # 用于统计变量的锁


# 预加载所有数据包到内存
def load_packets():
    global packet_list
    for filename in os.listdir(input_dir):
        if filename.endswith('.txt'):
            filepath = os.path.join(input_dir, filename)
            with open(filepath, 'rb') as f:
                packet_list.append(f.read())
    print(f"已预加载 {len(packet_list)} 个数据包到内存")


# 监控线程函数
def monitor_stats():
    global last_packet_count, last_bytes_sent, should_exit

    while not should_exit:
        time.sleep(1)  # 每秒更新一次

        with stats_lock:
            current_packets = packet_count - last_packet_count
            current_bytes = total_bytes_sent - last_bytes_sent
            last_packet_count = packet_count
            last_bytes_sent = total_bytes_sent

        # 计算速率
        packets_per_sec = current_packets
        mb_per_sec = current_bytes / (1024 * 1024)

        # 显示统计信息
        stats = (f"发送速率: {packets_per_sec} pkts/s | "
                 f"带宽: {mb_per_sec:.2f} MB/s | "
                 f"总包数: {packet_count} | "
                 f"总数据: {total_bytes_sent / (1024 * 1024):.2f} MB")
        print(f"\r{stats}", end="", flush=True)


# 主发送函数
def send_packets():
    global packet_count, total_bytes_sent, should_exit
    index = 0

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    try:
        while not should_exit:
            if not packet_list:
                print("错误：没有加载任何数据包！")
                break

            # 发送数据包
            packet = packet_list[index]
            sock.sendto(packet, (receiver_ip, receiver_port))

            # 更新统计信息
            with stats_lock:
                packet_count += 1
                total_bytes_sent += len(packet)

            index = (index + 1) % len(packet_list)

    except KeyboardInterrupt:
        should_exit = True
    finally:
        sock.close()
        print("\n正在关闭程序...")


if __name__ == "__main__":
    # 检查数据包目录
    if not os.path.exists(input_dir):
        print(f"错误：目录 {input_dir} 不存在！")
        exit(1)

    # 加载数据包
    load_packets()

    # 启动监控线程
    monitor_thread = Thread(target=monitor_stats)
    monitor_thread.daemon = True
    monitor_thread.start()

    # 开始发送
    try:
        send_packets()
    except KeyboardInterrupt:
        should_exit = True
        monitor_thread.join()