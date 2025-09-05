import random
import struct
import os
from concurrent.futures import ThreadPoolExecutor

# 数据包基本参数
payload_size = 4096
header_size = 8
packet_size = header_size + payload_size

# 输出文件夹
#output_dir = "/home/gao_xf/LX/GPU/udp_packet"
output_dir = "./udp_packet"
os.makedirs(output_dir, exist_ok=True)  # 创建输出文件夹

def generate_voltage_packet(sequence_number):
    """
    生成一个包含偏振A或偏振B的电压采样的数据包。

    参数:
    sequence_number (int): 数据包的序列号。

    返回:
    bytes: 生成的数据包，包含数据头和数据载荷。
    """


    # 随机生成bit 6（单波束或多波束）
    bit6 = random.randint(0, 1)

    # 随机生成bit 5-1
    if bit6 == 1:
        # 多波束模式，bit 5-1的取值范围为[1, 19]
        bit5_1 = random.randint(1, 19)
    else:
        # 单波束模式，bit 5-3表示feed id，bit 2-1表示subband id
        feed_id = random.randint(1, 2)  # 1: Caltech, 2: Cetc16
        subband_id = random.randint(0, 3)  # 0-3
        bit5_1 = (feed_id << 2) | subband_id

    bit0 = random.randint(0, 1)

    # 组合数据包头的前8位
    # bit 7: 1 (pure ADC sample)
    # bit 6: 随机生成
    # bit 5-1: 随机生成
    # bit 0: 随机生成
    header_bits = 0b10000000  # bit 7: 1 (pure ADC sample)
    header_bits |= (bit6 << 6)  # bit 6: 随机生成
    header_bits |= (bit5_1 << 1)  # bit 5-1: 随机生成
    header_bits |= bit0  # bit 0: 随机生成

    # 数据包头的后56位为序列号
    header = (header_bits << 56) | sequence_number
    print(bin(header))

    # 将数据包头打包为小端格式的64位整数
    header_bytes = struct.pack('<Q', header)

    # 生成4096个随机电压采样（8位有符号整数）
    voltage_samples = [random.randint(-128, 127) for _ in range(4096)]
    payload_bytes = struct.pack('4096b', *voltage_samples)

    # 组合数据头和数据载荷
    packet = header_bytes + payload_bytes

    return packet

# 定义生成数据包并保存为txt文件的函数
def save_packet_to_file(sequence_number):
    udp_packet = generate_voltage_packet(sequence_number)


    # 保存数据包到txt文件
    packet_filename = os.path.join(output_dir, f"packet_udp_{sequence_number:06d}.txt")
    with open(packet_filename, 'wb') as f:
        f.write(udp_packet)

    print(f"Saved packet {sequence_number} to {packet_filename}")

# 包生成的总数
#total_packets = 523267
total_packets = 20000
batch_size = 10240  # 每个线程处理的包数

try:
    # 启动线程池并行生成数据包
    with ThreadPoolExecutor(max_workers=16) as executor:  # 增加线程数来提升并发
        # 提交任务
        executor.map(save_packet_to_file, range(total_packets))

except KeyboardInterrupt:
    print("生成已停止。")
