# openmv_test.py
import pyb
from pyb import UART
import struct
import time

# UART配置
uart = UART(1, 115200)  # 使用UART1，波特率115200
uart.init(115200, bits=8, parity=None, stop=1)

# 测试数据
test_cases = [
    # (找到目标, u坐标, v坐标)
    (True, 160, 120),   # 中心位置
    (True, 0, 0),       # 左上角
    (True, 319, 239),   # 右下角
    (False, 0, 0),      # 未找到目标
]

# 通信协议
HEADER = 0x3A
TAIL = 0x0A
PKT_FORMAT = '<BHH'  # flag(1B) + u(2B) + v(2B)

print("OpenMV通信测试开始...")

while True:
    for flag, u, v in test_cases:
        # 数据打包
        data = struct.pack(PKT_FORMAT, 1 if flag else 0, u, v)
        packet = bytes([HEADER]) + data + bytes([TAIL])

        # 发送数据
        uart.write(packet)
        print(f"发送: 目标={flag}, u={u}, v={v}")

        # 等待2秒
        time.sleep(1)  # 毫秒
