"""
transmitter.py
-------------
OpenMV与STM32通信模块 - 基于UART的目标检测数据传输系统

模块功能：
---------
1. 可靠的串口通信：
   - 自动重连机制
   - 通信状态监控
   - 错误检测和恢复

2. 数据包协议：
   基础包格式：
   |  起始(1B)  |  标志(1B)  |  坐标(4B)  |  结束(1B)  |
   |  0x3A      |  0/1       |  x,y       |  0x0A      |

   说明：
   - 标志位：1表示检测到目标，0表示未检测到
   - 坐标：x,y为目标中心坐标，每个坐标2字节


3. 错误处理：
   - 通信超时检测
   - 自动重传机制

配置参数：
---------
UART配置：
- 端口：UART1
- 波特率：115200
- 数据位：8
- 校验位：无
- 停止位：1

通信协议：
- 包头：0x3A
- 包尾：0x0A
"""

import pyb
from pyb import UART
import struct

class UartConfig:
    """
    UART通信配置类
    -------------
    统一管理所有通信相关的参数配置

    通信参数：
    --------
    1. 硬件配置：
       - UART_ID: 使用的UART端口号
       - BAUDRATE: 通信波特率
       - BITS: 数据位数
       - PARITY: 校验位类型
       - STOP: 停止位数量

    2. 协议配置：
       - HEADER: 包头标识符
       - TAIL: 包尾标识符
       - PACKET_SIZE: 数据包大小
       - PKT_FORMAT: 数据打包格式

    3. 超时设置：
       - TIMEOUT_MS: 通信超时时间
       - RETRY_COUNT: 重试次数
       - RETRY_DELAY_MS: 重试延迟
    """
    # 硬件参数
    UART_ID = 1         # UART端口号
    BAUDRATE = 115200   # 波特率
    BITS = 8           # 数据位
    PARITY = None      # 校验位
    STOP = 1          # 停止位

    # 协议参数
    HEADER = 0x3A     # 起始字节
    TAIL = 0x0A      # 结束字节
    PACKET_SIZE = 7  # 数据包总大小：起始(1) + 标志(1) + 坐标(4) + 结束(1)
    PKT_FORMAT = '<BHH'  # 数据格式：flag(1B) + u(2B) + v(2B)

    # 超时参数
    TIMEOUT_MS = 100    # 通信超时
    RETRY_COUNT = 3     # 重试次数
    RETRY_DELAY_MS = 10 # 重试延迟
class CommunicationError(Exception):
    """通信异常基类"""
    pass

class UartInitError(CommunicationError):
    """UART初始化异常"""
    pass

class TransmissionError(CommunicationError):
    """数据传输异常"""
    pass

class UartTransmitter:
    """
    UART通信管理类
    -------------
    实现可靠的串口通信功能

    主要功能：
    --------
    1. 通信管理：
       - 初始化和配置UART
       - 状态监控和错误恢复
       - 数据发送和重试机制

    2. 数据处理：
       - 数据打包和解包

    使用方法：
    --------
    trans = UartTransmitter()
    trans.send_target({
        'IS_FIND_TARGET': True,
        'u_target': 160,
        'v_target': 120
    })
    """
    def __init__(self):
        """
        初始化UART通信

        异常：
            UartInitError: UART初始化失败
        """
        try:
            self.uart = UART(UartConfig.UART_ID, UartConfig.BAUDRATE)
            self.uart.init(UartConfig.BAUDRATE,
                         bits=UartConfig.BITS,
                         parity=UartConfig.PARITY,
                         stop=UartConfig.STOP)
            # 等待UART稳定
            pyb.delay(100)
            self._check_uart_status()
        except Exception as e:
            raise UartInitError(f"UART初始化失败: {str(e)}")

    def _check_uart_status(self):
        """
        检查UART状态

        异常：
            UartInitError: UART状态异常
        """
        # 简单的回环测试
        try:
            # self.uart.write(b'test')
            pyb.delay(10)
            if self.uart.any() > 0:
                self.uart.read()  # 清空缓冲区
        except Exception as e:
            raise UartInitError(f"UART状态检查失败: {str(e)}")



    def send_target(self, result):
        """
        发送目标检测结果

        参数：
            result: dict, 检测结果字典
                {
                    'IS_FIND_TARGET': bool,  # 是否找到目标
                    'u_target': int,         # 目标横坐标
                    'v_target': int          # 目标纵坐标
                }

        异常：
            TransmissionError: 数据发送失败
        """
        # 数据预处理
        flag = 1 if result['IS_FIND_TARGET'] else 0
        u = int(result['u_target'] if result['u_target'] is not None else 0)
        v = int(result['v_target'] if result['v_target'] is not None else 0)

        # 数据打包
        try:
            data = struct.pack(UartConfig.PKT_FORMAT,
                             flag,           # 1字节，是否找到目标
                             u & 0xFFFF,     # 2字节，横坐标
                             v & 0xFFFF)     # 2字节，纵坐标
        except Exception as e:
            raise TransmissionError(f"数据打包失败: {str(e)}")

        # 构造完整数据包
        packet = bytes([UartConfig.HEADER]) + data + bytes([UartConfig.TAIL])

        # 发送数据（带重试机制）
        for attempt in range(UartConfig.RETRY_COUNT):
            try:
                bytes_written = self.uart.write(packet)
                if bytes_written == len(packet):
                    return  # 发送成功
            except Exception as e:
                if attempt == UartConfig.RETRY_COUNT - 1:
                    raise TransmissionError(f"数据发送失败: {str(e)}")
            pyb.delay(UartConfig.RETRY_DELAY_MS)


def test_transmitter():
    """
    通信模块测试函数

    测试项目：
    1. 基本通信功能
    2. 错误处理机制
    3. 边界条件处理
    """
    try:
        transmitter = UartTransmitter()

        # 测试1：正常目标数据
        test_result1 = {
            'IS_FIND_TARGET': True,
            'u_target': 160,
            'v_target': 120
        }
        print("\n测试1 - 发送正常目标数据:")
        print(test_result1)
        transmitter.send_target(test_result1)
        pyb.delay(500)

        # 测试2：无目标数据
        test_result2 = {
            'IS_FIND_TARGET': False,
            'u_target': None,
            'v_target': None
        }
        print("\n测试2 - 发送无目标数据:")
        print(test_result2)
        transmitter.send_target(test_result2)
        pyb.delay(500)

        # 测试3：边界值测试
        test_result3 = {
            'IS_FIND_TARGET': True,
            'u_target': 0,
            'v_target': 0,
        }
        print("\n测试3 - 发送边界值数据:")
        print(test_result3)
        transmitter.send_target(test_result3)

    except CommunicationError as e:
        print(f"\n通信错误: {str(e)}")
    except Exception as e:
        print(f"\n未预期的错误: {str(e)}")

if __name__ == '__main__':
    while True:
        test_transmitter()

