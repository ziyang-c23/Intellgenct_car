"""
transmitter.py
-------------
OpenMV与STM32通信模块，通过UART传输目标检测结果。

通信协议：
    起始字节: 0x3A
    数据包:
        - flag (1字节): 是否找到目标 (0/1)
        - u    (2字节): 目标中心横坐标
        - v    (2字节): 目标中心纵坐标
    校验和:  1字节 (所有数据字节的和取低8位)
    结束字节: 0x0A
"""

import pyb
from pyb import UART
import struct

class UartConfig:
    """UART通信配置"""
    UART_ID = 1         # UART端口号
    BAUDRATE = 115200   # 波特率
    PACKET_SIZE = 11    # 数据包大小（不含校验和和结束字节）
    
    # 数据包标识符
    HEADER = 0x3A       # 起始字节
    TAIL = 0x0A        # 结束字节
    
    # 数据类型和格式
    PKT_FORMAT = '<BHH'  # 格式：flag, u, v (little-endian)
    
class UartTransmitter:
    """UART通信类"""
    def __init__(self):
        """初始化UART通信"""
        self.uart = UART(UartConfig.UART_ID, UartConfig.BAUDRATE)
        self.uart.init(UartConfig.BAUDRATE, bits=8, parity=None, stop=1)
        pyb.delay(1000)  # 等待UART稳定
        
    def calculate_checksum(self, data):
        """计算校验和：所有字节的和取低8位"""
        return sum(data) & 0xFF
        
    def send_target(self, result):
        """
        发送目标检测结果到STM32
        
        参数：
            result: dict, 包含以下键值：
                IS_FIND_TARGET: bool, 是否找到目标
                u_target: int, 目标横坐标
                v_target: int, 目标纵坐标
                delta_u: int, 水平偏移
                d_norm: float, 归一化距离[0,1]
        """
        # 准备数据
        flag = 1 if result['IS_FIND_TARGET'] else 0
        u = int(result['u_target'] if result['u_target'] is not None else 0)
        v = int(result['v_target'] if result['v_target'] is not None else 0)
        
        # 打包数据
        data = struct.pack(UartConfig.PKT_FORMAT,
                         flag,           # 1字节，是否找到目标
                         u & 0xFFFF,     # 2字节，横坐标
                         v & 0xFFFF)     # 2字节，纵坐标
        
        # 计算校验和
        checksum = self.calculate_checksum(data)
        
        # 构造完整数据包
        packet = bytes([UartConfig.HEADER]) + \
                data + \
                bytes([checksum]) + \
                bytes([UartConfig.TAIL])
                
        # 发送数据
        self.uart.write(packet)

'''
对应的STM32解包代码：

#pragma pack(push, 1)
typedef struct {
    uint8_t  header;    // 0x3A
    uint8_t  flag;      // 0/1
    uint16_t u;         // 目标横坐标
    uint16_t v;         // 目标纵坐标
    uint8_t  checksum;  // 校验和
    uint8_t  tail;      // 0x0A
} __attribute__((packed)) TargetPkt_t;
#pragma pack(pop)

// 接收并解析数据
bool receive_target(UART_HandleTypeDef *huart, TargetPkt_t *pkt) {
    // 接收完整数据包
    if (HAL_UART_Receive(huart, (uint8_t*)pkt, sizeof(TargetPkt_t), 100) != HAL_OK) {
        return false;
    }
    
    // 验证包头和包尾
    if (pkt->header != 0x3A || pkt->tail != 0x0A) {
        return false;
    }
    
    // 计算并验证校验和
    uint8_t sum = 0;
    uint8_t *data = (uint8_t*)pkt + 1;  // 跳过header
    for (int i = 0; i < sizeof(TargetPkt_t) - 3; i++) {  // -3跳过checksum和tail
        sum += data[i];
    }
    sum &= 0xFF;
    
    if (sum != pkt->checksum) {
        return false;
    }
    
    return true;
}
'''

def test_transmitter():
    """测试UART通信"""
    transmitter = UartTransmitter()
    
    # 测试用例1：找到目标
    test_result1 = {
        'IS_FIND_TARGET': True,
        'u_target': 160,
        'v_target': 120
    }
    print("测试1 - 发送目标数据:")
    print(test_result1)
    transmitter.send_target(test_result1)
    pyb.delay(1000)
    
    # 测试用例2：未找到目标
    test_result2 = {
        'IS_FIND_TARGET': False,
        'u_target': None,
        'v_target': None
    }
    print("测试2 - 发送无目标数据:")
    print(test_result2)
    transmitter.send_target(test_result2)

if __name__ == '__main__':
    test_transmitter()

