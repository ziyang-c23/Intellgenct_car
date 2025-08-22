"""
蓝牙通信模块 - STM32通信版本
--------------------------
本模块用于通过蓝牙向STM32发送整型数据指令。

通信协议：
1. 数据格式：整型数字
2. 数据范围：根据STM32程序定义
3. 发送格式：每个数字单独发送，以换行结束

典型用法：
    ser = open_serial()
    send_int_command(ser, 123)  # 发送单个整数命令
    close_serial(ser)           # 关闭串口
"""

import serial
import time
from typing import Optional, Union

class SerialCommError(Exception):
    """串口通信异常类"""
    pass

def open_serial(port: str = 'COM6', baudrate: int = 9600, timeout: int = 1) -> serial.Serial:
    """
    打开串口连接
    
    参数:
        port: 串口号
        baudrate: 波特率
        timeout: 超时时间（秒）
    
    返回:
        serial.Serial对象
    
    异常:
        SerialCommError: 串口打开失败时抛出
    """
    try:
        ser = serial.Serial(port, baudrate, timeout=timeout)
        return ser
    except serial.SerialException as e:
        raise SerialCommError(f"串口打开失败: {str(e)}")

def close_serial(ser: serial.Serial) -> None:
    """
    安全关闭串口
    
    参数:
        ser: 要关闭的串口对象
    """
    if ser and ser.is_open:
        ser.close()

def send_int_command(ser: serial.Serial, value: int) -> bool:
    """
    发送整型数据到STM32
    
    参数:
        ser: 串口对象
        value: 要发送的整数值
    
    返回:
        bool: 发送成功返回True
    
    异常:
        SerialCommError: 发送失败时抛出
        ValueError: 数值无效时抛出
    """
    try:
        # 数据验证
        if not isinstance(value, int):
            raise ValueError("只能发送整数")
            
        # 转换为字符串并添加换行符
        message = f"{value}\n"
        
        # 编码并发送
        ser.write(message.encode())
        return True
        
    except (serial.SerialException, ValueError) as e:
        raise SerialCommError(f"发送失败: {str(e)}")

def receive_int_response(ser: serial.Serial, timeout: float = 1.0) -> Optional[int]:
    """
    接收STM32返回的整型数据
    
    参数:
        ser: 串口对象
        timeout: 等待超时时间（秒）
    
    返回:
        int: 接收到的整数，超时返回None
    
    异常:
        SerialCommError: 接收出错时抛出
    """
    try:
        # 设置超时
        ser.timeout = timeout
        
        # 读取一行数据
        response = ser.readline().decode(errors='ignore').strip()
        
        # 尝试转换为整数
        if response:
            try:
                return int(response)
            except ValueError:
                raise SerialCommError(f"收到非整数数据: {response}")
        return None
        
    except serial.SerialException as e:
        raise SerialCommError(f"接收失败: {str(e)}")

def test_connection(ser: serial.Serial, test_value: int = 123) -> bool:
    """
    测试与STM32的通信
    
    参数:
        ser: 串口对象
        test_value: 测试用的整数值
    
    返回:
        bool: 测试成功返回True
    """
    try:
        print(f"发送测试值: {test_value}")
        send_int_command(ser, test_value)
        
        print("等待响应...")
        response = receive_int_response(ser)
        
        if response is not None:
            print(f"收到响应: {response}")
            return True
        else:
            print("未收到响应")
            return False
            
    except SerialCommError as e:
        print(f"测试失败: {str(e)}")
        return False

def interactive_test():
    """交互式测试函数"""
    try:
        # 打开串口
        ser = open_serial()
        print('串口已打开，输入整数进行测试，输入q退出')
        
        while True:
            # 获取用户输入
            user_input = input('>>> ').strip()
            
            # 检查退出命令
            if user_input.lower() == 'q':
                break
                
            # 尝试转换为整数并发送
            try:
                value = int(user_input)
                send_int_command(ser, value)
                print(f"已发送: {value}")
                
                # 接收响应
                response = receive_int_response(ser)
                if response is not None:
                    print(f"收到: {response}")
                else:
                    print("未收到响应")
                    
            except ValueError:
                print("请输入有效的整数")
            except SerialCommError as e:
                print(f"错误: {str(e)}")
                
    except SerialCommError as e:
        print(f"串口错误: {str(e)}")
        
    finally:
        # 确保关闭串口
        close_serial(ser)
        print("串口已关闭")

if __name__ == '__main__':
    interactive_test()
