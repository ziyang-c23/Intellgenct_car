"""
蓝牙通信模块 - STM32通信版本
--------------------------
本模块用于通过蓝牙向STM32发送整型数据指令和结构化数据。主要用于智能小车项目中
视觉处理系统(CamoStudio)与STM32控制器之间的通信。

通信协议：
1. 基本指令格式：整型数字，以换行结束
2. 结构化数据格式：二进制数据包，包含包头、数据、包尾
3. 数据范围：根据STM32程序定义的参数限制

典型用法：
    # 基本命令发送
    ser = open_serial()
    send_int_command(ser, 123)  # 发送单个整数命令
    
    # 结构化数据发送
    data = {
        'SEARCH_OBJ_NUM': 1,     # 搜索到的物体数量
        'item_angle': -300,      # 物体角度*10 (-30.0度)
        'item_distance': 150,    # 物体距离(像素)
        'home_angle': 900,       # 家的角度*10 (90.0度)
        'home_distance': 500,    # 家的距离(像素)
        'item_out_of_bounds': 0  # 物体是否超出边界区域
    }
    send_camostudio_data(ser, data)
    
    # 完成后关闭
    close_serial(ser)           # 关闭串口
"""

import serial
import time
from typing import Optional, Union

class SerialCommError(Exception):
    """串口通信异常类，用于封装所有与串口通信相关的错误，提供统一的错误处理机制"""
    pass

def open_serial(port: str = 'COM7', baudrate: int = 115200, timeout: int = 1) -> serial.Serial:
    """
    打开串口连接
    
    参数:
        port: 串口号，默认为'COM7'（Windows系统）
        baudrate: 波特率，默认115200
        timeout: 超时时间（秒），默认1秒
    
    返回:
        serial.Serial对象：成功建立的串口连接
    
    异常:
        SerialCommError: 串口打开失败时抛出（如端口不存在或被占用）
    """
    try:
        ser = serial.Serial(port, baudrate, timeout=timeout)
        # 清空缓冲区
        ser.reset_input_buffer()
        ser.reset_output_buffer()
        return ser
    except serial.SerialException as e:
        raise SerialCommError(f"串口打开失败: {str(e)}")

def close_serial(ser: serial.Serial) -> None:
    """
    安全关闭串口
    
    参数:
        ser: 要关闭的串口对象
        
    说明:
        该函数会检查串口是否存在并已打开，然后安全关闭
        可以在任何情况下调用，即使串口未打开或为None
    """
    if ser and ser.is_open:
        ser.close()

def send_int_command(ser: serial.Serial, value: int) -> bool:
    """
    发送整型数据到STM32
    
    参数:
        ser: 串口对象，必须是已打开的串口
        value: 要发送的整数值
    
    返回:
        bool: 发送成功返回True
    
    异常:
        SerialCommError: 发送失败时抛出（如串口断开）
        ValueError: 数值无效时抛出（非整数类型）
    
    说明:
        发送的数据格式为文本，会自动在末尾添加换行符
    """
    try:
        # 数据验证
        if not isinstance(value, int):
            raise ValueError("只能发送整数")
            
        # 转换为字符串并添加换行符
        message = f"{value}\n"
        
        # 编码并发送
        ser.write(message.encode())
        ser.flush()  # 确保数据被发送出去
        return True
        
    except (serial.SerialException, ValueError) as e:
        raise SerialCommError(f"发送失败: {str(e)}")

def receive_str_response(ser: serial.Serial, timeout: float = 1.0) -> Optional[str]:
    """
    接收STM32返回的字符串数据
    
    参数:
        ser: 串口对象，必须是已打开的串口
        timeout: 等待超时时间（秒），默认1秒
    
    返回:
        str: 接收到的字符串
        None: 超时或未接收到数据时返回
    
    异常:
        SerialCommError: 接收出错时抛出（串口错误）
    
    说明:
        接收数据采用readline方法，预期数据末尾有换行符
        使用decode时忽略可能出现的解码错误，提高通信稳定性
    """
    try:
        # 设置超时
        ser.timeout = timeout
        
        # 读取一行数据
        response = ser.readline().decode(errors='ignore').strip()
        
        # 检查是否接收到数据
        if response:
            return response
        return None
        
    except serial.SerialException as e:
        raise SerialCommError(f"接收失败: {str(e)}")

def receive_int_response(ser: serial.Serial, timeout: float = 1.0) -> Optional[int]:
    """
    接收STM32返回的整型数据
    
    参数:
        ser: 串口对象，必须是已打开的串口
        timeout: 等待超时时间（秒），默认1秒
    
    返回:
        int: 接收到的整数
        None: 超时或未接收到数据时返回
    
    异常:
        SerialCommError: 接收出错时抛出（串口错误或接收到非整数数据）
    
    说明:
        接收数据采用readline方法，预期数据末尾有换行符
        使用decode时忽略可能出现的解码错误，提高通信稳定性
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

def send_camostudio_data(ser: serial.Serial, data_dict: dict, header: int = 0xAA, tail: int = 0x55) -> bool:
    """
    发送特定结构的数据包到STM32
    
    数据格式:
        {
            'SEARCH_OBJ_NUM': int,      # 当前搜索到的物体数量
            'item_angle': int,          # 到物体的相对角度*10，范围[-1800,1800]
            'item_distance': int,       # 到物体的像素距离，无效时=0
            'home_angle': int,          # 到家的相对角度*10，范围[-1800,1800]
            'home_distance': int,       # 到家的像素距离，无效时=0
            'item_out_of_bounds': int   # 物体是否超出内收矩形区域，0=否，1=是
        }
    
    通信协议:
        二进制数据包格式: <头部1字节><物体数量1字节><物体超界1字节><目标角度2字节><目标距离2字节><尾部1字节>
        使用小端序(little-endian)排列
        
        * 当物体数量为0时，传输家的信息（家的角度和距离）
        * 当物体数量大于0时，传输最近物体的信息（物体的角度和距离）
    
    参数:
        ser: 串口对象，必须是已打开的串口
        data_dict: 按上述格式组织的数据字典
        header: 包头标识符，默认为0xAA (170)
        tail: 包尾标识符，默认为0x55 (85)
    
    返回:
        bool: 发送成功返回True
    
    异常:
        SerialCommError: 发送失败时抛出（如串口错误、数据打包失败）
        ValueError: 数据格式错误或数值范围超出限制时抛出
    """
    try:
        # 验证数据结构
        if not isinstance(data_dict, dict):
            raise ValueError("数据必须是字典类型")

        required_keys = ['SEARCH_OBJ_NUM', 'item_angle', 'item_distance', 
                         'self_home_angle', 'self_home_distance', 
                         'oppo_home_angle', 'oppo_home_distance',
                         'item_out_of_bounds']
        for key in required_keys:
            if key not in data_dict:
                raise ValueError(f"数据字典缺少必要的键: {key}")
        
        # 从字典中提取数据
        search_obj_num = data_dict['SEARCH_OBJ_NUM']
        item_angle = data_dict['item_angle']
        item_distance = data_dict['item_distance']
        self_home_angle = data_dict['self_home_angle']
        self_home_distance = data_dict['self_home_distance']
        oppo_home_angle = data_dict['oppo_home_angle']
        oppo_home_distance = data_dict['oppo_home_distance']
        item_out_of_bounds = data_dict['item_out_of_bounds']
        
        # 验证头部和尾部标识符
        if not isinstance(header, int) or not (0 <= header <= 255):
            raise ValueError(f"包头标识符必须是0-255之间的整数: {header}")
        if not isinstance(tail, int) or not (0 <= tail <= 255):
            raise ValueError(f"包尾标识符必须是0-255之间的整数: {tail}")
        
            
        # 构建二进制数据包
        # 格式: <头部1字节><物体数量1字节><物体超界1字节><目标角度2字节><目标距离2字节><尾部1字节>
        # 注意: 使用小端序(little-endian)，h表示16位有符号整数
        import struct
        
        # 根据搜索到的物体数量决定传输哪些信息
        target_angle = 0
        target_distance = 0
        
        # if search_obj_num > 0:
        #     # 如果有物体，传输物体信息
        #     target_angle = item_angle
        #     target_distance = item_distance
        #     print(f"打包物体信息: 角度={target_angle}, 距离={target_distance}")
        # else:
        #     # 如果没有物体，传输家的信息
        #     target_angle = home_angle
        #     target_distance = home_distance
        #     print(f"打包家的信息: 角度={target_angle}, 距离={target_distance}")
        
        packet = struct.pack('<BBBhhhhhhB',
                            header & 0xFF,
                            search_obj_num & 0xFF,
                            item_out_of_bounds & 0xFF,
                            item_angle,
                            item_distance,
                            self_home_angle,
                            self_home_distance,
                            oppo_home_angle,
                            oppo_home_distance,
                            tail & 0xFF)
        
        # 发送数据包
        bytes_sent = ser.write(packet)
        return bytes_sent == len(packet)
        
    except Exception as e:
        if "struct" in str(e):
            raise SerialCommError(f"数据打包失败: {str(e)}")
        elif isinstance(e, serial.SerialException):
            raise SerialCommError(f"发送失败: {str(e)}")
        else:
            raise

def test_connection(ser: serial.Serial, test_value: int = 123) -> bool:
    """
    测试与STM32的通信连接
    
    参数:
        ser: 串口对象，必须是已打开的串口
        test_value: 测试用的整数值，默认为123
    
    返回:
        bool: 测试成功返回True，失败返回False
    
    说明:
        该函数会发送一个测试值并等待响应，适用于检查通信链路是否畅通
        会在控制台输出测试过程的日志信息
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
    """
    交互式测试函数
    
    说明:
        该函数提供命令行交互界面，允许用户输入整数进行发送测试
        输入'q'退出测试
        每次发送指令后会短暂等待设备响应
        
    使用方法:
        在命令行中运行: python camostudio_comm.py
    """
    try:
        # 打开串口
        ser = open_serial()
        print('串口已打开，输入整数进行测试，输入q退出')
        
        # 添加初始化延时，确保蓝牙设备准备就绪
        time.sleep(0.5)
        
        while True:
            # 获取用户输入
            user_input = input('>>> ').strip()
            
            # 检查退出命令
            if user_input.lower() == 'q':
                break
                
            # 尝试转换为整数并发送
            try:
                value = int(user_input)
                
                # 设置较短的超时时间防止长时间阻塞
                original_timeout = ser.timeout
                ser.timeout = 0.5
                
                send_int_command(ser, value)
                print(f"已发送: {value}")
                
                # 恢复原始超时设置
                ser.timeout = original_timeout
                
                # # 接收响应
                # response = receive_int_response(ser)
                # if response is not None:
                #     print(f"收到: {response}")
                # else:
                #     print("未收到响应")
                    
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

def test_camostudio_data():
    """
    测试结构化数据传输函数
    
    说明:
        该函数测试结构化数据包的发送功能，包含三个测试用例:
        1. 物体数量 > 0，传输物体信息（发送物体的角度和距离）
        2. 物体数量 = 0，传输家的信息（发送家的角度和距离）
        3. 使用自定义包头和包尾进行通信测试
        
    在每个测试后，会自动检查设备响应情况
        
    使用方法:
        在命令行中运行: python camostudio_comm.py --camostudiodata
    """
    ser = None
    try:
        # 打开串口
        print("正在打开串口...")
        ser = open_serial()
        print("串口已打开")
        
        # 测试数据1 - 有物体的情况
        data1 = {
            'SEARCH_OBJ_NUM': 3,     # 当前搜索到的物体数量
            'item_angle': -400,      # -40.0度
            'item_distance': 320,    # 320像素
            'home_angle': 1200,      # 120.0度
            'home_distance': 280,    # 280像素
            'item_out_of_bounds': 1  # 物体超出内收矩形区域
        }
        
        # 测试数据2 - 没有物体的情况
        data2 = {
            'SEARCH_OBJ_NUM': 0,     # 当前搜索到的物体数量
            'item_angle': 0,         # 物体角度无效
            'item_distance': 0,      # 物体距离无效
            'home_angle': 900,       # 90.0度
            'home_distance': 500,    # 500像素
            'item_out_of_bounds': 0  # 物体超出标志无效
        }
        
        # 测试1: 有物体的情况
        print("\n测试1: 有物体的情况（传输物体信息）")
        print(f"发送结构化数据: {data1}")
        success = send_camostudio_data(ser, data1)
        
        if success:
            print("数据发送成功")
            check_response(ser)
        
        # 测试2: 没有物体的情况
        print("\n测试2: 没有物体的情况（传输家的信息）")
        print(f"发送结构化数据: {data2}")
        success = send_camostudio_data(ser, data2)
        
        if success:
            print("数据发送成功")
            check_response(ser)
        
        if success:
            print("数据发送成功")
            check_response(ser)
            
    except SerialCommError as e:
        print(f"通信错误: {str(e)}")
    except ValueError as e:
        print(f"数据错误: {str(e)}")
    finally:
        # 确保关闭串口
        if ser is not None:
            close_serial(ser)
            print("串口已关闭")

def check_response(ser):
    """
    检查串口响应
    
    参数:
        ser: 串口对象，必须是已打开的串口
        
    说明:
        该函数会等待0.5秒，然后读取串口缓冲区中的所有数据
        如果收到响应，会尝试解析第一个字节作为状态码
        状态码0表示成功，其他值表示错误
        
    返回:
        无返回值，但会在控制台打印响应状态
    """
    print("等待响应...")
    time.sleep(0.5)  # 等待响应
    
    if ser.in_waiting > 0:
        response = ser.read(ser.in_waiting)
        print(f"收到原始响应: {response.hex()}")
        
        # 尝试解析响应
        try:
            if len(response) >= 2:
                status = response[0]
                print(f"状态码: {status}")
                if status == 0:
                    print("设备报告: 成功接收")
                else:
                    print(f"设备报告: 错误代码 {status}")
        except Exception as e:
            print(f"解析响应失败: {str(e)}")
    else:
        print("未收到响应")

if __name__ == '__main__':
    import sys
    
    # 检查命令行参数
    if len(sys.argv) > 1 and sys.argv[1] == '--camostudiodata':
        # 结构化数据测试
        test_camostudio_data()
    else:
        # 默认交互式测试
        print("使用方法:")
        print("  python camostudio_comm.py           # 交互式测试（发送基本命令）")
        print("  python camostudio_comm.py --camostudiodata  # 结构化数据测试（发送数据包）")
        print("\n启动交互式测试...\n")
        interactive_test()
