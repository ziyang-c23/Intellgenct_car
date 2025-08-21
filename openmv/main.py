"""
main.py
-------
OpenMV主程序：车载前视相机物体检测与通信模块

模块说明：
    本程序用于智能小车的视觉检测系统，实现物体检测和数据传输的集成。
    作为车载前视相机，为小车提供实时的目标检测和位置引导。

功能特点：
    1. 物体检测：
       - 支持红色和黄色物体的实时检测
       - 基于LAB颜色空间的鲁棒分割算法
       - 使用形态学处理进行图像去噪
       - 智能的目标选择策略
    
    2. 数据通信：
       - 基于UART的可靠串口通信
       - 紧凑的数据包格式设计
       - 包含校验和的传输协议
       - 实时的数据反馈机制
    
    3. 实时处理：
       - 实时帧率监控与显示
       - 检测状态实时反馈
       - 完善的异常处理机制
       - 可配置的系统参数

运行流程：
    1. 系统初始化：
       - 加载配置参数
       - 初始化摄像头
       - 建立UART通信
    
    2. 主循环执行：
       - 图像采集
       - 目标检测
       - 结果处理
       - 数据发送
    
    3. 状态监控：
       - 帧率统计
       - 检测结果显示
       - 异常捕获与处理

使用说明：
    1. 确保硬件连接：
       - OpenMV摄像头正确固定
       - UART连接到STM32对应接口
    
    2. 参数调整：
       - 通过SystemConfig类配置参数
       - 可根据实际环境调整阈值
    
    3. 运行监控：
       - 观察系统输出的状态信息
       - 检查检测结果的准确性
       - 监控通信是否正常

作者：Ziyang Chen
版本：1.0
日期：2025-08-21
"""

import sensor
import image
import time
from OpenmvDetector import detect_object, DetectorConfig
from transmitter import UartTransmitter, UartConfig

class SystemConfig:
    """
    系统配置类
    集中管理所有系统参数，包括检测器、通信和摄像头的配置
    
    配置项：
        1. 检测器配置 (DETECTOR_CONFIG)
           - 面积阈值：目标最小面积占比
           - 形态学参数：去噪核大小
           - 颜色阈值：LAB空间的颜色范围
        
        2. 通信配置 (UART_CONFIG)
           - 串口参数：波特率、端口等
           - 协议参数：包头包尾标识
        
        3. 摄像头配置 (CAMERA_CONFIG)
           - 图像参数：格式、分辨率
           - 传感器参数：增益、白平衡
    """
    
    # 检测器参数配置
    DETECTOR_CONFIG = {
        'area_threshold': 0.005,    # 最小面积比例（占图像面积的0.5%）
        'morph_kernel': 3,          # 形态学核大小（用于开闭运算）
        'min_pixels': 30,           # 最小像素数（初步噪点过滤）
        'merge_blobs': True,        # 合并相邻区域（连通域处理）
        # LAB颜色空间阈值：(L_min, L_max, a_min, a_max, b_min, b_max)
        'threshold_red': [(15, 41, 11, 55, 4, 37)],      # 红色LAB阈值
        'threshold_yellow': [(35, 75, -12, 7, 36, 72)],  # 黄色LAB阈值
    }
    
    # 通信参数配置
    UART_CONFIG = {
        'uart_id': 1,              # UART端口号（与STM32连接）
        'baudrate': 115200,        # 波特率（与STM32需保持一致）
        'header': 0x3A,            # 数据包起始标识符（固定为0x3A）
        'tail': 0x0A,             # 数据包结束标识符（固定为0x0A）
    }
    
    # 摄像头参数配置
    CAMERA_CONFIG = {
        'pixformat': sensor.RGB565,  # 图像格式（用于色彩识别）
        'framesize': sensor.QVGA,    # 分辨率：320x240像素
        'skip_frames': 2000,         # 跳过帧数（等待自动参数稳定）
        'auto_gain': False,          # 关闭自动增益（保持图像稳定）
        'auto_whitebal': False,      # 关闭自动白平衡（保持颜色稳定）
    }

def init_camera():
    """
    初始化摄像头
    
    功能：
        配置并初始化OpenMV摄像头，设置合适的参数以获得稳定的图像。
    
    配置参数：
        1. 图像格式：RGB565
           - 适用于颜色识别
           - 每像素16位，平衡性能和精度
        
        2. 分辨率：QVGA (320x240)
           - 提供足够的细节
           - 保证处理速度
        
        3. 图像稳定性：
           - 关闭自动增益：避免亮度自动调节
           - 关闭白平衡：保持颜色检测的稳定性
           - 跳过初始帧：等待图像参数稳定
    
    注意事项：
        - 确保镜头清洁
        - 环境光照要充足均匀
        - 等待足够时间使图像稳定
    """
    sensor.reset()
    sensor.set_pixformat(SystemConfig.CAMERA_CONFIG['pixformat'])
    sensor.set_framesize(SystemConfig.CAMERA_CONFIG['framesize'])
    sensor.skip_frames(time=SystemConfig.CAMERA_CONFIG['skip_frames'])
    sensor.set_auto_gain(SystemConfig.CAMERA_CONFIG['auto_gain'])
    sensor.set_auto_whitebal(SystemConfig.CAMERA_CONFIG['auto_whitebal'])

def configure_system():
    """
    配置系统参数
    
    功能：
        统一配置系统所有模块的参数，确保各模块协调工作。
    
    配置内容：
        1. 检测器参数：
           - 面积阈值：过滤小目标
           - 形态学参数：图像去噪
           - 颜色阈值：目标识别条件
        
        2. 通信参数：
           - 串口配置：确保与STM32通信正常
           - 协议配置：定义数据包格式
        
    配置流程：
        1. 载入默认配置
        2. 应用到各个模块
        3. 验证配置有效性
    
    注意事项：
        - 确保参数在有效范围内
        - 参数修改后需要验证系统性能
        - 通信参数需与STM32端匹配
    """
    # 配置检测参数
    DetectorConfig.area_threshold = SystemConfig.DETECTOR_CONFIG['area_threshold']
    DetectorConfig.morph_kernel = SystemConfig.DETECTOR_CONFIG['morph_kernel']
    DetectorConfig.min_pixels = SystemConfig.DETECTOR_CONFIG['min_pixels']
    DetectorConfig.merge_blobs = SystemConfig.DETECTOR_CONFIG['merge_blobs']
    DetectorConfig.threshold_red = SystemConfig.DETECTOR_CONFIG['threshold_red']
    DetectorConfig.threshold_yellow = SystemConfig.DETECTOR_CONFIG['threshold_yellow']
    
    # 配置通信参数
    UartConfig.UART_ID = SystemConfig.UART_CONFIG['uart_id']
    UartConfig.BAUDRATE = SystemConfig.UART_CONFIG['baudrate']
    UartConfig.HEADER = SystemConfig.UART_CONFIG['header']
    UartConfig.TAIL = SystemConfig.UART_CONFIG['tail']

def print_system_info():
    """打印系统配置信息"""
    print("\n系统配置信息:")
    print("-------------")
    print("检测器配置:")
    print("- 最小面积比例:", DetectorConfig.area_threshold)
    print("- 形态学核大小:", DetectorConfig.morph_kernel)
    print("- 最小像素数:", DetectorConfig.min_pixels)
    print("- 合并相邻区域:", DetectorConfig.merge_blobs)
    
    print("\n通信配置:")
    print("- 串口号:", UartConfig.UART_ID)
    print("- 波特率:", UartConfig.BAUDRATE)
    print("- 包头/包尾: 0x%02X/0x%02X" % (UartConfig.HEADER, UartConfig.TAIL))
    
    print("\n摄像头配置:")
    print("- 分辨率:", "QVGA (320x240)")
    print("- 像素格式:", "RGB565")
    print("-------------\n")

def process_detection_result(result, transmitter):
    """
    处理检测结果
    
    功能：
        处理目标检测的结果，进行数据传输和状态显示。
    
    参数：
        result: dict
            检测返回的结果字典，包含：
            - IS_FIND_TARGET: bool, 是否检测到目标
            - u_target: int, 目标中心横坐标
            - v_target: int, 目标中心纵坐标
            
        transmitter: UartTransmitter
            串口通信对象，用于发送数据到STM32
    
    处理流程：
        1. 检查结果有效性
        2. 通过UART发送数据
        3. 显示检测状态
    
    显示信息：
        - 是否找到目标
        - 目标位置坐标
        - 其他调试信息
    """
    if result:
        transmitter.send_target(result)
        if result['IS_FIND_TARGET']:
            print("找到目标 - u: %d, v: %d" % (
                result['u_target'],
                result['v_target']
            ))
        else:
            print("未找到目标")

def main():
    """
    主函数
    
    功能：
        作为程序入口，协调各个模块工作，实现完整的检测和通信功能。
    
    执行流程：
        1. 初始化阶段：
           - 载入系统配置
           - 初始化摄像头
           - 建立通信连接
        
        2. 主循环阶段：
           - 图像采集
           - 目标检测
           - 数据处理
           - 结果发送
        
        3. 状态监控：
           - 帧率统计
           - 运行状态
           - 错误处理
    
    异常处理：
        - 捕获键盘中断
        - 处理运行时错误
        - 确保正常退出
    
    使用方式：
        直接运行此文件，程序会自动执行完整流程。
        可通过Ctrl+C终止运行。
    """
    try:
        # 系统初始化
        print("配置系统参数...")
        configure_system()
        
        print("初始化摄像头...")
        init_camera()
        
        print("初始化通信...")
        transmitter = UartTransmitter()
        
        # 打印系统信息
        print_system_info()
        
        # 主循环
        clock = time.clock()
        print("开始检测循环...\n")
        
        while True:
            clock.tick()  # 开始计时
            
            # 捕获图像
            img = sensor.snapshot()
            
            # 检测物体并处理结果
            result = detect_object(img)
            process_detection_result(result, transmitter)
            
            # 打印帧率
            print("FPS: %.1f" % clock.fps())
            
    except KeyboardInterrupt:
        print("\n程序已停止")
    except Exception as e:
        print("\n发生错误:", str(e))
    finally:
        print("程序结束")

if __name__ == "__main__":
    main()
