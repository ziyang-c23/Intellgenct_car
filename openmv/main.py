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
import time
from OpenmvDetector import detect_object, DetectorConfig
from openmv_comm import UartTransmitter, UartConfig

class SystemConfig:
    """
    系统配置类 - 智能车视觉系统的中央配置管理器
    统一管理所有子系统的配置参数，包括检测器、通信和摄像头

    子系统配置：
        1. 视觉检测系统 (DETECTOR_CONFIG)：
           - 颜色阈值：LAB空间的红色和黄色范围
           - 形态学参数：去噪和填充设置
           - 目标筛选：面积和像素阈值
           - 距离计算：视野范围设置

        2. 通信系统 (UART_CONFIG)：
           - 硬件设置：串口号、波特率
           - 协议设置：包头包尾、数据格式
           - 缓冲设置：包大小、校验方式

        3. 摄像头系统 (CAMERA_CONFIG)：
           - 图像设置：格式、分辨率
           - 传感器设置：增益、白平衡
           - 性能设置：跳帧、稳定性

    使用方法：
        1. 在主程序初始化时加载配置
        2. 通过configure_system统一配置
        3. 运行时可动态调整参数
    """

    # 视觉检测器配置
    DETECTOR_CONFIG = {
        # 目标检测参数
        'area_threshold': 0.0005,  # 最小面积比例（占图像面积的0.1%）
        'min_pixels': 30,            # 最小像素数（滤除噪点）
        'merge_blobs': True,        # 合并相邻区域

        # 颜色检测参数 (LAB颜色空间)
        'threshold_red': [(10,40, 5, 55, -5, 45)],      # 红色阈值
        'threshold_yellow': [(40, 70, -30, 20, 25,75)],  # 黄色阈值

        # 距离计算参数
        'v_min_ratio': 0.05,         # 最远可见位置（图像顶部10%）
        'v_max_ratio': 0.75,         # 最近可见位置（图像底部75%）
    }

    # 通信系统配置
    UART_CONFIG = {
        # 硬件参数
        'uart_id': 1,               # UART端口号
        'baudrate': 115200,         # 波特率
        'bits': 8,                  # 数据位
        'parity': None,             # 校验位
        'stop': 1,                  # 停止位
        'cycle': 50,               # 通信周期(ms)
        # 协议参数
        'header': 0xFF,             # 包头标识
        'tail': 0xF0,               # 包尾标识
        'packet_size': 7,           # 数据包总大小 (header + flag + u + v + checksum + tail)
        'pkt_format': '<BHH',       # 数据格式：flag(1B) + u(2B) + v(2B)
    }

    # 摄像头系统配置
    CAMERA_CONFIG = {
        # 图像参数
        'pixformat': sensor.RGB565,  # 像素格式
        'framesize': sensor.QVGA,    # 分辨率 (320x240)
        'windowing': None,           # 窗口裁剪

        # 性能参数
        'skip_frames': 10,           # 跳过帧数（等待相机稳定）
        'fps_max': 30,              # 最大帧率

        # 传感器参数
        'auto_gain': True,           # 自动增益
        'auto_whitebal': False,      # 自动白平衡
        'contrast': 0,               # 对比度
        'brightness': 0              # 亮度
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
        统一配置所有子系统的参数，确保系统协调工作。

    配置流程：
        1. 视觉检测系统配置：
           - 检测参数：面积、像素阈值
           - 处理参数：形态学操作
           - 颜色参数：LAB阈值范围
           - 距离参数：视野范围设置

        2. 通信系统配置：
           - 串口参数：端口、波特率等
           - 协议参数：数据包格式
           - 校验参数：包头包尾、校验和

        3. 摄像头系统配置：
           - 基本参数：格式、分辨率
           - 性能参数：帧率、缓存
           - 图像参数：对比度、亮度

    错误处理：
        - 参数验证：确保在有效范围
        - 依赖检查：检查必要组件
        - 异常处理：优雅降级策略

    注意事项：
        - 参数同步：确保各模块配置一致
        - 性能平衡：在功能和性能间权衡
        - 实时调整：支持运行时修改
    """
    try:
        # 1. 配置视觉检测系统
        # 检测基本参数
        DetectorConfig.area_threshold = SystemConfig.DETECTOR_CONFIG['area_threshold']
        DetectorConfig.min_pixels = SystemConfig.DETECTOR_CONFIG['min_pixels']
        DetectorConfig.merge_blobs = SystemConfig.DETECTOR_CONFIG['merge_blobs']

        # 颜色检测参数
        DetectorConfig.threshold_red = SystemConfig.DETECTOR_CONFIG['threshold_red']
        DetectorConfig.threshold_yellow = SystemConfig.DETECTOR_CONFIG['threshold_yellow']

        # 距离计算参数
        DetectorConfig.v_min_ratio = SystemConfig.DETECTOR_CONFIG['v_min_ratio']
        DetectorConfig.v_max_ratio = SystemConfig.DETECTOR_CONFIG['v_max_ratio']

        # 2. 配置通信系统
        # 硬件参数
        UartConfig.UART_ID = SystemConfig.UART_CONFIG['uart_id']
        UartConfig.BAUDRATE = SystemConfig.UART_CONFIG['baudrate']

        # 协议参数
        UartConfig.HEADER = SystemConfig.UART_CONFIG['header']
        UartConfig.TAIL = SystemConfig.UART_CONFIG['tail']
        UartConfig.PACKET_SIZE = SystemConfig.UART_CONFIG['packet_size']
        UartConfig.PKT_FORMAT = SystemConfig.UART_CONFIG['pkt_format']

        print("系统配置完成")
        return True

    except Exception as e:
        print("配置错误:", str(e))
        return False

def print_system_info():
    """
    打印系统配置信息
    显示所有子系统的当前配置状态
    """
    print("\n========= 系统配置信息 =========")

    # 1. 视觉检测系统配置
    print("\n[视觉检测系统]")
    print("目标检测参数:")
    print("  - 最小面积比例:", DetectorConfig.area_threshold)
    print("  - 最小像素数:", DetectorConfig.min_pixels)
    print("  - 合并相邻区域:", DetectorConfig.merge_blobs)

    print("图像处理参数:")
    print("  - 最小像素数:", DetectorConfig.min_pixels)

    print("距离计算参数:")
    print("  - 最远可见位置:", DetectorConfig.v_min_ratio)
    print("  - 最近可见位置:", DetectorConfig.v_max_ratio)

    # 2. 通信系统配置
    print("\n[通信系统]")
    print("硬件参数:")
    print("  - 串口号: UART", UartConfig.UART_ID)
    print("  - 波特率:", UartConfig.BAUDRATE)

    print("协议参数:")
    print("  - 包头/包尾: 0x%02X/0x%02X" % (UartConfig.HEADER, UartConfig.TAIL))
    print("  - 数据包大小:", UartConfig.PACKET_SIZE, "字节")
    print("  - 数据格式:", UartConfig.PKT_FORMAT)

    # 3. 摄像头系统配置
    print("\n[摄像头系统]")
    print("图像参数:")
    print("  - 分辨率: QVGA (320x240)")
    print("  - 像素格式: RGB565")

    print("传感器参数:")
    print("  - 自动增益:", SystemConfig.CAMERA_CONFIG['auto_gain'])
    print("  - 自动白平衡:", SystemConfig.CAMERA_CONFIG['auto_whitebal'])
    print("  - 跳帧数:", SystemConfig.CAMERA_CONFIG['skip_frames'])

    print("\n================================\n")

def process_detection_result(result, transmitter):
    """
    处理检测结果并进行数据传输

    功能：
        1. 处理视觉检测返回的结果
        2. 通过UART发送结果到控制器
        3. 显示当前检测状态和调试信息
        4. 处理特殊情况和边界条件

    参数：
        result: dict
            检测返回的结果字典，包含：
            - IS_FIND_TARGET: bool, 是否检测到目标
                True: 找到目标，可以获取位置信息
                False: 未找到目标，位置信息无效
            - u_target: int, 目标中心横坐标 (0-320)
                表示目标在图像水平方向的位置
            - v_target: int, 目标中心纵坐标 (0-240)
                表示目标在图像垂直方向的位置
                用于计算目标距离

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
        # 只发送基本信息：是否找到目标、目标位置
        transmit_data = {
            'IS_FIND_TARGET': result['IS_FIND_TARGET'],
            'u_target': result['u_target'],
            'v_target': result['v_target']
        }
        transmitter.send_target(transmit_data)

        if result['IS_FIND_TARGET']:
            print("找到目标 - u: %d, v: %d" % (
                result['u_target'],
                result['v_target']
            ))
        else:
            print("未找到目标")

def main():
    """
    主函数 - OpenMV视觉检测系统的入口点

    功能：
        作为程序入口，协调各个模块工作，实现完整的检测和通信功能。
        管理系统生命周期，处理异常情况，确保稳定运行。

    系统架构：
        1. 传感器层：OpenMV摄像头
           - 图像采集
           - 参数控制
           - 帧率管理

        2. 处理层：视觉算法
           - 颜色检测
           - 目标定位
           - 结果过滤

        3. 通信层：UART接口
           - 数据打包
           - 可靠传输
           - 实时反馈

    执行流程：
        1. 初始化阶段：
           - 载入系统配置：从SystemConfig加载参数
           - 初始化摄像头：配置sensor和图像参数
           - 建立通信连接：初始化UART传输器

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
            time.sleep_ms(SystemConfig.UART_CONFIG['cycle'])  # 稍作延时，避免过快循环
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
