import serial
import json
import numpy as np
import time

# =========== 1. 打开蓝牙串口 ===========
# 在 Windows “设备管理器→端口” 里查看 HC-06 对应的 COM 号
ser = serial.Serial(port='COM7',           # 改成你的端口号
                    baudrate=115200,
                    timeout=1)            # 读超时，可忽略

# =========== 2. 模拟检测结果 ===========
def fake_detect(frame_bgr, car_pose=None):
    """仅演示用，替换为你的真实检测函数"""
    return {
        'fence_quad': np.array([[100, 80], [200, 80], [200, 180], [100, 180]]),
        'target_center_uv': (150, 130),
        'target_box': np.array([[140, 120], [160, 120], [160, 140], [140, 140]]),
        'objects': [{'cls': 'cone', 'uv': (120, 110)}, {'cls': 'ball', 'uv': (170, 150)}],
        'nearest_obj': {'cls': 'cone', 'uv': (120, 110), 'dist': 0.8},
        'car_center_uvθ': (160, 120, 1.57)
    }

# =========== 3. 序列化：numpy → list ===========
def serialize(result):
    """把 numpy 数组转成 list，再整体 dump 成单行 JSON"""
    serializable = {}
    for k, v in result.items():
        if isinstance(v, np.ndarray):
            serializable[k] = v.tolist()
        elif isinstance(v, tuple):
            serializable[k] = list(v)  # 元组→列表，方便 STM32 解析
        else:
            serializable[k] = v
    return json.dumps(serializable, separators=(',', ':'))   # 压缩空格

# =========== 4. 主循环 ===========
if __name__ == '__main__':
    try:
        while True:
            # 1. 获取一帧（这里用 dummy）
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            result = fake_detect(frame)

            # 2. 序列化并发送
            packet = serialize(result) + '\n'
            ser.write(packet.encode('utf-8'))
            print(f"[PC] sent: {packet.strip()}")

            time.sleep(0.2)   # 20 ms 一帧，可按需求调整
    except KeyboardInterrupt:
        ser.close()
        print("Serial closed.")