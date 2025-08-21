
"""
OpenmvDetector.py
------------------
车载相机（OpenMV）物体检测模块
用于智能小车前向斜视相机的目标检测与抓取引导

功能特点：
    1. 支持红色和黄色物体的实时检测
    2. 基于LAB颜色空间的鲁棒物体分割
    3. 完整的形态学处理去噪
    4. 智能的目标选择策略
    5. 实时的抓取位置引导

整体流程：
    1. 图像预处理：
       - LAB颜色空间转换
       - 开运算去噪 + 闭运算填充
    2. 物体检测：
       - 红色和黄色的LAB阈值分割
       - 形态学处理消除噪点
    3. 目标筛选：
       - 面积阈值（>图像面积0.5%）
       - 按距离排序（v坐标）
    4. 特征提取：
       - 目标中心像素坐标(u,v)
       - 水平偏移量Δu
       - 归一化距离d_norm

配置说明：
    DetectorConfig类：
        threshold_red    : 红色LAB阈值 [(L_min,L_max,a_min,a_max,b_min,b_max)]
        threshold_yellow : 黄色LAB阈值
        morph_kernel    : 形态学处理核大小
        area_threshold  : 最小目标面积比例
        v_min_ratio    : 最远可见位置（图像高度比例）
        v_max_ratio    : 最近可见位置（图像高度比例）

返回数据：
    IS_FIND_TARGET : 是否找到目标物体
    u_target      : 目标中心横坐标
    v_target      : 目标中心纵坐标
    delta_u       : 水平偏移量（相对图像中心线）
    d_norm        : 归一化距离（0最近，1最远）
    area          : 目标面积（像素数）
    type          : 物体类型（'red'或'yellow'）

使用示例：
    result = detect_object(img)
    if result['IS_FIND_TARGET']:
        print(f"发现{result['type']}物体，距离：{result['d_norm']:.2f}")
"""

import sensor, time

# 摄像头配置
sensor.reset()
sensor.set_pixformat(sensor.RGB565)
sensor.set_framesize(sensor.QVGA)
sensor.skip_frames(10)
sensor.set_auto_gain(False)      # 固定增益
sensor.set_auto_whitebal(False)  # 固定白平衡
clock = time.clock()


class DetectorConfig:
    """
    检测器配置类
    统一管理所有物体检测相关的参数配置
    
    颜色阈值参数：
        threshold_red    : 红色物体的LAB阈值
            - L: 亮度通道 (0~100)
            - a: 红绿对立通道 (+红/-绿)
            - b: 黄蓝对立通道 (+黄/-蓝)
        threshold_yellow : 黄色物体的LAB阈值
            格式：[(L_min,L_max, a_min,a_max, b_min,b_max)]
    
    形态学处理：
        morph_kernel    : 形态学操作的核大小
            - 用于开运算（去噪）和闭运算（填充）
            - 较大的核会导致更强的滤波效果
    
    目标筛选参数：
        area_threshold  : 最小面积比例（占总图像面积）
            - 默认0.005，即0.5%
            - 用于过滤小面积噪点
        min_pixels     : 最小像素数（绝对值）
            - 默认30像素
            - 用于find_blobs初步筛选
        merge_blobs    : 是否合并相邻色块
            - True: 合并邻近的同色区域
            - 有助于处理目标被分割的情况
    
    距离计算参数：
        v_min_ratio    : 最远可见位置（图像顶部）
            - 默认0.1，即10%处
        v_max_ratio    : 最近可见位置（图像底部）
            - 默认0.9，即90%处
            - 用于计算归一化距离
    """
    # LAB颜色阈值（可根据实际调整）
    threshold_red = [(15, 41, 11, 55, 4, 37)]      # 红色LAB阈值
    threshold_yellow = [(35, 75, -12, 7, 36, 72)]  # 黄色LAB阈值
    
    # 形态学处理参数
    morph_kernel = 3      # 形态学核大小
    
    # 物体筛选参数
    area_threshold = 0.005  # 最小面积比例（占图像面积）
    min_pixels = 30        # 最小像素数
    merge_blobs = True     # 是否合并相邻区域
    
    # 距离计算参数
    v_min_ratio = 0.1     # 最远可见行（占图像高度比例）
    v_max_ratio = 0.9     # 最近可见行（占图像高度比例）

# 物体检测主流程


def detect_object(img):
    """
    检测图像中的红色和黄色物体，提供实时抓取引导信息

    主要功能：
        1. 物体检测：识别图像中的红色和黄色目标
        2. 目标筛选：按面积和距离选择最优目标
        3. 位置引导：计算目标的中心位置和偏移量
        4. 距离估计：基于图像坐标的归一化距离

    参数：
        img : OpenMV图像对象
            - 要求：RGB565格式
            - 建议：固定曝光和白平衡以提高稳定性

    处理流程：
        1. 图像预处理：
           - 复制原图避免修改
           - 开运算去除小噪点
           - 闭运算填充小孔洞
        2. 颜色检测：
           - 分别检测红色和黄色区域
           - 使用LAB空间的阈值分割
        3. 目标筛选：
           - 面积阈值筛选
           - 合并同类色块
        4. 目标选择：
           - 按v坐标排序
           - 选取最远目标
        5. 特征计算：
           - 计算目标中心坐标
           - 计算水平偏移量
           - 计算归一化距离

    返回值：
        dict: 包含以下键值对的字典
            IS_FIND_TARGET : bool, 是否检测到目标
            u_target      : int, 目标中心横坐标
            v_target      : int, 目标中心纵坐标
            delta_u       : int, 相对中心线的水平偏移
            d_norm        : float, 归一化距离[0,1]
            area          : int, 目标像素面积
            type          : str, 物体类型('red'/'yellow')

    注意事项：
        1. 未检测到目标时返回全None的字典
        2. 归一化距离中0表示最近，1表示最远
        3. 水平偏移量为正表示目标在图像右侧
    """
    # 复制图像并进行形态学处理
    temp = img.copy()
    kernel = DetectorConfig.morph_kernel
    temp.open(kernel, kernel)   # 开运算：去除小噪点
    temp.close(kernel, kernel)  # 闭运算：填充小孔
    
    # 计算面积阈值
    img_area = img.width() * img.height()
    min_area = int(img_area * DetectorConfig.area_threshold)
    
    # 检测红色物体
    red_blobs = temp.find_blobs(
        DetectorConfig.threshold_red,
        pixels_threshold=DetectorConfig.min_pixels,
        area_threshold=DetectorConfig.min_pixels,
        merge=DetectorConfig.merge_blobs
    )
    red_blobs = [b for b in red_blobs if b.pixels() > min_area] if red_blobs else []
    for b in red_blobs:
        b._color_type = 'red'
        
    # 检测黄色物体
    yellow_blobs = temp.find_blobs(
        DetectorConfig.threshold_yellow,
        pixels_threshold=DetectorConfig.min_pixels,
        area_threshold=DetectorConfig.min_pixels,
        merge=DetectorConfig.merge_blobs
    )
    yellow_blobs = [b for b in yellow_blobs if b.pixels() > min_area] if yellow_blobs else []
    for b in yellow_blobs:
        b._color_type = 'yellow'

    # 合并所有目标并检查是否找到物体
    all_blobs = red_blobs + yellow_blobs
    IS_FIND_TARGET = bool(all_blobs)
    if not IS_FIND_TARGET:
        return {
            'IS_FIND_TARGET': False,
            'u_target': None,
            'v_target': None,
            'delta_u': None,
            'd_norm': None,
            'area': None,
            'type': None
        }

    # 按v坐标升序排序（v越小越靠近图像上方，即越远）
    all_blobs.sort(key=lambda b: b.cy())
    target = all_blobs[0]  # 选取最远的目标（v最小）

    # 提取目标特征
    u_target = target.cx()                   # 目标中心横坐标
    v_target = target.cy()                   # 目标中心纵坐标
    u_centerline = img.width() // 2          # 图像中心线
    delta_u = u_target - u_centerline        # 水平偏移量

    # 计算归一化距离
    v_min = int(img.height() * DetectorConfig.v_min_ratio)  # 最远可见行
    v_max = int(img.height() * DetectorConfig.v_max_ratio)  # 最近可见行
    
    # 将v值限制在有效区间内并归一化（0表示最近，1表示最远）
    v_clipped = max(min(v_target, v_max), v_min)
    d_norm = 1.0 - (v_clipped - v_min) / (v_max - v_min)
    
    # 返回检测结果字典
    return {
        'IS_FIND_TARGET': IS_FIND_TARGET,  # 是否找到目标物体
        'u_target': u_target,      # 目标中心横坐标
        'v_target': v_target,      # 目标中心纵坐标
        'delta_u': delta_u,        # 水平偏移量
        'd_norm': d_norm,          # 归一化距离（0最近，1最远）
        'area': target.pixels(),   # 目标面积
        'type': target._color_type # 物体类型
    }

def test_camera_detection():
    clock = time.clock()
    print("持续采集并检测物体，按 Ctrl+C 停止...")
    while True:
        clock.tick()
        img = sensor.snapshot()
        result = detect_object(img)
        if result:
            print("检测到物体：", end=' ')
            print("中心(u,v):", result['u_target'], result['v_target'], end='; ')
            print("Δu:", result['delta_u'], end='; ')
            print("d_norm:", result['d_norm'], end='; ')
            print("面积:", result['area'], end='; ')
            print("类型:", result['type'])
        else:
            print("未检测到目标物体。")
        time.sleep(0.1)

def test_image_detection():
    import image
    img_path = "test.jpg"  # 直接在此处指定图片路径
    try:
        img = image.Image(img_path)
    except Exception as e:
        print(f"无法读取图片: {img_path}")
        print(e)
        return
    result = detect_object(img)
    if result:
        print("检测到物体：", end=' ')
        print("中心(u,v):", result['u_target'], result['v_target'], end='; ')
        print("Δu:", result['delta_u'], end='; ')
        print("d_norm:", result['d_norm'], end='; ')
        print("面积:", result['area'], end='; ')
        print("类型:", result['type'])
    else:
        print("未检测到目标物体。")

    clock = time.clock()

    print("持续采集并检测物体，按 Ctrl+C 停止...")
    while True:
        clock.tick()
        img = sensor.snapshot()
        result = detect_object(img)
        if result:
            print("检测到物体：", end=' ')
            print("中心(u,v):", result['u_target'], result['v_target'], end='; ')
            print("Δu:", result['delta_u'], end='; ')
            print("d_norm:", result['d_norm'], end='; ')
            print("面积:", result['area'], end='; ')
            print("类型:", result['type'])
        else:
            print("未检测到目标物体。")
        time.sleep(0.1)

# 直接运行测试
if __name__ == "__main__":
    # 直接在此处指定模式，无需 input。可选 'c'（摄像头）或 'i'（图片），其他为退出。
    mode = 'i'  # 修改为 'c' 或 'i'，如 mode = 'c' 进行摄像头实时检测
    if mode == 'c':
        test_camera_detection()
    elif mode == 'i':
        test_image_detection()
    else:
        print("已退出。")
