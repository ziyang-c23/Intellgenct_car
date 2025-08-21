
"""
OpenmvDetector.py
------------------
智能小车视觉导航模块 - 目标检测与位置引导系统
基于OpenMV实现的实时物体检测与抓取导航

模块概述：
    本模块为智能小车提供实时的视觉导航能力，通过前向斜视相机
    实现对红色和黄色目标物体的检测、定位和抓取引导。

核心功能：
    1. 双色目标检测：
       - 红色物体实时识别
       - 黄色物体实时识别
       - 基于LAB颜色空间的稳定检测

    2. 智能目标选择：
       - 基于距离的优先级排序
       - 自动选择最近目标
       - 面积过滤去除干扰

    3. 精确位置引导：
       - 实时目标坐标定位
       - 水平偏移量计算
       - 距离估计与标准化

    4. 可视化调试：
       - 实时目标标记显示
       - 距离与位置可视化
       - 检测状态实时反馈

系统流程：
    1. 图像获取：
       - 320x240分辨率
       - RGB565颜色格式
       - 固定曝光参数

    2. 目标检测：
       - LAB颜色空间变换
       - 双通道并行检测
       - 区域特征提取

    3. 智能筛选：
       - 面积阈值过滤
       - 最近目标优先
       - 噪声干扰消除

    4. 位置计算：
       - 目标中心定位(u,v)
       - 偏移量分析
       - 距离归一化处理

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
sensor.set_auto_gain(True)      # 固定增益
sensor.set_auto_whitebal(False)  # 固定白平衡
clock = time.clock()


class DetectorConfig:
    """
    视觉检测配置类
    -------------
    集中管理视觉检测系统的所有可调参数，实现灵活的检测策略调整

    颜色检测参数：
    -------------
    1. 红色目标检测 (threshold_red):
        LAB颜色空间参数:
        - L: 亮度分量 [0-100]
          * 较大范围可适应不同光照
          * 建议范围：20-80
        - a: 红绿对立分量 [+128/-128]
          * 正值代表红色倾向
          * 负值代表绿色倾向
          * 建议范围：15-70
        - b: 黄蓝对立分量 [+128/-128]
          * 正值代表黄色倾向
          * 负值代表蓝色倾向
          * 建议范围：-20-40

    2. 黄色目标检测 (threshold_yellow):
        LAB参数特点:
        - L: 通常需要较高亮度
        - a: 允许适度红色分量
        - b: 需要较强黄色分量

    目标筛选配置：
    -------------
    1. 面积筛选 (area_threshold):
        - 含义：目标最小面积占图像比例
        - 单位：小数（如0.0005表示0.05%）
        - 作用：过滤小面积噪点
        - 建议：根据实际目标大小调整

    2. 像素阈值 (min_pixels):
        - 含义：目标最小像素数
        - 单位：像素
        - 作用：初步快速筛选
        - 建议：保持较小值，与面积筛选配合

    3. 区域合并 (merge_blobs):
        - 功能：合并相邻同色区域
        - 设置：True/False
        - 场景：目标可能被分割时使用
        - 注意：可能影响检测精度

    距离估计参数：
    -------------
    1. 视野范围 (v_min_ratio, v_max_ratio):
        - v_min_ratio: 最远可见位置（图像顶部）
        - v_max_ratio: 最近可见位置（图像底部）
        - 范围：[0-1]的浮点数
        - 作用：计算目标距离
        - 建议：根据相机安装角度调整
    """
    # LAB颜色阈值（优化后的阈值）
    threshold_red = [(20, 80, 15, 70, -20, 40)]      # 红色LAB阈值（扩大检测范围）
    threshold_yellow = [(50, 100, -30, 20, 25, 75)]  # 黄色LAB阈值（扩大检测范围）

    # 物体筛选参数
    area_threshold = 0.00005  # 最小面积比例（占图像面积）
    min_pixels =  5       # 最小像素数
    merge_blobs = True     # 是否合并相邻区域

    # 距离计算参数
    v_min_ratio = 0.1     # 最远可见行（占图像高度比例）
    v_max_ratio = 0.9     # 最近可见行（占图像高度比例）

# 物体检测主流程


def detect_object(img):
    """
    目标检测与定位核心函数
    ---------------------
    实现对图像中红色和黄色目标的检测、选择和位置分析

    功能描述：
    ---------
    1. 双色目标检测：
       - 红色目标：使用LAB空间阈值分割
       - 黄色目标：使用LAB空间阈值分割
       - 同时处理多个目标

    2. 智能目标选择：
       - 面积过滤：去除小面积噪点
       - 距离优先：选择最近的目标
       - 分类标记：区分红色和黄色目标

    3. 精确定位分析：
       - 中心点坐标：确定目标精确位置
       - 水平偏移：计算与中线距离
       - 距离估计：基于垂直位置归一化

    参数说明：
    ---------
    img : OpenMV图像对象
        格式要求：
        - 颜色：RGB565格式
        - 分辨率：320x240（QVGA）
        - 配置：建议固定曝光和白平衡

    工作流程：
    ---------
    1. 目标检测阶段：
       - 并行检测红色和黄色目标
       - 应用面积和像素数阈值
       - 记录每个目标的颜色类型

    2. 目标筛选阶段：
       - 合并两种颜色的检测结果
       - 按垂直位置（距离）排序
       - 选择最近的有效目标

    3. 特征提取阶段：
       - 计算目标中心坐标(u,v)
       - 测量水平偏移量(Δu)
       - 估算归一化距离(d_norm)

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
    # 计算图像尺寸和区域阈值
    img_area = img.width() * img.height()
    min_area = int(img_area * DetectorConfig.area_threshold)

    # 检测红色物体
    red_blobs = img.find_blobs(
        DetectorConfig.threshold_red,
        pixels_threshold=min(DetectorConfig.min_pixels, min_area),  # 使用较小的值
        area_threshold=min(DetectorConfig.min_pixels, min_area),  # 使用较小的值
        merge=DetectorConfig.merge_blobs,  # 是否合并区域
        margin=1  # 允许边缘检测
    )
    # 将blob对象和颜色类型打包在元组中
    red_blobs = [(b, 'red') for b in red_blobs if b.pixels() > min_area] if red_blobs else []

    # 检测黄色物体
    yellow_blobs = img.find_blobs(
        DetectorConfig.threshold_yellow,
        pixels_threshold=min(DetectorConfig.min_pixels, min_area),  # 使用较小的值
        area_threshold=min(DetectorConfig.min_pixels, min_area),  # 使用较小的值
        merge=DetectorConfig.merge_blobs,  # 是否合并区域
        margin=1  # 允许边缘检测
    )
    # 将blob对象和颜色类型打包在元组中
    yellow_blobs = [(b, 'yellow') for b in yellow_blobs if b.pixels() > min_area] if yellow_blobs else []

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

    # 按v坐标降序排序（v越大越靠近图像下方，即越近）
    all_blobs.sort(key=lambda x: -x[0].cy())  # 使用负值实现降序，选择y坐标最大的
    target_blob, target_type = all_blobs[0]  # 选取最近的目标（v最大）

    # 提取目标特征
    u_target = target_blob.cx()                   # 目标中心横坐标
    v_target = target_blob.cy()                   # 目标中心纵坐标
    u_centerline = img.width() // 2              # 图像中心线
    delta_u = u_target - u_centerline            # 水平偏移量

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
        'area': target_blob.pixels(),   # 目标面积
        'type': target_type        # 物体类型
    }

def test_camera_detection():
    """
    相机实时检测测试函数
    ------------------
    提供实时视觉检测的可视化界面，用于调试和参数优化

    显示内容：
    --------
    1. 检测标记：
       - 红色目标：红色矩形框 + 品红十字
       - 黄色目标：黄色矩形框 + 青色十字
       - 选中目标：绿色加粗十字

    2. 参考线：
       - 蓝色水平线：最远和最近检测范围
       - 白色十字线：图像中心参考线
       - 绿色连线：目标到中心的偏移

    3. 状态信息：
       - 左上角：检测到的目标数量
       - 右上角：实时帧率
       - 底部：选中目标的详细信息
    """
    clock = time.clock()
    print("持续采集并检测物体，按 Ctrl+C 停止...")
    
    # 定义绘制颜色（RGB565格式）
    RED = (255, 0, 0)      # 红色目标框
    GREEN = (0, 255, 0)    # 选中目标标记
    BLUE = (0, 0, 255)     # 范围参考线
    YELLOW = (255, 255, 0) # 黄色目标框
    WHITE = (255, 255, 255)# 文字和中心线
    CYAN = (0, 255, 255)   # 黄色目标中心点
    MAGENTA = (255, 0, 255)# 红色目标中心点
    
    while True:
        clock.tick()
        img = sensor.snapshot()
        
        # 绘制视野范围线
        v_min = int(img.height() * DetectorConfig.v_min_ratio)  # 最远可见行
        v_max = int(img.height() * DetectorConfig.v_max_ratio)  # 最近可见行
        img.draw_line(0, v_min, img.width(), v_min, color=BLUE)  # 最远线
        img.draw_line(0, v_max, img.width(), v_max, color=BLUE)  # 最近线
        
        # 绘制中心十字线
        center_x = img.width() // 2
        center_y = img.height() // 2
        img.draw_line(center_x, 0, center_x, img.height(), color=WHITE)  # 垂直线
        img.draw_line(0, center_y, img.width(), center_y, color=WHITE)  # 水平线
        
        # 直接进行颜色检测，跳过形态学处理
        # 检测红色物体
        red_blobs = img.find_blobs(
            DetectorConfig.threshold_red, # type: ignore
            pixels_threshold=min(DetectorConfig.min_pixels, int(img.width() * img.height() * DetectorConfig.area_threshold)),
            area_threshold=min(DetectorConfig.min_pixels, int(img.width() * img.height() * DetectorConfig.area_threshold)),
            merge=False,
            margin=1
        )
        
        # 检测黄色物体
        yellow_blobs = img.find_blobs(
            DetectorConfig.threshold_yellow, # type: ignore
            pixels_threshold=min(DetectorConfig.min_pixels, int(img.width() * img.height() * DetectorConfig.area_threshold)),
            area_threshold=min(DetectorConfig.min_pixels, int(img.width() * img.height() * DetectorConfig.area_threshold)),
            merge=False,
            margin=1
        )        # 标记所有检测到的物体
        detected_count = {'red': 0, 'yellow': 0}
        
        # 绘制所有红色物体
        if red_blobs:
            for blob in red_blobs:
                # 绘制矩形框
                img.draw_rectangle(blob.x(), blob.y(), blob.w(), blob.h(), color=RED)
                # 绘制十字标记
                img.draw_cross(blob.cx(), blob.cy(), color=MAGENTA, size=10)
                # 显示面积信息
                area_percent = (blob.pixels() / (img.width() * img.height())) * 100
                img.draw_string(blob.x(), blob.y()-10, 
                              f"R{detected_count['red']+1}: {area_percent:.1f}%", 
                              color=RED, scale=1)
                detected_count['red'] += 1

        # 绘制所有黄色物体
        if yellow_blobs:
            for blob in yellow_blobs:
                # 绘制矩形框
                img.draw_rectangle(blob.x(), blob.y(), blob.w(), blob.h(), color=YELLOW)
                # 绘制十字标记
                img.draw_cross(blob.cx(), blob.cy(), color=CYAN, size=10)
                # 显示面积信息
                area_percent = (blob.pixels() / (img.width() * img.height())) * 100
                img.draw_string(blob.x(), blob.y()-10, 
                              f"Y{detected_count['yellow']+1}: {area_percent:.1f}%", 
                              color=YELLOW, scale=1)
                detected_count['yellow'] += 1

        # 运行标准检测流程
        result = detect_object(img)
        if result['IS_FIND_TARGET']:
            # 标记选中的目标
            x, y = result['u_target'], result['v_target']
            # 绘制大十字标记表示选中的目标
            img.draw_cross(x, y, color=GREEN, size=15, thickness=2)
            
            # 绘制到中心线的偏移
            if result['delta_u'] != 0:
                img.draw_line(x, y, center_x, y, color=GREEN, thickness=2)

            # 显示选中目标的详细信息
            info_y = img.height() - 60
            img.draw_string(5, info_y, 
                          f"Selected: {result['type'].upper()}", 
                          color=WHITE, scale=1)
            img.draw_string(5, info_y + 15, 
                          f"Dist: {result['d_norm']:.2f}", 
                          color=WHITE, scale=1)
            img.draw_string(5, info_y + 30, 
                          f"dU: {result['delta_u']}", 
                          color=WHITE, scale=1)

            # 打印调试信息
            print(f"选中目标: {result['type']} ", end='')
            print(f"位置({x},{y}) ", end='')
            print(f"距离{result['d_norm']:.2f} ", end='')
            print(f"偏移{result['delta_u']}")
        else:
            img.draw_string(5, img.height()-30, "No Target Selected", 
                          color=WHITE, scale=1)

        # 显示检测统计和帧率
        img.draw_string(5, 10, f"Red: {detected_count['red']} Yellow: {detected_count['yellow']}", 
                       color=WHITE, scale=1)
        fps = clock.fps()
        img.draw_string(img.width()-50, 10, f"{fps:.1f}fps", color=WHITE, scale=1)

        time.sleep(0.01)  # 短暂延时，避免刷新太快

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
    mode = 'c'  # 修改为 'c' 或 'i'，如 mode = 'c' 进行摄像头实时检测
    if mode == 'c':
        test_camera_detection()
    elif mode == 'i':
        test_image_detection()
    else:
        print("已退出。")
