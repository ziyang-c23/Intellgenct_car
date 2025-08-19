
"""
OpenmvDetector.py
------------------
车载相机（OpenMV）物体检测模块。
实现红色和黄色物体的实时识别与抓取引导。

整体流程：
    每帧执行：颜色分割（LAB模型）→ 形态学去噪 → 面积筛选 → 目标排序 → 特征输出。
    支持红色和黄色物体，输出抓取目标的中心坐标、水平偏移量、归一化距离和物体类型。

主要变量说明：
    threshold_red    红色物体的 LAB 颜色阈值（元组列表），用于颜色分割。
    threshold_yellow 黄色物体的 LAB 颜色阈值（元组列表），用于颜色分割。

主要函数说明：
    detect_object(img)
        检测图像中的红色和黄色物体，返回最近目标的中心坐标、水平偏移、归一化距离和物体类型。
        参数：img（OpenMV 图像对象）
        返回：dict，包括 u_target, v_target, delta_u, d_norm, area, type
        type 为 'red' 或 'yellow'。
"""

import sensor, time

# 摄像头配置
sensor.reset()
sensor.set_pixformat(sensor.RGB565)
sensor.set_framesize(sensor.QVGA)
sensor.skip_frames(10)
sensor.set_auto_whitebal(False)
clock = time.clock()


# 红色和黄色物体 LAB 阈值（可根据实际调整）
threshold_red = [(30, 70, 20, 60, 10, 50)]
threshold_yellow = [(60, 90, -10, 30, 40, 80)]


# 物体检测主流程


def detect_object(img):
    """
    检测红色和黄色物体，返回最近目标的中心(u,v)、水平偏移Δu、归一化距离d_norm、物体类型。

    参数：
        img : OpenMV 图像对象

    流程：
        1. 形态学开运算去噪（img.open(2,2)）
        2. 分别用红色和黄色阈值进行颜色分割，得到红色和黄色的 blobs。
        3. 筛选面积大于 0.5% 图像面积的 blobs。
        4. 合并所有目标，按中心 v 坐标（y）降序排序，选取 v 最大（最靠近下方）的 blob 作为抓取目标。
        5. 计算目标中心坐标 (u_target, v_target)、水平偏移量 (delta_u)、归一化距离 (d_norm)，并返回物体类型（'red' 或 'yellow'）。

    返回：
        dict，包括如下键值：
            'u_target' : 目标中心横坐标
            'v_target' : 目标中心纵坐标
            'delta_u'  : 目标中心与图像中心线的水平偏移量
            'd_norm'   : 归一化距离（0 最近，1 最远）
            'area'     : 目标像素面积
            'type'     : 物体类型（'red' 或 'yellow'）
    """
    img.open(2, 2)  # 形态学去噪，去除小噪点
    img_area = img.width() * img.height()
    min_area = int(img_area * 0.005)  # 面积筛选阈值

    # 检测红色物体
    red_blobs = img.find_blobs(threshold_red, pixels_threshold=30, area_threshold=30, merge=True)
    red_blobs = [b for b in red_blobs if b.pixels() > min_area] if red_blobs else []
    for b in red_blobs:
        b._color_type = 'red'

    # 检测黄色物体
    yellow_blobs = img.find_blobs(threshold_yellow, pixels_threshold=30, area_threshold=30, merge=True)
    yellow_blobs = [b for b in yellow_blobs if b.pixels() > min_area] if yellow_blobs else []
    for b in yellow_blobs:
        b._color_type = 'yellow'

    # 合并所有目标
    all_blobs = red_blobs + yellow_blobs
    if not all_blobs:
        return None

    # 按中心 v 坐标降序排序，选取最近目标
    all_blobs.sort(key=lambda b: b.cy(), reverse=True)
    target = all_blobs[0]

    # 特征输出
    u_target = target.cx()  # 目标中心横坐标
    v_target = target.cy()  # 目标中心纵坐标
    u_centerline = img.width() // 2  # 图像中心线
    delta_u = u_target - u_centerline  # 水平偏移量

    v_min = int(img.height() * 0.1)
    v_max = int(img.height() * 0.9)
    d_norm = 1 - (v_target - v_min) / (v_max - v_min)  # 距离归一化
    d_norm = max(0, min(1, d_norm))

    return {
        'u_target': u_target,
        'v_target': v_target,
        'delta_u': delta_u,
        'd_norm': d_norm,
        'area': target.pixels(),
        'type': getattr(target, '_color_type', 'unknown')
    }
