"""
HomeDetector 家区域检测模块
------------------------
本模块实现了基于OpenCV的智能小车比赛中"家"区域（黑色区域）的检测功能。
通过HSV颜色分割和形状分析来定位和验证目标区域。

主要功能：
    - 黑色区域的HSV阈值分割
    - 形状特征提取和分析
    - 区域有效性验证
    - 检测结果可视化

核心类：
    HomeDetector：主检测器类
        - 支持自定义检测class HomeDetector:
    
    家区域检测器类，用于智能小车比赛中检测和分析黑色目标区class HomeDetector:
    ""
    HomeDetector类用于检测画面中的"家"区域（黑色区域），同时支持自己家（左下角）
    和对手家（右上角）的检测。通过HSV颜色分割、轮廓分析和几何特征提取，能够精确
    识别目标区域并验证其有效性。
    
    主要方法包括：检测(detect)、结果绘制(draw_results)、角度计算(calculate_angles)
    以及区域验证(validate_self_home/validate_opponent_home)等。
    
    适用于智能小车比赛中的区域识别、机器人导航定位以及边界检测与跟踪等场景。
    ""
    主要功能：
        1. 图像处理和分割：
           - HSV颜色空间转换
           - 自适应阈值分割
           - 形态学操作去噪
        
        2. 特征提取：
           - 轮廓检测和分析
           - 最小外接矩形计算
           - 角点检测和排序
           
        3. 结果验证：
           - 面积范围验证
           - 形状特征分析
           - 角度一致性检查
           
        4. 可视化显示：
           - 绘制检测框
           - 显示关键点
           - 输出检测参数
    
    属性：
        config: HomeConfig
            检测器配置参数
        
        kernel_open: np.ndarray
            开运算核，用于去除小噪点
            
        kernel_close: np.ndarray
            闭运算核，用于填充小孔
            
        last_result: Optional[HomeInfo]
            最近一次的检测结果
            
    方法：
        detect(hsv: np.ndarray) -> Optional[HomeInfo]
            主检测方法，处理输入图像并返回检测结果
            
        draw_results(frame: np.ndarray, result: Optional[HomeInfo]) -> np.ndarray
            在图像上绘制检测结果和相关信息
            
        calculate_angles(quad: np.ndarray) -> list
            计算四边形的四个内角
            
        validate_home(info: HomeInfo) -> bool
            验证检测结果是否有效
        - 提供实时检测和结果验证
        - 包含结果可视化功能

数据类：
    HomeConfig：检测参数配置类
        - HSV颜色范围
        - 面积阈值范围
        - 形状约束参数
        - 验证阈值设置

    HomeInfo：检测结果信息类
        - 位置和形状信息
        - 几何特征参数
        - 验证结果状态

技术特点：
    1. 鲁棒的颜色分割：
       - 使用HSV色彩空间
       - 形态学操作去噪
       - 自适应阈值处理

    2. 精确的形状分析：
       - 轮廓提取和筛选
       - 最小外接矩形计算
       - 角点排序和验证

    3. 完善的验证机制：
       - 面积范围验证
       - 宽高比检查
       - 角度一致性验证

使用示例：
    # 创建检测器实例
    config = HomeConfig(
        black_lower=(0, 0, 0),
        black_upper=(180, 255, 50),
        min_area=1000
    )
    detector = HomeDetector(config)

    # 检测处理
    result = detector.detect(hsv_image)
    if result and result.valid:
        print(f"检测到有效家区域，中心点：{result.center}")

    # 可视化结果
    frame = detector.draw_results(frame, result)

依赖项：
    - OpenCV (cv2)
    - NumPy
    - Python 3.6+

作者：Ziyang Chen
版本：1.0.0
"""
import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict


@dataclass
class HomeConfig:
    """
    家区域检测的配置参数类

    属性：
        black_lower (tuple): 黑色区域HSV下界，默认(0,0,0)
            - H: 0-180，色调
            - S: 0-255，饱和度
            - V: 0-255，明度
            
        black_upper (tuple): 黑色区域HSV上界，默认(180,255,50)
            - 范围宽泛以适应不同光照
            
        min_area (int): 最小区域面积，默认1000像素
            - 用于过滤小噪声区域
            
        max_area (int): 最大区域面积，默认100000像素
            - 用于过滤过大区域
            
        aspect_ratio_range (tuple): 宽高比范围，默认(0.5,2.0)
            - 用于验证区域形状
            - 过滤过于狭长的区域
            
        angle_threshold (float): 角度偏差容差，默认15度
            - 用于验证矩形程度
            - 理想矩形四个角应接近90度
    
    注意：
        - HSV范围应根据实际光照条件调整
        - 面积阈值应根据摄像头距离调整
        - 宽高比和角度阈值影响检测的严格程度
    """
    # HSV颜色范围
    black_lower: tuple = (0, 0, 0)
    black_upper: tuple = (180, 255, 50)
    # black_lower: tuple = (35, 50, 50)  # 绿色下界
    # black_upper: tuple = (85, 255, 255)  # 绿色上界
    # 面积范围（像素）
    min_area: int = 100
    max_area: int = 1000000
    # 宽高比范围
    aspect_ratio_range: tuple = (0.5, 2.0)
    # 角度阈值（度）
    angle_threshold: float = 15.0
    
@dataclass
class HomeInfo:
    """
    家区域检测结果信息类，同时包含自己家和对方家的完整检测信息
    
    自己家区域属性(Self_开头)：
        Self_Quad: np.ndarray
            自己家四边形顶点坐标数组
            - 形状为(4,2)，每行表示一个顶点的[x,y]坐标
            - 顺序为：左上、右上、右下、左下
            - 数据类型为整数
            
        Self_Center: tuple
            自己家区域中心点坐标(x,y)
            - 使用质心或顶点平均值计算
            - 用于定位和导航
            
        Self_Area: float
            自己家区域面积（像素）
            - 通过轮廓计算
            - 用于验证检测结果
            
        Self_Perimeter: float
            自己家区域周长（像素）
            - 轮廓的弧长
            - 用于形状分析
            
        Self_Aspect_Ratio: float
            自己家区域宽高比
            - 最小外接矩形的宽高比
            - 用于验证形状
            
        Self_Angles: list
            自己家区域四个角的角度列表
            - 按顶点顺序存储
            - 单位为度
            - 理想值应接近90度
            
        Self_Valid: bool
            自己家检测结果的有效性标志
            - True：满足所有验证条件
            - False：未通过验证
            
    对方家区域属性(Opponent_开头)：
        与自己家区域属性结构相同，但表示对方家区域的信息
        
    使用示例：
        result = HomeInfo(
            Self_Quad=np.array([[x1,y1], [x2,y2], [x3,y3], [x4,y4]]),
            Self_Center=(cx,cy),
            Self_Area=1000.0,
            Self_Perimeter=200.0,
            Self_Aspect_Ratio=1.2,
            Self_Angles=[88.5, 91.2, 89.8, 90.5],
            Self_Valid=True,
            Opponent_Quad=np.array([[x1,y1], [x2,y2], [x3,y3], [x4,y4]]),
            Opponent_Center=(cx,cy),
            Opponent_Area=1000.0,
            Opponent_Perimeter=200.0,
            Opponent_Aspect_Ratio=1.2,
            Opponent_Angles=[88.5, 91.2, 89.8, 90.5],
            Opponent_Valid=True
        )
    """
    Self_Quad: np.ndarray       # 自己区域四边形顶点坐标 (4,2)
    Self_Center: tuple         # 自己区域中心点坐标 (x,y)
    Self_Area: float          # 自己区域面积
    Self_Perimeter: float     # 自己区域周长
    Self_Aspect_Ratio: float  # 自己区域宽高比
    Self_Angles: list         # 自己区域四个角的角度
    Self_Valid: bool          # 自己区域是否是有效的区域
    Opponent_Quad: np.ndarray  # 对方区域四边形顶点坐标 (4,2)
    Opponent_Center: tuple     # 对方区域中心点坐标 (x,y)
    Opponent_Area: float       # 对方区域面积
    Opponent_Perimeter: float  # 对方区域周长
    Opponent_Aspect_Ratio: float  # 对方区域宽高比
    Opponent_Angles: list      # 对方区域四个角的角度
    Opponent_Valid: bool       # 对方区域是否是有效的区域


# 通用工具函数

def order_quad_points(pts: np.ndarray) -> np.ndarray:
    """
    对四边形的四个顶点进行排序，返回顺序为：左上、右上、右下、左下。
    输入参数:
        pts: 四个顶点的坐标，形状为(4,2)
    返回:
        排序后的顶点坐标，形状为(4,2)
    排序规则：
        - 左上：坐标和最小
        - 右下：坐标和最大
        - 右上：坐标差最小
        - 左下：坐标差最大
    """
    pts = np.array(pts, dtype=np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).reshape(-1)
    ordered = np.zeros((4, 2), dtype=np.float32)
    ordered[0] = pts[np.argmin(s)]
    ordered[2] = pts[np.argmax(s)]
    ordered[1] = pts[np.argmin(diff)]
    ordered[3] = pts[np.argmax(diff)]
    return ordered

def contour_min_area_rect_points(contour: np.ndarray) -> np.ndarray:
    """
    获取轮廓的最小外接矩形的四个顶点，并按顺序返回。
    输入参数:
        contour: 轮廓点集
    返回:
        排序后的最小外接矩形四个顶点坐标，形状为(4,2)
    """
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    return order_quad_points(box)

def compute_centroid(contour: np.ndarray):
    """
    计算轮廓的质心（中心点坐标）。
    输入参数:
        contour: 轮廓点集
    返回:
        (cx, cy): 质心坐标，若无法计算则返回None
    """
    M = cv2.moments(contour)
    if M["m00"] == 0:
        return None
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    return (cx, cy)

class HomeDetector:
    """
    HomeDetector类用于检测画面中的“家”区域（黑色区域），并返回其最小外接矩形和中心点。
    方法：
        - detect: 检测并返回家区域的中心和矩形顶点
        - draw_results: 在画面上绘制检测结果
    """
    def __init__(self, config: HomeConfig = HomeConfig()):
        """
        初始化HomeDetector。
        参数:
            config: HomeConfig对象，包含检测参数
        """
        self.config = config
        self.kernel_open = np.ones((3, 3), np.uint8)
        self.kernel_close = np.ones((5, 5), np.uint8)
        self.last_result = None

    def calculate_angles(self, quad: np.ndarray) -> list:
        """
        计算四边形的四个内角角度。

        算法步骤：
            1. 遍历四个顶点
            2. 对每个顶点：
               - 获取相邻的两个向量
               - 计算向量夹角
               - 转换为角度值
               
        参数：
            quad: np.ndarray
                四边形顶点坐标，shape=(4,2)
                要求顺序为：左上、右上、右下、左下
                
        返回：
            list：四个内角的角度值（度）
            - 顺序与顶点顺序对应
            - 理想矩形应接近90度
            - 值域范围：[0, 180]
            
        实现细节：
            - 使用向量内积计算夹角
            - 使用arccos获取角度
            - 应用clip避免数值误差
        """
        angles = []
        for i in range(4):
            p1 = quad[i]
            p2 = quad[(i+1)%4]
            p3 = quad[(i+2)%4]
            v1 = p1 - p2
            v2 = p3 - p2
            cosine = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            angle = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
            angles.append(angle)
        return angles

    def validate_self_home(self, info: HomeInfo) -> bool:
        """
        验证检测到的区域是否为有效的家区域。

        验证标准：
            1. 面积检查
               - 最小面积：{self.config.min_area} 像素
               - 最大面积：{self.config.max_area} 像素
               - 过滤过小或过大的区域
               
            2. 形状验证
               - 宽高比范围：{self.config.aspect_ratio_range}
               - 检查区域是否过于狭长
               - 保证形状近似方形
               
            3. 角度验证
               - 容差范围：±{self.config.angle_threshold}度
               - 检查四个角是否接近90度
               - 确保形状为矩形
               
        参数：
            info: HomeInfo
                待验证的检测结果对象
                必须包含所有几何特征参数
                
        返回：
            bool：验证结果
            - True: 满足所有验证条件
            - False: 任一条件不满足
            
        重要性：
            - 面积验证：过滤噪声和干扰
            - 形状验证：确保区域规则性
            - 角度验证：保证矩形特征
        """
        if info.Self_Area < self.config.min_area or info.Self_Area > self.config.max_area:
            return False
        if not (self.config.aspect_ratio_range[0] <= info.Self_Aspect_Ratio <= self.config.aspect_ratio_range[1]):
            return False
        for angle in info.Self_Angles:
            if abs(angle - 90) > self.config.angle_threshold:
                return False
        return True

    def validate_opponent_home(self, info: HomeInfo) -> bool:
        """
        验证检测到的对方区域是否为有效的家区域。

        验证标准：
            1. 面积检查
               - 最小面积：{self.config.min_area} 像素
               - 最大面积：{self.config.max_area} 像素
               - 过滤过小或过大的区域
               
            2. 形状验证
               - 宽高比范围：{self.config.aspect_ratio_range}
               - 检查区域是否过于狭长
               - 保证形状近似方形
               
            3. 角度验证
               - 容差范围：±{self.config.angle_threshold}度
               - 检查四个角是否接近90度
               - 确保形状为矩形
               
        参数：
            info: HomeInfo
                待验证的检测结果对象
                必须包含所有对方区域的几何特征参数
                
        返回：
            bool：验证结果
            - True: 满足所有验证条件
            - False: 任一条件不满足
        """
        if info.Opponent_Area < self.config.min_area or info.Opponent_Area > self.config.max_area:
            return False
        if not (self.config.aspect_ratio_range[0] <= info.Opponent_Aspect_Ratio <= self.config.aspect_ratio_range[1]):
            return False
        for angle in info.Opponent_Angles:
            if abs(angle - 90) > self.config.angle_threshold:
                return False
        return True

    def detect(self, hsv: np.ndarray) -> Optional[HomeInfo]:
        """
        检测画面中的自己家（左下角）和对手家（右上角），并返回检测结果。
        
        处理流程：
            1. 颜色分割
               - 使用HSV阈值分割黑色区域
               - 应用形态学操作去除噪点、填充空洞
               
            2. 轮廓检测
               - 提取外部轮廓
               - 过滤小面积轮廓
               
            3. 特征提取
               - 计算最小外接矩形
               - 提取中心点和几何特征
               - 计算角度信息
               
            4. 区域分类
               - 基于象限划分自己家和对手家
               - 以画面中心为参考点
               - 左下象限为自己家
               - 右上象限为对手家
               
            5. 结果验证
               - 调用验证函数检查有效性
               - 保存最终结果
        
        参数：
            hsv: np.ndarray
                输入的HSV格式图像
                
        返回：
            Optional[HomeInfo]：检测结果对象
            - 包含自己家和对手家的所有特征信息
            - 如果两区域均未检测到，返回None
        """

        # 颜色阈值处理
        mask = cv2.inRange(hsv, np.array(self.config.black_lower), np.array(self.config.black_upper))
        # 形态学操作
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel_open, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel_close, iterations=1)

        # 查找轮廓
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        h, w = hsv.shape[:2]
        mid_x, mid_y = w // 2, h // 2   # 屏幕中点，用来判断象限

        # 初始化结果
        self_info = {
            "quad": None, "center": None, "area": 0, "perimeter": 0,
            "aspect_ratio": 0, "angles": [], "valid": False
        }
        opp_info = {
            "quad": None, "center": None, "area": 0, "perimeter": 0,
            "aspect_ratio": 0, "angles": [], "valid": False
        }

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.config.min_area:
                continue

            quad = contour_min_area_rect_points(cnt).astype(int)
            center = compute_centroid(cnt)
            if center is None:
                center = tuple(np.mean(quad, axis=0).astype(int))

            perimeter = cv2.arcLength(cnt, True)
            rect = cv2.minAreaRect(cnt)
            aspect_ratio = max(rect[1]) / (min(rect[1]) + 1e-6)
            angles = self.calculate_angles(quad)

            cx, cy = center

            # 判断归属
            if cx < mid_x and cy > mid_y:  # 左下角 → 自己家
                if area > self_info["area"]:   # 取面积最大的一个
                    self_info.update({
                        "quad": quad,
                        "center": center,
                        "area": area,
                        "perimeter": perimeter,
                        "aspect_ratio": aspect_ratio,
                        "angles": angles
                    })
            elif cx > mid_x and cy < mid_y:  # 右上角 → 对手家
                if area > opp_info["area"]:
                    opp_info.update({
                        "quad": quad,
                        "center": center,
                        "area": area,
                        "perimeter": perimeter,
                        "aspect_ratio": aspect_ratio,
                        "angles": angles
                    })

        # 如果两边都没检测到，就返回 None
        if self_info["quad"] is None and opp_info["quad"] is None:
            return None

        # 构造 HomeInfo
        info = HomeInfo(
            Self_Quad=self_info["quad"],
            Self_Center=self_info["center"],
            Self_Area=self_info["area"],
            Self_Perimeter=self_info["perimeter"],
            Self_Aspect_Ratio=self_info["aspect_ratio"],
            Self_Angles=self_info["angles"],
            Self_Valid=False,
            Opponent_Quad=opp_info["quad"],
            Opponent_Center=opp_info["center"],
            Opponent_Area=opp_info["area"],
            Opponent_Perimeter=opp_info["perimeter"],
            Opponent_Aspect_Ratio=opp_info["aspect_ratio"],
            Opponent_Angles=opp_info["angles"],
            Opponent_Valid=False
        )

        # 验证
        if info.Self_Quad is not None:
            info.Self_Valid = self.validate_self_home(info)
        if info.Opponent_Quad is not None:
            info.Opponent_Valid = self.validate_opponent_home(info)

        self.last_result = info
        return info


    def draw_results(self, frame: np.ndarray, result: Optional[HomeInfo]) -> np.ndarray:
        """
        在图像上绘制检测结果（同时支持自己家和对手家）。
        
        绘制内容：
            1. 四边形轮廓
               - 自己家（有效：绿色，无效：红色）
               - 对手家（有效：蓝色，无效：橙色）
               
            2. 关键点标记
               - 四个顶点（P1-P4）
               - 中心点（红色）
               
            3. 文本信息
               - 区域名称
               - 中心点坐标
               - 面积大小
               - 宽高比
               - 有效性状态
               
        参数：
            frame: np.ndarray
                要绘制的原始图像
                
            result: Optional[HomeInfo]
                检测结果，包含两个区域的信息
                若为None，则显示"No Home Detected"
                
        返回：
            np.ndarray：添加了可视化信息的图像
            
        注意：
            - 顶点编号按左上、右上、右下、左下顺序
            - 文本信息靠近区域左上角显示
            - 自己家和对手家文本分开显示避免重叠
        """
        if result is None:
            cv2.putText(frame, "No Home Detected", (20, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            return frame

        def draw_home(quad, center, area, aspect_ratio, valid, name, color_valid, color_invalid, text_offset=0):
            """
            在图像上绘制单个家区域的详细信息。
            
            参数:
                quad: 四边形顶点坐标
                center: 中心点坐标
                area: 区域面积
                aspect_ratio: 宽高比
                valid: 区域有效性
                name: 区域名称标识
                color_valid: 有效区域的颜色
                color_invalid: 无效区域的颜色
                text_offset: 文本信息的垂直偏移量
            """
            if quad is None or center is None:
                return
            color = color_valid if valid else color_invalid

            # 绘制四边形
            cv2.polylines(frame, [quad], isClosed=True, color=color, thickness=2)

            # 绘制顶点
            for i, pt in enumerate(quad):
                cv2.circle(frame, tuple(pt), 5, color, -1)
                cv2.putText(frame, f"P{i+1}", (pt[0]+5, pt[1]-5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            # 绘制中心点
            cv2.circle(frame, center, 6, (0,0,255), -1)

            # 绘制信息
            min_x = np.min(quad[:,0])
            min_y = np.min(quad[:,1])
            text_x = max(min_x - 10, 10)
            text_y = max(min_y - 40, 30) + text_offset

            info_text = [
                f"{name}:",
                f"  Center: {center}",
                f"  Area: {area:.0f}",
                f"  Aspect: {aspect_ratio:.2f}",
                f"  Valid: {valid}"
            ]
            for i, text in enumerate(info_text):
                cv2.putText(frame, text, (int(text_x), int(text_y + 20*i)),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # 绘制自己家（左下）
        draw_home(
            result.Self_Quad, result.Self_Center, result.Self_Area,
            result.Self_Aspect_Ratio, result.Self_Valid,
            "Self Home", (0,255,0), (0,0,255), text_offset=0
        )

        # 绘制对手家（右上）
        draw_home(
            result.Opponent_Quad, result.Opponent_Center, result.Opponent_Area,
            result.Opponent_Aspect_Ratio, result.Opponent_Valid,
            "Opponent Home", (255,0,0), (0,165,255), text_offset=100
        )

        return frame

    
if __name__ == "__main__":
    """
    主程序：打开摄像头，实时检测并显示家区域。
    按 q 键退出。
    """
    # 创建检测器实例，可以自定义配置参数
    config = HomeConfig(
        black_lower=(0, 0, 0),
        black_upper=(180, 255, 50),
        min_area=1000,
        max_area=100000,
        aspect_ratio_range=(0.5, 2.0),
        angle_threshold=15.0
    )
    detector = HomeDetector(config)
    
    # 尝试打开摄像头
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        print("已打开摄像头，开始实时检测... (按 q 退出)")
        while True:
            ret, frame = cap.read()
            if not ret:
                print("无法读取摄像头")
                break
                
            # 检测处理
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            result = detector.detect(hsv)
            
            # 绘制结果
            frame = detector.draw_results(frame, result)

            # 控制台输出（每 30 帧打印一次）
            if result and cap.get(cv2.CAP_PROP_POS_FRAMES) % 30 == 0:
                if result.Self_Quad is not None:
                    print(result.Self_Quad)
                    print(f"[Self] Center={result.Self_Center}, Area={result.Self_Area:.0f}, Valid={result.Self_Valid}")
                if result.Opponent_Quad is not None:
                    print(f"[Opponent] Center={result.Opponent_Center}, Area={result.Opponent_Area:.0f}, Valid={result.Opponent_Valid}")
            
            # 显示
            cv2.imshow("Home Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        cap.release()
        cv2.destroyAllWindows()
        
    else:
        print("无法打开摄像头，可通过读取图片进行测试。")
        img_path = input("请输入要检测的图片路径（如 test.jpg），直接回车跳过：").strip()
        if img_path:
            img = cv2.imread(img_path)
            if img is None:
                print(f"无法读取图片: {img_path}")
            else:
                # 检测处理
                hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                result = detector.detect(hsv)
                
                # 绘制结果
                img = detector.draw_results(img, result)
                
                # 保存并显示
                cv2.imwrite("camostudio/Home.jpg", img)
                cv2.imshow("Home Detection", img)
                print("检测结果已保存到 camostudio/Home.jpg")

                if result:
                    if result.Self_Quad is not None:
                        print("\n=== 自己家 (Self Home) ===")
                        print(f"- 中心点: {result.Self_Center}")
                        print(f"- 面积: {result.Self_Area:.0f}")
                        print(f"- 宽高比: {result.Self_Aspect_Ratio:.2f}")
                        print(f"- 四个角度: {[f'{a:.1f}°' for a in result.Self_Angles]}")
                        print(f"- 有效性: {result.Self_Valid}")

                    if result.Opponent_Quad is not None:
                        print("\n=== 对手家 (Opponent Home) ===")
                        print(f"- 中心点: {result.Opponent_Center}")
                        print(f"- 面积: {result.Opponent_Area:.0f}")
                        print(f"- 宽高比: {result.Opponent_Aspect_Ratio:.2f}")
                        print(f"- 四个角度: {[f'{a:.1f}°' for a in result.Opponent_Angles]}")
                        print(f"- 有效性: {result.Opponent_Valid}")
                else:
                    print("未检测到任何家区域。")
                    
                print("\n按任意键关闭窗口...")
                cv2.waitKey(0)
                cv2.destroyAllWindows()
        else:
            print("未输入图片路径，跳过测试。")

