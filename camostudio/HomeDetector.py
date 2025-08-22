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
    
    家区域检测器类，用于智能小车比赛中检测和分析黑色目标区域。
    
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
    # 面积范围（像素）
    min_area: int = 1000
    max_area: int = 100000
    # 宽高比范围
    aspect_ratio_range: tuple = (0.5, 2.0)
    # 角度阈值（度）
    angle_threshold: float = 15.0
    
@dataclass
class HomeInfo:
    """
    家区域检测结果信息类
    
    属性说明：
        quad: np.ndarray
            四边形顶点坐标数组
            - 形状为(4,2)，每行表示一个顶点的[x,y]坐标
            - 顺序为：左上、右上、右下、左下
            - 数据类型为整数
            
        center: tuple
            区域中心点坐标(x,y)
            - 使用质心或顶点平均值计算
            - 用于定位和导航
            
        area: float
            区域面积（像素）
            - 通过轮廓计算
            - 用于验证检测结果
            
        perimeter: float
            区域周长（像素）
            - 轮廓的弧长
            - 用于形状分析
            
        aspect_ratio: float
            宽高比
            - 最小外接矩形的宽高比
            - 用于验证形状
            
        angles: list
            四个角的角度列表
            - 按顶点顺序存储
            - 单位为度
            - 理想值应接近90度
            
        valid: bool
            检测结果的有效性标志
            - True：满足所有验证条件
            - False：未通过验证
    
    使用示例：
        result = HomeInfo(
            quad=np.array([[x1,y1], [x2,y2], [x3,y3], [x4,y4]]),
            center=(cx,cy),
            area=1000.0,
            perimeter=200.0,
            aspect_ratio=1.2,
            angles=[88.5, 91.2, 89.8, 90.5],
            valid=True
        )
    """
    quad: np.ndarray       # 四边形顶点坐标 (4,2)
    center: tuple         # 中心点坐标 (x,y)
    area: float          # 面积
    perimeter: float     # 周长
    aspect_ratio: float  # 宽高比
    angles: list         # 四个角的角度
    valid: bool          # 是否是有效的区域

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

    def validate_home(self, info: HomeInfo) -> bool:
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
        if info.area < self.config.min_area or info.area > self.config.max_area:
            return False
        if not (self.config.aspect_ratio_range[0] <= info.aspect_ratio <= self.config.aspect_ratio_range[1]):
            return False
        for angle in info.angles:
            if abs(angle - 90) > self.config.angle_threshold:
                return False
        return True

    def detect(self, hsv: np.ndarray) -> Optional[HomeInfo]:
        """
        检测画面中的家区域（黑色区域）并进行验证。
        
        检测流程：
            1. 颜色分割：
               - 使用HSV阈值分割黑色区域
               - 形态学操作去除噪声
               
            2. 轮廓处理：
               - 提取所有轮廓
               - 选择最大面积轮廓
               - 初步面积验证
               
            3. 形状分析：
               - 计算最小外接矩形
               - 提取顶点并排序
               - 计算几何特征
               
            4. 结果验证：
               - 面积范围检查
               - 宽高比验证
               - 角度一致性验证
        
        参数：
            hsv: np.ndarray
                HSV格式的输入图像
                需要通过cv2.cvtColor预处理
                
        返回：
            Optional[HomeInfo]：检测结果对象
            - 如果检测成功：返回包含位置和特征的HomeInfo对象
            - 如果检测失败：返回None
            
        注意：
            - 输入图像必须是HSV格式
            - 检测结果包含验证状态
            - 可通过last_result属性获取最近结果
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
            
        # 获取最大轮廓
        cnt = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(cnt)
        if area < self.config.min_area:
            return None
            
        # 获取四边形顶点
        quad = contour_min_area_rect_points(cnt).astype(int)
        center = compute_centroid(cnt)
        if center is None:
            center = tuple(np.mean(quad, axis=0).astype(int))
        
        # 计算特征
        perimeter = cv2.arcLength(cnt, True)
        rect = cv2.minAreaRect(cnt)
        aspect_ratio = max(rect[1]) / (min(rect[1]) + 1e-6)
        angles = self.calculate_angles(quad)
        
        # 创建检测结果对象
        info = HomeInfo(
            quad=quad,
            center=center,
            area=area,
            perimeter=perimeter,
            aspect_ratio=aspect_ratio,
            angles=angles,
            valid=False
        )
        
        # 验证结果
        info.valid = self.validate_home(info)
        self.last_result = info
        
        return info

    def draw_results(self, frame: np.ndarray, result: Optional[HomeInfo]) -> np.ndarray:
        """
        在图像上绘制检测结果和相关信息的可视化显示。

        绘制内容：
            1. 检测框
               - 有效检测：绿色
               - 无效检测：红色
               - 线宽：2像素
               
            2. 关键点
               - 四个顶点：绿色圆点(5像素)
               - 顶点编号：P1-P4
               - 中心点：红色圆点(7像素)
               
            3. 状态信息
               - 位置坐标
               - 面积大小
               - 宽高比
               - 验证结果
               
        参数：
            frame: np.ndarray
                原始图像，BGR格式
                直接在其上绘制，会修改原图
                
            result: Optional[HomeInfo]
                检测结果对象
                如果为None则显示未检测到的信息
                
        返回：
            np.ndarray：绘制完成的图像
            
        注意：
            - 所有文本使用简单字体
            - 信息显示在区域左上方
            - 颜色编码表示检测状态
        """
        if result is not None:
            # 绘制四边形
            cv2.polylines(frame, [result.quad], isClosed=True, 
                         color=(0,255,0) if result.valid else (0,0,255), 
                         thickness=2)
            
            # 绘制顶点及坐标
            for i, pt in enumerate(result.quad):
                cv2.circle(frame, tuple(pt), 5, (0,255,0), -1)
                # 根据顶点位置调整标签位置
                if i == 0:  # 左上角点
                    label_x = pt[0] - 25
                    label_y = pt[1] - 10
                elif i == 1:  # 右上角点
                    label_x = pt[0] + 5
                    label_y = pt[1] - 10
                elif i == 2:  # 右下角点
                    label_x = pt[0] + 5
                    label_y = pt[1] + 20
                else:  # 左下角点
                    label_x = pt[0] - 25
                    label_y = pt[1] + 20
                
                cv2.putText(frame, f"P{i+1}", (label_x, label_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
            
            # 显示中心点
            cv2.circle(frame, result.center, 7, (0,0,255), -1)
            
            # 获取框的左上角坐标
            min_x = np.min(result.quad[:, 0])
            min_y = np.min(result.quad[:, 1])
            
            # 确保文本不会超出图像边界
            text_x = max(min_x - 10, 10)
            text_y = max(min_y - 20, 30)
            
            # 显示信息
            info_text = [
                f"Center: {result.center}",
                f"Area: {result.area:.0f}",
                f"Aspect Ratio: {result.aspect_ratio:.2f}",
                f"Valid: {result.valid}"
            ]
            
            for i, text in enumerate(info_text):
                cv2.putText(frame, text, (int(text_x), int(text_y + 25*i)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
                           (0,255,0) if result.valid else (0,0,255), 2)
                
        else:
            cv2.putText(frame, "No Home Detected", (20, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        return frame
    
if __name__ == "__main__":
    """
    主程序：打开摄像头，实时检测并显示家区域。
    按q键退出。
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
    cap = cv2.VideoCapture(2)
    if cap.isOpened():
        print("已打开摄像头，开始实时检测...")
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
            
            # 显示
            cv2.imshow("Home Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):  # 按q退出
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
                print("检测结果已保存到 Home.jpg")
                if result and result.valid:
                    print(f"检测到有效的家区域:")
                    print(f"- 中心点: {result.center}")
                    print(f"- 面积: {result.area:.0f}")
                    print(f"- 宽高比: {result.aspect_ratio:.2f}")
                    print(f"- 四个角度: {[f'{angle:.1f}°' for angle in result.angles]}")
                else:
                    print("未检测到有效的家区域")
                    
                print("\n按任意键关闭窗口...")
                cv2.waitKey(0)
                cv2.destroyAllWindows()
        else:
            print("未输入图片路径，跳过测试。")
