"""
FenceDetector 围栏检测模块
------------------------
本模块实现了基于OpenCV的蓝色围栏区域检测功能。主要用于智能小车比赛中识别和定位蓝色围栏区域。

主要功能：
    - 检测画面中的蓝色围栏区域
    - 提取围栏区域的四边形轮廓
    - 计算并验证围栏特征（面积、宽高比、角度等）
    - 在图像上可视化检测结果

核心类：
    FenceDetector：主检测器类
        - 支持自定义检测参数配置
        - 提供实时检测和结果验证
        - 包含结果可视化功能

数据类：
    FenceConfig：检测参数配置类
        - HSV颜色范围
        - 面积阈值
        - 宽高比范围
        - 角度容差

    FenceInfo：检测结果信息类
        - 围栏位置和形状信息
        - 几何特征参数
        - 验证结果

使用示例：
    detector = FenceDetector()
    result = detector.detect(hsv_image)
    if result and result.valid:
        print(f"检测到有效围栏，中心点：{result.center}")

作者：Ziyang Chen
版本：1.0.0
"""


import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional

@dataclass
class FenceConfig:
    """
    围栏检测的配置参数
    参数：
        - hsv_ranges: HSV颜色范围
        - min_area: 最小轮廓面积
        - max_area: 最大轮廓面积
        - aspect_ratio_range: 宽高比范围 (min, max)
        - angle_threshold: 角度阈值（度）
        - shrink_ratio: 内收边界比例，用于避障
    """
    # HSV颜色范围
    blue_lower: tuple = (100, 50, 50)
    blue_upper: tuple = (130, 255, 255)
    # 面积范围（像素）
    min_area: int = 1000
    max_area: int = 1500000
    # 宽高比范围
    aspect_ratio_range: tuple = (0.5, 2.0)
    # 角度阈值（度）
    angle_threshold: float = 15.0
    # 内收边界比例（0-1之间的值）
    shrink_ratio: float = 0.20

@dataclass
class FenceInfo:
    """
    围栏检测结果信息
    属性：
        - quad: 四边形顶点坐标 (4,2)
        - inner_rect: 向内收缩的矩形坐标 [x1, y1, x2, y2]，用于避障，表示左上角和右下角坐标
        - center: 中心点坐标 (x,y)
        - area: 面积
        - perimeter: 周长
        - aspect_ratio: 宽高比
        - angles: 四个角的角度
        - valid: 是否是有效的围栏
    """
    quad: np.ndarray
    inner_rect: tuple  # 简化为[x1, y1, x2, y2]格式的矩形，表示左上角和右下角坐标
    center: tuple
    area: float
    perimeter: float
    aspect_ratio: float
    angles: list
    valid: bool

# 通用工具函数
def order_quad_points(pts: np.ndarray) -> np.ndarray:
    """
    对四边形的四个顶点进行标准化排序。

    排序算法：
        1. 计算每个点的坐标和(x+y)和坐标差(x-y)
        2. 根据以下规则分配位置：
           - 左上点：坐标和最小
           - 右下点：坐标和最大
           - 右上点：坐标差最小
           - 左下点：坐标差最大
        
    参数:
        pts: 四个顶点坐标数组
            - shape必须为(4,2)
            - 每行为一个点[x,y]
            - 输入顺序可以任意
    
    返回:
        np.ndarray: 排序后的顶点坐标数组
            - shape=(4,2)
            - dtype=np.float32
            - 顺序：[左上,右上,右下,左下]
    
    应用场景：
        - 标准化检测到的四边形顶点顺序
        - 确保后续处理（如角度计算）的一致性
        - 便于可视化和结果展示
    
    注意：
        - 输入点必须组成一个凸四边形
        - 算法假设图像坐标系（原点在左上）
        - 返回的数组是新的内存空间
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

class FenceDetector:
    """
    FenceDetector类用于检测画面中的蓝色围栏区域，返回其四边形顶点坐标。
    方法：
        - detect: 检测并返回围栏四边形顶点
        - draw_results: 在画面上绘制检测结果
    """
    def __init__(self, config: FenceConfig = FenceConfig()):
        """
        初始化FenceDetector。
        参数:
            config: FenceConfig对象，包含检测参数
        """
        self.config = config
        # 形态学操作的核
        self.kernel_open = np.ones((3, 3), np.uint8)
        self.kernel_close = np.ones((5, 5), np.uint8)
        # 存储最近一次的检测结果
        self.last_result = None

    def draw_results(self, frame: np.ndarray, result: Optional[FenceInfo]) -> np.ndarray:
        """
        在frame上绘制检测到的围栏区域及其信息。

        绘制内容：
            1. 围栏轮廓
               - 有效围栏：绿色
               - 无效围栏：红色
               - 线宽：2像素
            
            2. 特征点
               - 四个顶点：绿色圆点(半径5像素)
               - 顶点编号：P1-P4
               - 中心点：红色圆点(半径7像素)
            
            3. 信息显示
               - 中心点坐标
               - 面积
               - 宽高比
               - 验证结果
            
        参数:
            frame: BGR格式的原始图像
                  会直接在此图像上绘制，需要保留原图请先复制
            result: FenceInfo对象，包含检测结果信息
                   如果为None，将显示"No Fence Detected"
        
        返回:
            np.ndarray: 绘制完成的图像（与输入是同一个对象）
        
        注意：
            - 所有文字使用OpenCV默认字体
            - 信息显示在图像左上角
            - 坐标系：原点在左上角，x向右，y向下
        """
        if result is not None:
            # 绘制四边形
            cv2.polylines(frame, [result.quad], isClosed=True, 
                         color=(0,255,0) if result.valid else (0,0,255), 
                         thickness=2)
            
            # 绘制内部收缩的矩形（用于避障）
            if result.valid:
                x1, y1, x2, y2 = result.inner_rect  # 左上角和右下角的坐标
                cv2.rectangle(frame, (x1, y1), (x2, y2), 
                             color=(255,165,0),  # 橙色
                             thickness=2, lineType=cv2.LINE_AA)
            
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
            
            # 显示信息
            info_text = [
                f"Center: {result.center}",
                f"Area: {result.area:.0f}",
                f"Aspect Ratio: {result.aspect_ratio:.2f}",
                f"Valid: {result.valid}",
                f"Inner Rect: Avoid Obstacle Zone"
            ]
            
            # 获取框的左上角坐标
            min_x = np.min(result.quad[:, 0])
            min_y = np.min(result.quad[:, 1])
            
            # 确保文本不会超出图像边界
            text_x = min_x   
            text_y = max(min_y - 10, 30)  # 向上偏移10像素，但不小于30

            for i, text in enumerate(info_text):
                cv2.putText(frame, text, (int(text_x), int(text_y + 25*i)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
                           (0,255,0) if result.valid else (0,0,255), 2)
                
        else:
            cv2.putText(frame, "No Fence Detected", (20, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        return frame

    def calculate_angles(self, quad: np.ndarray) -> list:
        """
        计算四边形的四个内角角度。

        算法步骤：
            1. 对每个顶点：
               - 获取相邻的两个向量
               - 计算向量夹角（内积）
               - 将弧度转换为角度
            
        参数:
            quad: 四边形顶点坐标数组，shape=(4,2)
                 顺序必须为：左上、右上、右下、左下
        
        返回:
            list[float]: 四个内角的角度值（度）
                        顺序与顶点顺序对应
        
        实现细节：
            - 使用numpy的dot计算向量内积
            - 使用arccos计算夹角
            - 使用clip防止数值误差导致的定义域错误
            
        注意：
            - 返回角度值范围：[0, 180]
            - 对于标准矩形，应该都接近90度
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

    def validate_fence(self, info: FenceInfo) -> bool:
        """
        验证检测到的区域是否为有效的围栏。
        
        验证标准：
            1. 面积检查：
               - 最小面积：{self.config.min_area} 像素
               - 最大面积：{self.config.max_area} 像素
            
            2. 形状检查：
               - 宽高比范围：{self.config.aspect_ratio_range}
               - 要求近似矩形（四个角接近90度）
               - 角度容差：±{self.config.angle_threshold}度
        
        参数:
            info: FenceInfo对象，包含待验证的围栏信息
                 必须包含：area、aspect_ratio、angles等属性
        
        返回:
            bool: 验证结果
                - True: 满足所有验证条件
                - False: 任一条件不满足
        
        注意：
            - 宽高比检查用于过滤过于狭长的区域
            - 角度检查用于确保形状接近矩形
            - 面积检查用于过滤噪声和过大/过小的区域
        """
        if info.area < self.config.min_area or info.area > self.config.max_area:
            return False
        if not (self.config.aspect_ratio_range[0] <= info.aspect_ratio <= self.config.aspect_ratio_range[1]):
            return False
        # 检查角度是否接近90度
        for angle in info.angles:
            if abs(angle - 90) > self.config.angle_threshold:
                return False
        return True

    def detect(self, hsv: np.ndarray) -> Optional[FenceInfo]:
        """
        检测画面中的蓝色围栏区域。
        
        检测流程：
            1. HSV颜色空间分割
            2. 形态学操作去噪
            3. 轮廓检测和筛选
            4. 多边形拟合和顶点排序
            5. 特征计算和结果验证

        参数:
            hsv: HSV格式的图像。要求非空且为三通道HSV格式。
                可通过cv2.cvtColor(BGR_image, cv2.COLOR_BGR2HSV)转换得到。

        返回:
            FenceInfo对象：包含以下信息
                - quad: 围栏四边形顶点坐标
                - center: 中心点坐标
                - area: 面积
                - perimeter: 周长
                - aspect_ratio: 宽高比
                - angles: 四个角的角度
                - valid: 是否为有效围栏
            如果未检测到则返回None

        注意：
            - 返回的坐标都是整数类型
            - 顶点顺序：左上、右上、右下、左下
            - valid字段表示检测结果是否满足所有验证条件
        """
        # 颜色阈值处理
        mask = cv2.inRange(hsv, np.array(self.config.blue_lower), np.array(self.config.blue_upper))
        # 形态学操作
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel_open, iterations=2)
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
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) < 4:
            quad = contour_min_area_rect_points(cnt)
        else:
            if len(approx) > 4:
                hull = cv2.convexHull(approx)
                quad = contour_min_area_rect_points(hull)
            else:
                quad = order_quad_points(approx.reshape(-1, 2))
        
        quad = quad.astype(int)
        center = tuple(np.mean(quad, axis=0).astype(int))
        
        # 计算向内收缩的矩形（用于避障）
        # 获取围栏的边界框
        x_min, y_min = np.min(quad, axis=0)
        x_max, y_max = np.max(quad, axis=0)
        
        # 根据shrink_ratio计算内收矩形的左上角和右下角坐标
        shrink_pixels_x = int((x_max - x_min) * self.config.shrink_ratio)
        shrink_pixels_y = int((y_max - y_min) * self.config.shrink_ratio)
        
        # 内收矩形表示为左上角和右下角的坐标 (x1, y1, x2, y2)
        inner_rect = (
            x_min + shrink_pixels_x,  # x1 - 左上角x坐标
            y_min + shrink_pixels_y,  # y1 - 左上角y坐标
            x_max - shrink_pixels_x,  # x2 - 右下角x坐标
            y_max - shrink_pixels_y   # y2 - 右下角y坐标
        )
        
        # 计算特征
        perimeter = cv2.arcLength(cnt, True)
        rect = cv2.minAreaRect(cnt)
        aspect_ratio = max(rect[1]) / (min(rect[1]) + 1e-6)  # 防止除零
        angles = self.calculate_angles(quad)
        
        # 创建检测结果对象
        info = FenceInfo(
            quad=quad,
            inner_rect=inner_rect,
            center=center,
            area=area,
            perimeter=perimeter,
            aspect_ratio=aspect_ratio,
            angles=angles,
            valid=False  # 先设置为False，后面验证
        )
        
        # 验证结果
        info.valid = self.validate_fence(info)
        self.last_result = info
        
        return info
    
if __name__ == "__main__":
    """
    主程序：打开摄像头，实时检测并显示围栏区域。
    按q键退出。
    """
    # 创建检测器实例，可以自定义配置参数
    config = FenceConfig(
        blue_lower=(100, 50, 50),
        blue_upper=(130, 255, 255),
        min_area=1000,
        max_area=1500000,
        aspect_ratio_range=(0.5, 2.0),
        angle_threshold=15.0,
        shrink_ratio=0.10  # 向内收缩15%
    )
    detector = FenceDetector(config)
    
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
            cv2.imshow("Fence Detection", frame)
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
                cv2.imwrite("camostudio/Fence.jpg", img)
                cv2.imshow("Fence Detection", img)
                print("检测结果已保存到 Fence.jpg")
                if result and result.valid:
                    print(f"检测到有效的围栏区域:")
                    print(f"- 中心点: {result.center}")
                    print(f"- 面积: {result.area:.0f}")
                    print(f"- 宽高比: {result.aspect_ratio:.2f}")
                    print(f"- 四个角度: {[f'{angle:.1f}°' for angle in result.angles]}")
                else:
                    print("未检测到有效的围栏区域")
                    
                print("\n按任意键关闭窗口...")
                cv2.waitKey(0)
                cv2.destroyAllWindows()
        else:
            print("未输入图片路径，跳过测试。")
