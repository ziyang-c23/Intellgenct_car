
"""
ItemDetector 物体检测模块
------------------------
本模块实现了基于OpenCV的红色和黄色物体检测系统。通过HSV颜色空间分割和形态学处理，
实现对目标物体的精确识别和特征提取。

主要功能：
1. 颜色检测：支持红色（双阈值）和黄色的HSV空间检测
2. 特征提取：计算物体的面积、中心点、边界框、周长、宽高比等特征
3. 有效性验证：基于面积和形状约束进行物体筛选
4. 可视化：支持检测结果的实时可视化，包括边界框、中心点和详细信息

核心类：
    ItemConfig: 检测器配置类，包含颜色阈值、形状约束等参数
    ItemInfo: 检测结果数据类，存储物体的所有特征信息
    ItemDetector: 主检测器类，实现核心检测和绘制功能

典型用法：
    # 创建检测器实例
    config = ItemConfig(min_area=100, max_area=10000)
    detector = ItemDetector(config)
    
    # 处理图像
    items = detector.detect(hsv_img)  # 检测物体
    frame = detector.draw_results(frame, items)  # 绘制结果

高级功能：
    - 支持距离计算（nearest_to方法）
    - 提供可自定义的验证规则
    - 兼容字典格式输出（向后兼容）
    - 支持结果排序和过滤
"""
import cv2
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple


@dataclass
class ItemConfig:
    """
    物体检测器的配置参数类
    
    属性:
        yellow_lower (Tuple[int,int,int]): 黄色HSV下限 (色相,饱和度,明度)
        yellow_upper (Tuple[int,int,int]): 黄色HSV上限
        red_lower1 (Tuple[int,int,int]): 红色HSV下限1（低色相段）
        red_upper1 (Tuple[int,int,int]): 红色HSV上限1
        red_lower2 (Tuple[int,int,int]): 红色HSV下限2（高色相段）
        red_upper2 (Tuple[int,int,int]): 红色HSV上限2
        min_area (int): 最小有效物体面积（像素）
        max_area (int): 最大有效物体面积（像素）
        min_aspect_ratio (float): 最小宽高比（宽/高）
        max_aspect_ratio (float): 最大宽高比
        kernel_size (Tuple[int,int]): 形态学操作核大小
        morph_iterations (int): 形态学操作迭代次数
    """
    # HSV颜色范围 - 黄色
    yellow_lower: Tuple[int, int, int] = (20, 120, 110)
    yellow_upper: Tuple[int, int, int] = (30, 255, 255)
    # HSV颜色范围 - 红色（两段）
    red_lower1: Tuple[int, int, int] = (0, 100, 70)
    red_upper1: Tuple[int, int, int] = (10, 255, 255)
    red_lower2: Tuple[int, int, int] = (175, 100, 70)
    red_upper2: Tuple[int, int, int] = (180, 255, 255)
    # 面积范围
    min_area: int = 0
    max_area: int = 200
    # 形状约束
    min_aspect_ratio: float = 0.5
    max_aspect_ratio: float = 2.0
    # 其他参数
    kernel_size: Tuple[int, int] = (5, 5)
    morph_iterations: int = 1

@dataclass
class ItemInfo:
    """
    物体检测结果信息类
    
    属性:
        type (str): 物体类型，可选值：'red' 或 'yellow'
        center (Tuple[int,int]): 物体中心点坐标 (x, y)
        area (float): 物体面积（像素数）
        bbox (Tuple[int,int,int,int]): 边界框 (x, y, width, height)
        contour (np.ndarray): 物体轮廓点集
        aspect_ratio (float): 宽高比 (width/height)
        perimeter (float): 物体轮廓周长
        valid (bool): 物体是否通过验证
        dist_px (Optional[float]): 到参考点的距离（像素），若未计算则为None
    
    方法:
        to_dict() -> Dict: 将对象转换为字典格式（用于兼容性）
    """
    type: str              # 物体类型：'red' 或 'yellow'
    center: Tuple[int, int]  # 中心点坐标
    area: float           # 面积
    bbox: Tuple[int, int, int, int]  # 边界框 (x, y, w, h)
    contour: np.ndarray   # 轮廓点集
    aspect_ratio: float   # 宽高比
    perimeter: float      # 周长
    valid: bool = True    # 是否为有效检测
    dist_px: Optional[float] = None  # 到参考点的距离（如果计算）

    def to_dict(self) -> Dict:
        """转换为字典格式，用于兼容旧代码"""
        return {
            "type": self.type,
            "center": self.center,
            "area": self.area,
            "bbox": self.bbox,
            "contour": self.contour,
            "aspect_ratio": self.aspect_ratio,
            "perimeter": self.perimeter,
            "valid": self.valid,
            "dist_px": self.dist_px
        }

def compute_centroid(contour: np.ndarray) -> Optional[Tuple[int, int]]:
    """
    计算轮廓的质心（中心点坐标）
    
    基于图像矩（moments）计算给定轮廓的质心坐标。当轮廓面积为0时返回None。
    
    参数:
        contour (np.ndarray): OpenCV格式的轮廓点集
        
    返回:
        Optional[Tuple[int, int]]: 
            - 成功时返回质心坐标 (cx, cy)
            - 计算失败时返回None
            
    实现细节:
        使用OpenCV的moments()函数计算图像矩，然后通过一阶矩和零阶矩的比值
        计算质心坐标：cx = M10/M00, cy = M01/M00
    """
    M = cv2.moments(contour)
    if M["m00"] == 0:
        return None
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    return (cx, cy)


class ItemDetector:
    """
    红黄物体检测器类
    
    本类实现了基于HSV颜色空间的红色和黄色物体检测。通过颜色分割、形态学处理
    和特征提取，实现对目标物体的识别和特征计算。
    
    主要特点:
        1. 支持红色（双阈值）和黄色的HSV空间检测
        2. 提供完整的物体特征提取（面积、中心点、边界框等）
        3. 支持基于面积和形状的物体验证
        4. 提供结果可视化功能
        5. 支持最近物体查找
    
    主要方法:
        detect(hsv): 检测图像中的所有红黄物体
        draw_results(frame, items): 在图像上可视化检测结果
        validate_item(item): 验证物体是否满足约束条件
        nearest_to(items, point): 查找距离指定点最近的物体
    
    属性:
        config (ItemConfig): 检测器配置对象
        kernel (np.ndarray): 形态学操作使用的结构元素
        last_result (List[Dict]): 上一次的检测结果（可选）
    """
    def __init__(self, config: ItemConfig = ItemConfig()):
        """
        初始化ItemDetector实例
        
        参数:
            config (ItemConfig, optional): 检测器配置对象
                                         若未提供，将使用默认配置
        
        初始化过程:
            1. 存储配置对象
            2. 创建形态学操作用的结构元素
            3. 初始化结果缓存
        """
        self.config = config
        self.kernel = np.ones(config.kernel_size, np.uint8)
        self.last_result = None

    def validate_item(self, item: ItemInfo) -> bool:
        """
        验证检测到的物体是否满足约束条件
        
        验证规则:
            1. 面积约束: min_area <= area <= max_area
            2. 形状约束: min_aspect_ratio <= width/height <= max_aspect_ratio
        
        参数:
            item (ItemInfo): 待验证的物体信息对象
            
        返回:
            bool: True表示物体有效，False表示无效
        """
        # 面积检查
        if not (self.config.min_area <= item.area <= self.config.max_area):
            return False
            
        # 形状检查
        if not (self.config.min_aspect_ratio <= item.aspect_ratio <= self.config.max_aspect_ratio):
            return False
            
        return True

    def _mask_color(self, hsv: np.ndarray, color: str) -> np.ndarray:
        """
        根据指定颜色在HSV空间生成二值掩码
        
        处理流程:
            1. 根据颜色类型选择对应的HSV阈值
            2. 使用cv2.inRange生成初始掩码
            3. 应用形态学开运算去除噪点
            4. 应用形态学闭运算填充小孔
        
        参数:
            hsv (np.ndarray): HSV格式的输入图像
            color (str): 颜色类型，可选值: 'red' 或 'yellow'
                        - red: 使用双阈值范围
                        - yellow: 使用单阈值范围
        
        返回:
            np.ndarray: 二值掩码图像，目标区域为255，背景为0
            
        异常:
            ValueError: 当指定的颜色类型不支持时抛出
        """
        if color == "red":
            m1 = cv2.inRange(hsv, np.array(self.config.red_lower1), np.array(self.config.red_upper1))
            m2 = cv2.inRange(hsv, np.array(self.config.red_lower2), np.array(self.config.red_upper2))
            mask = cv2.bitwise_or(m1, m2)
        elif color == "yellow":
            mask = cv2.inRange(hsv, np.array(self.config.yellow_lower), np.array(self.config.yellow_upper))
        else:
            raise ValueError(f"Unsupported color for ItemDetector: {color}")
            
        # 形态学操作
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel, 
                              iterations=self.config.morph_iterations)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel, 
                              iterations=self.config.morph_iterations)
        return mask

    def _extract_objects(self, mask: np.ndarray, label: str) -> List[ItemInfo]:
        """
        从二值掩码中提取物体信息
        
        处理流程:
            1. 使用findContours提取轮廓
            2. 对每个轮廓:
                - 计算面积并初步筛选
                - 计算中心点
                - 提取边界框和其他特征
                - 创建ItemInfo对象
                - 进行有效性验证
            3. 对有效物体按面积降序排序
        
        参数:
            mask (np.ndarray): 二值掩码图像
            label (str): 物体类型标签('red'或'yellow')
        
        返回:
            List[ItemInfo]: 检测到的有效物体列表，按面积降序排序
            
        特征计算:
            - 面积: contourArea
            - 中心点: moments方法
            - 边界框: boundingRect
            - 宽高比: width/height
            - 周长: arcLength
        """
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        items = []
        
        for c in contours:
            # 计算基本特征
            area = cv2.contourArea(c)
            if area < self.config.min_area:
                continue
                
            center = compute_centroid(c)
            if center is None:
                continue
                
            # 计算边界框和其他特征
            x, y, w, h = cv2.boundingRect(c)
            aspect_ratio = float(w) / float(h) if h != 0 else 0
            perimeter = cv2.arcLength(c, True)
            
            # 创建ItemInfo对象
            item = ItemInfo(
                type=label,
                center=(int(center[0]), int(center[1])),
                area=float(area),
                bbox=(int(x), int(y), int(w), int(h)),
                contour=c,
                aspect_ratio=aspect_ratio,
                perimeter=perimeter,
                valid=True  # 初始设为True，后面验证
            )
            
            # 验证物体
            item.valid = self.validate_item(item)
            if item.valid:  # 只添加有效的物体
                items.append(item)
                
        # 按面积降序排序
        items.sort(key=lambda x: x.area, reverse=True)
        return items

    def detect(self, hsv: np.ndarray) -> List[Dict]:
        """
        检测画面中的红色和黄色物体
        
        处理流程:
            1. 分别处理红色和黄色:
                - 生成颜色掩码
                - 提取物体信息
            2. 合并两种颜色的检测结果
            3. 转换为字典格式（兼容性考虑）
        
        参数:
            hsv (np.ndarray): HSV格式的输入图像
            
        返回:
            List[Dict]: 检测到的所有物体信息，每个物体表示为一个字典
            包含字段:
                - type: 物体类型 ('red'或'yellow')
                - center: 中心点坐标 (x,y)
                - area: 面积
                - bbox: 边界框 (x,y,w,h)
                - 其他特征...
        """
        # 处理不同颜色
        red_mask = self._mask_color(hsv, "red")
        yellow_mask = self._mask_color(hsv, "yellow")
        
        # 提取物体
        red_items = self._extract_objects(red_mask, "red")
        yellow_items = self._extract_objects(yellow_mask, "yellow")
        
        # 组合结果并转换为字典格式（兼容性）
        return [item.to_dict() for item in (red_items + yellow_items)]

    def draw_results(self, frame: np.ndarray, items: List[Dict]) -> np.ndarray:
        """
        在图像上可视化检测结果
        
        绘制内容:
            1. 每个物体的:
                - 边界框（矩形）
                - 中心点（圆点）
                - 详细信息（文本）
                    * 类型（红/黄）
                    * 面积
                    * 中心点坐标
                    * 宽高比（可选）
                    * 距离（可选）
                    * 有效性状态
            2. 统计信息:
                - 总物体数
                - 有效物体数
        
        参数:
            frame (np.ndarray): 原始图像
            items (List[Dict]): 物体信息列表
            
        返回:
            np.ndarray: 绘制结果后的图像
            
        注意:
            - 有效物体使用正常颜色显示
            - 无效物体使用灰色显示
            - 文本位置自动调整以避免超出图像边界
        """
        for item in items:
            # 获取基本信息
            x, y, w, h = item["bbox"]
            cx, cy = item["center"]
            valid = item.get("valid", True)  # 兼容旧数据
            
            # 设置颜色
            base_color = (0, 0, 255) if item["type"] == "red" else (0, 255, 255)
            color = base_color if valid else (128, 128, 128)
            
            # 绘制边界框
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            
            # 绘制中心点
            cv2.circle(frame, (cx, cy), 4, color, -1)
            
            # 准备显示信息
            info_parts = [
                f"{item['type'].upper()}",
                f"A:{int(item['area'])}",
                f"C:({cx},{cy})"
            ]
            if "aspect_ratio" in item:
                info_parts.append(f"AR:{item['aspect_ratio']:.2f}")
            if "dist_px" in item:
                info_parts.append(f"D:{item['dist_px']}px")
            if "valid" in item:
                info_parts.append(f"V:{valid}")
                
            # 计算文本位置
            text_x = x
            text_y = y - 10 if y > 30 else y + h + 20
            
            # 绘制信息
            cv2.putText(frame, " ".join(info_parts), 
                       (text_x, text_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
        # 显示总数和有效数
        valid_count = sum(1 for item in items if item.get("valid", True))
        status_text = f"Total: {len(items)} Valid: {valid_count}"
        cv2.putText(frame, status_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
            
        return frame

    @staticmethod
    def nearest_to(point: Tuple[int, int], items: List[Dict]) -> Optional[Dict]:
        """
        计算距离指定点最近的物体。
        参数:
            point: 参考点坐标
            items: 物体信息列表
        返回:
            距离最近的物体信息（包含距离dist_px），若无则返回None
        """
        if not items:
            return None
        px, py = point
        best = None
        best_d2 = float("inf")
        for it in items:
            cx, cy = it["center"]
            d2 = (cx - px) ** 2 + (cy - py) ** 2
            if d2 < best_d2:
                best = it
                best_d2 = d2
        if best is not None:
            best = dict(best)
            best["dist_px"] = int(np.sqrt(best_d2))

        return best



# 摄像头实时检测测试程序
if __name__ == "__main__":
    """
    主程序：打开摄像头，实时检测并显示红色和黄色物体。
    按q键退出。
    """
    cap = cv2.VideoCapture(0)
    detector = ItemDetector()
    if cap.isOpened():
        while True:
            ret, frame = cap.read()
            if not ret:
                print("无法读取摄像头")
                break
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            items = detector.detect(hsv)
            h_, w_ = frame.shape[:2]
            center_point = (w_ // 2, h_ // 2)
            nearest = ItemDetector.nearest_to(center_point, items)
            if nearest:
                for i, item in enumerate(items):
                    if item["center"] == nearest["center"]:
                        items[i]["dist_px"] = nearest["dist_px"]
            frame = detector.draw_results(frame, items)
            cv2.circle(frame, center_point, 6, (255,0,0), -1)
            cv2.putText(frame, "Center", (center_point[0]+10, center_point[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)
            cv2.imshow("Item Detection", frame)
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
                hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                items = detector.detect(hsv)
                h_, w_ = img.shape[:2]
                center_point = (w_ // 2, h_ // 2)
                nearest = ItemDetector.nearest_to(center_point, items)
                if nearest:
                    for i, item in enumerate(items):
                        if item["center"] == nearest["center"]:
                            items[i]["dist_px"] = nearest["dist_px"]
                img = detector.draw_results(img, items)
                cv2.circle(img, center_point, 6, (255,0,0), -1)
                cv2.putText(img, "Center", (center_point[0]+10, center_point[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)
                cv2.imshow("Item Image Detection", img)
                cv2.imwrite("camostudio/Item.jpg", img)
                print("按任意键关闭窗口...")
                cv2.waitKey(0)
                cv2.destroyAllWindows()
        else:
            print("未输入图片路径，跳过测试。")
