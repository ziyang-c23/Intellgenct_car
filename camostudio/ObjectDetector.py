"""
智能机电系统 - 物体检测与定位模块
用于识别和定位场地内的目标物体：
- 黄色圆柱体
- 红色正立方体  
- 黑色目标区域
- 蓝色围栏
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional

class ObjectDetector:
    def __init__(self):
        # 颜色阈值范围 (HSV色彩空间)
        self.color_ranges = {
            'yellow': {
                'lower': np.array([20, 100, 100]),   # 黄色下界
                'upper': np.array([30, 255, 255])    # 黄色上界
            },
            'red': {
                # 红色在HSV中跨越0度，需要两个范围
                'lower1': np.array([0, 120, 70]),
                'upper1': np.array([10, 255, 255]),
                'lower2': np.array([170, 120, 70]),
                'upper2': np.array([180, 255, 255])
            },
            'black': {
                'lower': np.array([0, 0, 0]),        # 黑色下界
                'upper': np.array([180, 255, 50])    # 黑色上界
            },
            'blue': {
                'lower': np.array([100, 50, 50]),    # 蓝色下界
                'upper': np.array([130, 255, 255])   # 蓝色上界
            }
        }
        
        # 物体形状参数
        self.min_contour_area = 500      # 最小轮廓面积
        self.cylinder_aspect_ratio = 0.8 # 圆柱体长宽比阈值
        self.cube_aspect_ratio = 0.7     # 立方体长宽比阈值

    def preprocess_image(self, frame: np.ndarray) -> np.ndarray:
        """
        图像预处理
        Args:
            frame: 输入BGR图像
        Returns:
            processed_frame: 处理后的HSV图像
        """
        # 高斯模糊减少噪声
        blurred = cv2.GaussianBlur(frame, (5, 5), 0)
        # 转换到HSV色彩空间
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        return hsv

    def detect_color_mask(self, hsv_frame: np.ndarray, color: str) -> np.ndarray:
        """
        根据颜色创建掩膜
        Args:
            hsv_frame: HSV图像
            color: 颜色名称 ('yellow', 'red', 'black', 'blue')
        Returns:
            mask: 颜色掩膜
        """
        if color == 'red':
            # 红色需要特殊处理
            mask1 = cv2.inRange(hsv_frame, 
                               self.color_ranges['red']['lower1'],
                               self.color_ranges['red']['upper1'])
            mask2 = cv2.inRange(hsv_frame, 
                               self.color_ranges['red']['lower2'],
                               self.color_ranges['red']['upper2'])
            mask = cv2.bitwise_or(mask1, mask2)
        else:
            mask = cv2.inRange(hsv_frame, 
                              self.color_ranges[color]['lower'],
                              self.color_ranges[color]['upper'])
        
        # 形态学操作去除噪声
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        return mask

    def analyze_contour_shape(self, contour: np.ndarray) -> str:
        """
        分析轮廓形状，判断是圆柱体还是立方体
        Args:
            contour: 轮廓点
        Returns:
            shape: 'cylinder', 'cube', 或 'unknown'
        """
        # 计算轮廓的外接矩形
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = float(w) / h
        
        # 计算轮廓面积和外接矩形面积的比值
        contour_area = cv2.contourArea(contour)
        rect_area = w * h
        extent = float(contour_area) / rect_area
        
        # 计算周长和圆形度
        perimeter = cv2.arcLength(contour, True)
        if perimeter > 0:
            circularity = 4 * np.pi * contour_area / (perimeter * perimeter)
        else:
            circularity = 0
        
        # 根据特征判断形状
        if circularity > 0.7 and 0.7 < aspect_ratio < 1.3:
            return 'cylinder'  # 圆形且长宽比接近1
        elif extent > 0.8 and 0.7 < aspect_ratio < 1.3:
            return 'cube'      # 矩形且长宽比接近1
        else:
            return 'unknown'

    def detect_objects(self, frame: np.ndarray) -> Dict[str, List[Dict]]:
        """
        检测所有目标物体
        Args:
            frame: 输入BGR图像
        Returns:
            objects: 检测结果字典
        """
        hsv = self.preprocess_image(frame)
        results = {
            'yellow_cylinders': [],
            'red_cubes': [],
            'black_target_areas': [],
            'blue_barriers': []
        }
        
        # 检测黄色圆柱体
        yellow_mask = self.detect_color_mask(hsv, 'yellow')
        yellow_objects = self.find_objects_in_mask(yellow_mask, 'yellow')
        for obj in yellow_objects:
            if obj['shape'] == 'cylinder' or obj['shape'] == 'unknown':
                results['yellow_cylinders'].append(obj)
        
        # 检测红色立方体
        red_mask = self.detect_color_mask(hsv, 'red')
        red_objects = self.find_objects_in_mask(red_mask, 'red')
        for obj in red_objects:
            if obj['shape'] == 'cube' or obj['shape'] == 'unknown':
                results['red_cubes'].append(obj)
        
        # 检测黑色目标区域
        black_mask = self.detect_color_mask(hsv, 'black')
        black_objects = self.find_objects_in_mask(black_mask, 'black')
        results['black_target_areas'] = black_objects
        
        # 检测蓝色围栏
        blue_mask = self.detect_color_mask(hsv, 'blue')
        blue_objects = self.find_objects_in_mask(blue_mask, 'blue')
        results['blue_barriers'] = blue_objects
        
        return results

    def find_objects_in_mask(self, mask: np.ndarray, color: str) -> List[Dict]:
        """
        在掩膜中查找物体
        Args:
            mask: 颜色掩膜
            color: 颜色名称
        Returns:
            objects: 物体信息列表
        """
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        objects = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.min_contour_area:
                continue
                
            # 计算中心点和边界框
            M = cv2.moments(contour)
            if M['m00'] != 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
            else:
                continue
                
            x, y, w, h = cv2.boundingRect(contour)
            
            # 分析形状
            shape = self.analyze_contour_shape(contour)
            
            # 计算最小外接圆
            (circle_x, circle_y), radius = cv2.minEnclosingCircle(contour)
            
            obj_info = {
                'center': (cx, cy),
                'bounding_box': (x, y, w, h),
                'area': area,
                'contour': contour,
                'shape': shape,
                'color': color,
                'circle_center': (int(circle_x), int(circle_y)),
                'radius': int(radius)
            }
            
            objects.append(obj_info)
        
        # 按面积排序（大到小）
        objects.sort(key=lambda x: x['area'], reverse=True)
        return objects

    def get_object_positions(self, objects_dict: Dict[str, List[Dict]]) -> Dict[str, List[Tuple[int, int]]]:
        """
        获取所有物体的位置坐标
        Args:
            objects_dict: 物体检测结果
        Returns:
            positions: 位置坐标字典
        """
        positions = {}
        for obj_type, obj_list in objects_dict.items():
            positions[obj_type] = [obj['center'] for obj in obj_list]
        return positions

    def draw_detections(self, frame: np.ndarray, objects_dict: Dict[str, List[Dict]]) -> np.ndarray:
        """
        在图像上绘制检测结果
        Args:
            frame: 输入图像
            objects_dict: 物体检测结果
        Returns:
            annotated_frame: 标注后的图像
        """
        annotated = frame.copy()
        
        # 定义颜色
        colors = {
            'yellow_cylinders': (0, 255, 255),   # 黄色
            'red_cubes': (0, 0, 255),           # 红色
            'black_target_areas': (255, 255, 255), # 白色（用于黑色物体）
            'blue_barriers': (255, 0, 0)        # 蓝色
        }
        
        for obj_type, obj_list in objects_dict.items():
            color = colors.get(obj_type, (0, 255, 0))
            
            for i, obj in enumerate(obj_list):
                center = obj['center']
                bbox = obj['bounding_box']
                x, y, w, h = bbox
                
                # 绘制边界框
                cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)
                
                # 绘制中心点
                cv2.circle(annotated, center, 5, color, -1)
                
                # 绘制标签
                label = f"{obj_type.replace('_', ' ')} {i+1}"
                if obj['shape'] != 'unknown':
                    label += f" ({obj['shape']})"
                
                cv2.putText(annotated, label, (x, y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                # 绘制坐标信息
                coord_text = f"({center[0]}, {center[1]})"
                cv2.putText(annotated, coord_text, (center[0] + 10, center[1] + 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        return annotated

    def find_target_area(self, objects_dict: Dict[str, List[Dict]]) -> Optional[Tuple[int, int]]:
        """
        找到最大的黑色目标区域
        Args:
            objects_dict: 物体检测结果
        Returns:
            target_center: 目标区域中心坐标，如果没找到则返回None
        """
        black_areas = objects_dict.get('black_target_areas', [])
        if not black_areas:
            return None
        
        # 返回面积最大的黑色区域的中心
        largest_area = max(black_areas, key=lambda x: x['area'])
        return largest_area['center']

    def calculate_pickup_sequence(self, objects_dict: Dict[str, List[Dict]], 
                                 target_area: Tuple[int, int]) -> List[Dict]:
        """
        计算物体抓取顺序（基于距离目标区域的远近）
        Args:
            objects_dict: 物体检测结果
            target_area: 目标区域中心坐标
        Returns:
            sequence: 按抓取顺序排列的物体列表
        """
        all_objects = []
        
        # 收集所有需要抓取的物体
        for obj in objects_dict.get('yellow_cylinders', []):
            obj['type'] = 'yellow_cylinder'
            all_objects.append(obj)
        
        for obj in objects_dict.get('red_cubes', []):
            obj['type'] = 'red_cube'
            all_objects.append(obj)
        
        # 计算到目标区域的距离
        for obj in all_objects:
            distance = np.sqrt((obj['center'][0] - target_area[0])**2 + 
                             (obj['center'][1] - target_area[1])**2)
            obj['distance_to_target'] = distance
        
        # 按距离排序（远的先抓）
        all_objects.sort(key=lambda x: x['distance_to_target'], reverse=True)
        
        return all_objects

def detect_field_objects(frame: np.ndarray) -> Dict:
    """
    便捷函数：检测场地中的所有物体
    Args:
        frame: 输入图像
    Returns:
        detection_result: 完整的检测结果
    """
    detector = ObjectDetector()
    objects = detector.detect_objects(frame)
    target_area = detector.find_target_area(objects)
    positions = detector.get_object_positions(objects)
    
    # 如果找到目标区域，计算抓取顺序
    pickup_sequence = []
    if target_area:
        pickup_sequence = detector.calculate_pickup_sequence(objects, target_area)
    
    return {
        'objects': objects,
        'positions': positions,
        'target_area': target_area,
        'pickup_sequence': pickup_sequence,
        'total_items': len(objects.get('yellow_cylinders', [])) + len(objects.get('red_cubes', []))
    }

if __name__ == "__main__":
    # 测试代码
    print("物体检测模块加载成功")
    
    detector = ObjectDetector()
    
    # 测试摄像头
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        print("摄像头已打开，按 'q' 退出，按 's' 保存当前检测结果")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # 检测物体
            result = detect_field_objects(frame)
            objects = result['objects']
            target_area = result['target_area']
            
            # 绘制检测结果
            annotated = detector.draw_detections(frame, objects)
            
            # 显示目标区域
            if target_area:
                cv2.circle(annotated, target_area, 20, (255, 255, 255), 3)
                cv2.putText(annotated, "TARGET", (target_area[0] - 30, target_area[1] - 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            # 显示统计信息
            info_text = [
                f"Yellow Cylinders: {len(objects['yellow_cylinders'])}",
                f"Red Cubes: {len(objects['red_cubes'])}",
                f"Target Areas: {len(objects['black_target_areas'])}",
                f"Blue Barriers: {len(objects['blue_barriers'])}"
            ]
            
            for i, text in enumerate(info_text):
                cv2.putText(annotated, text, (10, 30 + i * 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            cv2.imshow('Object Detection', annotated)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                cv2.imwrite('detection_result.jpg', annotated)
                print("检测结果已保存为 detection_result.jpg")
                print(f"检测摘要: {result}")
        
        cap.release()
        cv2.destroyAllWindows()
    else:
        print("无法打开摄像头，跳过测试")
