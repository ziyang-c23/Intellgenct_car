
"""
ArUco 标记检测模块
-------------------
本模块用于检测 ArUco 标记的位置和朝向，支持单个/多个/指定ID标记检测，并可在图像上绘制检测结果。
主要类和方法：
    - ArucoDetector: 主检测器类
        - detect_single: 检测单个标记
        - detect_all: 检测所有标记
        - detect_by_id: 检测指定ID标记
        - draw_detections: 在图像上绘制检测结果
        - get_detector_info: 获取检测器参数信息

返回数据格式：
    - 单个标记信息（字典）：
        - id: 标记ID
        - center: (x, y) 中心坐标
        - angle: 朝向角度（度）
        - corners: 四个角点坐标
        - area: 标记面积（像素）
        - perimeter: 标记周长（像素）
        - aspect_ratio: 宽高比
        - valid: 标记有效性验证结果
"""

import cv2
import numpy as np
from typing import Optional, List, Tuple, Union, Dict

class ArucoDetector:
    """
    ArUco 标记检测器类
    用于检测和定位 ArUco 标记，支持多种检测模式和结果绘制。
    """
    
    def __init__(self, dict_type=cv2.aruco.DICT_6X6_50):
        """
        初始化 ArUco 检测器
        
        Args:
            dict_type: ArUco 字典类型，默认为 DICT_6X6_50
        """
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(dict_type)
        self.parameters = cv2.aruco.DetectorParameters()
        self.dict_type = dict_type
    
    def detect_single(self, frame: np.ndarray) -> Optional[Dict]:
        """
        检测单个 ArUco 标记（返回第一个检测到的）
        
        Args:
            frame: 输入的BGR图像帧
            
        Returns:
            dict: 包含标记信息的字典，或 None
                字典包含以下键值：
                - id: 标记ID
                - center: (x, y) 中心坐标
                - angle: 朝向角度（度）
                - corners: 四个角点坐标
                - area: 标记面积（像素）
                - perimeter: 标记周长（像素）
                - aspect_ratio: 宽高比
                - valid: 标记有效性验证结果
        """
        if frame is None:
            return None
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, rejected = cv2.aruco.detectMarkers(gray, self.aruco_dict, parameters=self.parameters)
        
        if ids is None or len(ids) == 0:
            return None
        
        # 使用detect_all的结果并返回第一个
        all_markers = self.detect_all(frame)
        if all_markers:
            return all_markers[0]
            
        return None
    
    def detect_all(self, frame: np.ndarray) -> List[Dict]:
        """
        检测图像中所有的 ArUco 标记
        
        Args:
            frame: 输入的BGR图像帧
            
        Returns:
            list: 包含所有检测到的标记信息的列表
                每个元素是一个字典，包含以下键值：
                - id: 标记ID
                - center: (x, y) 中心坐标
                - angle: 朝向角度（度）
                - corners: 四个角点坐标
                - area: 标记面积（像素）
                - perimeter: 标记周长（像素）
                - aspect_ratio: 宽高比
                - valid: 标记有效性验证结果
        """
        if frame is None:
            return []
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, rejected = cv2.aruco.detectMarkers(gray, self.aruco_dict, parameters=self.parameters)
        
        if ids is None or len(ids) == 0:
            return []
        
        results = []
        for i, marker_id in enumerate(ids.flatten()):
            marker_corners = corners[i].reshape(4, 2)
            
            # 1. 计算中心点
            center = np.mean(marker_corners, axis=0).astype(int)
            
            # 2. 计算朝向角度
            dx = marker_corners[1][0] - marker_corners[0][0]
            dy = marker_corners[1][1] - marker_corners[0][1]
            angle = np.degrees(np.arctan2(dy, dx))
            
            # 3. 计算面积
            area = cv2.contourArea(marker_corners)
            
            # 4. 计算周长
            perimeter = cv2.arcLength(marker_corners.astype(np.float32), True)
            
            # 5. 计算宽高比
            rect = cv2.minAreaRect(marker_corners)
            aspect_ratio = max(rect[1]) / (min(rect[1]) + 1e-6)  # 防止除零
            
            # 6. 验证标记的有效性
            valid = self._validate_marker(marker_corners, area, aspect_ratio)
            
            marker_info = {
                'id': int(marker_id),
                'center': tuple(center),
                'angle': float(angle),
                'corners': marker_corners.tolist(),
                'area': float(area),
                'perimeter': float(perimeter),
                'aspect_ratio': float(aspect_ratio),
                'valid': valid
            }
            
            results.append(marker_info)
        
        return results
    
    def detect_by_id(self, frame: np.ndarray, target_id: int) -> Optional[Dict]:
        """
        检测指定ID的 ArUco 标记
        
        Args:
            frame: 输入的BGR图像帧
            target_id: 目标标记ID
            
        Returns:
            dict: 包含标记信息的字典，或 None
        """
        all_markers = self.detect_all(frame)
        for marker in all_markers:
            if marker['id'] == target_id:
                return marker
        return None
    
    def draw_detections(self, frame: np.ndarray, detection_result: Union[Dict, List[Dict], None]) -> np.ndarray:
        """
        在图像上绘制 ArUco 检测结果。
        支持单个标记或多个标记的检测结果。
        保证每个标记都绘制完整信息。
        参数:
            frame: 输入的BGR图像帧
            detection_result: detect_single 或 detect_all 的返回结果
                - detect_single: 包含标记信息的字典
                - detect_all: 包含标记信息的字典列表
        返回:
            np.ndarray: 绘制了检测结果的图像
        """
        output_frame = frame.copy()
        if detection_result is None:
            return output_frame
            
        if isinstance(detection_result, dict):
            self._draw_single_marker(output_frame, detection_result)
        elif isinstance(detection_result, list):
            for marker in detection_result:
                self._draw_single_marker(output_frame, marker)
                
        return output_frame
    
    def _draw_single_marker(self, frame: np.ndarray, marker: Dict):
        """
        在图像上绘制单个 ArUco 标记的检测结果
        参数:
            frame: 图像
            marker: 包含标记信息的字典
        """
        center = marker['center']
        angle = marker['angle']
        marker_id = marker.get('id')
        
        # 1. 绘制角点和轮廓
        if 'corners' in marker:
            corners = np.array(marker['corners'], dtype=np.int32)
            cv2.polylines(frame, [corners], True, (0, 255, 0), 2)
            # 标记四个角点
            for i, corner in enumerate(corners):
                cv2.circle(frame, tuple(corner), 3, (0, 0, 255), -1)
        
        # 2. 绘制中心点
        cv2.circle(frame, center, 5, (0, 255, 0), -1)
        
        # 3. 绘制朝向箭头
        arrow_length = 70
        end_x = int(center[0] + arrow_length * np.cos(np.radians(angle)))
        end_y = int(center[1] + arrow_length * np.sin(np.radians(angle)))
        cv2.arrowedLine(frame, center, (end_x, end_y), (255, 0, 0), 2)
        
        # 4. 显示标记信息
        texts = []
        if marker_id is not None:
            texts.append(f"ID:{marker_id}")
        texts.append(f"({center[0]},{center[1]})")
        texts.append(f"{angle:.1f}deg")
        
        # 添加面积信息
        if 'area' in marker:
            texts.append(f"Area:{marker['area']:.0f}")
        
        # 添加有效性标记
        if 'valid' in marker:
            valid_text = "Valid" if marker['valid'] else "Invalid"
            texts.append(valid_text)
        
        # 绘制所有文本信息
        text = " | ".join(texts)
        cv2.putText(frame, text,
                    (center[0] + 10, center[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    def _validate_marker(self, corners: np.ndarray, area: float, aspect_ratio: float) -> bool:
        """
        验证ArUco标记的有效性
        
        参数:
            corners: 标记的四个角点坐标
            area: 标记面积
            aspect_ratio: 宽高比
            
        返回:
            bool: 标记是否有效
        """
        # 1. 检查面积是否在合理范围内
        MIN_AREA = 100  # 最小面积（像素）
        MAX_AREA = 100000  # 最大面积（像素）
        if area < MIN_AREA or area > MAX_AREA:
            return False
            
        # 2. 检查宽高比是否接近1（正方形）
        if aspect_ratio > 1.5 or aspect_ratio < 0.67:
            return False
            
        # 3. 检查四边是否大致相等
        edges = []
        for i in range(4):
            next_i = (i + 1) % 4
            edge = np.linalg.norm(corners[next_i] - corners[i])
            edges.append(edge)
        
        mean_edge = np.mean(edges)
        edge_ratios = [edge/mean_edge for edge in edges]
        
        # 所有边长与平均值的比例应在0.8-1.2之间
        if any(ratio < 0.8 or ratio > 1.2 for ratio in edge_ratios):
            return False
            
        return True
        
    def get_detector_info(self) -> Dict:
        """
        获取当前使用的 ArUco 检测器信息
        
        Returns:
            dict: 包含字典类型和参数信息，包括：
                - dict_type: 使用的ArUco字典类型名称
                - marker_size: 标记的大小（如6x6）
                - total_markers: 字典中的标记总数
                - min_marker_size: 建议的最小标记尺寸（像素）
        """
        dict_names = {
            cv2.aruco.DICT_4X4_50: "DICT_4X4_50",
            cv2.aruco.DICT_4X4_100: "DICT_4X4_100",
            cv2.aruco.DICT_4X4_250: "DICT_4X4_250",
            cv2.aruco.DICT_4X4_1000: "DICT_4X4_1000",
            cv2.aruco.DICT_5X5_50: "DICT_5X5_50",
            cv2.aruco.DICT_5X5_100: "DICT_5X5_100",
            cv2.aruco.DICT_5X5_250: "DICT_5X5_250",
            cv2.aruco.DICT_5X5_1000: "DICT_5X5_1000",
            cv2.aruco.DICT_6X6_50: "DICT_6X6_50",
            cv2.aruco.DICT_6X6_100: "DICT_6X6_100",
            cv2.aruco.DICT_6X6_250: "DICT_6X6_250",
            cv2.aruco.DICT_6X6_1000: "DICT_6X6_1000",
            cv2.aruco.DICT_7X7_50: "DICT_7X7_50",
            cv2.aruco.DICT_7X7_100: "DICT_7X7_100",
            cv2.aruco.DICT_7X7_250: "DICT_7X7_250",
            cv2.aruco.DICT_7X7_1000: "DICT_7X7_1000"
        }
        
        dict_name = dict_names.get(self.dict_type, "UNKNOWN")
        
        return {
            "dictionary": dict_name,
            "dict_type": self.dict_type,
            "parameters": "DetectorParameters"
        }


if __name__ == "__main__":
    """
    ArUco标记检测器测试程序
    功能：
    1. 摄像头实时检测模式
       - 自动检测和显示所有ArUco标记
       - 显示每个标记的详细信息
       - 提供视觉反馈
    2. 图片检测模式（当摄像头不可用时）
       - 从文件读取图片进行检测
       - 保存检测结果
    3. 交互功能：
       - 'q': 退出程序
       - 's': 切换检测模式（单个/所有标记）
    """
    print("\n=== ArUco标记检测测试程序 ===")
    print("初始化检测器...")
    detector = ArucoDetector()
    info = detector.get_detector_info()
    print(f"使用字典类型: {info['dictionary']}")
    
    # 1. 尝试打开摄像头
    print("\n尝试打开摄像头...")
    cap = cv2.VideoCapture(0)  # 使用默认摄像头
    
    if cap.isOpened():
        # 设置摄像头分辨率
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        print("\n摄像头模式：")
        print("- 按'q'退出程序")
        print("- 按's'切换检测模式")
        
        detect_all_mode = True  # 默认检测所有标记
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("错误: 无法读取摄像头画面")
                break
                
            # 根据模式进行检测
            if detect_all_mode:
                markers = detector.detect_all(frame)
                if markers:
                    frame = detector.draw_detections(frame, markers)
                    # 显示检测到的标记数量
                    cv2.putText(frame, f"Detected: {len(markers)} markers", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    
                    # 显示每个标记的信息
                    y = 60
                    for marker in markers:
                        if marker.get('valid', True):
                            text = (f"ID {marker['id']} | "
                                   f"Pos ({marker['center'][0]},{marker['center'][1]}) | "
                                   f"Angle {marker['angle']:.1f} deg")
                            cv2.putText(frame, text, (10, y),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                            y += 25
            else:
                marker = detector.detect_single(frame)
                if marker:
                    frame = detector.draw_detections(frame, marker)
                    
            # 显示当前模式
            mode_text = "Mode: " + ("Detect All" if detect_all_mode else "Detect Single")
            cv2.putText(frame, mode_text, (frame.shape[1] - 300, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                       
            # 显示图像
            cv2.imshow('ArUco Detection', frame)
            
            # 处理键盘输入
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                detect_all_mode = not detect_all_mode
                print(f"切换到{'检测所有标记' if detect_all_mode else '检测单个标记'}模式")
        
        # 清理资源
        print("\n程序结束")
        cap.release()
        cv2.destroyAllWindows()
        
    else:
        # 2. 图片检测模式
        print("摄像头不可用，切换到图片检测模式")
        img_path = input("\n请输入图片路径（如 test.jpg），直接回车退出：").strip()
        
        if img_path:
            img = cv2.imread(img_path)
            if img is None:
                print(f"错误: 无法读取图片 {img_path}")
            else:
                # 处理图片
                markers = detector.detect_all(img)
                if markers:
                    print(f"\n检测到 {len(markers)} 个标记:")
                    for marker in markers:
                        print(f"  ID {marker['id']}: "
                              f"位置=({marker['center'][0]},{marker['center'][1]}), "
                              f"角度={marker['angle']:.1f}度, "
                              f"面积={marker['area']:.0f}像素")
                              
                    # 绘制检测结果
                    img = detector.draw_detections(img, markers)
                else:
                    print("\n未检测到任何标记")
                                
                # 显示结果
                cv2.imshow('ArUco Detection Result', img)   
                cv2.imwrite('camostudio/ArUco.jpg', img)
                print("\n按任意键关闭窗口...")
                cv2.waitKey(0)
                cv2.destroyAllWindows()
        else:
            print("未输入图片路径，程序退出。")
