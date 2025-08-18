"""
ArUco 标记检测模块
用于检测 ArUco 标记的位置和朝向
"""

import cv2
import numpy as np
from typing import Optional, List, Tuple, Union, Dict

class ArucoDetector:
    """
    ArUco 标记检测器类
    用于检测和定位 ArUco 标记
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
    
    def detect_single(self, frame: np.ndarray) -> Optional[Tuple[Tuple[int, int], float]]:
        """
        检测单个 ArUco 标记（返回第一个检测到的）
        
        Args:
            frame: 输入的BGR图像帧
            
        Returns:
            tuple: (center, angle) 或 None
                center: (x, y)  int 元组，标记中心坐标
                angle:  float    车体朝向角度（度）
        """
        if frame is None:
            return None
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, rejected = cv2.aruco.detectMarkers(gray, self.aruco_dict, parameters=self.parameters)
        
        if ids is None or len(ids) == 0:
            return None
        
        # 取第一个检测到的标记
        marker_corners = corners[0].reshape(4, 2)
        
        # 计算中心点
        center = np.mean(marker_corners, axis=0).astype(int)
        
        # 计算朝向角度（使用标记的第一条边：左上->右上）
        # ArUco角点顺序：左上、右上、右下、左下
        dx = marker_corners[1][0] - marker_corners[0][0]
        dy = marker_corners[1][1] - marker_corners[0][1]
        angle = np.degrees(np.arctan2(dy, dx))
        
        return tuple(center), angle
    
    def detect_all(self, frame: np.ndarray) -> List[Tuple[int, Tuple[int, int], float]]:
        """
        检测图像中所有的 ArUco 标记
        
        Args:
            frame: 输入的BGR图像帧
            
        Returns:
            list: 包含所有检测到的标记信息的列表
                每个元素是 (id, center, angle) 元组
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
            
            # 计算中心点
            center = np.mean(marker_corners, axis=0).astype(int)
            
            # 计算朝向角度
            dx = marker_corners[1][0] - marker_corners[0][0]
            dy = marker_corners[1][1] - marker_corners[0][1]
            angle = np.degrees(np.arctan2(dy, dx))
            
            results.append((marker_id, tuple(center), angle))
        
        return results
    
    def detect_by_id(self, frame: np.ndarray, target_id: int) -> Optional[Tuple[Tuple[int, int], float]]:
        """
        检测指定ID的 ArUco 标记
        
        Args:
            frame: 输入的BGR图像帧
            target_id: 目标标记ID
            
        Returns:
            tuple: (center, angle) 或 None
        """
        all_markers = self.detect_all(frame)
        for marker_id, center, angle in all_markers:
            if marker_id == target_id:
                return center, angle
        return None
    
    def draw_detections(self, frame: np.ndarray, detection_result: Union[Tuple, List, None]) -> np.ndarray:
        """
        在图像上绘制 ArUco 检测结果
        
        Args:
            frame: 输入的BGR图像帧
            detection_result: detect_single 或 detect_all 的返回结果
            
        Returns:
            np.ndarray: 绘制了检测结果的图像
        """
        output_frame = frame.copy()
        
        if detection_result is None:
            return output_frame
        
        # 处理单个检测结果
        if isinstance(detection_result, tuple) and len(detection_result) == 2:
            center, angle = detection_result
            self._draw_single_marker(output_frame, center, angle)
        
        # 处理多个检测结果
        elif isinstance(detection_result, list):
            for marker_id, center, angle in detection_result:
                self._draw_single_marker(output_frame, center, angle, marker_id)
        
        return output_frame
    
    def _draw_single_marker(self, frame: np.ndarray, center: Tuple[int, int], angle: float, marker_id: Optional[int] = None):
        """
        绘制单个标记
        
        Args:
            frame: 图像
            center: 中心点
            angle: 角度
            marker_id: 标记ID（可选）
        """
        # 绘制中心点
        cv2.circle(frame, center, 5, (0, 255, 0), -1)
        
        # 绘制朝向箭头
        arrow_length = 50
        end_x = int(center[0] + arrow_length * np.cos(np.radians(angle)))
        end_y = int(center[1] + arrow_length * np.sin(np.radians(angle)))
        cv2.arrowedLine(frame, center, (end_x, end_y), (255, 0, 0), 2)
        
        # 显示信息
        if marker_id is not None:
            text = f"ID:{marker_id} {angle:.1f}°"
        else:
            text = f"Angle: {angle:.1f}°"
        
        cv2.putText(frame, text, 
                   (center[0] + 10, center[1] - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    def get_detector_info(self) -> Dict:
        """
        获取当前使用的 ArUco 检测器信息
        
        Returns:
            dict: 包含字典类型和参数信息
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
    # 测试代码
    print("ArUco 检测模块加载成功")
    
    # 创建检测器实例
    detector = ArucoDetector()
    print("检测器信息:", detector.get_detector_info())
    
    # 简单的测试，需要摄像头或图像文件
    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    if cap.isOpened():
        print("摄像头已打开，按 'q' 退出测试，按 's' 切换检测模式")
        detect_all_mode = False
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if detect_all_mode:
                # 检测所有 ArUco 标记
                results = detector.detect_all(frame)
                if results:
                    print(f"检测到 {len(results)} 个标记:")
                    for marker_id, center, angle in results:
                        print(f"  ID={marker_id}: 中心=({center[0]}, {center[1]}), 角度={angle:.1f}°")
                    frame = detector.draw_detections(frame, results)
                
                # 显示模式信息
                cv2.putText(frame, "Mode: Detect All (Press 's' to switch)", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            else:
                # 检测单个 ArUco 标记
                result = detector.detect_single(frame)
                if result:
                    center, angle = result
                    print(f"检测到标记: 中心=({center[0]}, {center[1]}), 角度={angle:.1f}°")
                    frame = detector.draw_detections(frame, result)
                
                # 显示模式信息
                cv2.putText(frame, "Mode: Detect Single (Press 's' to switch)", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            cv2.imshow('ArUco Detection Test', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                detect_all_mode = not detect_all_mode
                print(f"切换到 {'检测所有标记' if detect_all_mode else '检测单个标记'} 模式")
        
        cap.release()
        cv2.destroyAllWindows()
    else:
        print("无法打开摄像头，跳过测试")
        
        # 演示如何使用类
        print("\n使用示例:")
        print("# 创建检测器")
        print("detector = ArucoDetector()")
        print("\n# 检测单个标记")
        print("result = detector.detect_single(frame)")
        print("\n# 检测所有标记")
        print("all_results = detector.detect_all(frame)")
        print("\n# 检测指定ID的标记")
        print("target_result = detector.detect_by_id(frame, target_id=5)")
        print("\n# 绘制检测结果")
        print("annotated_frame = detector.draw_detections(frame, result)")
