
"""
HomeDetector 家区域检测模块
------------------------
本模块用于检测画面中的“家”区域（黑色区域），返回其最小外接矩形和中心点，并可在图像上绘制检测结果。
主要类和方法：
    - HomeDetector: 主检测器类
        - detect: 检测家区域，返回中心点和矩形顶点
        - draw_results: 在画面上绘制检测结果
典型用法：
    detector = HomeDetector()
    result = detector.detect(hsv_img)
    frame = detector.draw_results(frame, result)
"""
import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict


@dataclass
class HSVRanges:
    """
    存储黑色区域的HSV阈值范围。
    black_lower: 黑色的HSV下界
    black_upper: 黑色的HSV上界
    """
    black_lower: tuple = (0, 0, 0)
    black_upper: tuple = (180, 255, 50)

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
    def __init__(self, hsv_ranges: HSVRanges = HSVRanges()):
        """
        初始化HomeDetector。
        参数:
            hsv_ranges: HSVRanges对象，指定黑色区域的HSV阈值
        """
        self.r = hsv_ranges
        self.kernel_open = np.ones((3, 3), np.uint8)
        self.kernel_close = np.ones((5, 5), np.uint8)

    def detect(self, hsv: np.ndarray) -> Optional[Dict[str, np.ndarray]]:
        """
        检测画面中的家区域（黑色区域），返回中心点和最小外接矩形顶点。
        参数:
            hsv: HSV格式的图像
        返回:
            dict: {"center": 中心点坐标, "box": 四个顶点坐标}
            若未检测到则返回None
        """
        mask = cv2.inRange(hsv, np.array(self.r.black_lower), np.array(self.r.black_upper))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel_open, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel_close, iterations=1)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        cnt = max(contours, key=cv2.contourArea)
        box = contour_min_area_rect_points(cnt).astype(int)
        center = compute_centroid(cnt)
        if center is None:
            center = tuple(np.mean(box, axis=0).astype(int))
        return {"center": np.array(center, dtype=int), "box": box}

    def draw_results(self, frame: np.ndarray, result: Optional[Dict[str, np.ndarray]]) -> np.ndarray:
        """
        在frame上绘制检测结果，包括最小外接矩形、中心点、坐标等信息。
        参数:
            frame: 原始画面
            result: 检测结果（中心点和顶点坐标）
        返回:
            绘制后的画面
        """
        if result is not None:
            box = result["box"]
            center = tuple(result["center"])
            # 绘制最小外接矩形
            cv2.polylines(frame, [box], isClosed=True, color=(0,255,0), thickness=2)
            # 绘制中心点
            cv2.circle(frame, center, 6, (0,0,255), -1)
            # 显示坐标信息
            cv2.putText(frame, f"Center: {center}", (center[0]+10, center[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
            # 显示四个顶点坐标
            for i, pt in enumerate(box):
                cv2.circle(frame, tuple(pt), 4, (255,0,0), -1)
                cv2.putText(frame, f"P{i+1}:({pt[0]},{pt[1]})", (pt[0], pt[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1)
        else:
            cv2.putText(frame, "No Home Detected", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        return frame
    
if __name__ == "__main__":
    """
    主程序：打开摄像头，实时检测并显示家区域。
    按q键退出。
    """
    cap = cv2.VideoCapture(0)
    detector = HomeDetector()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("无法读取摄像头")
            break

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        result = detector.detect(hsv)
        frame = detector.draw_results(frame, result)

        cv2.imshow("Home Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):  # 按q退出
            break

    cap.release()
    cv2.destroyAllWindows()
