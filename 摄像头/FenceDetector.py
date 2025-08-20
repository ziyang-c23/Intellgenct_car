"""
FenceDetector 围栏检测模块
------------------------
本模块用于检测画面中的蓝色围栏区域，返回其四边形顶点坐标，并可在图像上绘制检测结果。
主要类和方法：
    - FenceDetector: 主检测器类
        - detect: 检测围栏区域
        - draw_results: 在画面上绘制检测结果
"""


import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional

@dataclass
class HSVRanges:
    """
    存储蓝色围栏的HSV阈值范围。
    blue_lower/blue_upper: 蓝色的HSV下界和上界
    """
    blue_lower: tuple = (100, 50, 50)
    blue_upper: tuple = (130, 255, 255)

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

class FenceDetector:
    """
    FenceDetector类用于检测画面中的蓝色围栏区域，返回其四边形顶点坐标。
    方法：
        - detect: 检测并返回围栏四边形顶点
        - draw_results: 在画面上绘制检测结果
    """
    def __init__(self, hsv_ranges: HSVRanges = HSVRanges()):
        """
        初始化FenceDetector。
        参数:
            hsv_ranges: HSVRanges对象，指定蓝色阈值
        """
        self.r = hsv_ranges
        self.kernel_open = np.ones((3, 3), np.uint8)
        self.kernel_close = np.ones((5, 5), np.uint8)

    def draw_results(self, frame: np.ndarray, quad: Optional[np.ndarray]) -> np.ndarray:
        """
        在frame上绘制检测到的围栏区域，包括四边形、顶点坐标、中心点等信息。
        参数:
            frame: 原始画面
            quad: 检测到的四边形顶点坐标
        返回:
            绘制后的画面
        """
        if quad is not None:
            # 绘制四边形
            cv2.polylines(frame, [quad], isClosed=True, color=(255,0,0), thickness=2)
            # 绘制顶点及坐标
            for i, pt in enumerate(quad):
                cv2.circle(frame, tuple(pt), 5, (0,255,0), -1)
                cv2.putText(frame, f"P{i+1}:({pt[0]},{pt[1]})", (pt[0]+5, pt[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
            # 显示中心点
            center = tuple(np.mean(quad, axis=0).astype(int))
            cv2.circle(frame, center, 7, (0,0,255), -1)
            cv2.putText(frame, f"Center: {center}", (center[0]+10, center[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
        else:
            cv2.putText(frame, "No Fence Detected", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        return frame

    def detect(self, hsv: np.ndarray) -> dict:
        """
        检测画面中的蓝色围栏区域，返回包含多项信息的字典。
        参数:
            hsv: HSV格式的图像
        返回:
            dict，包含如下内容：
                - quad: 四边形顶点坐标，若未检测到则为None
                - center: 四边形中心点坐标，若未检测到则为None
                - area: 区域面积，若未检测到则为0
                - contour: 最大轮廓点集，若未检测到则为None
                - mask: 二值掩码
        """
        mask = cv2.inRange(hsv, np.array(self.r.blue_lower), np.array(self.r.blue_upper))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel_open, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel_close, iterations=1)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        result = {
            "quad": None,
            "center": None,
            "area": 0,
            "contour": None,
            "mask": mask
        }
        if not contours:
            return result
        cnt = max(contours, key=cv2.contourArea)
        result["contour"] = cnt
        result["area"] = cv2.contourArea(cnt)
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
        result["quad"] = quad
        result["center"] = tuple(np.mean(quad, axis=0).astype(int)) if quad is not None else None
        return result
if __name__ == "__main__":
    """
    主程序：打开摄像头，实时检测并显示围栏区域。
    按q键退出。
    """
    cap = cv2.VideoCapture(2)
    detector = FenceDetector()
    if cap.isOpened():
        while True:
            ret, frame = cap.read()
            if not ret:
                print("无法读取摄像头")
                break
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            result = detector.detect(hsv)
            frame = detector.draw_results(frame, result['quad'])
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
                hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                result = detector.detect(hsv)
                img = detector.draw_results(img, result['quad'])
                cv2.imshow("Fence Image Detection", img)
                cv2.imwrite("Fence.jpg", img)
                print("按任意键关闭窗口...")
                cv2.waitKey(0)
                cv2.destroyAllWindows()
        else:
            print("未输入图片路径，跳过测试。")
