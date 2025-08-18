import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional

@dataclass
class HSVRanges:
    blue_lower: tuple = (100, 50, 50)
    blue_upper: tuple = (130, 255, 255)

# 通用工具函数
def order_quad_points(pts: np.ndarray) -> np.ndarray:
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
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    return order_quad_points(box)

class FenceDetector:
    def __init__(self, hsv_ranges: HSVRanges = HSVRanges()):
        self.r = hsv_ranges
        self.kernel_open = np.ones((3, 3), np.uint8)
        self.kernel_close = np.ones((5, 5), np.uint8)

    def detect(self, hsv: np.ndarray) -> Optional[np.ndarray]:
        mask = cv2.inRange(hsv, np.array(self.r.blue_lower), np.array(self.r.blue_upper))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel_open, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel_close, iterations=1)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        cnt = max(contours, key=cv2.contourArea)
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
        return quad.astype(int)
