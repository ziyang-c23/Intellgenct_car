import cv2
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

@dataclass
class HSVRanges:
    yellow_lower: Tuple[int, int, int] = (20, 100, 100)
    yellow_upper: Tuple[int, int, int] = (30, 255, 255)
    red_lower1: Tuple[int, int, int] = (0, 120, 70)
    red_upper1: Tuple[int, int, int] = (10, 255, 255)
    red_lower2: Tuple[int, int, int] = (170, 120, 70)
    red_upper2: Tuple[int, int, int] = (180, 255, 255)

@dataclass
class ItemDetectorConfig:
    min_area: int = 100

def compute_centroid(contour: np.ndarray) -> Optional[Tuple[int, int]]:
    M = cv2.moments(contour)
    if M["m00"] == 0:
        return None
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    return (cx, cy)

class ItemDetector:
    def __init__(self, cfg: ItemDetectorConfig = ItemDetectorConfig(), hsv_ranges: HSVRanges = HSVRanges()):
        self.cfg = cfg
        self.r = hsv_ranges
        self.kernel = np.ones((5, 5), np.uint8)

    def _mask_color(self, hsv: np.ndarray, color: str) -> np.ndarray:
        if color == "red":
            m1 = cv2.inRange(hsv, np.array(self.r.red_lower1), np.array(self.r.red_upper1))
            m2 = cv2.inRange(hsv, np.array(self.r.red_lower2), np.array(self.r.red_upper2))
            mask = cv2.bitwise_or(m1, m2)
        elif color == "yellow":
            mask = cv2.inRange(hsv, np.array(self.r.yellow_lower), np.array(self.r.yellow_upper))
        else:
            raise ValueError("Unsupported color for ItemDetector: %s" % color)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel, iterations=1)
        return mask

    def _extract_objects(self, mask: np.ndarray, label: str) -> List[Dict]:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        items: List[Dict] = []
        for c in contours:
            area = cv2.contourArea(c)
            if area < self.cfg.min_area:
                continue
            center = compute_centroid(c)
            if center is None:
                continue
            x, y, w, h = cv2.boundingRect(c)
            items.append({
                "type": label,
                "center": (int(center[0]), int(center[1])),
                "area": float(area),
                "bbox": (int(x), int(y), int(w), int(h)),
                "contour": c
            })
        items.sort(key=lambda d: d["area"], reverse=True)
        return items

    def detect(self, hsv: np.ndarray) -> List[Dict]:
        red_mask = self._mask_color(hsv, "red")
        yellow_mask = self._mask_color(hsv, "yellow")
        red_items = self._extract_objects(red_mask, "red")
        yellow_items = self._extract_objects(yellow_mask, "yellow")
        return red_items + yellow_items

    @staticmethod
    def nearest_to(point: Tuple[int, int], items: List[Dict]) -> Optional[Dict]:
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
