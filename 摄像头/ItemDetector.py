
"""
ItemDetector 物体检测模块
------------------------
本模块用于检测画面中的红色和黄色物体，返回其类别、中心点、面积、边界框等信息，并可在图像上绘制检测结果。
主要类和方法：
    - ItemDetector: 主检测器类
        - detect: 检测所有红黄物体，返回物体信息列表
        - draw_results: 在画面上绘制检测结果
        - nearest_to: 计算距离指定点最近的物体
典型用法：
    detector = ItemDetector()
    items = detector.detect(hsv_img)
    frame = detector.draw_results(frame, items)
"""
import cv2
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple


@dataclass
class HSVRanges:
    """
    存储红色和黄色物体的HSV阈值范围。
    yellow_lower/yellow_upper: 黄色的HSV下界和上界
    red_lower1/red_upper1, red_lower2/red_upper2: 红色的HSV下界和上界（分两段处理）
    """
    yellow_lower: Tuple[int, int, int] = (20, 100, 100)
    yellow_upper: Tuple[int, int, int] = (30, 255, 255)
    red_lower1: Tuple[int, int, int] = (0, 120, 70)
    red_upper1: Tuple[int, int, int] = (10, 255, 255)
    red_lower2: Tuple[int, int, int] = (170, 120, 70)
    red_upper2: Tuple[int, int, int] = (180, 255, 255)


@dataclass
class ItemDetectorConfig:
    """
    ItemDetector的配置参数。
    min_area: 检测物体的最小面积阈值，过滤小噪声
    """
    min_area: int = 100

def compute_centroid(contour: np.ndarray) -> Optional[Tuple[int, int]]:
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


class ItemDetector:
    """
    ItemDetector类用于检测画面中的红色和黄色物体，返回其类别、中心点、面积、边界框等信息。
    方法：
        - detect: 检测并返回所有物体的信息
        - draw_results: 在画面上绘制检测结果
        - nearest_to: 计算距离指定点最近的物体
    """
    def __init__(self, cfg: ItemDetectorConfig = ItemDetectorConfig(), hsv_ranges: HSVRanges = HSVRanges()):
        """
        初始化ItemDetector。
        参数:
            cfg: ItemDetectorConfig对象，指定检测参数
            hsv_ranges: HSVRanges对象，指定颜色阈值
        """
        self.cfg = cfg
        self.r = hsv_ranges
        self.kernel = np.ones((5, 5), np.uint8)

    def _mask_color(self, hsv: np.ndarray, color: str) -> np.ndarray:
        """
        根据指定颜色生成掩码。
        参数:
            hsv: HSV格式的图像
            color: "red"或"yellow"
        返回:
            掩码图像
        """
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
        """
        从掩码中提取物体信息。
        参数:
            mask: 掩码图像
            label: 物体类别（"red"或"yellow"）
        返回:
            物体信息列表
        """
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
        """
        检测画面中的红色和黄色物体。
        参数:
            hsv: HSV格式的图像
        返回:
            物体信息列表
        """
        red_mask = self._mask_color(hsv, "red")
        yellow_mask = self._mask_color(hsv, "yellow")
        red_items = self._extract_objects(red_mask, "red")
        yellow_items = self._extract_objects(yellow_mask, "yellow")
        return red_items + yellow_items

    def draw_results(self, frame: np.ndarray, items: List[Dict]) -> np.ndarray:
        """
        在frame上绘制检测结果，显示类别、面积、中心点坐标、距离最近物体等信息。
        参数:
            frame: 原始画面
            items: 物体信息列表
        返回:
            绘制后的画面
        """
        for item in items:
            x, y, w, h = item["bbox"]
            cx, cy = item["center"]
            color = (0, 0, 255) if item["type"] == "red" else (0, 255, 255)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            info = f"{item['type']} Area:{int(item['area'])} Center:({cx},{cy})"
            if "dist_px" in item:
                info += f" Dist:{item['dist_px']}px"
            cv2.putText(frame, info, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.circle(frame, (cx, cy), 4, color, -1)
        # 显示总数
        cv2.putText(frame, f"Total: {len(items)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
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
    cap = cv2.VideoCapture(2)
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
                cv2.imwrite("Item.jpg", img)
                print("按任意键关闭窗口...")
                cv2.waitKey(0)
                cv2.destroyAllWindows()
        else:
            print("未输入图片路径，跳过测试。")
