"""
智能机电系统 - 全局相机检测模块（重构版）
将围栏、目标区域、物体分别封装为三个类：
- FenceDetector（围栏，蓝色，输出四边形顶点并排序）
- TargetAreaDetector（目标区，黑色，输出最小外接矩形中心与四点）
- ItemDetector（物体，红/黄，输出所有物体质心、并可结合小车位置选最近目标）

整体协调器 GlobalDetector 负责按顺序执行：小车（ArUco 另行处理）→ 围栏 → 目标区 → 物体，
并汇总为状态字典，供上位机/STM32读取。

与用户提供版本相比，本文件做了以下改动：
1) 将不同目标的检测逻辑解耦为类，便于单元测试和参数独立调优；
2) 明确每个检测器的输入/输出，统一像素级坐标系；
3) 补充顶点排序、距离度量等通用工具函数；
4) 形态学参数与阈值符合需求描述（围栏：开3x3×2 + 闭5x5×1；目标区：黑色HSV阈；物体：红/黄HSV + 面积阈100px²）。
"""

from __future__ import annotations
import cv2
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

# --------------------------- 通用工具函数 ---------------------------

def order_quad_points(pts: np.ndarray) -> np.ndarray:
    """将四边形顶点按左上→右上→右下→左下排序。
    Args:
        pts: (4,2) ndarray，float或int
    Returns:
        (4,2) ndarray，按顺序排序后的顶点
    """
    pts = np.array(pts, dtype=np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).reshape(-1)
    ordered = np.zeros((4, 2), dtype=np.float32)
    ordered[0] = pts[np.argmin(s)]  # 左上
    ordered[2] = pts[np.argmax(s)]  # 右下
    ordered[1] = pts[np.argmin(diff)]  # 右上
    ordered[3] = pts[np.argmax(diff)]  # 左下
    return ordered


def contour_min_area_rect_points(contour: np.ndarray) -> np.ndarray:
    """获取轮廓的最小外接矩形四个顶点（已排序）。"""
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    return order_quad_points(box)


def compute_centroid(contour: np.ndarray) -> Optional[Tuple[int, int]]:
    M = cv2.moments(contour)
    if M["m00"] == 0:
        return None
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    return (cx, cy)


# --------------------------- 阈值配置 ---------------------------

@dataclass
class HSVRanges:
    yellow_lower: Tuple[int, int, int] = (20, 100, 100)
    yellow_upper: Tuple[int, int, int] = (30, 255, 255)
    red_lower1: Tuple[int, int, int] = (0, 120, 70)
    red_upper1: Tuple[int, int, int] = (10, 255, 255)
    red_lower2: Tuple[int, int, int] = (170, 120, 70)
    red_upper2: Tuple[int, int, int] = (180, 255, 255)
    black_lower: Tuple[int, int, int] = (0, 0, 0)
    black_upper: Tuple[int, int, int] = (180, 255, 50)
    blue_lower: Tuple[int, int, int] = (100, 50, 50)
    blue_upper: Tuple[int, int, int] = (130, 255, 255)


# --------------------------- 检测器：围栏 ---------------------------

class FenceDetector:
    """蓝色围栏检测：输出可通行区域四边形顶点（左上→右上→右下→左下）。"""

    def __init__(self, hsv_ranges: HSVRanges = HSVRanges()):
        self.r = hsv_ranges
        # 固定的形态学核
        self.kernel_open = np.ones((3, 3), np.uint8)
        self.kernel_close = np.ones((5, 5), np.uint8)

    def detect(self, hsv: np.ndarray) -> Optional[np.ndarray]:
        """检测围栏，返回四边形顶点 (4,2) 的 ndarray（int）。"""
        mask = cv2.inRange(hsv, np.array(self.r.blue_lower), np.array(self.r.blue_upper))
        # 开运算 3x3 两次
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel_open, iterations=2)
        # 闭运算 5x5 一次
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel_close, iterations=1)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        cnt = max(contours, key=cv2.contourArea)
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

        if len(approx) < 4:
            # 退化：使用最小外接矩形
            quad = contour_min_area_rect_points(cnt)
        else:
            # 若>=4，选面积最大的四边形近似，再排序
            if len(approx) > 4:
                # 取凸包再最小外接矩形，保证四点输出
                hull = cv2.convexHull(approx)
                quad = contour_min_area_rect_points(hull)
            else:
                quad = order_quad_points(approx.reshape(-1, 2))
        return quad.astype(int)


# --------------------------- 检测器：目标区域 ---------------------------

class TargetAreaDetector:
    """黑色目标区域检测：输出中心坐标与最小外接矩形四点。"""

    def __init__(self, hsv_ranges: HSVRanges = HSVRanges()):
        self.r = hsv_ranges
        self.kernel_open = np.ones((3, 3), np.uint8)
        self.kernel_close = np.ones((5, 5), np.uint8)

    def detect(self, hsv: np.ndarray) -> Optional[Dict[str, np.ndarray]]:
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
            # 使用矩形中心作为兜底
            center = tuple(np.mean(box, axis=0).astype(int))
        return {"center": np.array(center, dtype=int), "box": box}


# --------------------------- 检测器：物体 ---------------------------

@dataclass
class ItemDetectorConfig:
    min_area: int = 100  # 面积阈值（px^2）


class ItemDetector:
    """红/黄物体检测：返回所有物体质心；可计算相对某点最近者。"""

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
        # 大到小排序
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
            best = dict(best)  # 拷贝并附加距离
            best["dist_px"] = int(np.sqrt(best_d2))
        return best


# --------------------------- 协调器：全局相机 ---------------------------

class GlobalDetector:
    """按顺序执行围栏→目标区→物体的检测，并输出统一状态字典。
    小车的 (u_car, v_car, theta) 由 ArUco 模块外部提供后可传入 compute_state()。
    """

    def __init__(self):
        self.fence = FenceDetector()
        self.target = TargetAreaDetector()
        self.items = ItemDetector()

    @staticmethod
    def _to_hsv(frame_bgr: np.ndarray) -> np.ndarray:
        blurred = cv2.GaussianBlur(frame_bgr, (5, 5), 0)
        return cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    def compute_state(
        self,
        frame_bgr: np.ndarray,
        car_pose: Optional[Tuple[int, int, float]] = None,
    ) -> Dict:
        """运行一帧检测并返回状态字典。
        Args:
            frame_bgr: 顶视 BGR 帧
            car_pose: 可选 (u_car, v_car, theta)
        Returns:
            dict:
              {
                'fence_quad': np.ndarray(4,2) or None,
                'target_center_uv': (u, v) or None,
                'target_box': np.ndarray(4,2) or None,
                'objects': List[Dict],
                'nearest_obj': Dict or None,
                'car_center_uvθ': (u,v,θ) or None
              }
        """
        hsv = self._to_hsv(frame_bgr)

        # 1) 围栏
        fence_quad = self.fence.detect(hsv)

        # 2) 目标区域
        target_info = self.target.detect(hsv)
        target_center = tuple(target_info["center"]) if target_info is not None else None
        target_box = target_info["box"] if target_info is not None else None

        # 3) 物体
        objects = self.items.detect(hsv)

        # 最近物体（若有车位姿）
        nearest = None
        if car_pose is not None:
            u_car, v_car, _ = car_pose
            nearest = self.items.nearest_to((u_car, v_car), objects)

        state = {
            "fence_quad": fence_quad,
            "target_center_uv": target_center,
            "target_box": target_box,
            "objects": objects,
            "nearest_obj": nearest,
            "car_center_uvθ": car_pose,
        }
        return state

    # ------------------- 可视化辅助 -------------------
    @staticmethod
    def draw_state(frame: np.ndarray, state: Dict) -> np.ndarray:
        img = frame.copy()
        # 围栏四点
        quad = state.get("fence_quad")
        if quad is not None and len(quad) == 4:
            quad = quad.reshape(-1, 2)
            for i in range(4):
                p1 = tuple(quad[i].astype(int))
                p2 = tuple(quad[(i + 1) % 4].astype(int))
                cv2.line(img, p1, p2, (255, 0, 0), 3)
                cv2.circle(img, p1, 5, (255, 0, 0), -1)

        # 目标区
        if state.get("target_box") is not None:
            box = state["target_box"].reshape(-1, 2).astype(int)
            for i in range(4):
                p1 = tuple(box[i])
                p2 = tuple(box[(i + 1) % 4])
                cv2.line(img, p1, p2, (255, 255, 255), 2)
        if state.get("target_center_uv") is not None:
            cv2.circle(img, tuple(state["target_center_uv"]), 6, (255, 255, 255), -1)
            cv2.putText(img, "TARGET", (state["target_center_uv"][0] + 8, state["target_center_uv"][1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # 物体
        for obj in state.get("objects", []):
            x, y, w, h = obj["bbox"]
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255) if obj["type"] == "yellow" else (0, 0, 255), 2)
            cx, cy = obj["center"]
            cv2.circle(img, (cx, cy), 4, (0, 255, 0), -1)
            cv2.putText(img, f"{obj['type']} ({cx},{cy})", (x, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # 最近物体
        if state.get("nearest_obj") is not None:
            cx, cy = state["nearest_obj"]["center"]
            cv2.circle(img, (cx, cy), 10, (0, 165, 255), 3)
            cv2.putText(img, "NEAREST", (cx + 10, cy + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)

        # 小车位姿（若提供）
        if state.get("car_center_uvθ") is not None:
            u, v, theta = state["car_center_uvθ"]
            cv2.circle(img, (int(u), int(v)), 6, (0, 255, 0), -1)
            L = 40
            end_x = int(u + L * np.cos(np.deg2rad(theta)))
            end_y = int(v + L * np.sin(np.deg2rad(theta)))
            cv2.arrowedLine(img, (int(u), int(v)), (end_x, end_y), (0, 255, 0), 2)
            cv2.putText(img, f"theta={theta:.1f}deg", (int(u) + 8, int(v) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        return img


# --------------------------- 便捷测试 ---------------------------
if __name__ == "__main__":
    print("全局相机检测模块（重构版）加载成功。按 'q' 退出，'s' 保存一帧。")
    det = GlobalDetector()
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise SystemExit("无法打开摄像头")

    # 示例：若需要结合 ArUco 结果，请在外部调用 ArucoDetector 计算 car_pose 后传入 compute_state()
    car_pose = None  # 例如 (u_car, v_car, theta)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        state = det.compute_state(frame, car_pose=car_pose)
        vis = det.draw_state(frame, state)

        # HUD 文本
        hud = [
            f"Fence: {'OK' if state['fence_quad'] is not None else 'None'}",
            f"Target: {state['target_center_uv']}",
            f"Objects: {len(state['objects'])}",
            f"Nearest: {state['nearest_obj']['center'] if state['nearest_obj'] else None}",
        ]
        for i, t in enumerate(hud):
            cv2.putText(vis, t, (10, 28 + i * 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.imshow("Global Camera Detection (Refactored)", vis)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            cv2.imwrite("global_det_frame.jpg", vis)
            print("已保存 global_det_frame.jpg")

    cap.release()
    cv2.destroyAllWindows()
