"""
智能小车视觉系统主程序
---------------------
本模块是视觉系统的核心集成模块，实现了以下功能：

1. 系统集成
    - 整合所有检测器（ArUco、围栏、家、物体）
    - 统一的图像处理流程
    - 集中的结果管理和通信接口

2. 实时检测功能
    - ArUco标记检测：确定车体位置和朝向（提供全局坐标系）
    - 围栏检测：识别蓝色围栏区域（场地边界）
    - 家区域检测：识别黑色返回区域（任务终点）
    - 物体检测：识别红色和黄色目标物体（任务目标）

3. 导航辅助
    - 计算车体与目标物体的相对位置（角度和距离）
    - 自动选择最近目标物体
    - 提供到家区域的导航信息

4. 可视化与调试
    - 实时显示检测结果和导航引导线
    - 支持图片测试模式
    - 提供详细的状态信息和性能指标

5. 通信接口
    - 与STM32控制器实时通信
    - 发送结构化数据包
    - 支持错误处理和异常恢复

坐标系说明：
    - 图像坐标系：原点在左上角，X轴向右，Y轴向下
    - 角度定义：0°指向右侧，逆时针增加为正方向
    - 相对角度：以车头朝向为0°，右侧为正，左侧为负，范围[-180°, 180°]

使用方法：
    1. 摄像头模式：python main.py
    2. 图片测试：python main.py -i <图片路径>
    3. 指定摄像头：python main.py -c <摄像头ID>
    4. 调试模式：python main.py --debug
"""

import cv2
import time
import numpy as np
from ArucoDetector import ArucoDetector
from FenceDetector import FenceDetector
from HomeDetector import HomeDetector
from ItemDetector import ItemDetector
from camostudio_comm import open_serial, close_serial, send_camostudio_data, SerialCommError
from typing import Dict, List, Optional, Tuple

cnt = 0

def extract_vision_data(vision_results: Dict) -> Dict[str, int]:
    """
    从视觉检测结果中提取需要传输的关键数据，并转换为整数格式
    
    数据处理过程:
    1. 从检测结果字典中提取各类信息
    2. 转换为适合通信的整数格式
    3. 对无效数据设置特定值
    
    主要数据项:
    1. 检测到的物体数量 (1个整数: SEARCH_OBJ_NUM)
        - 范围：[0, N]，表示当前视野中识别到的目标物体数量
    
    2. 相对最近物体的方位角 (1个整数: item_angle)
        - 乘以10以保留一位小数精度
        - 范围：[-1800, 1800]，对应[-180.0°, 180.0°]
        - 相对车头方向的角度：
          * 0° = 正前方
          * 90° = 右侧
          * -90° = 左侧
          * ±180° = 后方
        - 无效值：0（表示没有有效角度数据）
    
    3. 到最近物体的距离 (1个整数: item_distance)
        - 单位：像素
        - 从车体中心（ArUco标记中心）到目标物体中心的直线距离
        - 无效值：0（表示没有有效距离数据）
        
    4. 到家的相对方位角 (1个整数: home_angle)
        - 乘以10以保留一位小数精度
        - 范围：[-1800, 1800]，对应[-180.0°, 180.0°]
        - 无效值：0（表示没有有效角度数据）
    
    5. 到家的距离 (1个整数: home_distance)
        - 单位：像素
        - 从车体中心（ArUco标记中心）到家中心的直线距离
        - 无效值：0（表示没有有效距离数据）
        
    6. 物体是否超出内收矩形区域 (1个整数: item_out_of_bounds)
        - 0 = 未超出区域
        - 1 = 超出区域
        - 当物体超出内收矩形区域时，视觉系统会将目标点调整到内收矩形边界上
        - 无效值：0（表示没有有效的物体或物体在内收矩形内）
    
    参数:
        vision_results: 包含所有检测结果的字典，包括：
            - items: 检测到的物体列表
            - navigation: 导航信息（车辆位置和目标相对位置）
    
    返回:
        Dict[str, int]: 包含处理后整数数据的字典：
        {
            'SEARCH_OBJ_NUM': int,      # 当前检测到的物体数量
            'fence_x1': int, ...,       # 围栏四个顶点的x坐标
            'fence_y1': int, ...,       # 围栏四个顶点的y坐标
            'home_x': int, 'home_y': int, # 家区域中心点坐标
            'item_angle': int,          # 到物体的相对角度*10，范围[-1800,1800]
            'item_distance': int,       # 到物体的像素距离，无效时=0
            'home_angle': int,          # 到家的相对角度*10，范围[-1800,1800]
            'home_distance': int,       # 到家的像素距离，无效时=0
            'item_out_of_bounds': int   # 物体是否超出内收矩形区域，0=否，1=是
        }
        
        注意：当处理失败或数据无效时，会返回所有值为0的字典，而不是返回None
    """
    try:
        result = {
            'SEARCH_OBJ_NUM': 0,
            'fence_x1': 0, 'fence_y1': 0,
            'fence_x2': 0, 'fence_y2': 0,
            'fence_x3': 0, 'fence_y3': 0,
            'fence_x4': 0, 'fence_y4': 0,
            'home_x': 0, 'home_y': 0,
            'item_angle': 0,
            'item_distance': 0,
            'home_angle': 0,
            'home_distance': 0,
            'item_out_of_bounds': 0  # 表示最近物体是否超出内收矩形区域，0=否，1=是
        }
        
        # 1. 提取围栏顶点（按顺时针顺序）
        if vision_results['fence'] and vision_results['fence'].quad is not None:
            quad = vision_results['fence'].quad
            # quad是一个4x2的numpy数组，每行是一个顶点的[x,y]坐标
            for i in range(4):
                x, y = quad[i]
                result[f'fence_x{i+1}'] = int(round(x))
                result[f'fence_y{i+1}'] = int(round(y))
        else:
            print("警告: 未检测到有效围栏")
            
        # 2. 提取目标区域（家）中心点
        if vision_results['home'] and vision_results['home'].Self_Center:
            hx, hy = vision_results['home'].Self_Center
            result['home_x'] = int(round(hx))
            result['home_y'] = int(round(hy))
        else:
            print("警告: 未检测到有效家区域")

        result['SEARCH_OBJ_NUM'] = len(vision_results['items'])

        # 3. 提取到最近物体的的相对位置
        nav_info = vision_results['navigation']
        if nav_info['item_relative_angle'] is not None and nav_info['item_distance'] is not None:
            # 相对角度（乘以10以保留一位小数）
            angle = nav_info['item_relative_angle']
            result['item_angle'] = int(round(angle * 10))
            
            # 距离（像素）
            result['item_distance'] = int(round(nav_info['item_distance']))
            
            # 设置是否超出内收矩形区域的标志
            if nav_info['nearest_item'] and 'out_of_bounds' in nav_info['nearest_item']:
                result['item_out_of_bounds'] = 1 if nav_info['nearest_item']['out_of_bounds'] else 0
        else:
            # 没有有效的导航信息，将无效值设为0
            result['item_angle'] = 0  # 无效角度设为0
            result['item_distance'] = 0  # 无效距离设为0
            result['item_out_of_bounds'] = 0  # 无效物体超界标志为0
            
        # 4. 提取到家的相对位置信息
        if nav_info['home_relative_angle'] is not None and nav_info['home_distance'] is not None:
            # 相对角度（乘以10以保留一位小数）
            home_angle = nav_info['home_relative_angle']
            result['home_angle'] = int(round(home_angle * 10))
            
            # 距离（像素）
            result['home_distance'] = int(round(nav_info['home_distance']))
        else:
            # 没有有效的导航信息，将无效值设为0
            result['home_angle'] = 0  # 无效角度设为0
            result['home_distance'] = 0  # 无效距离设为0
            
        return result
        
    except Exception as e:
        print(f"数据提取错误: {str(e)}")
        # 出现异常时返回所有值为0的字典，确保通信不会因为None而出错
        return {
            'SEARCH_OBJ_NUM': 0,
            'fence_x1': 0, 'fence_y1': 0,
            'fence_x2': 0, 'fence_y2': 0,
            'fence_x3': 0, 'fence_y3': 0,
            'fence_x4': 0, 'fence_y4': 0,
            'home_x': 0, 'home_y': 0,
            'item_angle': 0,
            'item_distance': 0,
            'home_angle': 0,
            'home_distance': 0,
            'item_out_of_bounds': 0
        }


class VisionSystem:
    """
    视觉系统集成类
    
    该类整合了所有检测器，实现了完整的视觉处理流程，包括：
    1. 图像获取和预处理
    2. 多目标检测和识别
    3. 结果分析和位置计算
    4. 导航数据生成
    5. 通信数据封装
    6. 可视化显示
    
    主要组件：
        - ArUco检测器：识别定位标记，提供全局位置和朝向
        - 围栏检测器：识别蓝色场地边界，提供环境范围
        - 家检测器：识别黑色返回区域，提供导航终点
        - 物体检测器：识别红色和黄色目标物体，提供任务目标
        
    数据流程：
        1. 获取图像 → 2. 运行检测器 → 3. 整合结果
        4. 计算导航信息 → 5. 封装通信数据 → 6. 发送至控制器
        
    坐标系说明：
        - 图像坐标系：原点在左上角，X轴向右，Y轴向下
        - 角度定义：0°指向右侧，逆时针为正
        - 相对角度：以车头方向为基准，右侧为正，左侧为负
    """
    
    def __init__(self):
        """
        初始化视觉系统
        
        完成以下初始化工作：
        1. 创建各个检测器实例
           - ArUco检测器：识别标记ID和位置
           - 围栏检测器：基于HSV颜色空间识别蓝色区域
           - 家检测器：基于HSV颜色空间识别黑色区域
           - 物体检测器：基于HSV颜色空间识别红色和黄色物体
           
        2. 初始化结果存储结构
           - 使用嵌套字典统一管理检测结果
           - 包含原始检测数据和计算的导航信息
           - 设置所有值的默认状态为None或空列表
           
        3. 设置性能计数器
           - 帧率计算器
           - 执行时间统计
        """
        self.aruco_detector = ArucoDetector()
        self.fence_detector = FenceDetector()
        self.home_detector = HomeDetector()
        self.item_detector = ItemDetector()
        
        # 用于存储最新的检测结果
        self.results = {
            # ArUco标记信息列表，每个标记包含：
            # - id: int - 标记ID
            # - center: Tuple[float, float] - 标记中心坐标
            # - corners: np.ndarray - 标记四个角点坐标
            # - angle: float - 标记旋转角度（度数）
            # - valid: bool - 标记是否有效
            'aruco': [],      
            
            # 围栏区域信息
            # - center: Tuple[float, float] - 围栏中心坐标
            # - area: float - 围栏面积（像素）
            # - quad: np.ndarray - 围栏四个顶点坐标，顺时针排列
            # - valid: b检测结果是否有效ool - 
            'fence': None,    
            
            # 目标区域（家）信息
            # - center: Tuple[float, float] - 区域中心坐标
            # - area: float - 区域面积（像素）
            # - valid: bool - 检测结果是否有效
            'home': None,     
            
            # 检测到的物体列表，每个物体包含：
            # - type: str - 物体类型（'red' 或 'yellow'）
            # - center: Tuple[float, float] - 物体中心坐标
            # - area: float - 物体面积（像素）
            # - aspect_ratio: float - 宽高比
            # - valid: bool - 物体是否有效
            # - out_of_bounds: bool - 物体是否超出内收矩形范围
            # - adjusted_center: Tuple[float, float] - 调整后的中心坐标（仅当out_of_bounds=True时）
            'items': [],      
            
            # 导航相关信息
            'navigation': {   
                # 车体位置（从ArUco标记获得）
                'car_pos': None,          # Tuple[float, float] - 像素坐标 (u, v)
                'car_angle': None,        # float - 车头朝向角度，范围[-180, 180]
                
                # 最近目标物体信息
                'nearest_item': None,     # Dict - 参考items中的物体格式
                                          # 增加了distance_px、out_of_bounds和adjusted_center字段
                
                # 相对位置信息（相对于车头方向）
                'item_relative_angle': None,   # float - 相对方位角（度数），范围[-180, 180]
                                        # 正值表示目标在车头右侧，负值表示在左侧
                'item_distance': None,         # float - 相对距离（像素）
                
                # 到家的相对位置信息
                'home_relative_angle': None,  # float - 到家的相对方位角（度数），范围[-180, 180]
                'home_distance': None         # float - 到家的相对距离（像素）
            }
        }
        
        # 性能指标
        self.fps = 0.0  # float - 当前帧率

    def find_nearest_item(self):
        """
        寻找距离小车最近的有效物体，并更新导航信息
        
        算法流程：
        1. 检查必要条件：
           - 车辆位置信息存在（通过ArUco标记检测获得）
           - 检测到的物体列表不为空
           - 检测到有效的围栏内收矩形
           
        2. 计算每个物体到车辆的距离：
           - 仅考虑有效的物体（valid=True）
           - 如果物体超出围栏内收矩形范围，则将坐标缩小到矩形边界上
           - 计算欧氏距离：sqrt((x_car - x_item)^2 + (y_car - y_item)^2)
           - 对每个物体添加distance_px字段存储距离值
           - 对超出内收矩形的物体添加out_of_bounds标识
           
        3. 选择最近的物体：
           - 按距离升序排序
           - 取第一个（最近的）物体
           - 如果没有有效物体，返回None
           
        结果更新：
            更新self.results['navigation']中的信息：
            - nearest_item: 最近物体的完整信息，包括：
              * type: str - 物体类型（'red'/'yellow'）
              * center: Tuple[float, float] - 中心坐标
              * area: float - 面积
              * distance_px: float - 到车体的距离
              * valid: bool - 是否有效
              * out_of_bounds: bool - 是否超出内收矩形范围
              * adjusted_center: Tuple[float, float] - 调整后的中心坐标（如果在边界外）
              如果没有找到有效物体，设为None
        
        注意事项：
            - 距离计算使用像素坐标系
            - 优先级不考虑物体类型（红色/黄色）
            - 对超出内收矩形范围的物体进行特殊处理，以便避障
        """
        nav_info = self.results['navigation']
        
        # 验证必要条件
        if not nav_info['car_pos'] or not self.results['items']:
            nav_info['nearest_item'] = None
            return
            
        # 提取车辆位置
        u_car, v_car = nav_info['car_pos']
        
        # 提取围栏内收矩形（如果存在）
        inner_rect = None
        if self.results['fence'] and self.results['fence'].valid:
            inner_rect = self.results['fence'].inner_rect
        
        # 计算每个有效物体的距离
        valid_items = []
        for item in self.results['items']:
            if not item.get('valid', True):
                continue

                
            u_item, v_item = item['center']

            # 计算距离（使用调整后的坐标）
            dist = np.sqrt((u_item - u_car)**2 + (v_item - v_car)**2)

            out_of_bounds = False
            adjusted_center = (u_item, v_item)  # 默认不调整
            

            # 检查物体是否超出内收矩形范围，如果是则调整坐标
            if inner_rect :
                x1, y1, x2, y2 = inner_rect  # 内收矩形的左上角和右下角坐标
                
                extra_dist = 100
                dist_lim = 100
                # 调整x坐标（如果超出范围）
                if u_item < x1:
                    u_item = x1 + extra_dist
                    out_of_bounds = True
                elif u_item > x2:
                    u_item = x2 - extra_dist
                    out_of_bounds = True
                    
                # 调整y坐标（如果超出范围）
                if v_item < y1:
                    v_item = y1 + extra_dist
                    out_of_bounds = True
                elif v_item > y2:
                    v_item = y2 - extra_dist
                    out_of_bounds = True
                    
                # 如果坐标被调整，保存调整后的坐标
                if out_of_bounds:
                    adjusted_center = (u_item, v_item)
                    # 重新计算距离
                    dist = np.sqrt((u_item - u_car)**2 + (v_item - v_car)**2)

                if dist < dist_lim:
                    out_of_bounds = False  # 靠近车体时不认为超出边界

                
            
            
            valid_items.append({
                **item,
                'distance_px': dist,  # 统一使用distance_px作为距离字段
                'out_of_bounds': out_of_bounds,  # 标记是否超出内收矩形
                'adjusted_center': adjusted_center  # 保存调整后的坐标
            })
            
        # 按距离排序并选择最近的
        if valid_items:
            valid_items.sort(key=lambda x: x['distance_px'])
            if len(valid_items) > 2 and abs( (valid_items[0]['distance_px'] - valid_items[1]['distance_px']) )< 20:
                valid_items = valid_items[:2]
                valid_items.sort(key=lambda x: x['center'][0])
            nav_info['nearest_item'] = valid_items[0]
        else:
            nav_info['nearest_item'] = None

    def compute_item_relative_position(self):
        """
        计算小车到最近物体的相对方位信息
        
        计算原理：
        1. 位移向量计算
           - 计算物体相对车体的位移向量(Δu, Δv)
           - Δu = u_item - u_car（水平位移）
           - Δv = v_item - v_car（垂直位移）
           - 如果物体超出内收矩形范围，使用调整后的坐标
        
        2. 方位角计算
           - 计算物体相对车体的绝对方位角α = arctan2(Δv, Δu)
           - 计算相对方位角β = α - θ（θ为车头朝向）
           - 结果归一化到[-180°, 180°]范围内
           - 正值表示物体在车头右侧，负值表示在左侧
        
        3. 距离计算
           - 计算车辆到物体的直线距离d = sqrt(Δu² + Δv²)
           - 单位为像素
           
        应用场景：
        - 导航决策：确定转向方向和角度
        - 路径规划：避障和目标接近
        - 通信发送：提供给STM32控制器的导航数据
            
        返回:
            Dict: 包含相对方位信息的字典，无法计算时返回None
                - relative_angle_deg: float, 相对方位角（度）
                - distance_px: float, 像素距离
                - delta_uv: Tuple[float, float], 位移向量
        """
        nav_info = self.results['navigation']
        
        # 验证必要条件
        if (not nav_info['car_pos'] or 
            nav_info['car_angle'] is None or 
            not nav_info['nearest_item'] or 
            'center' not in nav_info['nearest_item']):
            return None
            
        # 提取坐标
        u_car, v_car = nav_info['car_pos']
        
        # 获取物体坐标 - 如果物体超出内收矩形，使用调整后的坐标
        nearest_item = nav_info['nearest_item']
        if nearest_item.get('out_of_bounds', False) and 'adjusted_center' in nearest_item:
            u_item, v_item = nearest_item['adjusted_center']
        else:
            u_item, v_item = nearest_item['center']
        
        # 计算位移向量
        delta_u = float(u_item - u_car)
        delta_v = float(v_item - v_car)
        
        # 计算绝对方位角（度数）
        angle = np.degrees(np.arctan2(delta_v, delta_u))
        
        # 计算相对方位角（度数）
        relative_angle = angle - nav_info['car_angle']
        # 归一化到 [-180, 180]
        relative_angle = (relative_angle + 180) % 360 - 180
        
        # 计算距离
        distance = np.hypot(delta_u, delta_v)
        
        # 返回计算结果
        return {
            'relative_angle_deg': float(relative_angle),
            'distance_px': float(distance),
            'delta_uv': (delta_u, delta_v)
        }

    def compute_home_relative_position(self):
        """
        计算小车到家的相对方位信息
        
        计算原理：
        1. 位移向量计算
           - 计算家相对车体的位移向量(Δu, Δv)
           - Δu = u_home - u_car（水平位移）
           - Δv = v_home - v_car（垂直位移）
        
        2. 方位角计算
           - 计算家相对车体的绝对方位角α = arctan2(Δv, Δu)
           - 计算相对方位角β = α - θ（θ为车头朝向）
           - 结果归一化到[-180°, 180°]范围内
           - 正值表示家在车头右侧，负值表示在左侧
        
        3. 距离计算
           - 计算车辆到家的直线距离d = sqrt(Δu² + Δv²)
           - 单位为像素
           
        应用场景：
        - 任务完成后返回家的导航
        - 定位状态评估
        - 通信发送：提供给STM32控制器的导航数据
            
        返回:
            Dict: 包含相对方位信息的字典，无法计算时返回None
                - relative_angle_deg: float, 相对方位角（度）
                - distance_px: float, 像素距离
                - delta_uv: Tuple[float, float], 位移向量
        """
        global cnt
        nav_info = self.results['navigation']
        
        # 验证必要条件
        # if (not nav_info['car_pos'] or 
        #     nav_info['car_angle'] is None or 
        #     not self.results['home'] or 
        #     not self.results['home'].Self_Center):
        #     return None
            
        # 提取坐标
        u_car, v_car = nav_info['car_pos']
        u_home, v_home = self.results['home'].Self_Center
        # if cnt == 0:
        #     u_last_home, v_last_home = u_home, v_home
        #     cnt = 1
        # if cnt == 1:
        #     u_home, v_home = u_last_home, v_last_home
        # u_home, v_home = 300, 650

        # 计算位移向量
        delta_u = float(u_home - u_car)
        delta_v = float(v_home - v_car)
        
        # 计算绝对方位角（度数）
        angle = np.degrees(np.arctan2(delta_v, delta_u))
        
        # 计算相对方位角（度数）
        relative_angle = angle - nav_info['car_angle']
        # 归一化到 [-180, 180]
        relative_angle = (relative_angle + 180) % 360 - 180
        
        # 计算距离
        distance = np.hypot(delta_u, delta_v)
        
        # 返回计算结果
        return {
            'relative_angle_deg': float(relative_angle),
            'distance_px': float(distance),
            'delta_uv': (delta_u, delta_v)
        }

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        处理单帧图像并运行所有检测器
        
        处理流程：
        1. 图像预处理
           - 检查图像有效性
           - 转换颜色空间：BGR -> HSV（用于颜色检测）
        
        2. 目标检测（按优先级顺序）：
           a) ArUco标记检测
              - 确定车辆位置和朝向
              - 提供全局定位基准
              - 角度定义：0°指向右侧，逆时针为正
           
           b) 围栏检测
              - 识别蓝色围栏区域
              - 提取四边形轮廓和顶点
              - 顶点按顺时针顺序排列
           
           c) 家区域检测
              - 识别黑色返回区域
              - 计算中心点和面积
           
           d) 物体检测
              - 识别红色和黄色目标物体
              - 计算位置、面积和形状特征
              - 过滤无效检测结果
        
        3. 导航计算
           a) 车体状态更新
              - 从ArUco检测更新位置
              - 计算车头朝向角度（度数）
           
           b) 目标分析
              - 查找最近的有效物体
              - 计算相对方位角（[-180°, 180°]）
              - 计算直线距离（像素）
              
           c) 家位置分析
              - 计算车辆到家的相对方位角
              - 计算车辆到家的直线距离
           
           d) 数据提取
              - 处理所有结果为整数格式
              - 准备用于传输的数据包
        
        4. 可视化
           - 绘制检测结果和位置标记
           - 显示导航引导线
           - 添加状态信息和帧率
        
        参数:
            frame: BGR格式的输入图像帧
                - shape: (height, width, 3)
                - dtype: uint8
                - 范围: [0, 255]
            
        返回:
            np.ndarray: 处理后的图像帧
                - 包含可视化的检测和导航结果
                - 与输入图像相同的尺寸和格式
        
        注意事项:
            - 处理顺序很重要，ArUco检测必须先于其他检测
            - 导航计算依赖于检测结果的有效性
            - 可视化部分不影响结果计算，仅用于调试
        """
        if frame is None:
            return None
            
        # 转换HSV色彩空间(用于围栏、家和物体检测)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # 1. ArUco标记检测
        aruco_result = self.aruco_detector.detect_all(frame)
        self.results['aruco'] = aruco_result
        
        # 2. 围栏检测
        fence_result = self.fence_detector.detect(hsv)
        self.results['fence'] = fence_result
        
        # 3. 家区域检测
        home_result = self.home_detector.detect(hsv)
        self.results['home'] = home_result
        
        # 4. 物体检测 - 现在返回ItemInfo字典列表
        items_result = self.item_detector.detect(hsv)
        self.results['items'] = [item for item in items_result if item.get('valid', True)]
        
        # 更新车体位置和朝向（从ArUco检测结果）
        nav_info = self.results['navigation']
        if aruco_result and aruco_result[0].get('valid', True):
            # 取第一个有效的ArUco标记作为车体位置
            marker_data = aruco_result[0]
            nav_info['car_pos'] = marker_data['center']
            nav_info['car_angle'] = marker_data['angle']  # 保存角度值
        else:
            nav_info['car_pos'] = None
            nav_info['car_angle'] = None
            
        # 查找最近的有效物体
        self.find_nearest_item()
        
        # 如果找到有效物体，计算相对位置
        if nav_info['nearest_item'] is not None:
            # 计算相对位置并更新导航信息
            pos_info = self.compute_item_relative_position()
            if pos_info:
                nav_info['item_relative_angle'] = pos_info['relative_angle_deg']
                nav_info['item_distance'] = pos_info['distance_px']
            else:
                nav_info['item_relative_angle'] = None
                nav_info['item_distance'] = None
        
        # 计算到家的相对位置
        home_pos_info = self.compute_home_relative_position()
        if home_pos_info:
            nav_info['home_relative_angle'] = home_pos_info['relative_angle_deg']
            nav_info['home_distance'] = home_pos_info['distance_px']
        else:
            nav_info['home_relative_angle'] = None
            nav_info['home_distance'] = None
        
        # 在图像上绘制所有检测结果
        output = self.draw_all_results(frame.copy())
        
        # 提取要传输的数据，包含以下信息:
        # - 围栏四个顶点坐标 (fence_x1~x4, fence_y1~y4)
        # - 目标区域中心点 (home_x, home_y)
        # - 到最近物体的方位角 (item_angle, 乘以10保留一位小数)
        # - 到最近物体的距离 (item_distance, 像素单位)
        # - 到家的方位角 (home_angle, 乘以10保留一位小数)
        # - 到家的距离 (home_distance, 像素单位)
        # 所有数据都转换为整数格式，便于通信传输
        self.transmission_data = extract_vision_data(self.results)
        
        return output
        
    def draw_all_results(self, frame: np.ndarray) -> np.ndarray:
        """
        在图像上绘制所有检测结果和系统状态
        
        绘制内容：
        1. 检测结果标记
           a) 围栏区域
              - 蓝色轮廓线
              - 顶点标记
              - 中心点十字标记
           
           b) 家（目标）区域
              - 绿色轮廓线
              - 中心点标记
              - 面积数值
           
           c) 目标物体
              - 红/黄色边界框
              - 类型标签
              - 中心点标记
           
           d) ArUco标记
              - ID标签
              - 方向指示箭头
              - 边框标记
        
        2. 导航信息显示
           a) 物体位置引导
              - 车辆到目标的绿色连接线
              - 方向指示箭头
              - 状态信息（方位角、距离、物体类型等）
              
           b) 家位置引导
              - 车辆到家的蓝色连接线
              - 状态信息（方位角、距离）
           
           c) 系统信息
              - 帧率计数器（FPS）
              - 检测统计信息
        
        参数说明：
            frame: BGR格式的原始图像帧
                - 要求未经任何绘制的原始帧
                - 会创建副本进行绘制
        
        返回值：
            numpy.ndarray: 绘制完成的图像帧
                - 包含所有可视化元素
                - 保持原始图像尺寸不变
        
        注意事项：
        1. 文本显示：
           - 统一使用英文避免编码问题
           - 字体：FONT_HERSHEY_SIMPLEX
           - 缩放：0.7
           - 粗细：2
        
        2. 颜色方案：
           - 正常状态：绿色 (0, 255, 0)
           - 警告状态：黄色 (0, 255, 255)
           - 错误状态：红色 (0, 0, 255)
           - 辅助信息：灰色 (128, 128, 128)
        
        3. 布局规则：
           - 文本信息从左上角开始排列
           - 行间距：30像素
           - 边距：10像素
           - FPS显示固定在右上角
        
        4. 性能考虑：
           - 仅在有效数据时绘制
           - 使用整数坐标避免抗锯齿
           - 优化文本位置计算
        """
        h, w = frame.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX
        margin = 10
        line_height = 30
        current_y = margin + line_height
        
        # 1. 绘制围栏
        if self.results['fence'] is not None:
            frame = self.fence_detector.draw_results(frame, self.results['fence'])
            
        # 2. 绘制家区域
        if self.results['home'] is not None:
            frame = self.home_detector.draw_results(frame, self.results['home'])
            
        # 3. 绘制物体
        if self.results['items']:
            frame = self.item_detector.draw_results(frame, self.results['items'])
            
        # 4. 绘制ArUco标记
        if self.results['aruco']:
            frame = self.aruco_detector.draw_detections(frame, self.results['aruco'])
            
        # 5. 添加导航信息（最近物体）
        nav_info = self.results['navigation']
        if nav_info['car_pos'] and nav_info['nearest_item']:
            nearest_item = nav_info['nearest_item']
            
            # 绘制导航线（从车到物体实际中心）
            cv2.line(frame, 
                    tuple(map(int, nav_info['car_pos'])), 
                    tuple(map(int, nearest_item['center'])), 
                    (0, 255, 0), 2)
            
            # 如果物体超出边界，绘制调整后的位置和导航线
            if nearest_item.get('out_of_bounds', False) and 'adjusted_center' in nearest_item:
                # 绘制调整后的中心点（用橙色圆圈标识）
                adjusted_center = tuple(map(int, nearest_item['adjusted_center']))
                cv2.circle(frame, adjusted_center, 8, (0, 165, 255), 2)  # 橙色圆圈
                
                # 从物体实际中心到调整后中心点的虚线
                actual_center = tuple(map(int, nearest_item['center']))
                cv2.line(frame, actual_center, adjusted_center, 
                        (0, 165, 255), 2, cv2.LINE_AA)
                
                # 绘制从车到调整后位置的导航线（虚线）
                pts = np.array([nav_info['car_pos'], adjusted_center], np.int32).reshape((-1, 1, 2))
                cv2.polylines(frame, [pts], False, (0, 165, 255), 2, cv2.LINE_AA)

            # 添加方位信息
            info_lines = []
            
            # 方位角
            if nav_info['item_relative_angle'] is not None:
                info_lines.append(f"Item Angle: {nav_info['item_relative_angle']:.1f} deg")
            
            # 距离
            if nav_info['item_distance'] is not None:
                info_lines.append(f"Item Dist: {nav_info['item_distance']:.1f}px")

            # 物体信息
            info_lines.append(f"Type: {nearest_item['type'].upper()}")
            
            # 物体状态
            valid = nearest_item.get('valid', True)
            info_lines.append(f"Status: {'VALID' if valid else 'INVALID'}")
            
            # 是否超出内收矩形
            if 'out_of_bounds' in nearest_item:
                out_of_bounds = nearest_item['out_of_bounds']
                info_lines.append(f"Out of Bounds: {out_of_bounds}")
                
                # 如果超出边界，显示调整后的坐标
                if out_of_bounds and 'adjusted_center' in nearest_item:
                    x, y = nearest_item['adjusted_center']
                    info_lines.append(f"Adjusted Pos: ({int(x)}, {int(y)})")
            
            # 绘制信息
            color = (0, 255, 0)  # 绿色（正常）
            # 如果超出边界，使用橙色
            if nearest_item.get('out_of_bounds', False):
                color = (0, 165, 255)  # 橙色
                
            font = cv2.FONT_HERSHEY_SIMPLEX
            for line in info_lines:
                cv2.putText(frame, line, (margin, current_y), font, 0.7, color, 2)
                current_y += line_height
                
        # 6. 添加到家的导航线
        if nav_info['car_pos'] and self.results['home'] and self.results['home'].Self_Center:
            # 绘制到家的导航线（使用蓝色区分）
            cv2.line(frame, 
                    tuple(map(int, nav_info['car_pos'])), 
                    tuple(map(int, self.results['home'].Self_Center)), 
                    (255, 0, 0), 2)  # 蓝色线
                    
            # 添加到家的导航信息
            home_info_lines = []
            
            # 方位角
            if nav_info['home_relative_angle'] is not None:
                home_info_lines.append(f"Home Angle: {nav_info['home_relative_angle']:.1f} deg")
            
            # 距离
            if nav_info['home_distance'] is not None:
                home_info_lines.append(f"Home Dist: {nav_info['home_distance']:.1f}px")
            
            # 绘制信息
            color = (255, 0, 0)  # 蓝色
            for line in home_info_lines:
                cv2.putText(frame, line, (margin, current_y), font, 0.7, color, 2)
                current_y += line_height        # 6. 添加帧率显示（显示在右上角）
        if hasattr(self, 'fps'):
            fps_text = f"FPS: {self.fps:.1f}"
            # 获取文本大小
            (text_w, text_h), _ = cv2.getTextSize(fps_text, 
                                                 cv2.FONT_HERSHEY_SIMPLEX, 
                                                 0.7, 2)
            # 在右上角显示
            cv2.putText(frame, fps_text,
                       (w - text_w - margin, margin + text_h),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return frame

def process_image(image_path: str, save_result: bool = True) -> np.ndarray:
    """
    处理单张测试图片
    
    功能描述：
        1. 读取指定图片
        2. 运行完整的视觉处理流程
        3. 在控制台输出详细结果
        4. 保存处理后的图片（可选）
    
    参数说明：
        image_path: 输入图片的完整路径
        save_result: 是否将结果保存为新图片
            - True: 自动生成"*_result.jpg"结果图片
            - False: 仅显示不保存
    
    返回值：
        处理后的图像帧（BGR格式）
    
    使用示例：
        result = process_image("test.jpg")
        result = process_image("test.jpg", save_result=False)
    """
    # 初始化视觉系统
    vision = VisionSystem()
    
    # 读取图片
    frame = cv2.imread(image_path)
    if frame is None:
        raise ValueError(f"无法读取图片: {image_path}")
        
    # 处理图片
    result = vision.process_frame(frame)
    

    # 打印检测结果
    print("\n所有检测结果:")
    print("-" * 40)
    
    if vision.results['aruco']:
        print("\nArUco标记:")
        for i, marker in enumerate(vision.results['aruco']):
            print(f"  标记 {i+1}:")
            print(f"    ID: {marker['id']}")
            print(f"    位置: {marker['center']}")
            print(f"    角度: {marker['angle']:.1f}°")
    
    if vision.results['fence'] is not None:
        print("\n围栏区域:")
        print(f"  中心点: {vision.results['fence'].center}")
        print(f"  面积: {vision.results['fence'].area:.1f}")
        print(f"  有效性: {vision.results['fence'].valid}")
        if vision.results['fence'].quad is not None:
            print("  四边形顶点:")
            for i, (x, y) in enumerate(vision.results['fence'].quad, 1):
                print(f"    顶点{i}: ({x:.1f}, {y:.1f})")
    
    if vision.results['home'] is not None:
        print("\n家区域:")
        print(f"  中心点: {vision.results['home'].Self_Center}")
        print(f"  面积: {vision.results['home'].Self_Area:.1f}")
        print(f"  有效性: {vision.results['home'].Self_Valid}")

    if vision.results['items']:
        print("\n检测到的物体:")
        for i, item in enumerate(vision.results['items']):
            if item.get('valid', True):  # 只显示有效物体
                print(f"\n  物体 {i+1} ({item['type']}):")
                print(f"    中心点: {item['center']}")
                print(f"    面积: {item['area']:.1f}")
                print(f"    宽高比: {item.get('aspect_ratio', 'N/A')}")
                if 'distance_px' in item:
                    print(f"    距离: {item['distance_px']:.1f}像素")
                if 'relative_angle_deg' in item:
                    print(f"    相对角度: {item['relative_angle_deg']:.1f}°")
    
        # 打印导航信息
    print("\n导航状态:")
    nav_info = vision.results['navigation']
    if nav_info['car_pos'] is not None:
        print(f"  车体位置: {nav_info['car_pos']}")
        print(f"  车体朝向: {nav_info['car_angle']:.1f}°")
    else:
        print("  未检测到车体位置")
    
    if nav_info['nearest_item'] is not None:
        print(f"  最近目标距离: {nav_info['item_distance']:.1f}像素")
        print(f"  相对方位角: {nav_info['item_relative_angle']:.1f}°")
        
        # 打印物体是否超出内收矩形区域的信息
        out_of_bounds = nav_info['nearest_item'].get('out_of_bounds', False)
        print(f"  超出内收矩形: {'是' if out_of_bounds else '否'}")
        
        # 如果超出边界，显示调整后的坐标
        if out_of_bounds and 'adjusted_center' in nav_info['nearest_item']:
            x, y = nav_info['nearest_item']['adjusted_center']
            print(f"  调整后坐标: ({int(x)}, {int(y)})")
    else:
        print("  未检测到有效目标")
        
    print("\n到家的相对位置:")
    if nav_info['home_relative_angle'] is not None and nav_info['home_distance'] is not None:
        print(f"  到家距离: {nav_info['home_distance']:.1f}像素")
        print(f"  相对方位角: {nav_info['home_relative_angle']:.1f}°")
    else:
        print("  未能计算到家的相对位置")
    
    # 保存结果
    if save_result:
        # 生成输出文件名
        base_name = image_path.rsplit('.', 1)[0]
        output_path = f"{base_name}_visionsystem.jpg"
        cv2.imwrite(output_path, result)
        print(f"\n结果已保存至: {output_path}")
    
    return result

def process_camera(camera_id: int = 0, serial_debug: int = 0):
    """
    使用摄像头进行实时视觉检测
    
    功能特点：
    1. 实时处理
       - 连续捕获和处理视频帧
       - 实时显示检测结果
       - 动态更新FPS显示
    
    2. 交互控制
       - 'q'键退出程序
       - 's'键保存当前帧
       - 实时显示系统状态
    
    3. 异常处理
       - 摄像头打开失败处理
       - 帧获取异常处理
       - 程序退出保护
    
    参数说明：
        camera_id: 摄像头设备ID
            - 0: 默认摄像头（通常是内置摄像头）
            - 1,2...: 外接摄像头
    
    使用方法：
        process_camera()  # 使用默认摄像头
        process_camera(1) # 使用外接摄像头
    """
    print("初始化视觉系统...")
    vision = VisionSystem()
    
    print(f"打开摄像头 (ID: {camera_id})...")
    cap = cv2.VideoCapture(camera_id)
    
    if not cap.isOpened():
        print(f"无法打开摄像头 {camera_id}!")
        return
        
    print("开始实时检测，按'q'退出，'s'保存当前帧...")
    
    # FPS计算变量
    fps = 0
    frame_count = 0
    start_time = cv2.getTickCount()
    frame_save_count = 0  # 用于生成保存文件名
    SER_DEBUG = serial_debug  # 是否启用串口调试
    if SER_DEBUG:
        ser = open_serial()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("无法读取摄像头画面!")
                break
                
            # 处理图像
            output = vision.process_frame(frame)

            if SER_DEBUG:
                send_camostudio_data(ser, vision.transmission_data)
            # 更新FPS
            frame_count += 1
            if frame_count >= 30:
                current_time = cv2.getTickCount()
                time_diff = (current_time - start_time) / cv2.getTickFrequency()
                fps = frame_count / time_diff
                vision.fps = fps
                frame_count = 0
                start_time = current_time
                
            # 显示结果
            cv2.imshow('Vision System', output)
            
            # 键盘控制
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):  # 退出
                break
            elif key == ord('s'):  # 保存当前帧
                frame_save_count += 1
                save_path = f"camera_frame_{frame_save_count}.jpg"
                cv2.imwrite(save_path, output)
                print(f"\n当前帧已保存至: {save_path}")
            time.sleep(0.10)

    finally:
        # 清理资源
        cap.release()
        if SER_DEBUG:
            close_serial(ser)
        cv2.destroyAllWindows()
        print("\n程序结束")

def test_vision_system():
    """
    视觉系统测试入口
    
    功能：
    1. 参数解析
       - -i/--image：图片处理模式
       - -c/--camera：摄像头模式
    
    2. 测试模式
       - 图片测试：处理单张图片并显示结果
       - 实时检测：使用摄像头实时处理
    
    3. 错误处理
       - 文件不存在检查
       - 运行时异常捕获
       - 用户友好的错误提示
    
    使用示例：
        python main.py            # 默认使用摄像头测试
        python main.py -i test.jpg # 使用图片测试
        python main.py -c 1        # 使用指定摄像头测试

    注意：
        这是独立的视觉测试模块，不包含数据传输功能。
        完整的主程序将在后续加入数据传输部分。
    """
    import argparse
    import os
    
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='智能小车视觉系统')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('-i', '--image', type=str, help='要处理的图片路径')
    group.add_argument('-c', '--camera', type=int, default=0, help='摄像头ID（默认0）')
    group.add_argument('-t', '--test_serial', type=int, default=0, help='启用测试模式')
    args = parser.parse_args()

    SER_DEBUG = args.test_serial  # 是否启用串口调试
    print(SER_DEBUG)
    try:
        if args.image:  # 图片模式
            if not os.path.exists(args.image):
                print(f"错误: 图片文件不存在: {args.image}")
                return
            
            print(f"正在处理图片: {args.image}")
            result = process_image(args.image)
            
            # 显示结果
            cv2.imshow('Result', result)
            print("\n按任意键关闭窗口...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
        else:  # 摄像头模式
            process_camera(args.camera, serial_debug=SER_DEBUG)

    except Exception as e:
        print(f"\n发生错误: {str(e)}")
        return

if __name__ == "__main__":
    # 暂时使用视觉测试作为入口
    # TODO: 后续添加完整的主程序，包含数据传输功能
    test_vision_system()