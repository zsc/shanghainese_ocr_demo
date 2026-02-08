#!/usr/bin/env python3
"""
上海方言词典页面处理脚本 - 简化版
流程：
1. 检测中间竖线，分割左右两栏
2. 连通域检测获取所有文字框（高 recall）
3. 按 y 坐标聚类合并同一行的框
4. 可视化并提取小图像
"""

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os
from typing import List, Tuple, Dict
from collections import defaultdict


def detect_vertical_line(image: np.ndarray) -> int:
    """检测中间竖线位置"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, 
                            minLineLength=height//3, maxLineGap=10)
    
    if lines is None:
        return width // 2
    
    center_x = width // 2
    best_x = center_x
    min_distance = float('inf')
    
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if abs(x1 - x2) < 20:
            line_x = (x1 + x2) // 2
            if width * 0.4 < line_x < width * 0.6:
                distance = abs(line_x - center_x)
                if distance < min_distance:
                    min_distance = distance
                    best_x = line_x
    
    return best_x


def detect_text_boxes_raw(image: np.ndarray) -> List[Tuple[int, int, int, int]]:
    """
    使用连通域检测获取文字框（高 recall 模式）
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 二值化
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # 轻微膨胀以连接字符但不连接不同行
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    binary = cv2.dilate(binary, kernel, iterations=1)
    
    # 连通域检测
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    
    boxes = []
    for i in range(1, num_labels):  # 跳过背景
        x, y, w, h, area = stats[i]
        
        # 宽松过滤条件 - 高 recall
        if 8 < h < 100 and 10 < w < 600 and area > 30:
            aspect = w / h if h > 0 else 1
            if 0.3 < aspect < 30:
                boxes.append((x, y, x+w, y+h))
    
    return boxes


def cluster_by_y(boxes: List[Tuple[int, int, int, int]], 
                 y_tolerance: int = 12) -> List[List[Tuple[int, int, int, int]]]:
    """
    按 y 坐标聚类，将同一行的框分到一组
    
    Args:
        boxes: 检测框列表
        y_tolerance: y 坐标差异容忍度（像素）
    
    Returns:
        分组后的框列表
    """
    if not boxes:
        return []
    
    # 计算每个框的中心 y
    box_centers = []
    for box in boxes:
        y_center = (box[1] + box[3]) / 2
        box_centers.append((y_center, box))
    
    # 按 y_center 排序
    box_centers.sort(key=lambda x: x[0])
    
    # 聚类
    clusters = []
    current_cluster = [box_centers[0][1]]
    current_y = box_centers[0][0]
    
    for i in range(1, len(box_centers)):
        y_center, box = box_centers[i]
        
        if abs(y_center - current_y) <= y_tolerance:
            # 同一行
            current_cluster.append(box)
            # 更新当前 y（使用平均值）
            current_y = sum((b[1] + b[3]) / 2 for b in current_cluster) / len(current_cluster)
        else:
            # 新行
            clusters.append(current_cluster)
            current_cluster = [box]
            current_y = y_center
    
    if current_cluster:
        clusters.append(current_cluster)
    
    return clusters


def merge_boxes_in_cluster(clusters: List[List[Tuple[int, int, int, int]]]) -> List[Tuple[int, int, int, int]]:
    """
    将每个聚类中的框合并成一个长框
    """
    merged = []
    
    for cluster in clusters:
        if len(cluster) == 1:
            merged.append(cluster[0])
        else:
            # 按 x 排序
            cluster = sorted(cluster, key=lambda b: b[0])
            
            # 合并所有框
            min_x = min(b[0] for b in cluster)
            min_y = min(b[1] for b in cluster)
            max_x = max(b[2] for b in cluster)
            max_y = max(b[3] for b in cluster)
            
            merged.append((min_x, min_y, max_x, max_y))
    
    # 按 y 排序
    merged = sorted(merged, key=lambda b: b[1])
    return merged


def process_column(image: np.ndarray, column_name: str) -> Tuple[List[Dict], np.ndarray]:
    """处理单列图像"""
    height, width = image.shape[:2]
    
    # 1. 检测所有文字框
    boxes = detect_text_boxes_raw(image)
    print(f"  {column_name} column: detected {len(boxes)} raw boxes")
    
    # 2. 按 y 聚类
    clusters = cluster_by_y(boxes, y_tolerance=12)
    print(f"  {column_name} column: clustered into {len(clusters)} rows")
    
    # 3. 合并每行
    merged_boxes = merge_boxes_in_cluster(clusters)
    print(f"  {column_name} column: merged to {len(merged_boxes)} boxes")
    
    # 4. 创建 debug 可视化图
    debug_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(debug_img)
    
    try:
        font = ImageFont.truetype("/System/Library/Fonts/PingFang.ttc", 16)
    except:
        font = ImageFont.load_default()
    
    # 准备提取信息
    extract_infos = []
    
    for idx, box in enumerate(merged_boxes):
        x1, y1, x2, y2 = box
        
        # 绘制框
        draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=2)
        
        # 添加编号
        label = f"{idx+1}"
        draw.text((x1, y1-18), label, fill=(255, 0, 0), font=font)
        
        extract_infos.append({
            'box': box,
            'column': column_name,
            'local_id': idx + 1,
            'global_id': None
        })
    
    return extract_infos, np.array(debug_img)


def process_dictionary_page(image_path: str, output_dir: str) -> List[str]:
    """处理词典页面主函数"""
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Loading image: {image_path}")
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Cannot load image: {image_path}")
    
    height, width = image.shape[:2]
    print(f"Image size: {width} x {height}")
    
    # 1. 检测竖线并分割
    print("Detecting vertical line...")
    line_x = detect_vertical_line(image)
    print(f"Vertical line at x={line_x}")
    
    left_img = image[:, :line_x]
    right_img = image[:, line_x:]
    print(f"Left: {left_img.shape[1]} x {left_img.shape[0]}, Right: {right_img.shape[1]} x {right_img.shape[0]}")
    
    # 2. 处理两栏
    print("\nProcessing left column...")
    left_infos, _ = process_column(left_img, "left")
    
    print("\nProcessing right column...")
    right_infos, _ = process_column(right_img, "right")
    
    # 3. 合并并编号
    all_infos = left_infos + right_infos
    for i, info in enumerate(all_infos):
        info['global_id'] = i + 1
    
    print(f"\nTotal: {len(all_infos)} boxes")
    
    # 4. 创建完整 debug 图
    debug_full = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(debug_full)
    
    try:
        font = ImageFont.truetype("/System/Library/Fonts/PingFang.ttc", 14)
    except:
        font = ImageFont.load_default()
    
    # 画分割线
    draw.line([(line_x, 0), (line_x, height)], fill=(0, 255, 255), width=2)
    
    # 画所有框和编号
    for info in all_infos:
        if info['column'] == 'left':
            x1, y1, x2, y2 = info['box']
        else:
            bx1, by1, bx2, by2 = info['box']
            x1, y1, x2, y2 = bx1 + line_x, by1, bx2 + line_x, by2
        
        draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=2)
        
        label = str(info['global_id'])
        bbox = draw.textbbox((0, 0), label, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        draw.rectangle([x1, y1-th-2, x1+tw+4, y1], fill=(255, 0, 0))
        draw.text((x1+2, y1-th-1), label, fill=(255, 255, 255), font=font)
    
    debug_path = os.path.join(output_dir, "debug_visualization.png")
    debug_full.save(debug_path)
    print(f"Debug image saved: {debug_path}")
    
    # 5. 提取小图像
    print("\nExtracting small images...")
    extracted_paths = []
    
    for info in all_infos:
        if info['column'] == 'left':
            x1, y1, x2, y2 = info['box']
            src_img = left_img
        else:
            bx1, by1, bx2, by2 = info['box']
            x1, y1, x2, y2 = bx1, by1, bx2, by2
            src_img = right_img
        
        # 加边距
        pad = 3
        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(src_img.shape[1], x2 + pad)
        y2 = min(src_img.shape[0], y2 + pad)
        
        crop = src_img[y1:y2, x1:x2]
        
        output_name = f"box_{info['global_id']:03d}_{info['column']}_{info['local_id']:02d}.png"
        output_path = os.path.join(output_dir, output_name)
        cv2.imwrite(output_path, crop)
        extracted_paths.append(output_path)
    
    print(f"Extracted {len(extracted_paths)} images")
    return extracted_paths


if __name__ == "__main__":
    INPUT_IMAGE = "first_page.png"
    OUTPUT_DIRECTORY = "extracted_boxes"
    
    try:
        result_paths = process_dictionary_page(INPUT_IMAGE, OUTPUT_DIRECTORY)
        print(f"\n✅ Success! Total {len(result_paths)} images.")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
