#!/usr/bin/env python3
"""
通用VLM-R1-OVD格式到XLABEL格式转换工具

该脚本将VLM-R1-OVD格式的JSONL文件转换为X-AnyLabeling可读的XLABEL格式文件。
VLM-R1-OVD格式包含图像路径和对话历史，其中assistant的回答中包含了检测框信息。
"""

import sys
import os
import json
import re
import argparse
from PIL import Image


def get_image_size(image_path):
    """
    获取图像尺寸
    
    Args:
        image_path (str): 图像文件路径
        
    Returns:
        tuple: (width, height) 图像尺寸
    """
    try:
        with Image.open(image_path) as img:
            return img.size[0], img.size[1]
    except Exception as e:
        print(f"无法获取图像尺寸 {image_path}: {e}")
        return 1024, 538  # 默认尺寸


def extract_bbox_data(assistant_content):
    """
    从assistant的回复中提取边界框数据
    
    Args:
        assistant_content (str): assistant的回复内容
        
    Returns:
        list: 包含边界框信息的字典列表
    """
    bbox_data = []
    
    # 方法1: 尝试直接解析整个内容为JSON
    try:
        data = json.loads(assistant_content)
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict) and "bbox_2d" in item and "label" in item:
                    bbox_data.append(item)
        elif isinstance(data, dict) and "bbox_2d" in data and "label" in data:
            bbox_data.append(data)
    except json.JSONDecodeError:
        pass
    
    # 方法2: 尝试提取代码块中的JSON
    if not bbox_data:
        code_block_pattern = r'```(?:\w+)?\s*(.*?)\s*```'
        code_blocks = re.findall(code_block_pattern, assistant_content, re.DOTALL)
        for block in code_blocks:
            try:
                data = json.loads(block)
                if isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict) and "bbox_2d" in item and "label" in item:
                            bbox_data.append(item)
                elif isinstance(data, dict) and "bbox_2d" in data and "label" in data:
                    bbox_data.append(data)
                break  # 找到第一个有效的代码块就停止
            except json.JSONDecodeError:
                continue
    
    # 方法3: 尝试提取<answer>标签中的内容
    if not bbox_data:
        answer_pattern = r'<answer>(.*?)</answer>'
        answer_matches = re.findall(answer_pattern, assistant_content, re.DOTALL)
        for match in answer_matches:
            try:
                data = json.loads(match)
                if isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict) and "bbox_2d" in item and "label" in item:
                            bbox_data.append(item)
                elif isinstance(data, dict) and "bbox_2d" in data and "label" in data:
                    bbox_data.append(data)
                break  # 找到第一个有效的answer就停止
            except json.JSONDecodeError:
                # 尝试在answer中查找代码块
                code_block_pattern = r'```(?:\w+)?\s*(.*?)\s*```'
                code_blocks = re.findall(code_block_pattern, match, re.DOTALL)
                for block in code_blocks:
                    try:
                        data = json.loads(block)
                        if isinstance(data, list):
                            for item in data:
                                if isinstance(item, dict) and "bbox_2d" in item and "label" in item:
                                    bbox_data.append(item)
                        elif isinstance(data, dict) and "bbox_2d" in data and "label" in data:
                            bbox_data.append(data)
                        break  # 找到第一个有效的代码块就停止
                    except json.JSONDecodeError:
                        continue
                if bbox_data:
                    break
    
    # 方法4: 尝试提取Qwen3-VL-Thinking格式的答案
    if not bbox_data:
        pattern = r'bbox2d\s+([^\s$]+?)\s*$(.*?)$'
        matches = re.findall(pattern, assistant_content)
        for label, coords_str in matches:
            try:
                coords = json.loads(f"[{coords_str}]")
                if len(coords) == 4:
                    bbox_data.append({
                        "label": label,
                        "bbox_2d": coords
                    })
            except json.JSONDecodeError:
                # 尝试手动解析坐标
                coord_pattern = r'(\d+(?:\.\d+)?)'
                coord_matches = re.findall(coord_pattern, coords_str)
                if len(coord_matches) == 4:
                    coords = [float(c) for c in coord_matches]
                    bbox_data.append({
                        "label": label,
                        "bbox_2d": coords
                    })
    
    return bbox_data


def convert_entry_to_xlabel(entry, output_dir, image_dir=None):
    """
    将单个VLM-R1-OVD条目转换为XLABEL格式
    
    Args:
        entry (dict): VLM-R1-OVD条目
        output_dir (str): 输出目录
        image_dir (str): 图像目录（可选）
        
    Returns:
        str: 生成的XLABEL文件路径
    """
    image_name = entry.get("image", "")
    if not image_name:
        print("警告: 条目中没有图像名称")
        return None
    
    # 构造完整的图像路径
    if image_dir:
        image_path = os.path.join(image_dir, os.path.basename(image_name))
    else:
        image_path = image_name
    
    # 获取图像尺寸
    image_width, image_height = get_image_size(image_path) if os.path.exists(image_path) else (1024, 538)
    
    # 构建XLABEL格式数据
    xlabel_data = {
        "version": "4.0",
        "flags": {},
        "description": "",
        "shapes": [],
        "imagePath": os.path.basename(image_name),
        "imageData": None,
        "imageWidth": image_width,
        "imageHeight": image_height
    }
    
    # 提取对话中的assistant回复
    conversations = entry.get("conversations", [])
    assistant_content = ""
    for conv in conversations:
        if conv.get("from") == "assistant":
            assistant_content = conv.get("value", "")
            break
    
    if not assistant_content:
        print(f"警告: 图像 {image_name} 没有assistant回复")
        # 仍然生成空的XLABEL文件
        output_file = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(image_name))[0]}.json")
        with open(output_file, 'w', encoding='utf-8') as out_f:
            json.dump(xlabel_data, out_f, ensure_ascii=False, indent=2)
        return output_file
    
    # 提取边界框数据
    bbox_data = extract_bbox_data(assistant_content)
    
    # 添加边界框到shapes
    for item in bbox_data:
        if "bbox_2d" in item and "label" in item:
            x1, y1, x2, y2 = item["bbox_2d"]
            
            # 确保坐标在有效范围内
            x1 = max(0, min(x1, image_width))
            y1 = max(0, min(y1, image_height))
            x2 = max(0, min(x2, image_width))
            y2 = max(0, min(y2, image_height))
            
            # 确保坐标顺序正确
            xmin, xmax = min(x1, x2), max(x1, x2)
            ymin, ymax = min(y1, y2), max(y1, y2)
            
            label = item["label"]
            
            # 创建矩形框的四个点（左上、右上、右下、左下）
            shape = {
                "label": label,
                "points": [
                    [xmin, ymin],  # 左上
                    [xmax, ymin],  # 右上
                    [xmax, ymax],  # 右下
                    [xmin, ymax]   # 左下
                ],
                "shape_type": "rectangle",
                "line_color": None,
                "fill_color": None,
                "group_id": None,
                "description": None,
                "difficult": False,
                "direction": 0,
                "flags": {}
            }
            xlabel_data["shapes"].append(shape)
    
    # 保存为XLABEL文件
    output_file = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(image_name))[0]}.json")
    with open(output_file, 'w', encoding='utf-8') as out_f:
        json.dump(xlabel_data, out_f, ensure_ascii=False, indent=2)
    
    return output_file


def convert_vlm_r1_ovd_to_xlabel(vlm_r1_ovd_path, output_dir, image_dir=None):
    """
    将VLM-R1-OVD格式的JSONL文件转换为XLABEL格式
    
    Args:
        vlm_r1_ovd_path (str): VLM-R1-OVD JSONL文件路径
        output_dir (str): 输出目录
        image_dir (str): 图像目录（可选）
    """
    print(f"开始转换: {vlm_r1_ovd_path}")
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 读取JSONL文件
    converted_count = 0
    with open(vlm_r1_ovd_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue
                
            try:
                # 解析JSON行
                entry = json.loads(line.strip())
            except json.JSONDecodeError as e:
                print(f"警告: 无法解析第 {line_num} 行为JSON: {e}")
                continue
            
            try:
                output_file = convert_entry_to_xlabel(entry, output_dir, image_dir)
                if output_file:
                    print(f"已转换 {entry.get('image', 'unknown')} 到 {output_file}")
                    converted_count += 1
            except Exception as e:
                print(f"错误: 转换第 {line_num} 行时出错: {e}")
                continue
    
    print(f"转换完成! 总共转换了 {converted_count} 个文件。")


def main():
    parser = argparse.ArgumentParser(description="将VLM-R1-OVD格式的JSONL文件转换为XLABEL格式")
    parser.add_argument("input", help="VLM-R1-OVD JSONL文件路径")
    parser.add_argument("output", help="输出目录")
    parser.add_argument("--image_dir", help="图像文件目录（可选）")
    
    args = parser.parse_args()
    
    convert_vlm_r1_ovd_to_xlabel(args.input, args.output, args.image_dir)


if __name__ == "__main__":
    main()