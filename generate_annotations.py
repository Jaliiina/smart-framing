import json
import re
from pathlib import Path
import cv2

def parse_readme(readme_path):
    """解析 README.md 表格，返回 {image_name: (cx_norm, cy_norm, w_norm, h_norm)}"""
    text = Path(readme_path).read_text(encoding='utf-8')
    # 匹配表格行：| A01.jpg | 0.356352 | 0.281752 | 0.5 | 0.5 |
    pattern = re.compile(r'\|\s*(\S+\.jpg)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|')
    data = {}
    for match in pattern.findall(text):
        name, cx, cy, w, h = match
        data[name] = (float(cx), float(cy), float(w), float(h))
    return data

def generate_annotations(image_dir, readme_path, output_json):
    """根据 README 和实际图片尺寸生成 annotations.json"""
    readme_data = parse_readme(readme_path)
    annotations = []
    for img_name, (cx_norm, cy_norm, w_norm, h_norm) in readme_data.items():
        img_path = Path(image_dir) / img_name
        if not img_path.exists():
            print(f"警告：图片 {img_path} 不存在，跳过")
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"警告：无法读取 {img_path}，跳过")
            continue
        h, w = img.shape[:2]
        # 计算 bbox 的像素坐标
        bbox_w = int(w_norm * w)
        bbox_h = int(h_norm * h)
        center_x = int(cx_norm * w)
        center_y = int(cy_norm * h)
        x1 = max(0, center_x - bbox_w // 2)
        y1 = max(0, center_y - bbox_h // 2)
        x2 = min(w, x1 + bbox_w)
        y2 = min(h, y1 + bbox_h)
        annotations.append({
            "image": img_name,
            "bbox": [x1, y1, x2, y2],
            "score": 1.0  # 可默认给 1.0
        })
    # 保存 JSON
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(annotations, f, indent=2, ensure_ascii=False)
    print(f"已生成 {len(annotations)} 条标注，保存至 {output_json}")

if __name__ == "__main__":
    # 请根据实际路径修改
    image_dir = "testA/testA"          # 存放 A01.jpg... 的目录
    readme_path = "testA/testA/README.md"  # README.md 的路径
    output_json = "testA/testA/annotations.json"
    generate_annotations(image_dir, readme_path, output_json)