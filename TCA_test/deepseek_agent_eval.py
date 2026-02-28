import os
import tempfile
from api_caller import call_sam2_api
import base64
import requests
from PIL import Image, ImageDraw
from yolo_agent2 import call_yolo_api  # YOLO接口
import uuid
import matplotlib
matplotlib.use("Agg")
import math
from collections import Counter
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from openai import OpenAI
import datetime
import cv2
import numpy as np


#sk-8a5add1a6785414a9ff1b2653e760880
temp_dir = tempfile.mkdtemp(prefix="deepseek_")
# ===== DeepSeek 初始化 =====
client = OpenAI(
    api_key="sk-8a5add1a6785414a9ff1b2653e760880",
    base_url="https://api.deepseek.com"
)
# ===== U-Net API 配置 =====
UNET_API_URL = "http://127.0.0.1:7000/unet/{model_id}/segment"

MODEL_MAPPING = {
    "unet_Fracture": {"color": (255, 0, 0)},          # 红色
    "unet_Induced_Fracture": {"color": (0, 0, 255)},  # 蓝色
    "unet_Vug": {"color": (0, 255, 0)}               # 绿色
}

# ===== 调用 U-Net API =====
def call_unet_api(model_id, image_path):
    with open(image_path, "rb") as f:
        files = {"roi": f}
        response = requests.post(UNET_API_URL.format(model_id=model_id), files=files)
    if response.status_code == 200:
        data = response.json()
        mask_base64 = data["mask"]
        mask_bytes = base64.b64decode(mask_base64)
        mask_path = f"{model_id}_{uuid.uuid4().hex}.png"

        #mask_path = os.path.join(temp_dir, f"{model_id}_{uuid.uuid4().hex}.png")
        with open(mask_path, "wb") as f:
            f.write(mask_bytes)
        return {"mask": mask_path}
    else:
        raise RuntimeError(f"❌ U-Net API调用失败: {response.text}")
def save_base64_mask(mask_b64, save_path):
    """将 base64 mask 保存为 PNG 文件"""
    with open(save_path, "wb") as f:
        f.write(base64.b64decode(mask_b64))
    return save_path
def call_sam2_box(image_path, box_coords):
    """
    使用 box 提示调用 SAM2 分割
    :param image_path: 输入图像路径
    :param box_coords: [x_min, y_min, x_max, y_max]
    :return: {"mask": mask_path}
    """
    with open(image_path, "rb") as f:
        files = {"file": (image_path, f, "image/jpeg")}
        data = {"prompt_type": "box", "box_coords": str(box_coords)}
        resp = requests.post("http://127.0.0.1:3000/predict", files=files, data=data)
        resp.raise_for_status()
        result = resp.json()

    # 保存 mask 文件
    mask_base64 = result.get("mask")
    mask_bytes = base64.b64decode(mask_base64)
    mask_path = f"sam2_box_{uuid.uuid4().hex}.png"
    #mask_path = os.path.join(temp_dir, f"sam2_box_{uuid.uuid4().hex}.png")
    with open(mask_path, "wb") as f:
        f.write(mask_bytes)

    return {"mask": mask_path}

# ===== Mask 预处理 =====
def preprocess_mask_for_analysis(mask_path, log_fn=None):
    mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask_img is None:
        raise RuntimeError(f"读取 mask 文件失败: {mask_path}")
    binary_mask = (mask_img > 0).astype(np.uint8) * 255
    nonzero_count = cv2.countNonZero(binary_mask)
    if log_fn:
        log_fn(f"🔹 mask 非零像素数: {nonzero_count}")
        if nonzero_count == 0:
            log_fn("⚠️ 当前 mask 全为空白，分析结果可能无效")
    temp_mask_path = f"temp_mask_{uuid.uuid4().hex}.png"
    #temp_mask_path = os.path.join(temp_dir, f"temp_mask_{uuid.uuid4().hex}.png")
    cv2.imwrite(temp_mask_path, binary_mask)
    return temp_mask_path

def split_mask_to_contours(mask_path):
    mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    binary_mask = (mask_img > 0).astype(np.uint8)
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    single_masks = []
    for contour in contours:
        mask_single = np.zeros_like(binary_mask)
        cv2.drawContours(mask_single, [contour], -1, 255, -1)
        single_masks.append(mask_single)
    return single_masks

# ===== 裂缝分析 API =====
def call_crack_api(mask_path, image_height_mm, image_width_mm):
    mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    binary_mask = (mask_img > 0).astype(np.uint8) * 255
    temp_mask_path = f"temp_mask_{uuid.uuid4().hex}.png"
    #temp_mask_path = os.path.join(temp_dir, f"temp_mask_{uuid.uuid4().hex}.png")
    cv2.imwrite(temp_mask_path, binary_mask)
    with open(temp_mask_path, "rb") as f:
        files = {"file": (temp_mask_path, f, "image/png")}
        data = {
            "image_height_mm": str(image_height_mm),
            "image_width_mm": str(image_width_mm)
        }
        resp = requests.post("http://127.0.0.1:8010/analyze_crack", files=files, data=data)
        if resp.status_code != 200:
            raise RuntimeError(f"裂缝分析失败: {resp.text}")
        result = resp.json()
    # 补全字段
    required_keys = ["A", "B", "C", "D", "倾角_deg", "倾向_deg", "走向_deg",
                     "裂缝宽度_FVA", "裂缝长度_FVTL", "裂缝密度_FVDC", "裂缝视孔隙度_FVPA"]
    for k in required_keys:
        if k not in result:
            result[k] = None
    return result

def call_vug_api(image_path, image_height_mm, image_width_mm, window_height_mm):
    with open(image_path, "rb") as f:
        files = {"file": (image_path, f, "image/png")}
        data = {
            "image_height_mm": str(image_height_mm),
            "image_width_mm": str(image_width_mm),
            "window_height_mm": str(window_height_mm)
        }
        resp = requests.post("http://127.0.0.1:8011/analyze_vug", files=files, data=data)
        if resp.status_code != 200:
            raise RuntimeError(f"孔洞分析失败: {resp.text}")
        return resp.json()
# ===== 颜色映射 =====


# SAM2 专用颜色映射
SAM2_MAPPING = {
    "sam2_fracture": {"color": (0, 255, 0)},   # 紫色
    "sam2_vug": {"color": (255, 165, 0)}         # 橙色
}


def overlay_masks(masks, model_ids, base_image_path):
    """
    将 U-Net（裂缝 / 孔洞）+ SAM2 的掩码叠加到原图上，并使用不同颜色区分
    """
    base_img = cv2.imread(base_image_path)
    if base_img is None:
        raise FileNotFoundError(f"无法读取底图: {base_image_path}")

    overlay = base_img.copy()

    for mask, model_id in zip(masks, model_ids):
        model_id_lower = model_id.lower()

        # ---------- 颜色策略 ----------
        if "fracture" in model_id_lower:
            color = (0, 0, 255)        # 🔴 裂缝：红
        elif "vug" in model_id_lower:
            color = (0, 255, 255)      # 🟡 孔洞：黄
        elif "sam2" in model_id_lower:
            color = (0, 255, 0)        # 🟢 SAM2：绿
        else:
            color = (255, 255, 255)    # ⚪ 兜底：白

        # ---------- 读取 mask ----------
        mask_path = mask.get("mask")
        if not mask_path or not os.path.exists(mask_path):
            continue

        mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask_img is None:
            continue

        # ---------- 二值化 ----------
        _, binary = cv2.threshold(mask_img, 1, 255, cv2.THRESH_BINARY)

        # ---------- 提取轮廓 ----------
        contours, _ = cv2.findContours(
            binary,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        # ---------- 绘制（填充） ----------
        cv2.drawContours(
            overlay,
            contours,
            -1,
            color,
            thickness=cv2.FILLED
        )

    out_path = base_image_path.replace(".png", "_overlay.png")
    cv2.imwrite(out_path, overlay)
    return out_path



def extract_coarse_mask(image, box):
    """
    在 YOLO box 内提取粗结构掩码
    """
    x, y, w, h = map(int, box)
    crop = image[y:y+h, x:x+w]

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    _, binary = cv2.threshold(edges, 0, 255,
                               cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return binary

def count_connected_components(binary_mask):
    """
    计算连通域数量（去除极小噪声）
    """
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary_mask, connectivity=8
    )
    # 过滤面积很小的噪声
    min_area = 20
    valid = sum(stats[i, cv2.CC_STAT_AREA] > min_area
                for i in range(1, num_labels))
    return valid
def contour_irregularity(binary_mask):
    contours, _ = cv2.findContours(
        binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    if not contours:
        return 0.0

    cnt = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)

    if area < 1:
        return 0.0

    irregularity = (perimeter ** 2) / (4 * np.pi * area)
    return irregularity  # ≥1，越大越复杂

def curvature_variance(contour):
    """
    近似计算边界曲率变化率
    """
    pts = contour.squeeze()
    if len(pts) < 10:
        return 0.0

    dx = np.gradient(pts[:, 0])
    dy = np.gradient(pts[:, 1])
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)

    curvature = np.abs(dx * ddy - dy * ddx) / (dx*dx + dy*dy + 1e-6) ** 1.5
    return np.var(curvature)
def compute_object_complexity(image, box):
    binary = extract_coarse_mask(image, box)
    cc_count = count_connected_components(binary)
    irregularity = contour_irregularity(binary)

    return {
        "connected_components": cc_count,
        "irregularity": irregularity
    }
def select_segmentation_model(
    object_class,
    complexity,
    cc_thresh=3,
    irregularity_thresh=1.5
):
    """
    根据目标类型 + 复杂度选择模型
    """
    cc = complexity["connected_components"]
    irr = complexity["irregularity"]

    # 规则 1：孔洞 → SAM2
    if object_class == "Vug":
        return "sam2_vug"

    # 规则 2：裂缝但结构复杂 → SAM2
    if cc > cc_thresh or irr > irregularity_thresh:
        return "sam2_fracture"

    # 规则 3：规则裂缝 → U-Net
    return "unet_Fracture"





# ===== 滑窗分析 =====
def sliding_window_unet_analysis(image_path, model_id, image_height_mm, image_width_mm, log_fn=None):
    img = cv2.imread(image_path)
    H, W = img.shape[:2]
    window_px = W  # 可以自定义滑窗高度
    n_blocks = math.ceil(H / window_px)
    masks_full = np.zeros((H, W), dtype=np.uint8)
    curves_metrics_all = []

    for i in range(n_blocks):
        start_y = i * window_px
        end_y = min((i+1) * window_px, H)
        img_block = img[start_y:end_y, :]
        block_path = f"temp_block_{uuid.uuid4().hex}.png"
        block_path = os.path.join(temp_dir, f"temp_block_{uuid.uuid4().hex}.png")
        cv2.imwrite(block_path, img_block)

        try:
            mask_result = call_unet_api(model_id, block_path)
            mask_block = cv2.imread(mask_result["mask"], cv2.IMREAD_GRAYSCALE)
            if mask_block is None:
                continue
            masks_full[start_y:end_y, :] = np.maximum(masks_full[start_y:end_y, :], mask_block)

            # 裂缝分析
            mask_path = preprocess_mask_for_analysis(mask_result["mask"], log_fn)
            single_masks = split_mask_to_contours(mask_path)
            for j, mask_single in enumerate(single_masks):
                temp_mask_path = f"temp_single_block_{i}_{j}.png"
                #temp_mask_path = os.path.join(temp_dir, f"temp_single_block_{i}_{j}.png")
                cv2.imwrite(temp_mask_path, mask_single)
                metrics = call_crack_api(temp_mask_path, image_height_mm, image_width_mm)
                metrics["y_offset"] = start_y  # ⚡ 添加块偏移
                curves_metrics_all.append(metrics)

        except Exception as e:
            if log_fn:
                log_fn(f"❌ 分块 U-Net 分析失败: {e}")

    mask_full_path = f"full_mask_{uuid.uuid4().hex}.png"
    mask_full_path = os.path.join(temp_dir, f"full_mask_{uuid.uuid4().hex}.png")
    # ✅ 强制归一化为 0-255，保证显示效果与 SAM2 一致
    if masks_full.max() <= 1:
        masks_full = (masks_full * 255).astype(np.uint8)
    else:
        masks_full = masks_full.astype(np.uint8)

    cv2.imwrite(mask_full_path, masks_full)
    return mask_full_path, curves_metrics_all

def sliding_window_vug_analysis(mask_path, image_height_mm, image_width_mm,
                                window_height_mm=100, log_fn=None):
    """
    对整张孔洞分割二值mask进行滑窗分析 + 全井汇总统计。

    参数:
        mask_path: str, 孔洞掩码路径 (二值图, 白色区域为Vug)
        image_height_mm: float, 图像对应的总深度 (mm)
        image_width_mm: float, 图像对应的井径宽度 (mm)
        window_height_mm: float, 滑窗高度 (mm)
        log_fn: callable, 日志记录函数，可为空

    返回:
        window_vug_list: list[dict], 每个滑窗的孔洞统计信息
        summary_metrics: dict, 全井汇总与平均指标
    """
    if log_fn:
        log_fn(f"📌 开始滑窗孔洞分析, window_height_mm={window_height_mm}")

    # ==== 读取并二值化掩码 ====
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"未找到掩码文件: {mask_path}")

    # 任何非0像素视为孔洞
    binary_mask = (mask > 0).astype(np.uint8)
    h_px, w_px = binary_mask.shape
    px_per_mm_h = h_px / image_height_mm
    num_windows = max(1, int(np.ceil(image_height_mm / window_height_mm)))

    window_vug_list = []
    total_vug_count = 0
    total_area_mm2 = 0
    all_cvs, all_dens, all_sizes = [], [], []

    # ==== 滑窗分析 ====
    for i in range(num_windows):
        start_px = int(i * window_height_mm * px_per_mm_h)
        end_px = int(min(h_px, (i + 1) * window_height_mm * px_per_mm_h))
        window_mask = binary_mask[start_px:end_px, :]

        # 保存临时mask
        temp_mask_path = f"temp_vug_window_{i}.png"
        Image.fromarray((window_mask * 255).astype(np.uint8)).save(temp_mask_path)

        try:
            resp = requests.post(
                "http://127.0.0.1:8011/analyze_vug",
                files={"file": open(temp_mask_path, "rb")},
                data={
                    "image_height_mm": str(window_height_mm),
                    "image_width_mm": str(image_width_mm),
                    "window_height_mm": str(window_height_mm)
                },
                timeout=30
            )
            resp.raise_for_status()
            result = resp.json()
            summary = result.get("summary", {})
            vugs = result.get("vugs", [])

            window_vug_list.append({
                "depth_start_mm": i * window_height_mm,
                "depth_end_mm": min((i + 1) * window_height_mm, image_height_mm),
                "vug_count": summary.get("vug_count", 0),
                "total_area_mm2": summary.get("total_area_mm2", 0),
                "CVPA": summary.get("CVPA", 0),
                "CDENS": summary.get("CDENS", 0),
                "CSIZE": summary.get("CSIZE", 0),
                "vugs": vugs
            })

            total_vug_count += summary.get("vug_count", 0)
            total_area_mm2 += summary.get("total_area_mm2", 0)
            all_cvs.append(summary.get("CVPA", 0))
            all_dens.append(summary.get("CDENS", 0))
            all_sizes.append(summary.get("CSIZE", 0))

            if log_fn:
                log_fn(f"✅ 滑窗 {i+1}/{num_windows} 分析完成: {summary}")

        except Exception as e:
            if log_fn:
                log_fn(f"⚠️ 滑窗 {i+1} 孔洞分析失败: {e}")
            window_vug_list.append({
                "depth_start_mm": i * window_height_mm,
                "depth_end_mm": min((i + 1) * window_height_mm, image_height_mm),
                "vug_count": 0,
                "total_area_mm2": 0,
                "CVPA": 0,
                "CDENS": 0,
                "CSIZE": 0,
                "vugs": []
            })

    # 汇总全井统计
    summary_metrics = {
        "total_vug_count": total_vug_count,
        "total_area_mm2": total_area_mm2,
        "mean_CVPA": np.mean(all_cvs) if all_cvs else 0,
        "mean_CDENS": np.mean(all_dens) if all_dens else 0,
        "mean_CSIZE": np.mean(all_sizes) if all_sizes else 0
    }

    if log_fn:
        log_fn(f"📊 全井孔洞总体分析完成: {summary_metrics}")
        log_fn(f"📊 孔洞滑窗分析完成, 总孔洞数={total_vug_count}, 总面积={total_area_mm2:.2f} mm²")

    # ==== 全井整体分析（不分滑窗） ====
    try:
        resp_all = requests.post(
            "http://127.0.0.1:8011/analyze_vug",
            files={"file": open(mask_path, "rb")},
            data={
                "image_height_mm": str(image_height_mm),
                "image_width_mm": str(image_width_mm),
                "window_height_mm": str(image_height_mm)
            },
            timeout=60
        )
        resp_all.raise_for_status()
        all_vug_result = resp_all.json()
        summary_metrics["whole_vug_summary"] = all_vug_result.get("summary", {})
        if log_fn:
            log_fn(f"📊 全井孔洞总体分析完成: {summary_metrics['whole_vug_summary']}")
    except Exception as e:
        if log_fn:
            log_fn(f"⚠️ 全井孔洞总体分析失败: {e}")
        summary_metrics["whole_vug_summary"] = {}

    if log_fn:
        log_fn(f"📊 孔洞滑窗分析完成, 总孔洞数={total_vug_count}, 总面积={total_area_mm2:.2f} mm²")

    return window_vug_list, summary_metrics

# ===== SAM2 滑窗分析 =====
def sliding_window_sam2_analysis(image_path, image_height_mm, image_width_mm, log_fn=None):
    """
    滑窗调用 SAM2（prompt_free 模式），并拼接整体掩码
    """
    img = cv2.imread(image_path)
    H, W = img.shape[:2]
    window_px = W  # 与 U-Net 一致，纵向滑窗
    n_blocks = math.ceil(H / window_px)
    masks_full = np.zeros((H, W), dtype=np.uint8)
    curves_metrics_all = []

    for i in range(n_blocks):
        start_y = i * window_px
        end_y = min((i + 1) * window_px, H)
        img_block = img[start_y:end_y, :]
        block_path = f"temp_sam2_block_{uuid.uuid4().hex}.png"
        cv2.imwrite(block_path, img_block)

        try:
            # prompt_free 模式分割
            sam2_result = call_sam2_api(block_path, prompt_type="prompt_free")
            mask_b64 = sam2_result.get("mask", None)
            if not mask_b64:
                if log_fn:
                    log_fn(f"⚠️ SAM2 第 {i+1}/{n_blocks} 块未返回有效 mask")
                continue

            mask_bytes = base64.b64decode(mask_b64)
            #mask_block_path = f"sam2_block_mask_{uuid.uuid4().hex}.png"
            mask_block_path = os.path.join(temp_dir, f"sam2_block_mask_{uuid.uuid4().hex}.png")
            with open(mask_block_path, "wb") as f:
                f.write(mask_bytes)

            mask_block = cv2.imread(mask_block_path, cv2.IMREAD_GRAYSCALE)
            if mask_block is None:
                continue

            masks_full[start_y:end_y, :] = np.maximum(
                masks_full[start_y:end_y, :], mask_block
            )

            # 裂缝参数分析（同 U-Net）
            mask_path = preprocess_mask_for_analysis(mask_block_path, log_fn)
            single_masks = split_mask_to_contours(mask_path)
            for j, mask_single in enumerate(single_masks):
                temp_mask_path = f"temp_single_sam2_block_{i}_{j}.png"
                #temp_mask_path = os.path.join(temp_dir, f"temp_single_sam2_block_{i}_{j}.png")
                cv2.imwrite(temp_mask_path, mask_single)
                metrics = call_crack_api(temp_mask_path, image_height_mm, image_width_mm)
                metrics["y_offset"] = start_y
                curves_metrics_all.append(metrics)

            if log_fn:
                log_fn(f"✅ SAM2 滑窗块 {i+1}/{n_blocks} 分析完成，裂缝 {len(single_masks)} 条")

        except Exception as e:
            if log_fn:
                log_fn(f"❌ SAM2 滑窗块 {i+1}/{n_blocks} 失败: {e}")

    #mask_full_path = f"full_sam2_mask_{uuid.uuid4().hex}.png"
    mask_full_path = os.path.join(temp_dir, f"full_sam2_mask_{uuid.uuid4().hex}.png")
    cv2.imwrite(mask_full_path, masks_full)
    return mask_full_path, curves_metrics_all


# ===== 绘制最终结果（保持不变，使用 metrics["y_offset"]） =====
def draw_final_results(base_image_path, unet_results, yolo_results, H):
    base_img = Image.open(base_image_path).convert("RGB")
    img_np = np.array(base_img)
    draw = ImageDraw.Draw(base_img)

    # YOLO 框
    for det in yolo_results.get("detections", []):
        cls = det.get("class", "")
        conf = det.get("confidence", 0)
        bbox = list(map(int, det.get("bbox", [])))
        draw.rectangle(bbox, outline="red", width=2)
        draw.text((bbox[0], max(0, bbox[1]-12)), f"{cls} {conf:.2f}", fill="red")

    # U-Net 掩码 + 裂缝曲线
    for item in unet_results:
        cls = item.get("class", "").lower()
        mask_path = item.get("mask_result", {}).get("mask")
        if not mask_path:
            continue
        mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask_img is None:
            continue

        # 掩码轮廓
        contours, _ = cv2.findContours((mask_img>0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            pts = [tuple(pt[0]) for pt in contour]
            draw.line(pts, fill=(0, 255, 0), width=2)

        # 裂缝曲线
        if cls == "fracture":
            metrics_list = item.get("metrics_list", [])
            for metrics in metrics_list:
                if all(k in metrics for k in ["A", "B", "C", "D"]):
                    A = metrics["A"]
                    B = metrics["B"]
                    C = metrics["C"]
                    D = metrics["D"]
                    y_offset = metrics.get("y_offset", 0)
                    if None not in [A, B, C, D]:
                        w = img_np.shape[1]
                        x_fit = np.arange(w)
                        y_fit = A * np.sin(B * x_fit + C) + D + y_offset
                        points = [(int(x), int(y)) for x, y in zip(x_fit, y_fit) if 0 <= int(y) < H]
                        for i in range(len(points)-1):
                            draw.line([points[i], points[i+1]], fill=(255, 0, 255), width=3)

    out_path = base_image_path.replace(".png","_final.png")
    base_img.save(out_path)
    return out_path








