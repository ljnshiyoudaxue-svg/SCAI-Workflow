from api_caller import call_sam2_api
import base64
import requests
from PIL import Image, ImageDraw
from yolo_agent2 import call_yolo_api  # YOLO接口
import uuid
import matplotlib
matplotlib.use("Agg")
import math
from openai import OpenAI
import os
import cv2
import json
import numpy as np
#sk-8a5add1a6785414a9ff1b2653e760880
# ===== DeepSeek 初始化 =====
client = OpenAI(
    api_key="sk-8a5add1a6785414a9ff1b2653e760880",
    base_url="https://api.deepseek.com"
)
# ===== U-Net API 配置 =====
# API_URL = "http://127.0.0.1:1000/unet/unet_Vug/segment"
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
# 将 DeepSeek 输出 JSON 转换为 report_api 可用格式

def parse_deepseek_json(deepseek_json, px_to_m, start_depth_m):
    """
    解析 DeepSeek 智能体输出 JSON：
    - 裂缝：仅来自 U-Net
    - 孔洞：来自 vug_results
    - 深度统一由 px → m 换算
    """

    # ================= 裂缝解析（仅 U-Net） =================
    fractures = []

    for r in deepseek_json.get("unet_results", []):
        for m in r.get("metrics_list", []):
            y_offset = m.get("y_offset", 0)
            D_px = m.get("D", 0)

            depth_m = start_depth_m + (y_offset + D_px) * px_to_m

            fractures.append({
                "length_mm": m.get("length_mm", 0),
                "dip_angle_deg": m.get("倾角_deg", 0),
                "depth_m": round(depth_m, 3),
                "area_mm2": m.get("area_mm2", 0),
                "source": "unet"
            })

    # ================= 孔洞解析（正确版本） =================
    vugs = []

    vug_results = deepseek_json.get("vug_results")
    if vug_results:
        for v in vug_results.get("window_metrics", []):
            depth_start_mm = v.get("depth_start_mm", None)
            depth_end_mm = v.get("depth_end_mm", None)

            # 防御式校验
            if depth_start_mm is None or depth_end_mm is None:
                continue

            # 使用窗口中点作为代表深度（工程上最常用）
            depth_m = (depth_start_mm + depth_end_mm) / 2.0 / 1000.0  # mm → m

            vugs.append({
                "vug_count": v.get("vug_count", 0),
                "area_mm2": v.get("total_area_mm2", 0),
                "depth_m": round(depth_m, 3),
                "CVPA": v.get("CVPA", 0),
                "CDENS": v.get("CDENS", 0),
                "CSIZE": v.get("CSIZE", 0),
                "source": "unet_vug"
            })

    # ================= 汇总报告 =================
    report_json = {
        "timestamp": deepseek_json.get("timestamp"),
        "modules_used": ["YOLO", "UNet", "SAM2"],
        "params_used": deepseek_json.get("params_used", {}),
        "fractures": fractures,
        "vugs": vugs,
        "reliability_score": "高"
    }

    return report_json


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
        resp = requests.post("http://127.0.0.1:8017/analyze_crack", files=files, data=data)
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
    将 U-Net + SAM2 的掩码叠加到原图上，并使用不同颜色区分
    """
    base_img = cv2.imread(base_image_path)
    overlay = base_img.copy()

    for mask, model_id in zip(masks, model_ids):
        # 选择颜色：先查 U-Net，再查 SAM2，最后兜底黄色
        if model_id in MODEL_MAPPING:
            color = (0, 255, 0)
        elif model_id in SAM2_MAPPING:
            color = (0, 255, 0)
        else:
            color = (255, 255, 0)  # 默认黄色

        # 读入 mask 图像
        mask_img = cv2.imread(mask["mask"], cv2.IMREAD_GRAYSCALE)
        if mask_img is None:
            continue
        _, binary = cv2.threshold(mask_img, 1, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 画轮廓
        cv2.drawContours(overlay, contours, -1, color, thickness=cv2.FILLED)

    out_path = base_image_path.replace(".png", "_overlay.png")
    cv2.imwrite(out_path, overlay)
    return out_path



def deepseek_filter_curves_safe(curves_metrics, x_points_list, image_height_px, image_width_px,
                                min_points=200, max_retries=3, log_fn=None):
    """
    DeepSeek 曲线复核 + 拟合点数过滤（保证每条曲线都有分析记录）
    返回：
      curves_filtered: DeepSeek 判定有效的曲线列表
      analysis_log: 每条预处理曲线对应分析记录（长度 = len(pre_filtered)）
      pre_filtered: 预处理后保留的曲线列表
    """
    import os, json, re, threading
    from openai import OpenAI

    if not curves_metrics or not x_points_list:
        return [], [], []

    # ---- 预处理：拟合点数过滤 ----
    pre_filtered = [
        (i, m) for i, m in enumerate(curves_metrics)
        if i < len(x_points_list) and len(x_points_list[i]) >= min_points
    ]
    if not pre_filtered:
        if log_fn:
            log_fn("⚠️ 预处理结果为空，没有曲线满足 min_points 条件")
        return [], [], []

    if log_fn:
        log_fn(f"✅ 预处理完成，保留 {len(pre_filtered)} 条曲线: 索引列表 {[i for i,_ in pre_filtered]}")

    # ---- 限制最大复核曲线数（20条）----
    #if len(pre_filtered) > 20:
        #if log_fn:
            #log_fn(f"⚠️ 曲线数量 {len(pre_filtered)} 超过20，仅保留前20条复核")
        #pre_filtered = pre_filtered[:20]

    # ---- 提取 A/B/C/D 参数 ----
    metrics_simple = [
        {"A": m.get("A"), "B": m.get("B"), "C": m.get("C"), "D": m.get("D")}
        for _, m in pre_filtered
    ]

    # ---- UTF-8 安全处理 ----
    try:
        params_json = json.dumps(metrics_simple, ensure_ascii=False)
    except Exception as e:
        if log_fn:
            log_fn(f"⚠️ 参数编码失败: {e}")
        params_json = json.dumps(metrics_simple, ensure_ascii=True)

    # ---- 构造消息（使用 params_json，避免重复转义）----
    messages = [
        {
            "role": "system",
            "content": (
                "你是地质图像分析助手。"
                "请严格输出 JSON，不要有任何解释或额外文字。\n"
                "要求：对输入的每条曲线都生成一条日志（即使 invalid 也必须生成），在处理振幅参数A时，请确保取绝对值，因为振幅应该是正数。"
                "输出格式示例："
                "{'analysis_log':['曲线0: A=100 ✅','曲线1: A=600 ❌'],'valid_curves':[0],'invalid_curves':[1]}。"
                "判断标准：B <= 0.05 且 |A| <= 图像高度*2/3。"
            )
        },
        {
            "role": "user",
            "content": f"图像宽度={image_width_px}, 高度={image_height_px}, 参数={params_json}"
        }
    ]

    # ---- 调用 DeepSeek ----
    for attempt in range(1, max_retries + 1):
        try:
            if log_fn:
                log_fn(f"📡 调用 DeepSeek (尝试 {attempt})... 当前线程: {threading.current_thread().name}")

            # ✅ 每次重新初始化 Client（避免 AnyIO 线程冲突）
            local_client = OpenAI(
                api_key="sk-8a5add1a6785414a9ff1b2653e760880",
                base_url="https://api.deepseek.com"
            )

            resp = local_client.chat.completions.create(
                model="deepseek-chat",
                messages=messages,
                temperature=0
                  # ✅ 强制要求返回 JSON，防乱码
            )

            raw = getattr(resp.choices[0].message, "content", "").strip()
            if not raw:
                raise ValueError("返回内容为空")

            if log_fn:
                preview = raw[:500] + ('...' if len(raw) > 500 else '')
                log_fn(f"📝 DeepSeek原始回复 (尝试 {attempt}): {preview}")

            # ---- 尝试解析 JSON ----
            try:
                decision = json.loads(raw)
            except Exception:
                match = re.search(r"\{[\s\S]*\}", raw)
                if not match:
                    raise ValueError("找不到 JSON")
                decision = json.loads(match.group(0))

            valid_idx = decision.get("valid_curves", [])
            analysis_log = decision.get("analysis_log", [])

            # ---- 补全日志 ----
            if len(analysis_log) != len(pre_filtered):
                valid_set = set(valid_idx)
                full_log = []
                for idx, m in enumerate(pre_filtered):
                    A, B = m[1].get("A"), m[1].get("B")
                    valid_mark = "valid" if idx in valid_set else "invalid"
                    full_log.append(f"曲线{idx}: B={B}, A={A} => {valid_mark}")
                analysis_log = full_log

            curves_filtered = [pre_filtered[i][1] for i in valid_idx if i < len(pre_filtered)]
            return curves_filtered, analysis_log, pre_filtered

        except Exception as e:
            if log_fn:
                log_fn(f"⚠️ DeepSeek曲线复核尝试 {attempt} 失败: {e}")

    # ---- 兜底逻辑 ----
    if log_fn:
        log_fn("⚠️ 使用兜底逻辑，仅返回本地过滤结果")

    full_log = []
    curves_filtered = []
    for idx, m in pre_filtered:
        A = abs(m.get("A", 0))
        B = m.get("B", 1.0)  # 默认 B=1，确保不会误判
        if B <= 0.05 and A <= image_height_px / 3:
            valid_mark = "valid"
            curves_filtered.append(m)
        else:
            valid_mark = "invalid"
        full_log.append(f"曲线{idx}: B={B}, |A|={A} => {valid_mark}")

    return curves_filtered, full_log, pre_filtered


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
        #block_path = os.path.join(temp_dir, f"temp_block_{uuid.uuid4().hex}.png")
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
    #mask_full_path = os.path.join(temp_dir, f"full_mask_{uuid.uuid4().hex}.png")
    # ✅ 强制归一化为 0-255，保证显示效果与 SAM2 一致
    if masks_full.max() <= 1:
        masks_full = (masks_full * 255).astype(np.uint8)
    else:
        masks_full = masks_full.astype(np.uint8)

    cv2.imwrite(mask_full_path, masks_full)
    return mask_full_path, curves_metrics_all

def sliding_window_vug_analysis(
    image_path,
    model_id,
    image_height_mm,
    image_width_mm,
    window_height_mm=1000,   # 默认 1 m
    window_px=None,          # ✅ 支持显式像素滑窗
    log_fn=None
):
    """
    孔洞统一分析流程（与裂缝滑窗保持一致）：
    - 内部滑窗：像素
    - 统计单位：mm
    - 返回：mask_full_path, window_metrics, summary
    """

    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"未找到图像文件: {image_path}")

    H, W = img.shape[:2]

    # ===== px ↔ mm 映射 =====
    px_per_mm = H / image_height_mm
    mm_per_px = image_height_mm / H

    # ===== 滑窗优先级：window_px > window_height_mm =====
    if window_px is not None:
        window_px = int(window_px)
        window_height_mm = window_px * mm_per_px
    else:
        window_px = int(window_height_mm * px_per_mm)

    window_px = max(1, window_px)
    n_blocks = math.ceil(H / window_px)

    masks_full = np.zeros((H, W), dtype=np.uint8)

    window_vug_list = []
    total_vug_count = 0
    total_area_mm2 = 0
    all_CVPA, all_CDENS, all_CSIZE = [], [], []

    for i in range(n_blocks):
        start_y = i * window_px
        end_y = min(start_y + window_px, H)

        depth_start_mm = start_y * mm_per_px
        depth_end_mm = end_y * mm_per_px
        window_depth_mm = depth_end_mm - depth_start_mm

        img_block = img[start_y:end_y, :]
        block_path = f"temp_vug_block_{uuid.uuid4().hex}.png"
        cv2.imwrite(block_path, img_block)

        try:
            # ---------- ① U-Net 孔洞分割 ----------
            mask_result = call_unet_api(model_id, block_path)
            mask_block = cv2.imread(mask_result["mask"], cv2.IMREAD_GRAYSCALE)
            if mask_block is None:
                continue

            masks_full[start_y:end_y, :] = np.maximum(
                masks_full[start_y:end_y, :],
                mask_block
            )

            # ---------- ② 单滑窗孔洞统计 ----------
            cleaned_mask = preprocess_mask_for_analysis(mask_result["mask"], log_fn)

            result = call_vug_api(
                cleaned_mask,
                image_height_mm=window_depth_mm,
                image_width_mm=image_width_mm,
                window_height_mm=window_depth_mm
            )

            summary = result.get("summary", {})
            vugs = result.get("vugs", [])

            window_info = {
                "depth_start_mm": round(depth_start_mm, 2),
                "depth_end_mm": round(depth_end_mm, 2),
                "vug_count": summary.get("vug_count", 0),
                "total_area_mm2": summary.get("total_area_mm2", 0),
                "CVPA": summary.get("CVPA", 0),
                "CDENS": summary.get("CDENS", 0),
                "CSIZE": summary.get("CSIZE", 0),
                "vugs": vugs
            }

            window_vug_list.append(window_info)

            total_vug_count += window_info["vug_count"]
            total_area_mm2 += window_info["total_area_mm2"]
            all_CVPA.append(window_info["CVPA"])
            all_CDENS.append(window_info["CDENS"])
            all_CSIZE.append(window_info["CSIZE"])

            if log_fn:
                log_fn(f"✅ 孔洞滑窗 {i+1}/{n_blocks} 完成")

        except Exception as e:
            if log_fn:
                log_fn(f"⚠️ 分块孔洞分析失败: {e}")

    # ---------- 保存整图 mask ----------
    mask_full_path = f"full_mask_vug_{uuid.uuid4().hex}.png"
    if masks_full.max() <= 1:
        masks_full = (masks_full * 255).astype(np.uint8)
    else:
        masks_full = masks_full.astype(np.uint8)
    cv2.imwrite(mask_full_path, masks_full)

    # ---------- 整井汇总 ----------
    summary_metrics = {
        "total_vug_count": total_vug_count,
        "total_area_mm2": total_area_mm2,
        "mean_CVPA": np.mean(all_CVPA) if all_CVPA else 0,
        "mean_CDENS": np.mean(all_CDENS) if all_CDENS else 0,
        "mean_CSIZE": np.mean(all_CSIZE) if all_CSIZE else 0
    }

    if log_fn:
        log_fn(
            f"📊 孔洞滑窗分析完成: "
            f"总孔洞数={total_vug_count}, 总面积={total_area_mm2:.2f} mm²"
        )

    return mask_full_path, window_vug_list, summary_metrics




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
            mask_block_path = f"sam2_block_mask_{uuid.uuid4().hex}.png"
            #mask_block_path = os.path.join(temp_dir, f"sam2_block_mask_{uuid.uuid4().hex}.png")
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

    mask_full_path = f"full_sam2_mask_{uuid.uuid4().hex}.png"
    #mask_full_path = os.path.join(temp_dir, f"full_sam2_mask_{uuid.uuid4().hex}.png")
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
                            draw.line([points[i], points[i+1]], fill=(0, 0, 0), width=2)

    out_path = base_image_path.replace(".png","_final.png")
    base_img.save(out_path)
    return out_path
# ===== DeepSeek 决策 =====
def deepseek_decide_models(user_prompt, yolo_results):
    """
    根据用户输入与YOLO检测结果自动选择分割模型。
    优先级：
        1. 用户输入包含“裂缝”“孔洞”“诱导缝” → 强制对应模型
        2. 用户输入含“检测图片”“分析图片”等 → 根据YOLO结果自动匹配
        3. 若未检测到相关对象 → 默认unet_Fracture
    """

    user_prompt_lower = user_prompt.strip().lower()

    # === 1️⃣ 用户直接指定的情况 ===
    if "裂缝" in user_prompt_lower and "图片" not in user_prompt_lower:
        return ["unet_Fracture"], {}
    elif "孔洞" in user_prompt_lower and "图片" not in user_prompt_lower:
        return ["unet_Vug"], {}
    elif "诱导" in user_prompt_lower and "图片" not in user_prompt_lower:
        return ["unet_Induced_Fracture"], {}

    # === 2️⃣ 构造 DeepSeek 提示 ===
    system_prompt = """你是一位图像分割智能助手，负责根据用户的自然语言意图与YOLO检测结果自动选择最合适的地质分割模型。
你的任务：
1. 如果用户输入中出现“裂缝”，仅使用 ["unet_Fracture"]；
2. 如果出现“孔洞”，仅使用 ["unet_Vug"]；
3. 如果出现“诱导缝”或“钻井诱导裂缝”，仅使用 ["unet_Induced_Fracture"]；
4. 如果用户输入为“检测图片”或“分析图片”，则根据 YOLO 检测结果自动匹配：
   - YOLO检测结果中包含 Fracture → 使用 ["unet_Fracture"]
   - YOLO检测结果中包含 Induced_Fracture → 使用 ["unet_Induced_Fracture"]
   - YOLO检测结果中包含 Vug → 使用 ["unet_Vug"]
5. 如果检测结果包含多种类型，可同时输出多个模型；
6. 如果均不符合条件，则返回 ["unet_Fracture"]。
输出格式固定为 JSON：
{
  "models": ["unet_Fracture", "unet_Vug"],
  "parameters": {}
}
请只输出JSON内容，不要解释。"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"用户输入: {user_prompt}\nYOLO结果: {yolo_results}"}
    ]

    # === 3️⃣ 调用 DeepSeek ===
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            temperature=0.2,
            max_tokens=300
        )
        raw_content = response.choices[0].message.content.strip()

        # 尝试解析JSON
        decision = json.loads(raw_content)
        models_raw = decision.get("models", [])
        allowed_models = ["unet_Fracture", "unet_Induced_Fracture", "unet_Vug"]

        models = [m for m in models_raw if m in allowed_models]
        if not models:
            models = ["unet_Fracture"]  # 兜底

        return models, decision.get("parameters", {})

    except Exception as e:
        print(f"❌ DeepSeek解析失败: {e}")
        return ["unet_Fracture"], {}











