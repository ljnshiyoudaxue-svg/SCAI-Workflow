from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
import cv2
import math

app = Flask(__name__)

@app.route("/analyze_vug", methods=["POST"])
def analyze_vug():
    if "file" not in request.files:
        return jsonify({"error": "未提供掩码图文件"}), 400

    try:
        image_height_mm = float(request.form.get("image_height_mm"))
        image_width_mm = float(request.form.get("image_width_mm"))
        window_height_mm = float(request.form.get("window_height_mm"))  # 统计窗口深度
    except:
        return jsonify({"error": "请提供 image_height_mm, image_width_mm, window_height_mm 参数"}), 400

    file = request.files["file"]
    try:
        mask_img = Image.open(file.stream).convert("L")
        result = analyze_vug_mask(mask_img, image_height_mm, image_width_mm, window_height_mm)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": f"分析失败: {str(e)}"}), 500


def analyze_vug_mask(mask_img, image_height_mm, image_width_mm, window_height_mm):
    mask = np.array(mask_img)
    h_px, w_px = mask.shape
    px_per_mm_h = h_px / image_height_mm
    px_per_mm_w = w_px / image_width_mm
    px_area_per_mm2 = px_per_mm_h * px_per_mm_w

    binary_mask = (mask > 128).astype(np.uint8)
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    total_area_mm2 = 0
    vug_count = 0
    vug_areas = []
    vugs = []

    for contour in contours:
        area_px = cv2.contourArea(contour)
        if area_px < 5:
            continue

        vug_count += 1
        area_mm2 = area_px / px_area_per_mm2
        total_area_mm2 += area_mm2
        vug_areas.append(area_mm2)

        perimeter = cv2.arcLength(contour, True)
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = round(w / h, 2) if h != 0 else 0
        circularity = round(4 * math.pi * area_px / (perimeter ** 2), 3) if perimeter != 0 else 0
        equiv_radius = round(math.sqrt(area_mm2 / math.pi), 2)

        orientation = None
        if len(contour) >= 5:
            ellipse = cv2.fitEllipse(contour)
            orientation = round(ellipse[2], 1)

        vugs.append({
            "area_mm2": round(area_mm2, 2),
            "equiv_radius_mm": equiv_radius,
            "aspect_ratio": aspect_ratio,
            "circularity": circularity,
            "bbox": [int(x), int(y), int(x + w), int(y + h)],
            "orientation_deg": orientation
        })

    # 统计参数
    image_width_mm2 = image_width_mm
    CVPA = round(total_area_mm2 / (image_width_mm2 * window_height_mm), 5)  # 面孔率
    CDENS = round(vug_count / (window_height_mm / 1000), 3)  # 个数/米
    CSIZE = round(np.mean(vug_areas), 2) if vug_areas else 0

    return {
        "vugs": vugs,
        "summary": {
            "vug_count": vug_count,
            "total_area_mm2": round(total_area_mm2, 2),
            "CVPA": CVPA,
            "CDENS": CDENS,
            "CSIZE": CSIZE
        }
    }

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8011)

