
from PIL import Image
import numpy as np
import cv2
from scipy.optimize import curve_fit
import math


def analyze_fracture(mask_img: Image.Image):
    mask = np.array(mask_img.convert("L"))  # 灰度图转数组
    binary_mask = (mask > 128).astype(np.uint8)

    area = int(np.sum(binary_mask))  # 裂缝面积 = 像素为1的数量
    length = estimate_fracture_length(binary_mask)  # 轮廓长度估算

    params, _, _ = fit_sine_to_mask(binary_mask)
    if params is None:
        return {
            "area": area,
            "length": length,
            "A": None, "B": None, "C": None, "D": None
        }

    A, B, C, D = params
    return {
        "area": area,
        "length": length,
        "A": round(A, 2),
        "B": round(B, 4),
        "C": round(C, 2),
        "D": round(D, 2)
    }


def estimate_fracture_length(mask):
    """简单估算裂缝长度（用最外轮廓长度）"""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0
    max_contour = max(contours, key=cv2.contourArea)
    arc_length = cv2.arcLength(max_contour, False)
    return round(arc_length, 2)


def fit_sine_to_mask(mask):
    mask = cv2.medianBlur(mask.astype(np.uint8), 3)

    rows, cols = mask.shape
    x_points = []
    y_points = []

    for x in range(0, cols, 3):
        column = mask[:, x]
        y_indices = np.where(column == 1)[0]
        if len(y_indices) > 0:
            x_points.append(x)
            y_points.append(y_indices.min())

    if len(x_points) < 10:
        return None, None, None

    x_data = np.array(x_points)
    y_data = np.array(y_points)

    y_amp = (y_data.max() - y_data.min()) / 2
    y_mean = (y_data.max() + y_data.min()) / 2
    initial_B = 2 * np.pi / (x_data.max() - x_data.min())

    def sine_function(x, A, B, C, D):
        return A * np.sin(B * x + C) + D

    try:
        params, _ = curve_fit(
            sine_function,
            x_data, y_data,
            p0=[y_amp, initial_B, 0, y_mean],
            maxfev=10000
        )
    except RuntimeError as e:
        print(f"拟合失败: {str(e)}")
        return None, None, None

    x_fit = np.arange(cols)
    y_fit = sine_function(x_fit, *params)
    return params, x_fit, y_fit
