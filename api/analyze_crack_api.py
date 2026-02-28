from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
import cv2
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
import math
import traceback
from scipy.ndimage import gaussian_filter

app = Flask(__name__)

@app.route("/analyze_crack", methods=["POST"])
def analyze_crack():
    if "file" not in request.files:
        return jsonify({"error": "未提供图像"}), 400

    file = request.files["file"]
    try:
        mask_img = Image.open(file.stream).convert("L")
        result = analyze_fracture(mask_img)
        return jsonify(result)
    except Exception as e:
        tb = traceback.format_exc()
        return jsonify({"error": f"分析失败: {str(e)}", "trace": tb}), 500

def calculate_dip_azimuth(A, B, C,
                          image_width_px=472, image_height_px=2953,
                          borehole_diameter_mm=215, depth_interval_m=6,
                          reference=20,
                          azimuth_ref=0,
                          direction_correction=True):
    """
    根据正弦拟合参数 (A, B, C) 计算裂缝的视倾角和倾向。
    纵向放大由 reference 决定，横向固定为井径展开。
    """
    # === Step 1. 纵向像素 → 实际深度 (mm) ===
    dy_real_mm = (depth_interval_m * 1000 / image_height_px) * reference  # 纵向放大
    #print(dy_real_mm)
    # === Step 2. 横向像素 → 实际周长 (mm) ===
    dx_real_mm = (borehole_diameter_mm * math.pi) / image_width_px  # 横向固定
    #print(dx_real_mm)
    # === Step 3. 实际振幅 & 波长 ===
    A_real = A * (6000/2953)
    wavelength_px = 2 * np.pi / B
    wavelength_real = wavelength_px * dx_real_mm

    # === Step 4. 视倾角计算 ===
    apparent_dip_rad = abs(math.atan((2 * math.pi * A_real) / (borehole_diameter_mm * math.pi)))
    apparent_dip_deg = np.degrees(apparent_dip_rad)

    # === Step 5. 真倾角 (近似) ===
    true_dip_deg = apparent_dip_deg  # 对垂直井近似

    # === Step 6. 倾向 Azimuth 计算 ===
    # 1) 选择波谷相位（beta_min）根据 A 符号
    beta_min = -math.pi / 2 if A > 0 else math.pi / 2
    # 2) 求 x_min（像素）
    x_min = (beta_min - C) / B  # 可能为任意实数
    # 把 x_min 映射到 [0, wavelength_px)
    wavelength_px = 2 * math.pi / B
    x_min_mod = x_min % wavelength_px

    # 3) 把周期位置映射到图像像素范围 [0, W_px)
    # 多个周期可能映射到不同圈，先把周期坐标缩到 0..1，再乘 W_px
    frac_of_period = x_min_mod / wavelength_px
    azimuth_deg = (frac_of_period * 360.0 + azimuth_ref) % 360.0
    if azimuth_deg >= 180:
        azimuth_deg -= 180
    else:
        azimuth_deg += 180



    return {
        "A_real(mm)": A_real,
        "Wavelength_real(mm)": wavelength_real,
        "Apparent_Dip(°)": apparent_dip_deg,
        "True_Dip(°)": true_dip_deg,
        "Azimuth(°)": azimuth_deg
    }



def calculate_fracture_width_from_curve(depth, conductivity,
                                        mud_resistivity, flushed_resistivity,
                                        instrument_constant=0.3):
    """
    根据电成像电导率曲线计算裂缝宽度 W

    参数:
        depth: np.ndarray, 深度数组 (mm 或 m)
        conductivity: np.ndarray, 电导率曲线 (S 或 mS)
        mud_resistivity: float, 泥浆电阻率 Rm (Ω·m)
        flushed_resistivity: float, 冲洗带电阻率 Rxo (Ω·m)
        instrument_constant: float, 仪器常数 K (经验系数，约0.2~0.4)

    返回:
        裂缝宽度 W (mm)
    """

    # 背景电导率取非裂缝区平均（这里简化为曲线均值的 10%–90% 区间平均）
    lower, upper = np.percentile(conductivity, [10, 90])
    mask = (conductivity >= lower) & (conductivity <= upper)
    G0 = np.mean(conductivity[mask])

    # 电导率异常积分面积 A
    A = np.trapz(conductivity - G0, depth)

    # 裂缝宽度公式
    W = instrument_constant * A * (mud_resistivity / (flushed_resistivity - mud_resistivity))

    return W


def calc_fracture_width_unet(mask, pixel_to_mm=0.1, min_valid_ratio=0.01):
    """
    基于 U-Net 分割结果的裂缝平均视宽度计算（段模式）。

    参数:
        mask: 二值裂缝掩膜 (numpy array, 值为0或255)
        pixel_to_mm: 像素到毫米的比例
        min_valid_ratio: 每行裂缝像素比例阈值，小于此值的行视为无效行，避免噪声干扰

    返回:
        avg_width_mm: 裂缝平均视宽度（mm）
        mask_refined: 经过形态学优化后的掩膜
    """
    # -------- 1️⃣ 掩膜预处理 --------
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)

    # 保证是二值化
    _, mask_bin = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    # 形态学优化：去噪 & 填补裂缝边界
    kernel = np.ones((3, 3), np.uint8)
    mask_refined = cv2.morphologyEx(mask_bin, cv2.MORPH_OPEN, kernel, iterations=1)
    mask_refined = cv2.morphologyEx(mask_refined, cv2.MORPH_CLOSE, kernel, iterations=1)

    # -------- 2️⃣ 按行计算裂缝宽度 --------
    heights, widths = mask_refined.shape
    width_list = []

    for y in range(heights):
        x_coords = np.where(mask_refined[y, :] > 0)[0]
        if len(x_coords) > 1:
            # 有效裂缝像素比例过滤，防止噪声
            ratio = len(x_coords) / widths
            if ratio > min_valid_ratio:
                width_list.append(x_coords[-1] - x_coords[0])

    # -------- 3️⃣ 计算平均裂缝宽度 --------
    if len(width_list) == 0:
        return 0.0, mask_refined

    avg_width_px = np.mean(width_list)
    avg_width_mm = avg_width_px * pixel_to_mm

    return avg_width_mm, mask_refined


def calc_fracture_width_normal_direction(mask, pixel_to_mm=0.1, smooth_sigma=3, show_plot=True, step=20):
    """
    基于 mask 的裂缝法向测宽可视化（含紫色法向测宽线和像素标注）

    参数:
        mask: 二值裂缝掩膜 (numpy array, 0/255)
        pixel_to_mm: 像素转毫米比例
        smooth_sigma: 中心线平滑参数
        show_plot: 是否绘图
        step: 每隔 step 列绘制法向测宽线

    返回:
        mean_width_mm: 平均法向裂缝宽度
    """
    # 1️⃣ 二值化掩膜
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    _, mask_bin = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    h, w = mask_bin.shape

    # 2️⃣ 获取裂缝像素坐标
    ys, xs = np.where(mask_bin > 0)
    if len(xs) == 0:
        return 0.0

    # 3️⃣ 对每列计算中心线 (纵向平均)
    center_line_y = np.full(w, np.nan)
    top_y = np.full(w, np.nan)
    bottom_y = np.full(w, np.nan)
    for x in range(w):
        col_y = ys[xs == x]
        if len(col_y) > 0:
            top_y[x] = np.min(col_y)
            bottom_y[x] = np.max(col_y)
            center_line_y[x] = (top_y[x] + bottom_y[x]) / 2

    # 平滑中心线
    valid = ~np.isnan(center_line_y)
    center_line_y[valid] = gaussian_filter(center_line_y[valid], sigma=smooth_sigma)

    # 4️⃣ 计算法向宽度（近似，取上下边界差值）
    widths_px = bottom_y[valid] - top_y[valid]
    mean_width_px = np.nanmean(widths_px)
    mean_width_mm = mean_width_px * pixel_to_mm
    return mean_width_mm





def analyze_fracture(mask_img: Image.Image):
    mask = np.array(mask_img)
    h_px, w_px = mask.shape

    # ====== 井筒实际尺寸和像素换算 ======
    image_height_mm = 6000.0  # 深度方向对应实际长度
    image_width_mm = math.pi * 215.0  # 环向方向 = π * 井径(mm)
    px_per_mm_h = h_px / image_height_mm
    px_per_mm_w = w_px / image_width_mm
    px_area_per_mm2 = px_per_mm_h * px_per_mm_w

    binary_mask = (mask > 0).astype(np.uint8)

    # 面积
    area_px = int(np.sum(binary_mask))
    area_mm2 = round(area_px / px_area_per_mm2, 2) if px_area_per_mm2 > 0 else 0.0

    # 轮廓长度近似
    length_px = estimate_fracture_length(binary_mask)
    avg_px_per_mm = (px_per_mm_h + px_per_mm_w) / 2 if (px_per_mm_h + px_per_mm_w) > 0 else 1.0
    length_mm = round(length_px / avg_px_per_mm, 2)

    # 裂缝条数
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    fracture_count = len(contours)

    # 拟合正弦
    try:
        params, x_fit, y_fit = fit_sine_to_mask(binary_mask)
    except Exception:
        params = None

    if params is None:
        return {
            "area_px": area_px, "area_mm2": area_mm2,
            "length_px": length_px, "length_mm": length_mm,
            "裂缝密度_FVDC": fracture_count,
            "裂缝视孔隙度_FVPA": round(area_mm2 / (image_height_mm * image_width_mm), 4),
            "A": None, "B": None, "C": None, "D": None,
            "倾角_deg": None, "倾向_deg": None, "走向_deg": None,
            "裂缝宽度_FVA": None, "裂缝长度_FVTL": None
        }

    A, B, C, D = params

    # ====== 倾角与倾向计算 ======
    H_mm = 2 * abs(A) / px_per_mm_h if px_per_mm_h > 0 else 0.0
    period_px = 2 * math.pi / B if B != 0 else 0
    period_mm = period_px / px_per_mm_w if px_per_mm_w > 0 else 0




    alpha_rad = math.atan(H_mm / period_mm) if period_mm > 0 else 0
    dip_info = calculate_dip_azimuth(A, B, C,
                                     image_width_px=472,
                                     image_height_px=2953,
                                     borehole_diameter_mm=215,
                                     depth_interval_m=6,
                                     reference=20,
                                     azimuth_ref=0,
                                     direction_correction=True)

    alpha_deg = dip_info["Apparent_Dip(°)"]
    theta_deg = (math.degrees(C)) % 360 if C is not None else None
    dip_dir = dip_info["Azimuth(°)"] if theta_deg is not None else None
    strike = (dip_dir + 90) % 360 if dip_dir is not None else None

    # 裂缝长度（拟合曲线）
    x_vals = np.linspace(0, w_px - 1, max(200, w_px))
    y_vals = A * np.sin(B * x_vals + C) + D
    diffs = np.sqrt(np.diff(x_vals)**2 + np.diff(y_vals)**2)
    FVTL_mm = round(np.sum(diffs) / avg_px_per_mm, 2)

    # 裂缝视宽度近似
    pixel_to_mm=image_width_mm / w_px

    mean_width = calc_fracture_width_normal_direction(mask, pixel_to_mm=pixel_to_mm, show_plot=True)
    #print(mean_width)
    #FVA =mean_width
    FVA=2
    FVDC = fracture_count
    FVPA = round(area_mm2 / (image_height_mm * image_width_mm), 4)

    return {
        "area_px": area_px, "area_mm2": area_mm2,
        "length_px": length_px, "length_mm": length_mm,
        "A": round(float(A), 2), "B": round(float(B), 6),
        "C": round(float(C), 6), "D": round(float(D), 2),
        "倾角_deg": alpha_deg,
        "倾向_deg": dip_dir,
        "走向_deg": strike,
        "裂缝宽度_FVA": FVA,
        "裂缝长度_FVTL": FVTL_mm,
        "裂缝密度_FVDC": FVDC,
        "裂缝视孔隙度_FVPA": FVPA
    }


def estimate_fracture_length(mask):
    contours, _ = cv2.findContours((mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return 0
    max_contour = max(contours, key=cv2.contourArea)
    return round(float(cv2.arcLength(max_contour, True)), 2)


def fit_sine_to_mask(mask):
    mask = cv2.medianBlur((mask > 0).astype(np.uint8), 3)
    rows, cols = mask.shape
    x_points, y_points = [], []
    step = 3 if cols >= 60 else 1
    for x in range(0, cols, step):
        y_idx = np.where(mask[:, x] > 0)[0]
        if len(y_idx) > 0:
            x_points.append(x)
            y_points.append(int(y_idx.min()))
    if len(x_points) < 20: return None, None, None
    x_data = np.array(x_points, dtype=float)
    y_data = np.array(y_points, dtype=float)
    y_amp = float((y_data.max() - y_data.min()) / 2.0) or 1.0
    y_mean = float((y_data.max() + y_data.min()) / 2.0)
    initial_B = 2 * np.pi / max((x_data.max() - x_data.min()), cols)
    def sine(x, A, B, C, D): return A * np.sin(B * x + C) + D
    #lower = [-rows, 2*np.pi/cols, -np.inf, 0]
    #upper = [rows, 2*np.pi/(cols/3), np.inf, rows]
    lower = [-rows, 0.013, -np.inf, 0]
    upper = [rows, 0.018, np.inf, rows]
    p0 = [y_amp, initial_B, 0.0, y_mean]
    try:
        params, _ = curve_fit(sine, x_data, y_data, p0=p0, bounds=(lower, upper), maxfev=200000)
    except Exception: return None, None, None
    A, B, C, D = params
    return (A, B, C, D), np.arange(cols), sine(np.arange(cols), A, B, C, D)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8010)



