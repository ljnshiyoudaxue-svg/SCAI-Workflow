# yolo_api.py - 输出三种prompt (box, mask, prompt_free)

from flask import Flask, request, jsonify
from ultralytics import YOLO
import cv2
import numpy as np
import base64

# ======================
# 加载 YOLOv9 模型
# ======================
model_path = "D:/360Downloads/SCAI/api/weights/yolo/best.pt"
model = YOLO(model_path)

# 初始化 Flask 应用
app = Flask(__name__)

@app.route('/')
def home():
    return "Welcome to the YOLOv9 Flask API!"

@app.route('/analyze', methods=['POST'])
def analyze():
    # 检查是否有文件上传
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    # 读取图片文件
    file = request.files['image']
    image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)

    # YOLO 推理
    results = model(image)

    detections = []
    for result in results:
        # 遍历每个目标 (保证 box 和 mask 一一对应)
        for box, mask in zip(result.boxes, result.masks.data):
            # 边界框 (原图坐标)
            bbox = box.xyxy.tolist()[0]
            cls_name = model.names[int(box.cls)]
            conf = float(box.conf)

            # ========= 修改点：保证 mask 尺寸和原图一致 =========
            mask_np = mask.cpu().numpy().astype(np.uint8)  # [h, w], 0/1
            mask_resized = cv2.resize(
                mask_np,
                (image.shape[1], image.shape[0]),  # (W, H)
                interpolation=cv2.INTER_NEAREST
            )
            mask_resized = (mask_resized * 255).astype(np.uint8)

            # 编码 PNG -> base64
            _, mask_buf = cv2.imencode(".png", mask_resized)
            mask_b64 = base64.b64encode(mask_buf).decode("utf-8")

            # 保存检测结果
            detections.append({
                "class": cls_name,
                "confidence": conf,
                "bbox": bbox,        # 原图坐标系
                "mask": mask_b64     # 与原图尺寸一致
            })

    return jsonify({"detections": detections})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=2000, debug=True)
