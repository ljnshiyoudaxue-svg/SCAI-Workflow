from flask import Flask, request, jsonify
import torch
import cv2
import numpy as np
import base64
from unet import Unet  # 确保路径正确
import torch.nn as nn

app = Flask(__name__)

# 模型配置（修改为实际路径）
MODEL_CONFIG = {
    "unet_Fracture": {
        "model_path": "D:/360Downloads/SCAI/api/weights/unet_Fracture/best_epoch_weights.pth",
        "num_classes": 2,
        "backbone": "resnet50",
        "input_shape": [640, 640],
        "cuda": True
    },
    "unet_Induced_Fracture": {
        "model_path": "D:/360Downloads/SCAI/api/weights/unet_InducedFracture/best_epoch_weights.pth",
        "num_classes": 2,
        "backbone": "resnet50",
        "input_shape": [640, 640],
        "cuda": True
    },
    "unet_Vug": {
        "model_path": "D:/360Downloads/SCAI/api/weights/unet_Vug/best_epoch_weights.pth",
        "num_classes": 2,
        "backbone": "resnet50",
        "input_shape": [640, 640],
        "cuda": True
    }
}

# 全局模型缓存
models = {}


def load_model(model_id):
    """加载并缓存模型"""
    global models

    if model_id in models:
        return models[model_id]

    cfg = MODEL_CONFIG.get(model_id)
    if not cfg:
        print(f"[Error] 无效模型ID: {model_id}")
        return None

    try:
        # 使用 unet2 的 Unet 类初始化（会自动加载权重和设置设备）
        model = Unet(
            model_path=cfg['model_path'],
            num_classes=cfg['num_classes'],
            backbone=cfg['backbone'],
            input_shape=cfg['input_shape'],
            cuda=cfg['cuda']
        )
        models[model_id] = model
        print(f"✅ 成功加载模型: {model_id}")
        return model

    except Exception as e:
        print(f"❌ 模型加载失败: {str(e)}")
        return None


# ----------------- 健康检查接口 -----------------
@app.route('/')
def home():
    """健康检查接口，返回服务状态和可用模型列表"""
    return jsonify({
        "status": "U-Net API is running 🚀",
        "available_models": list(MODEL_CONFIG.keys()),
        "usage": "POST /unet/<model_id>/segment with form-data {roi: <image_file>}"
    }), 200


# ----------------- 分割接口 -----------------
@app.route('/unet/<model_id>/segment', methods=['POST'])
def segment(model_id):
    if model_id not in MODEL_CONFIG:
        return jsonify({"error": "无效模型ID"}), 400

    # 确保模型已加载
    model = load_model(model_id)
    if not model:
        return jsonify({"error": "模型加载失败"}), 500

    try:
        # 读取输入图像
        file = request.files['roi'].read()
        img = cv2.imdecode(np.frombuffer(file, np.uint8), cv2.IMREAD_COLOR)
        original_h, original_w = img.shape[:2]

        # 使用配置中的输入尺寸
        input_shape = MODEL_CONFIG[model_id]['input_shape']

        # 预处理
        img_resized = cv2.resize(img, (input_shape[1], input_shape[0]))
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img_tensor = torch.from_numpy(img_rgb).float().permute(2, 0, 1).unsqueeze(0) / 255
        img_tensor = img_tensor.to(model.device)

        # 推理
        with torch.no_grad():
            output = model.net(img_tensor)
            mask = torch.argmax(output, dim=1).squeeze().cpu().numpy()

        # 使用 Unet 内部的固定调色板
        colors = model.colors
        mask_img = np.zeros((input_shape[0], input_shape[1], 3), dtype=np.uint8)
        for i in range(model.num_classes):
            mask_img[mask == i] = colors[i]

        # 还原到原图尺寸
        mask_resized = cv2.resize(mask_img, (original_w, original_h), interpolation=cv2.INTER_NEAREST)

        # 编码为 base64
        _, buffer = cv2.imencode('.png', mask_resized)
        return jsonify({
            "mask": base64.b64encode(buffer).decode('utf-8'),
            "classes": model.num_classes,
            "input_size": input_shape
        })

    except Exception as e:
        return jsonify({"error": f"处理失败: {str(e)}"}), 500


if __name__ == '__main__':
    app.run(port=7000, host='0.0.0.0',debug=True)
