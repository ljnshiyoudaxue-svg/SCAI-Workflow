from flask import Flask, request, jsonify
import cv2
import numpy as np
import torch
import base64
from threading import Lock
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

app = Flask(__name__)

# 全局模型和权重锁
predictor = None
model_lock = Lock()
current_model_type = None

def init_model(model_type="fracture"):
    """根据模型类型初始化模型"""
    global predictor, current_model_type
    
    if predictor is not None and current_model_type == model_type:
        return
    
    weight_mapping = {
        "fracture": "./weights/sam2.1_hiera_s_fracture.pt",
        "vug": "./weights/sam2.1_hiera_s_vug.pt"
    }
    checkpoint_path = weight_mapping.get(model_type)
    if not checkpoint_path:
        raise ValueError(f"Invalid model type: {model_type}")
    
    sam2_checkpoint = "./weights/sam2.1_hiera_small.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_s.yaml"
    sam2_model = build_sam2(model_cfg, sam2_checkpoint, device="cuda")
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cuda", weights_only=True)
        sam2_model.load_state_dict(checkpoint)
        print(f"成功加载 {model_type} 模型权重")
    except Exception as e:
        print(f"自定义权重加载失败: {str(e)}，使用基础模型")
    
    predictor = SAM2ImagePredictor(sam2_model)
    current_model_type = model_type

# 应用启动时初始化模型
with app.app_context():
    init_model()

def predict_with_prompt(image, prompt_type, params):
    predictor.set_image(image)

    if prompt_type == "box":
        input_box = np.array([params["box"]], dtype=np.float32)
        masks, scores, _ = predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_box,
            multimask_output=True
        )

    elif prompt_type == "mask":
        mask_base64 = params.get("mask", "")
        mask_bytes = base64.b64decode(mask_base64)
        mask_array = cv2.imdecode(np.frombuffer(mask_bytes, np.uint8), cv2.IMREAD_GRAYSCALE)

        # === 保证 mask 和 image 尺寸一致 ===
        if mask_array.shape[:2] != image.shape[:2]:
            mask_array = cv2.resize(mask_array, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

        mask_array = (mask_array > 127).astype(np.float32)[None, :, :]  # [1,H,W]

        # === 空 mask 检查 ===
        if mask_array.sum() == 0:
            raise ValueError("收到空 mask，跳过该目标")

        print(f"[DEBUG] SAM2 输入尺寸：image={image.shape}, mask={mask_array.shape}")

        masks, scores, _ = predictor.predict(
            point_coords=None,
            point_labels=None,
            box=None,
            mask_input=mask_array,
            multimask_output=True
        )

    elif prompt_type == "prompt_free":
        masks, scores, _ = predictor.predict(
            point_coords=None,
            point_labels=None,
            box=None,
            multimask_output=True
        )
    else:
        raise ValueError(f"Unsupported prompt_type: {prompt_type}")

    return masks, scores


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "Missing image file"}), 400

    file = request.files["file"]
    image_bytes = file.read()
    image_array = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)

    prompt_type = request.form.get("prompt_type", "prompt_free")

    params = {}
    if prompt_type == "box":
        box_coords = request.form.get("box_coords", "")
        box_coords = list(map(float, box_coords.strip("[]").split(",")))
        if len(box_coords) != 4:
            return jsonify({"error": "Invalid box_coords format"}), 400
        params = {"box": box_coords}

    elif prompt_type == "mask":
        params = {"mask": request.form.get("mask", "")}

    try:
        masks, scores = predict_with_prompt(image_array, prompt_type, params)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    # 只返回第一个 mask
    mask = masks[0].astype(np.uint8) * 255
    _, mask_png = cv2.imencode(".png", mask)
    mask_base64 = base64.b64encode(mask_png.tobytes()).decode("utf-8")

    return jsonify({
        "prompt_type": prompt_type,
        "score": float(scores[0]),
        "mask": mask_base64
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=3000)

