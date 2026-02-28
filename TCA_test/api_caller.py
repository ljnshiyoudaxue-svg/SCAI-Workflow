import json
import requests

SAM2_API_URL = "http://127.0.0.1:3000/predict"
def call_sam2_api(image_path: str, prompt_type: str = "box", box_coords: list = None, mask_base64: str = None):
    """
    调用 SAM2 Flask API
    :param image_path: 图像路径
    :param prompt_type: 'box', 'mask', 'prompt_free'
    :param box_coords: [x_min, y_min, x_max, y_max] (仅 box)
    :param mask_base64: mask base64 (仅 mask)
    """
    with open(image_path, "rb") as f:
        files = {"file": (image_path, f, "image/jpeg")}
        data = {"prompt_type": prompt_type}

        if prompt_type == "box":
            if box_coords is None or len(box_coords) != 4:
                raise ValueError("box 类型必须提供 box_coords")
            data["box_coords"] = str(box_coords)

        elif prompt_type == "mask":
            if not mask_base64:
                raise ValueError("mask 类型必须提供 mask_base64")
            data["mask"] = mask_base64

        elif prompt_type == "prompt_free":
            pass

        else:
            raise ValueError(f"Unsupported prompt_type: {prompt_type}")

        resp = requests.post("http://127.0.0.1:3000/predict", files=files, data=data)
        resp.raise_for_status()
        return resp.json()





