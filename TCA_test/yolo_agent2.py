# yolo_agent.py
import requests

YOLO_API_URL = "http://localhost:2000/analyze"  # 若部署在远程，改为实际IP:端口

def call_yolo_api(image_path: str, params: dict = {}):
    """
    发送图像到 YOLO Flask API，返回检测结果（包含 class、bbox、confidence）
    :param image_path: 本地图像路径
    :param params: 可选参数，暂未启用（保留扩展）
    :return: dict {detections: [...]}
    """
    with open(image_path, 'rb') as f:
        files = {'image': f}
        try:
            response = requests.post(YOLO_API_URL, files=files, data=params)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            print(f"❌ 调用YOLO API失败：{e}")
            raise



