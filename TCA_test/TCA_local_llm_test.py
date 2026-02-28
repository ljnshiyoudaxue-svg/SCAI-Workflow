from TCA_local_llm import segment_image_TCA
from PIL import Image
import os
# ============================
# ✅ 测试入口
# ============================
if __name__ == "__main__":
    test_image = "D:/360Downloads/ultralytics-main/Fracture7.png"
    if not os.path.exists(test_image):
        print("⚠️ 请放置测试图片到 Fracture7.png")
    else:
        segment_image_TCA(
            user_prompt="请分析这张电成像图像的裂缝",
            image_file=Image.open(test_image),
            image_height_mm=2500,
            image_width_mm=215,
            enable_prompt_mode="hard-constraint",
            enable_sam2=True,
            enable_reflection=True,
            multi_model=True,
            clean_temp=False
        )

