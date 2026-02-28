````markdown
# SCAI
Semantic-constrained framework for reproducible fracture–vug interpretation from micro-resistivity imaging logs.

## Overview
This repository contains code developed for the study:

**Article**: A Semantic-Constrained Computational Framework for Reproducible Fracture–Vug Interpretation from Micro-Resistivity Imaging Logs  
**Corresponding Author**: Jianing Li

The pipeline performs:
- Micro-resistivity image segmentation and fracture/vug detection
- Feature extraction and parameter quantification
- Automated report generation

## Requirements
Python 3.9+ with packages:
```bash
pip install -r requirements.txt
````

Additionally, **OLLMA** is required for LLM-based processing.
It is recommended to use **DeepSeek 14b or higher** for inference.

## Model Weights

The framework relies on several pretrained models. You can download the weights directly:

| Model                   | Hugging Face Repository                                                                   | Direct Download                                                                                                     |
| ----------------------- | ----------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------- |
| UNet Induced Fracture   | [scai-unet-inducedfracture](https://huggingface.co/Lijiangning/scai-unet-inducedfracture) | [Download `.pth`](https://huggingface.co/Lijiangning/scai-unet-inducedfracture/resolve/main/best_epoch_weights.pth) |
| YOLO Fracture Detection | [scai-yolo](https://huggingface.co/Lijiangning/scai-yolo)                                 | [Download `.pth`](https://huggingface.co/Lijiangning/scai-yolo/resolve/main/best.pt)                 |
| SAM2 Segmentation       | [scai-sam2](https://huggingface.co/Lijiangning/scai-sam2)                                 | [Download `.pth`](https://huggingface.co/Lijiangning/scai-sam2/resolve/main/sam2.1_hiera_s_fracture.pt) （https://huggingface.co/Lijiangning/scai-sam2/resolve/main/sam2.1_hiera_s_induced_fracture.pt）（https://huggingface.co/Lijiangning/scai-sam2/resolve/main/sam2.1_hiera_s_vug.pt）（https://huggingface.co/Lijiangning/scai-sam2/resolve/main/sam2.1_hiera_small.pt）                |
| UNet Vug Detection      | [scai-unet-vug](https://huggingface.co/Lijiangning/scai-unet-vug)                         | [Download `.pth`](https://huggingface.co/Lijiangning/scai-unet-vug/resolve/main/best_epoch_weights.pth)             |
| UNet Fracture Detection | [scai-unet-fracture](https://huggingface.co/Lijiangning/scai-unet-fracture)               | [Download `.pth`](https://huggingface.co/Lijiangning/scai-unet-fracture/resolve/main/best_epoch_weights.pth)        |

### Python Download Example

You can also download weights programmatically:

```python
from huggingface_hub import hf_hub_download

# Example: UNet Fracture Detection
file_path = hf_hub_download(
    repo_id="Lijiangning/scai-unet-fracture",
    filename="best_epoch_weights.pth"
)
print("Downloaded file path:", file_path)
```

## Quick Test

To quickly test the pipeline:

1. Run the 6 API scripts located in `SCAI/api` in order.
2. Run the main pipeline:

```bash
cd SCAI/main
python planner_pipeline.py
```

3. Test images are available in the `test` folder.

The pipeline will automatically:

* Load the pretrained weights
* Run segmentation and detection
* Output results for visualization and analysis

---

This setup allows you to quickly reproduce the fracture–vug interpretation workflow using your own micro-resistivity images.

```

---

💡 说明与建议：  
1. 所有 Hugging Face 权重链接使用 `resolve/main/<filename>`，这样无论仓库是否更新文件都能保证直接下载最新版本。  
2. Python 下载示例可以快速嵌入任何脚本中，适合自动化测试。  
3. README 中明确了快速测试流程，用户只需运行 API 脚本 + `planner_pipeline.py` 即可。  

