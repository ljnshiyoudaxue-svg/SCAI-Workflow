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
| UNet Induced Fracture   | [scai-unet-inducedfracture](https://huggingface.co/Lijiangning/scai-unet-inducedfracture) | [best_epoch_weights.pth](https://huggingface.co/Lijiangning/scai-unet-inducedfracture/resolve/main/best_epoch_weights.pth) |
| YOLO Fracture Detection | [scai-yolo](https://huggingface.co/Lijiangning/scai-yolo)                                 | [best.pt](https://huggingface.co/Lijiangning/scai-yolo/resolve/main/best.pt)                                        |
| SAM2 Segmentation       | [scai-sam2](https://huggingface.co/Lijiangning/scai-sam2)                                 | - [sam2.1_hiera_s_fracture.pt](https://huggingface.co/Lijiangning/scai-sam2/resolve/main/sam2.1_hiera_s_fracture.pt)  <br> - [sam2.1_hiera_s_induced_fracture.pt](https://huggingface.co/Lijiangning/scai-sam2/resolve/main/sam2.1_hiera_s_induced_fracture.pt) <br> - [sam2.1_hiera_s_vug.pt](https://huggingface.co/Lijiangning/scai-sam2/resolve/main/sam2.1_hiera_s_vug.pt) <br> - [sam2.1_hiera_small.pt](https://huggingface.co/Lijiangning/scai-sam2/resolve/main/sam2.1_hiera_small.pt) |
| UNet Vug Detection      | [scai-unet-vug](https://huggingface.co/Lijiangning/scai-unet-vug)                         | [best_epoch_weights.pth](https://huggingface.co/Lijiangning/scai-unet-vug/resolve/main/best_epoch_weights.pth)     |
| UNet Fracture Detection | [scai-unet-fracture](https://huggingface.co/Lijiangning/scai-unet-fracture)               | [best_epoch_weights.pth](https://huggingface.co/Lijiangning/scai-unet-fracture/resolve/main/best_epoch_weights.pth) |

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

To quickly test the full pipeline:

1. Install all dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Install and configure **Ollama** ([https://ollama.com](https://ollama.com))

   Recommended LLM:

   ```
   deepseek-r1:14b or higher
   ```

3. Install and configure the **official YOLO and SAM2 frameworks**.

   ⚠️ **Important**
   This repository only provides wrapped inference interfaces and pretrained weights.
   Users must manually install and configure the official YOLO and SAM2 implementations before running the API scripts.

4. Start the six API services under,Run the 6 API scripts located in `SCAI/api` in order.:

   ```
   SCAI/api/
   ```

   (Run all required API scripts)

5. Then execute:

   ```bash
   python SCAI/main/planner_pipeline.py
   ```

6. Test images are provided in:

   ```
   main/test.jpg
   ```


> Users are required to independently install and configure the official YOLO and SAM2 frameworks. This repository provides only the encapsulated inference interfaces and pretrained model weights, not the original framework implementations.



Notes and Recommendations:

1. All Hugging Face weight links use the `resolve/main/<filename>` format to ensure that the latest version of the file can always be downloaded directly, even if the repository is updated.

2. The provided Python download examples can be easily embedded into any script, making them suitable for automated testing and deployment.

3. The README clearly describes the quick test workflow. Users must install Ollama independently (recommended model: `deepseek-r1:14b` or higher), as well as properly install and configure the official YOLO and SAM2 frameworks before running the API scripts and `planner_pipeline.py`.


