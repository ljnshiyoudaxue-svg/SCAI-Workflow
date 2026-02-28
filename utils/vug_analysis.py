
from PIL import Image
import numpy as np

def analyze_vug(mask_img: Image.Image):
    mask = np.array(mask_img.convert("L")) > 128
    area = np.sum(mask)
    total = mask.size
    porosity = area / total
    count = np.count_nonzero(mask)
    density = area / max(1, count)
    return {
        "porosity": round(porosity, 4),
        "avg_area": round(area / max(1, count), 2),
        "density": round(density, 2),
        "total_area": int(area)
    }
