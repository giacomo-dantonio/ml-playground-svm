import numpy as np
from PIL import Image
from PIL import ImageOps
import matplotlib.cm as cm

def load_image(filepath: str) -> np.array:
    img = Image.open(filepath)
    img.thumbnail((28, 28))
    gs_img = ImageOps.grayscale(img)
    imgdata = np.array([255.0 - val for val in gs_img.getdata()])
    return imgdata
