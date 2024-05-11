import numpy as np
from PIL import Image
from skimage import io, transform
import cv2

def resize_longest_side(image):
    """
    Expects a numpy array with shape HxWxC in uint8 format.
    """
    target_length = 512
    long_side_length = target_length
    oldh, oldw = image.shape[0], image.shape[1]
    scale = long_side_length * 1.0 / max(oldh, oldw)
    newh, neww = oldh * scale, oldw * scale
    neww, newh = int(neww + 0.5), int(newh + 0.5)
    target_size = (neww, newh)
    return cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)

def pad_image(image):
    """
    Expects a numpy array with shape HxWxC in uint8 format.
    """
    # Pad
    image_size=512
    h, w = image.shape[0], image.shape[1]
    padh = image_size - h
    padw = image_size - w
    if len(image.shape) == 3: ## Pad image
        image_padded = np.pad(image, ((0, padh), (0, padw), (0, 0)))
    else: ## Pad gt mask
        image_padded = np.pad(image, ((0, padh), (0, padw)))

    return image_padded

def save_numpy_as_png(array, file_path):
    """
    将 NumPy 数组转换为 PNG 图像并保存到指定路径。
    
    参数:
    - array: numpy.ndarray，代表图像数据。
    - file_path: str，保存图像的文件路径。
    
    注意：
    - 如果数组是二维的，假定为灰度图。
    - 如果数组是三维的，假定最后一个维度是颜色通道，例如 RGB。
    """
    # 检查数组类型，确保是 uint8（这是图像通常使用的数据类型）
    if array.dtype != np.uint8:
        raise ValueError("数组应该有 dtype 为 np.uint8")

    # 根据数组维度创建图像
    if array.ndim == 2:  # 灰度图
        image = Image.fromarray(array, 'L')  # 'L' mode for grayscale
    elif array.ndim == 3 and array.shape[2] == 3:  # 彩色图，假定为 RGB
        image = Image.fromarray(array, 'RGB')
    else:
        raise ValueError("数组维度不支持，支持二维灰度图或三维RGB图像")

    # 保存图像
    image.save(file_path, 'PNG')

# # 示例使用
# # 创建一个灰度图像和一个 RGB 图像
# gray_image = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
# color_image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)

# # 保存图像
# save_numpy_as_png(gray_image, 'gray_image.png')
# save_numpy_as_png(color_image, 'color_image.png')

image_input=np.load("./data/sample/npy/imgs/XRay_Chest-Xray-Masks-and-Labels_MCUCXR_0393_1.npy")
# image_input = transform.resize(
#     image_input, (512, 512), order=3, preserve_range=True, anti_aliasing=True
# ).astype(np.uint8)

img_resize = resize_longest_side(image_input)
# Resizing
img_resize = (img_resize - img_resize.min()) / np.clip(img_resize.max() - img_resize.min(), a_min=1e-8, a_max=None) # normalize to [0, 1], (H, W, 3
img_padded = pad_image(img_resize) # (256, 256, 3)
save_numpy_as_png((img_padded*255).astype(np.uint8), 'XRay_Chest-Xray-Masks-and-Labels_MCUCXR_0393_1.png')