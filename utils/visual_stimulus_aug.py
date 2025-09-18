import numpy as np
import cv2
from PIL import Image

def VSA(image, box, flag='image', tau=15):
    if flag=='arr':
        original_dtype=image.dtype
        if image.dtype != np.uint8:
            image=image.cpu().detach().numpy()
            img_arr = (image * 255).clip(0, 255).astype(np.uint8)
        img_arr = np.transpose(img_arr, (0, 2, 3, 1))
        img_arr = img_arr[0]  
        height, width = img_arr.shape[:2]
    elif flag=='image':
        img_arr = np.array(image)
        height, width = img_arr.shape[:2]
    else:
        raise ValueError('No support flag.')
    
    cw,ch=max(20,int(width*box[-2])),max(20,int(height*box[-1]))
    cx, cy = width // 2, height // 2
    left = cx -  cw// 2
    top = cy - ch // 2
    right, bottom = left + cw, top + ch

    x, y = np.meshgrid(np.arange(width), np.arange(height))
    dx = np.maximum(np.maximum(left - x, x - right), 0)
    dy = np.maximum(np.maximum(top - y, y - bottom), 0)
    dist = np.sqrt(dx**2 + dy**2)
    
    blur_map = np.uint8(1-1e-8+dist//tau)  
    blur_map = np.clip(blur_map, 0, None)         
    
    result = np.zeros_like(img_arr)
    unique_radii = np.unique(blur_map)
    
    for r in unique_radii:
        mask = (blur_map == r)
        if r == 0:
            result[mask] = img_arr[mask]
        else:
            kernel_size = 2 * int(1.5 * r) + 1
            blurred = cv2.GaussianBlur(img_arr, (kernel_size, kernel_size), r) 
            blurred = blurred.astype(np.uint8)        
            result[mask] = blurred[mask]
    
    if flag=='arr':
        if original_dtype != np.uint8:
            result = result.astype(np.float32) / 255.0
        result = result.transpose(2, 0, 1)[np.newaxis, ...]
    return result
