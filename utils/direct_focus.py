from PIL import Image
from utils.visual_stimulus_aug import VSA
def direct_img_focus(raw_image, box, crop_width):
    original_width, original_height = raw_image.size
    cx,cy,w,h=box
    cx=int(float(cx)*original_width)
    cy=int(float(cy)*original_height)

    left = int(max(0, cx - crop_width // 2))
    top = int(max(0, cy - crop_width // 2))
    right = int(min(original_width, cx + crop_width // 2))
    bottom = int(min(original_height, cy + crop_width // 2))
    
    visual_region=raw_image.crop((left, top, right, bottom))
    focused_image=VSA(visual_region,box)
    return focused_image