import cv2
import numpy as np
import os

def extract_resistor_grabcut(resistor_img):
    mask = np.zeros(resistor_img.shape[:2], np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    # Rectangle covers almost whole image; you can tweak margins
    height, width = resistor_img.shape[:2]
    rect = (10, 10, width - 20, height - 20)
    cv2.grabCut(resistor_img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
    # 1: probable foreground, 3: definite foreground
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    result = resistor_img * mask2[:, :, np.newaxis]
    return result, mask2

def augment_resistor_on_background(resistor_folder, background_folder, output_folder, num_images,
                                   scale_factor=0.4, x_offset=100, y_offset=100,
                                   upscale_background=False, bg_target_size=(1280, 720)):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    resistor_images = sorted([f for f in os.listdir(resistor_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])
    background_images = sorted([f for f in os.listdir(background_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])

    for i in range(num_images):
        resistor_path = os.path.join(resistor_folder, resistor_images[i])
        background_path = os.path.join(background_folder, background_images[i])

        resistor_img = cv2.imread(resistor_path)
        bg_img = cv2.imread(background_path)

        # Optionally upscale background for bigger result images
        if upscale_background:
            bg_img = cv2.resize(bg_img, bg_target_size, interpolation=cv2.INTER_LINEAR)
        bg_h, bg_w = bg_img.shape[:2]

        # Resize resistor
        new_w = int(bg_w * scale_factor)
        new_h = int(resistor_img.shape[0] * (new_w / resistor_img.shape[1]))
        resistor_img = cv2.resize(resistor_img, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # Use GrabCut to extract the resistor cleanly
        resistor_clean, resistor_mask = extract_resistor_grabcut(resistor_img)

        # Check offsets
        x_off = min(max(x_offset, 0), bg_w - new_w)
        y_off = min(max(y_offset, 0), bg_h - new_h)

        # Overlay with mask: only blend nonzero mask part
        roi = bg_img[y_off:y_off+new_h, x_off:x_off+new_w]
        mask_inv = np.where(resistor_mask == 1, 0, 1).astype('uint8')
        mask_inv_3 = np.stack([mask_inv]*3, axis=-1)

        # Background stays where mask is 0
        roi_bg = roi * mask_inv_3
        # Foreground is resistor where mask is 1
        resistor_fg = resistor_clean

        # Combine
        dst = cv2.add(roi_bg, resistor_fg)
        bg_img[y_off:y_off+new_h, x_off:x_offset+new_w] = dst

        output_path = os.path.join(output_folder, f'augmentedB_{i+1}.jpg')
        cv2.imwrite(output_path, bg_img)
        print(f"Saved augmented image: {output_path}")

# Example usage:
augment_resistor_on_background(
    resistor_folder = 'C:/Users/dovaj/Documents/masked_images/resistorB', 
    background_folder = 'C:/Users/dovaj/Documents/augmented_images/background_images',     # Folder of background images
    output_folder = 'C:/Users/dovaj/Documents/augmented_images/resistorB',      # Folder to save augmented images
    num_images=1099,
    scale_factor=0.5,
    x_offset=300,
    y_offset=500,
    upscale_background=True,
    bg_target_size=(1280, 720)
)