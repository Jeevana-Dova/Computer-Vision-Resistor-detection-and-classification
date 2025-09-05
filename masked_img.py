import cv2
import numpy as np
import os

def mask_background_black_folder(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    count = 1  # Initialize the counter    

    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, f"maskedA_{count}.jpg")

            image = cv2.imread(input_path)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            blurred = cv2.GaussianBlur(gray, (7, 7), 0)

            _, thresh = cv2.threshold(blurred, 90, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            largest_contour = max(contours, key=cv2.contourArea)

            mask = np.zeros_like(gray)
            cv2.drawContours(mask, [largest_contour], -1, 255, thickness=cv2.FILLED)

            result = cv2.bitwise_and(image, image, mask=mask)

            background_mask = cv2.bitwise_not(mask)
            result[background_mask == 255] = (0, 0, 0)  # Set background black

            cv2.imwrite(output_path, result)
            print(f"Processed and saved: {output_path}")

            count += 1

# Usage
input_folder = 'C:/Users/dovaj/Documents/actual_images/resistorA'
output_folder = 'C:/Users/dovaj/Documents/masked_images/resistorA'

mask_background_black_folder(input_folder, output_folder)