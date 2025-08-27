#!/usr/bin/env python
# coding: utf-8

# In[10]:


import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance

def low_pass_filter(image, kernel_size=(3, 3)):
    return cv2.GaussianBlur(image, kernel_size, 0)

def high_pass_filter(image, kernel_size=(3, 3)):
    kernel = np.array([[0, -1, 0],
                       [-1, 4, -1],
                       [0, -1, 0]], dtype=np.float32)
    return cv2.filter2D(image, -1, kernel)

def normalize_image(image):
    normalized_image = cv2.normalize(image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    return normalized_image

def process_image(input_path, output_dir, new_size=(800, 600), filter_type=cv2.INTER_LANCZOS4, alpha=1.5, beta=30, background_weight=1.2):
    original_image = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)

    # Enhance the contrast using ImageEnhance
    image_pil = Image.fromarray(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
    enhancer = ImageEnhance.Contrast(image_pil)
    enhanced_image = enhancer.enhance(4)

    # Normalize the enhanced image
    normalized_image = normalize_image(np.array(enhanced_image))

    # Resize and process the image
    resized_image = cv2.resize(normalized_image, new_size, interpolation=cv2.INTER_AREA)
    high_res_image = cv2.resize(resized_image, (original_image.shape[1], original_image.shape[0]), interpolation=filter_type)
    adjusted_image = cv2.convertScaleAbs(high_res_image, alpha=alpha, beta=beta)

    low_pass_result_1 = low_pass_filter(adjusted_image)
    high_pass_result = high_pass_filter(adjusted_image)
    low_pass_result_2 = low_pass_filter(high_pass_result)

    # Combine the results using addWeighted
    final_image = cv2.addWeighted(adjusted_image, background_weight, high_pass_result, 1.0, 0)
    final_image = cv2.addWeighted(final_image, 1.0, low_pass_result_2, 0.5, 0)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    file_name = os.path.splitext(os.path.basename(input_path))[0]
    new_name = f"{file_name}_processed.jpg"  # Change the extension to .jpg
    output_path = os.path.join(output_dir, new_name)
    cv2.imwrite(output_path, final_image, [int(cv2.IMWRITE_JPEG_QUALITY), 90])  # Adjust JPEG quality as needed

    # Return the original and processed images
    return original_image, final_image

if __name__ == "__main__":
    input_directory = r'D:\uni\propozal\pepar\Deep-learning-high-DNA-integrity\all_data\Donor1\brightfield\unprocessed'
    output_directory = r'D:\uni\propozal\pepar\Deep-learning-high-DNA-integrity\all_data\Donor1\brightfield\output_dir'
    new_size = (1200, 900)
    filter_type = cv2.INTER_LANCZOS4
    alpha = 2
    beta = 30
    background_weight = 1.4  # Adjust the weight to control the brightness of the background

    processed_images = []

    # Process all images and store the results
    for filename in os.listdir(input_directory):
        if filename.endswith('.tif'):
            input_path = os.path.join(input_directory, filename)
            original_img, processed_img = process_image(input_path, output_directory, new_size, filter_type, alpha, beta, background_weight)
            processed_images.append((original_img, processed_img))

    # Display the original and processed images of the first image as a sample
    if processed_images:
        original_img, processed_img = processed_images[0]

        # Display the original image
        plt.imshow(original_img, cmap='gray' if original_img.shape[-1] == 1 else None)
        plt.title("Original Image")
        plt.axis('off')
        plt.show()

        # Display the processed image
        plt.imshow(processed_img, cmap='gray' if processed_img.shape[-1] == 1 else None)
        plt.title("Processed Image")
        plt.axis('off')
        plt.show()


# In[ ]:




