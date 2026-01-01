import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

# Path to dataset
image_folder = "."

# Apple color range (red + yellow) in HSV
lower1 = np.array([0, 80, 50])
upper1 = np.array([10, 255, 255])

lower2 = np.array([10, 80, 50])
upper2 = np.array([25, 255, 255])

# Read images one by one
for filename in os.listdir(image_folder):
    if not filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
        continue
        
    img_path = os.path.join(image_folder, filename)
    
    img = cv2.imread(img_path)
    if img is None:
        continue

    img = cv2.resize(img, (400, 400))

    # Convert to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Create mask
    mask1 = cv2.inRange(hsv, lower1, upper1)
    mask2 = cv2.inRange(hsv, lower2, upper2)
    mask = mask1 + mask2

    # Remove noise with larger kernel
    kernel = np.ones((15, 15), np.uint8)  # Increased from 7x7 to 15x15

    # Closing first (fills holes inside the apple)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Opening (removes small noise outside)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Extract apple
    apple_only = cv2.bitwise_and(img, img, mask=mask)

    # Display result
    plt.figure(figsize=(10, 3))

    plt.subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title("Original")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(mask, cmap="gray")
    plt.title("Mask")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(cv2.cvtColor(apple_only, cv2.COLOR_BGR2RGB))
    plt.title("Apple Segmented")
    plt.axis("off")

    plt.show()

    break   # show only one image (important)