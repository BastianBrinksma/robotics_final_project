import cv2
import numpy as np

def detect_green_centroid(image):
    image = "C:/Users/Odin/Documents/GitHub/robotics_final_project/images/monstera-deliciosa.jpg"
    img = cv2.imread(image)
    if img is None:
        print(f"Error: Could not load image: {image}")
        return None

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# 2. define the green range
    lower_green = np.array([35, 40, 40])   
    upper_green = np.array([85, 255, 255])

# 3. threshold
    mask = cv2.inRange(hsv, lower_green, upper_green)
    
    # Clean up noise
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0:
        print("No green object detected.")
        return None

