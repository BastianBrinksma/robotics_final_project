from ultralytics import YOLO
import cv2

def detect_all_plants_yolo(image_path):
    # Load YOLOv11 nano model
    model = YOLO("yolo11n.pt")
    img = cv2.imread(image_path)
    
    if img is None:
        print("Error: cannot load image")
        return None

    results = model(img)[0]
    PLANT_CLASS_ID = 58   # "potted plant" in COCO dataset
    
    # Create an array to store all plants objects 
    plants = []
    
    for box in results.boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])

        if cls == PLANT_CLASS_ID:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            plants.append({
                "bbox": (x1, y1, x2, y2),
                "centroid": (cx, cy),
                "confidence": conf
            })

    print(f"Found {len(plants)} plants")
    return plants
