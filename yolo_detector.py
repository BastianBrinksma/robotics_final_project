from ultralytics import YOLO
import cv2
import PIL.Image as Image
from LLM import prompt_gemini  

def detect_all_plants_yolo(image_path):
    # Load YOLOv11 nano model
    model = YOLO("yolo11n.pt")
    img = cv2.imread(image_path)
    
    if img is None:
        print("Error: cannot load image")
        return None

    results = model(img)[0]
    PLANT_CLASS_ID = 58   
    yolo_count = sum(1 for b in results.boxes if int(b.cls[0]) == PLANT_CLASS_ID)
    print(f"YOLO detected {yolo_count} plant candidates.")

    # Create an array to store all plants objects 
    verified_plants = []

    for box in results.boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])

        if cls == PLANT_CLASS_ID:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            is_plant = llm_verify_plant(img, (x1, y1, x2, y2))

            if is_plant:
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2

                verified_plants.append({
                    "bbox": (x1, y1, x2, y2),
                    "centroid": (cx, cy),
                    "confidence": conf
                })
            else:
                print("YOLO false positive removed by LLM.")            
    print(f"Verified plants: {len(verified_plants)}")
    return verified_plants

def llm_verify_plant(img, bbox):
    x1, y1, x2, y2 = bbox
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)

    prompt_text = (
        f"Does the object inside bounding box ({x1}, {y1}, {x2}, {y2}) "
        "You are an expert botanist. Does the image show a plant? "
        "Answer stricly with either 'Yes' or 'No'."
    )

    response = prompt_gemini(
        input_prompt=[pil_img, prompt_text],
        temperature=0.0,
        schema=None,
        with_parts=False,
        with_tokens_info=False
    )

    answer = response.strip().lower()
    return answer == "yes"
