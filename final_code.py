import cv2
import torch
import pandas as pd
import pyttsx3  
import ssl

ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

engine = pyttsx3.init()

model_path = "C:/Users/Deepak/yolov5/runs/train/yolov5s_results11/weights/best.pt"
model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)
coco_model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

cam = "http://172.29.50.11:8080/video"
cap = cv2.VideoCapture(cam)

confidence_threshold = 0.7
focal_length = 230  
real_world_heights = {
    'person': 1.7,  
    'car': 1.5,    
    'motorcycle': 1.2, 
    'tree': 5.0,   
    'bus': 3.0 
}

def calculate_distance(real_height, pixel_height, focal_length):
    if pixel_height > 0:
        return (real_height * focal_length) / pixel_height
    return None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results_tree = model(frame)
    results_coco = coco_model(frame)

    df_tree = results_tree.pandas().xyxy[0]
    df_coco = results_coco.pandas().xyxy[0]

    if df_tree.empty and df_coco.empty:
        direction_text = "All Clear" 
        engine.say("All Clear")
        engine.runAndWait()
    
        cv2.putText(frame, f"Direction: {direction_text}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)

        height, width = frame.shape[:2]
        cv2.line(frame, (width // 3, 0), (width // 3, height), (255,255,255), 2)   
        cv2.line(frame, (2 * width // 3, 0), (2 * width // 3, height), (255,255,255), 2) 

        new_width = int(width * 0.6)  
        new_height = int(height * 0.6)  
        frame_resized = cv2.resize(frame, (new_width, new_height))

        cv2.imshow('Obstacle Detection and Direction Suggestion', frame_resized)
        continue

    combined_results = pd.concat(
        [df for df in [df_tree, df_coco] if not df.empty],
        ignore_index=True
    )

    left_count = center_count = right_count = 0
    direction_text = ""
    spoken_text = ""

    if not combined_results.empty:
        width = frame.shape[1]

        filtered_results = combined_results[ 
            (combined_results['name'].isin(['tree', 'car', 'person', 'motorcycle', 'bus'])) & 
            (combined_results['confidence'] >= confidence_threshold)
        ]

        for index, row in filtered_results.iterrows():
            x1, y1, x2, y2, conf, cls = row[['xmin', 'ymin', 'xmax', 'ymax', 'confidence', 'class']]
            object_class = row['name']

            pixel_height = y2 - y1
            real_height = real_world_heights.get(object_class)

            if real_height is not None:
                distance = calculate_distance(real_height, pixel_height, focal_length)
                label = f"{object_class} {conf:.2f}, {distance:.2f}m" if distance else f"{object_class} {conf:.2f}"
                spoken_text += f"{object_class} ahead {distance:.2f} meters. "   
            else:
                label = f"{object_class} {conf:.2f}"

            frame = cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
            frame = cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 3)

            if x2 < width / 3:      
                left_count += 1
            elif x1 > 2 * width / 3: 
                right_count += 1
            else:                    
                center_count += 1

    if left_count > 0 and center_count > 0 and right_count > 0:
        direction_text = "Stop"
    elif right_count > 0 and center_count > 0:
        direction_text = "Move Left"
    elif left_count > 0 and center_count > 0:
        direction_text = "Move Right"
    elif left_count > 0 and right_count > 0:
        direction_text = "Move Straight"
    elif left_count > 0:
        direction_text = "Move Straight"
    elif right_count > 0:
        direction_text = "Move Straight"
    elif center_count > 0:
        direction_text = "Move Left or Right"
    else:
        direction_text = "All Clear"

    spoken_text += f"{direction_text}."
    engine.say(spoken_text)
    engine.runAndWait()

    cv2.putText(frame, f"Direction: {direction_text}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)

    height, width = frame.shape[:2]
    cv2.line(frame, (width // 3, 0), (width // 3, height), (255, 255, 255), 2)   
    cv2.line(frame, (2 * width // 3, 0), (2 * width // 3, height), (255, 255, 255), 2) 

    new_width = int(width * 0.6)  
    new_height = int(height * 0.6)  
    frame_resized = cv2.resize(frame, (new_width, new_height))

    cv2.imshow('Obstacle Detection and Direction Suggestion', frame_resized)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()




