import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

import torch
import cv2
import pyttsx3
from ultralytics import YOLO

# Initialize text-to-speech engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)  # Speed of speech

# Load both models
general_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
custom_model = YOLO('dog_detector_final.pt')  # Your custom trained model

cap = cv2.VideoCapture(0)
engine.say("Dog Spotter Activated")
engine.runAndWait()

while True:
    ret, frame = cap.read()
    if not ret:
        break
        
    # Get general detections
    results = general_model(frame)
    
    # Get detections
    detections = results.xyxy[0]  # Get detections in xyxy format
    for det in detections:
        x1, y1, x2, y2, conf, cls = det.tolist()
        class_name = results.names[int(cls)]
        if class_name == 'dog' and conf >= 0.60:
            # Extract the dog region
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            dog_region = frame[y1:y2, x1:x2]
            
            # Check if it's your dog using custom model
            custom_results = custom_model(dog_region)
            
            # Get the confidence score from the custom model
            if len(custom_results) > 0 and custom_results[0].boxes is not None:
                custom_conf = custom_results[0].boxes.conf[0].item()  # Get confidence for the detection
                
                if custom_conf >= 0.90:
                    print(f"Annie dog detected! Confidence: {custom_conf:.2%}")
                    engine.say(f"Annie dog detected!")
                    engine.runAndWait()
                else:
                    print(f"Other dog detected. Your dog confidence: {custom_conf:.2%}")
                    engine.say(f"Other dog detected")
                    engine.runAndWait()
            else:
                print("No custom model detection")
    
    # Display the frame with detections
    cv2.imshow("Dog Detector", results.render()[0])
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cap.release()
cv2.destroyAllWindows()