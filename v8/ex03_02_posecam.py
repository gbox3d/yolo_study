import cv2
from ultralytics import YOLO
import time

# Load the YOLOv8 model
model = YOLO('yolov8n-pose.pt')

# Open the default camera
cap = cv2.VideoCapture(0)

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()
    if success:
        start_time = time.time()
        
        # Run YOLOv8 inference on the frame
        results = model(frame)
        
        # Get the keypoints and bounding boxes for each person
        for person in results:
            keypoints = person['keypoints']
            boxes = person['boxes']
            
            # Draw keypoints and bounding boxes on the frame
            for keypoint in keypoints:
                cv2.circle(frame, (keypoint[0], keypoint[1]), 3, (0, 0, 255), -1)
            cv2.rectangle(frame, (boxes[0], boxes[1]), (boxes[2], boxes[3]), (255, 0, 0), 2)
        
        end_time = time.time()
        fps = 1 / (end_time - start_time)
        # print("FPS :", fps)
        
        cv2.putText(frame, "FPS :"+str(int(fps)), (10, 50), cv2.FONT_HERSHEY_COMPLEX, 1.2, (255, 0, 255), 1, cv2.LINE_AA)
        
        # Display the annotated frame
        cv2.imshow("YOLOv8 Inference", frame)
        
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()
