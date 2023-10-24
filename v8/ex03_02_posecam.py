#%%
import cv2 as cv
from ultralytics import YOLO
import time


from IPython.display import display
import PIL.ImageFont as ImageFont
import PIL.ImageDraw as ImageDraw
import PIL.ImageColor as ImageColor
import PIL.Image as Image



# Load the YOLOv8 model
model = YOLO('yolov8s-pose.pt')

# Open the default camera
cap = cv.VideoCapture(0)
#%%
# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()
    if success:
        start_time = time.time()
        
        # Run YOLOv8 inference on the frame
        results = model(frame, conf=0.5,verbose=False)  # predict on an image
        
        for det in results:
            for kpts in det.keypoints.data:
                # Each keypoint is a 2D tensor
                for kpt in kpts:
                    x, y, _ = map(int, kpt)
                    cv.circle(frame, (x, y), 5, (0, 0, 255), -1)
        
        end_time = time.time()
        fps = 1 / (end_time - start_time)
        # print("FPS :", fps)
        
        cv.putText(frame, "FPS :"+str(int(fps)), (10, 50), cv.FONT_HERSHEY_COMPLEX, 1.2, (255, 0, 255), 1, cv.LINE_AA)
        
        # Display the annotated frame
        cv.imshow("posecamTest", frame)
        
        # Break the loop if 'q' is pressed
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv.destroyAllWindows()

# %%
