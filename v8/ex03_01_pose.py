#%%
import cv2 as cv

# import ultralytics
from ultralytics import YOLO,checks

from IPython.display import display
import PIL.ImageFont as ImageFont
import PIL.ImageDraw as ImageDraw
import PIL.ImageColor as ImageColor
import PIL.Image as Image


checks()
#%%
# Load a model
model = YOLO('yolov8s-pose.pt')  # load an official model
#%%
im = cv.imread('./soccer_4.jpg')
results = model(source=im,conf=0.5)  # predict on an image

#%%
for index,result in enumerate(results):
    print(f'result {index} :  Found {len(result)} detection(s)')
# %%
# Make a copy of the original image to draw bounding boxes and keypoints on
result_img = im.copy()

# Iterate over the results
for res in results:
    # Get the bounding box coordinates
    x1, y1, x2, y2, _, _ = map(int, res.boxes.data[0])

    # Draw the bounding box on the image
    cv.rectangle(result_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

     # Iterate over the keypoints for each detected person
    for person_kpts in res.keypoints.data:
        # Each keypoint is a 2D tensor
        for kpt in person_kpts:
            x, y, _ = map(int, kpt)
            # Draw the keypoint on the image
            cv.circle(result_img, (x, y), 5, (0, 0, 255), -1)
display(Image.fromarray( cv.cvtColor(result_img, cv.COLOR_BGR2RGB)))        
# %%
