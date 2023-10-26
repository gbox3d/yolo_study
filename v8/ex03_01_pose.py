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
model = YOLO('yolov8n-pose.pt')  # load an official model
#%%
im = cv.imread('bus.jpg')
results = model(source=im,conf=0.5)  # predict on an image

#%%
print(results)
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
print(results[0].keypoints[0].xyn.numpy())
print(results[0].keypoints[0].xyn.numpy().shape)

#%%
print(results[0].orig_img.shape)
# %%
np_keypoint = results[0].keypoints[0].xyn.numpy()
print(np_keypoint[0][9])

screen_pos_x, screen_pos_y = map(int, np_keypoint[0][9])

# %%

