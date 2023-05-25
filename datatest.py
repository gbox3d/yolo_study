#%%
import cv2 as cv
import os
import numpy as np
from IPython.display import display
import PIL.ImageFont as ImageFont
import PIL.ImageDraw as ImageDraw
import PIL.ImageColor as ImageColor
import PIL.Image as Image

import yaml

#%%

# Read coco128.yaml file
yaml_path = './datasets/coco128.yaml'
with open(yaml_path, 'r') as file:
    data = yaml.load(file, Loader=yaml.FullLoader)

# Extract label names
label_names = data['names']

# Print label names
for label_id, label_name in label_names.items():
    print(f"Label ID: {label_id}, Label Name: {label_name}")
    
#%%

# Read image
image_path = os.path.join('./datasets/coco128', 'images/train2017/000000000036.jpg')
image = cv.imread(image_path)
image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)

# Display the original image
display(Image.fromarray(image_rgb))

#%%
# Read YOLO labels
label_path = os.path.join('./datasets/coco128', 'labels/train2017/000000000036.txt')

with open(label_path, 'r') as file:
    labels = file.readlines()

# Draw labeled boxes on the image
image_height, image_width, _ = image.shape

for label in labels:
    class_id, x, y, width, height = map(float, label.split())
    
    # Convert relative coordinates to absolute coordinates
    x_min = int((x - width/2) * image_width)
    y_min = int((y - height/2) * image_height)
    x_max = int((x + width/2) * image_width)
    y_max = int((y + height/2) * image_height)
    
    # Draw the box
    cv.rectangle(image_rgb, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
    # Add class label
    class_label = label_names[int(class_id)]
    label_text = f"{class_label}: {int(class_id)}"
    label_position = (x_min, y_min - 10)
    cv.putText(image_rgb, label_text, label_position, cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
# Display the image with labeled boxes
display(Image.fromarray(image_rgb))

# %%
