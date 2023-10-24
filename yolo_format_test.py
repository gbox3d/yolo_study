#%% Import packages
import argparse
import cv2 as cv
import os
import yaml

# Set up command-line argument parsing
parser = argparse.ArgumentParser(description='Draw bounding boxes on images using labels.')
parser.add_argument('--basepath', required=True, help='The basepath of your files.')
parser.add_argument('--imageFile', required=True, help='The image file to draw bounding boxes on.')
parser.add_argument('--data', required=True, help='The path of the yaml file.')
args = parser.parse_args()

# Read coco128.yaml file
yaml_path = args.data
with open(yaml_path, 'r') as file:
    data = yaml.load(file, Loader=yaml.FullLoader)

# Extract label names
label_names = data['names']

# Print label names
#for label_id, label_name in label_names.items():
#    print(f"Label ID: {label_id}, Label Name: {label_name}")

#%% Read image
image_path = os.path.join(args.basepath, 'src', args.imageFile)
image = cv.imread(image_path)
image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)

# get file name without extension
label_file = os.path.splitext(args.imageFile)[0] + '.txt'
# Read YOLO labels
label_path = os.path.join(args.basepath, 'labels', label_file)

with open(label_path, 'r') as file:
    labels = file.readlines()

#%% Draw labeled boxes on the image
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
    
#%% Display the image
# Save the image with labeled boxes
output_path = os.path.join('./', 'output', args.imageFile)
cv.imwrite(output_path, cv.cvtColor(image_rgb, cv.COLOR_RGB2BGR))