import pygame
import cv2

from ultralytics import YOLO,checks

checks()

# screen dimensions
width, height = 640, 480

#pygame initialization
pygame.init()
screen_surface = pygame.display.set_mode((width, height))

clock = pygame.time.Clock()
font = pygame.font.SysFont(None,24)

#Yolo model initialization
model = YOLO("yolo11n.pt")

#load image
frame = cv2.imread("tutorial/chap03_OD/bus.jpg")

# resize the image to fit the screen dimensions 
orig_h, orig_w = frame.shape[:2]

# Target dimensions (your screen dimensions)
target_w = width
target_h = height

# Calculate the scaling ratio for width and height
ratio_w = target_w / orig_w
ratio_h = target_h / orig_h

# Choose the smaller scaling ratio to ensure the image fits
# within the target dimensions while maintaining aspect ratio
scale_ratio = min(ratio_w, ratio_h)

# Calculate the new dimensions based on the chosen scale_ratio
new_w = int(orig_w * scale_ratio)
new_h = int(orig_h * scale_ratio)

# Resize the image using the new dimensions and INTER_AREA for shrinking
frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

# Check if the image was loaded successfully
if frame is None:
    print("Error: Could not read image.")
    exit()
# Run inference on the image
results = model(frame,conf=0.7,verbose=False)


# render the image in pygame
frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
# Convert OpenCV image to Pygame surface
frame_surface = pygame.surfarray.make_surface(frame_rgb.swapaxes(0, 1))
# Blit the surface to the screen
screen_surface.blit(frame_surface, (0, 0))

# Draw bounding boxes and labels
for result in results :
    boxes = result.boxes
    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = int(box.conf * 100)
        cls = int(box.cls)
        label = f"{model.names[cls]} {conf}%"
        pygame.draw.rect(screen_surface, (0, 255, 0), (x1, y1, x2 - x1, y2 - y1), 2)
        text_surface = font.render(label, True, (255, 0, 0))
        screen_surface.blit(text_surface, (x1, y1 - 20))

# Update the display
pygame.display.flip()

bLoop = True
while bLoop:    
    clock.tick(30) # prevent high CPU usage
    
    # evenet handling
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            bLoop = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                bLoop = False
            
pygame.quit()
print("exit successfully")