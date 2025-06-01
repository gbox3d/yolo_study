import pygame
import cv2

from ultralytics import YOLO,checks

checks()

width, height = 640, 480

# camera initialization
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

#pygame initialization
pygame.init()
screen_surface = pygame.display.set_mode((width, height))

clock = pygame.time.Clock()
font = pygame.font.SysFont(None,24)

#Yolo model initialization
model = YOLO("yolo11n.pt")

while True:
    
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break
    
    results = model(frame,conf=0.7,verbose=False)
    
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Convert OpenCV image to Pygame surface
    frame_surface = pygame.surfarray.make_surface(frame_rgb.swapaxes(0, 1))
    
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
    pygame.display.flip()
    
    clock.tick(30) # prevent high CPU usage
    
    # evenet handling
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            cap.release()
            pygame.quit()
            exit()
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                cap.release()
                pygame.quit()
                exit()
            
    