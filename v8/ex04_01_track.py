import cv2 as cv
import pygame
from ultralytics import YOLO
import time
import numpy as np

import os
from dotenv import load_dotenv

# .env 파일에서 환경변수 읽기
load_dotenv()

camera_index = int(os.getenv("CAMERA_INDEX"))
screen_width = int(os.getenv("SCREEN_WIDTH"))
screen_height = int(os.getenv("SCREEN_HEIGHT"))

# Initialize Pygame
pygame.init()

# Set up the display
# window_width, window_height = 640, 480  # Set the dimensions as needed
window = pygame.display.set_mode(size=(screen_width, screen_height))
pygame.display.set_caption('YOLOv8 Tracking')

# Load the YOLOv8 model
model = YOLO("yolov8n.pt")

# Open the camera
cap = cv.VideoCapture(camera_index)  # Open the default camera

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()
    
time.sleep(1.5)  # Let the camera warm up

running = True
while running:
    
    #fps 계산
    last_time = time.time()
    
    # Read a frame from the camera
    ret, frame = cap.read()
    if not ret:
        continue

    # Run YOLOv8 tracking on the frame
    results = model.track(frame, persist=True,
                          verbose=False, # Don't print stats
                          classes=[0] # person class only
                          )
    if len(results) == 0:
        continue
    
    boxes = results[0].boxes.xyxyn.cpu().numpy()
    
    if results[0].boxes.id == None :
        continue
    
    ids = results[0].boxes.id.cpu().numpy().astype(int)

    # Convert the OpenCV frame to a Pygame surface
    frame_rgb = cv.flip(np.rot90(cv.cvtColor(frame, cv.COLOR_BGR2RGB)),0)
    
    
    frame_surface = pygame.surfarray.make_surface(frame_rgb)

    # Clear the window
    window.fill((0, 0, 0))

    # Blit the frame surface onto the window
    window.blit(pygame.transform.scale(frame_surface, (screen_width, screen_height)), (0, 0))

    for box, id in zip(boxes, ids):
        #screen 좌표계로 변환
        x1, y1, x2, y2 = box[0]*screen_width, box[1]*screen_height, box[2]*screen_width, box[3]*screen_height
        
        # Draw a rectangle around the object
        pygame.draw.rect(window, (255, 0, 0), (x1, y1, x2-x1, y2-y1), 2)
        font = pygame.font.Font(None, 36)
        text_surface = font.render(f"Id {id}", True, (0, 0, 255))
        window.blit(text_surface, (x1, y1-30))

    # Calculate FPS
    current_time = time.time()
    elapsed_time = current_time - last_time
    fps = 1 / elapsed_time if elapsed_time > 0 else 0
    last_time = current_time
    
    window.blit(font.render(f"FPS: {fps:.1f}", True, (0, 255, 0)), (10, 10))
    
    pygame.display.flip()

    # Handle events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False

# Release the camera and quit Pygame
cap.release()
pygame.quit()
