import cv2
from detector import Detector
import time
import numpy as np
from collections import defaultdict

# VIDEO_PATH = './Files/Videos/car passing by.mp4'
# VIDEO_PATH = './Files/Videos/CarsPassingBy2.mp4'
VIDEO_PATH = './Files/Videos/CarsPassingBy3.mp4'
CONFIABILITY_THRESHOLD = 0.45

def reproduzir_video(video_Path):
    cap = cv2.VideoCapture(video_Path)
    detector = Detector('./Files/Models/yolov8n.pt', CONFIABILITY_THRESHOLD)

    if not cap.isOpened():
        print(f"Fail! It was not possible to open this video {VIDEO_PATH}")
        return
    
    last_time = 0
    processing_fps = 0



    while cap.isOpened():
        ret, frame = cap.read()
        
        if not ret:
            print("Error: One frame was not able to read - Ending stream")
            break

        actual_time = time.time()
        if last_time != 0:
            processing_fps = 1 / (actual_time - last_time)
        last_time = actual_time

        processedFrame = detector.processImage(frame)

        cv2.putText(processedFrame, f'FPS: {processing_fps:.1f}', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Video', processedFrame)

        if cv2.waitKey(25) & 0xFF == ord('q'): # Press Q to exit
            break
    
    cap.release()
    cv2.destroyAllWindows()   

def main():
    reproduzir_video(VIDEO_PATH)

if __name__=='__main__':
    main()