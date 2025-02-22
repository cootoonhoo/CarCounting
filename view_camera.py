import cv2
from detector import Detector
import time

#Scenario 1
# VIDEO_PATH = './Files/Videos/car passing by.mp4'
# CONFIABILITY_THRESHOLD = 0.75
# LINE_P1 = 1000,300  #(x,y)
# LINE_P2 = 1000,1000   #(x,y)

#Scenario 2
# VIDEO_PATH = './Files/Videos/CarsPassingBy2.mp4'
# CONFIABILITY_THRESHOLD = 0.55
# LINE_P1 = 100,700  #(x,y)
# LINE_P2 = 1800,700   #(x,y)

#Scenario 3
VIDEO_PATH = './Files/Videos/CarsPassingBy3.mp4'
CONFIABILITY_THRESHOLD = 0.55
LINE_P1 = 420,410   #(x,y)
LINE_P2 = 1200,405  #(x,y)

def play_video(video_Path):
    countLine = LINE_P1,LINE_P2
    cap = cv2.VideoCapture(video_Path)
    detector = Detector('./Files/Models/yolov8l.pt', CONFIABILITY_THRESHOLD)

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

        cv2.line(frame, LINE_P1, LINE_P2, (255, 0, 0), 2)

        actual_time = time.time()
        if last_time != 0:
            processing_fps = 1 / (actual_time - last_time)
        last_time = actual_time

        processedFrame = detector.processImage(frame, countLine)

        cv2.putText(processedFrame, f'FPS: {processing_fps:.1f}', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f'Total cars: {detector.counter}', (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Video', processedFrame)

        if cv2.waitKey(25) & 0xFF == ord('q'): # Press Q to exit
            break
    
    cap.release()
    cv2.destroyAllWindows()   

def main():
    play_video(VIDEO_PATH)

if __name__=='__main__':
    main()