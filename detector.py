from ultralytics import YOLO
from collections import defaultdict
import numpy as np
import cv2

def getRandomColor(object_id):
    np.random.seed(object_id)
    randomColor = tuple(map(int, np.random.randint(50, 255, 3)))
    return randomColor

class Detector:
    def __init__(self, model_path, conf_threshold, trail_length = 20):
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.colors = {}
        self.lastId = 0
        self.history_positions = defaultdict(list)
        self.trail_length = trail_length

    def processImage(self,frame):
        results = self.model(frame, conf=self.conf_threshold)

        detections = self.traking_updater(results[0].boxes)

        for object_id, box, classe, conf in detections:
            x1, y1, x2, y2 = box
            cor = self.colors[object_id]
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), cor, 2)
            
            textWithId = f'ID:{object_id} {self.model.names[classe]} {conf:.2f}'
            cv2.putText(frame, textWithId, (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, cor, 2)
            
            # Draw trail
            positions = self.history_positions[object_id]
            for i in range(1, len(positions)):
                pt1 = tuple(map(int, positions[i-1]))
                pt2 = tuple(map(int, positions[i]))
                cv2.line(frame, pt1, pt2, cor, 2)

        processedFrame = results[0].plot()
        # For Debug purposes
        for r in results:
            for box in r.boxes:
                classe = self.model.names[int(box.cls)]
                conf = float(box.conf)
                print(f'Detected: {classe} (Confiability: {conf:.2f})')

        
        return frame
    
    def traking_updater(self, detections, max_dist=50):
        new_detections = []
        
        for box in detections:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            actual_center = ((x1 + x2) // 2, (y1 + y2) // 2)
            classe = int(box.cls)
            conf = float(box.conf)
            
            min_dist = float('inf')
            closest_id = None
            
            for object_id, positions in self.history_positions.items():
                if positions:
                    last_position = positions[-1]
                    dist = np.sqrt((actual_center[0] - last_position[0])**2 + 
                                      (actual_center[1] - last_position[1])**2)
                    
                    if dist < min_dist and dist < max_dist:
                        min_dist = dist
                        closest_id = object_id
            
            if closest_id is None:
                closest_id = self.lastId
                self.lastId += 1
                self.colors[closest_id] = getRandomColor(closest_id)
            
            self.history_positions[closest_id].append(actual_center)
            if len(self.history_positions[closest_id]) > self.trail_length:
                self.history_positions[closest_id].pop(0)
            
            new_detections.append((closest_id, (x1, y1, x2, y2), classe, conf))
        
        return new_detections

