import cv2
from ultralytics import YOLO
import time
import numpy as np
from collections import defaultdict

# VIDEO_PATH = './Files/car passing by.mp4'
# VIDEO_PATH = './Files/CarsPassingBy2.mp4'
VIDEO_PATH = './Files/CarsPassingBy3.mp4'


def reproduzir_video(video_Path):
    cap = cv2.VideoCapture(video_Path)
    modelo = YOLO('yolov8n.pt')



    if not cap.isOpened():
        print("Erro ao abrir o vídeo")
        return
    
    fps_original = cap.get(cv2.CAP_PROP_FPS)
    tempo_anterior = 0
    fps_processamento = 0



    while cap.isOpened():
        ret, frame = cap.read()
        
        if not ret:
            print("Erro na leitura do video")
            break

        tempo_atual = time.time()
        if tempo_anterior != 0:
            fps_processamento = 1 / (tempo_atual - tempo_anterior)
        tempo_anterior = tempo_atual

        resultados = modelo(frame, conf=0.25) 

        frame_anotado = resultados[0].plot()

        cv2.putText(frame_anotado, f'FPS: {fps_processamento:.1f}', 
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        for r in resultados:
            for box in r.boxes:
                classe = modelo.names[int(box.cls)]
                conf = float(box.conf)
                print(f'Detectado: {classe} (Confiança: {conf:.2f})')
        
        cv2.imshow('Video', frame_anotado)
        
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

# Exemplo de uso
    # Substitua 'seu_video.mp4' pelo caminho do seu vídeo
    

def main():

    reproduzir_video(VIDEO_PATH)

if __name__=='__main__':
    main()