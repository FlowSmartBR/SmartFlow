import torch
import cv2

# Carregar o modelo YOLOv5 
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # yolov5s é o modelo pequeno e rápido

# Definir a função para detectar os objetos
def detect_objects_in_video(video_path):
    # Inicializar captura de vídeo a partir de um arquivo de vídeo ou nossa webcam
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Erro ao abrir o vídeo: {video_path}")
        return

    # Definir as classes que desejamos detectar: pessoas, bicicletas, carros e caminhões
    classes_to_detect = [0, 1, 2, 7]  # Pessoa, Bicicleta, Carro, Caminhão
    class_names = {0: 'Pessoa', 1: 'Bicicleta', 2: 'Carro', 7: 'Caminhão'}

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Fim do vídeo ou falha na captura do frame!")
            break

        # Fazer a detecção com o modelo YOLO
        results = model(frame)

        # Obter as detecções 
        detections = results.xyxy[0]

        # Filtrar as classes 
        for *xyxy, conf, cls in detections:
            if int(cls) in classes_to_detect:
                label = class_names[int(cls)]
                cv2.rectangle(frame, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0, 255, 0), 2)
                cv2.putText(frame, f'{label} {conf:.2f}', (int(xyxy[0]), int(xyxy[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        # Exibir o resultado com as detecções
        cv2.imshow('Detecção de Objetos: Pessoas, Veículos, Bicicletas e Caminhões', frame)

        # Sair com a tecla 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    
    cap.release()
    cv2.destroyAllWindows()


video_path = 'bike.mp4'  
detect_objects_in_video(video_path)
