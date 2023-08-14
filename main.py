import cv2
import torch

# Carregar modelo YOLOv5
model = torch.hub.load("ultralytics/yolov5", "yolov5s")

# Inicializar a webcam
cap = cv2.VideoCapture(0)  # Use 0 para webcam interna, 1 para webcam externa

while True:
    # Capturar o vídeo da webcam
    ret, frame = cap.read()
    if not ret:
        break

    # Converter a imagem para o formato correto (BGR -> RGB)
    img_rgb = frame[:, :, [2, 1, 0]]

    # Realizar a detecção de objetos usando YOLOv5
    results = model(img_rgb)

    # Desenhar bounding boxes apenas para detecções de "person"
    for detection in results.pred[0]:
        class_id = int(detection[5])
        class_name = model.names[class_id]

        x1, y1, x2, y2 = map(int, detection[:4])
        color = (0, 255, 0)  # Cor do bounding box (verde)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            frame,
            f"{class_name}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
        )

    # Exibir o resultado
    cv2.imshow("Object Detection", frame)

    if cv2.waitKey(1) == 27:  # Pressione ESC para sair
        break

cap.release()
cv2.destroyAllWindows()
