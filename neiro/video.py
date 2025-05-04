import cv2
import numpy as np
import requests
from ultralytics import YOLO
import time

# Загрузка модели YOLOv8
model = YOLO('yolov8n.pt')  # или yolov8s.pt для большей точности

# URL получения кадра
url = "http://172.20.10.14/capture"

while True:
    try:
        # Получаем кадр с камеры
        response = requests.get(url, timeout=5)
        img_array = np.asarray(bytearray(response.content), dtype=np.uint8)
        frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        if frame is None:
            print("Ошибка декодирования кадра")
            continue

        # Детекция объектов
        results = model(frame, verbose=False)

        # Фильтрация только людей (класс 0 в COCO)
        people = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls = int(box.cls[0])
                if cls == 0:  # класс 'person'
                    people.append(box.xyxy[0].tolist())

        # Преобразование кадра в цветовое пространство HSV
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Диапазон зелёного цвета
        lower_green = (56, 95, 101)
        upper_green = (80, 255, 255)
        mask = cv2.inRange(hsv_frame, lower_green, upper_green)

        hit = False  # флаг попадания лазера

        # Проверка пересечения человека и зеленого цвета
        for person in people:
            x1, y1, x2, y2 = map(int, person)
            roi_mask = mask[y1:y2, x1:x2]
            if roi_mask.any():
                hit = True
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            else:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

        # Поиск контуров зелёных областей
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if cv2.contourArea(contour) > 100:
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Прицел
        height, width = frame.shape[:2]
        center_x, center_y = width // 2, height // 2
        cross_size = 10
        thickness = 2
        cv2.line(frame, (center_x - cross_size, center_y), (center_x + cross_size, center_y), (0, 0, 0), thickness)
        cv2.line(frame, (center_x, center_y - cross_size), (center_x, center_y + cross_size), (0, 0, 0), thickness)

        # Вывод сообщения о попадании
        if hit:
            cv2.putText(frame, "Hit", (width // 2 - 50, height // 2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Отображение кадра
        cv2.imshow('ESP32 Camera Detection', frame)

        # Выход по нажатию Q
        if cv2.waitKey(1) == ord('q'):
            break

        # Задержка между кадрами
        time.sleep(0.05)

    except Exception as e:
        print("Ошибка:", e)
        time.sleep(1)

cv2.destroyAllWindows()