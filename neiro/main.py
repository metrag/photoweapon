import cv2
from ultralytics import YOLO

# Загрузка модели YOLOv8 (nano-версия для лучшей производительности)
model = YOLO('yolov8n.pt')  # Автоматически скачает модель при первом запуске

# Инициализация веб-камеры
cap = cv2.VideoCapture(0)  # 0 - индекс камеры по умолчанию

# Проверка подключения камеры
if not cap.isOpened():
    print("Ошибка: Не удалось подключиться к камере!")
    exit()

while True:
    # Чтение кадра
    ret, frame = cap.read()

    height, width = frame.shape[:2]

    if not ret:
        print("Ошибка: Не удалось получить кадр!")
        break

    # Детекция объектов
    results = model(frame, verbose=False)  # Убрать вывод в консоль
    
    # Фильтрация только людей (класс 0 в COCO)
    people = []
    for result in results:
        boxes = result.boxes
        for box in boxes:
            cls = int(box.cls[0])
            if cls == 0:  # 0 - класс 'person' в COCO
                people.append(box)

    # Преобразование кадра в цветовое пространство HSV
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Определение диапазона зелёного цвета в HSV
    lower_green = (56, 95, 101)  # Нижняя граница зелёного цвета
    upper_green = (80, 255, 255)  # Верхняя граница зелёного цвета

    # Создание маски для зелёного цвета
    mask = cv2.inRange(hsv_frame, lower_green, upper_green)

    # Поиск контуров зелёных областей
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Визуализация результатов
    hit = False  # Флаг для отслеживания попадания лазера
    for person in people:
        # Получение координат прямоугольника
        x1, y1, x2, y2 = map(int, person.xyxy[0].tolist())
        
        # Вырезаем область прямоугольника из маски
        roi_mask = mask[y1:y2, x1:x2]
        
        # Проверяем, есть ли зелёные пиксели в области прямоугольника
        if roi_mask.any():  # Если есть хотя бы один зелёный пиксель
            hit = True
            # Отрисовка прямоугольника красным цветом
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        else:
            # Отрисовка прямоугольника синим цветом
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

    # Отрисовка контуров зелёных областей на исходном кадре
    for contour in contours:
        # Отфильтровываем маленькие контуры (шум)
        if cv2.contourArea(contour) > 100:  # Минимальная площадь контура
            # Получаем ограничивающий прямоугольник вокруг контура
            x, y, w, h = cv2.boundingRect(contour)
            # Рисуем прямоугольник вокруг зелёной области
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Рисование прицела (чёрного крестика) в центре кадра
    center_x = width // 2
    center_y = height // 2
    cross_size = 10  # Размер "крыльев" крестика
    thickness = 2    # Толщина линий

    # Горизонтальная линия
    cv2.line(frame, (center_x - cross_size, center_y), (center_x + cross_size, center_y), (0, 0, 0), thickness)
    # Вертикальная линия
    cv2.line(frame, (center_x, center_y - cross_size), (center_x, center_y + cross_size), (0, 0, 0), thickness)

    # Вывод сообщения о попадании
    if hit:
        cv2.putText(frame, "Hit", (width // 2 - 50, height // 2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Отображение кадра
    cv2.imshow('People Detection', frame)
    
    # Выход по нажатию 'q'
    if cv2.waitKey(1) == ord('q'):
        break

# Очистка ресурсов
cap.release()
cv2.destroyAllWindows()