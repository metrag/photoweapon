import cv2

# Функция для создания ползунков
def nothing(x):
    pass

# Инициализация веб-камеры
cap = cv2.VideoCapture(0)

# Создание окна для настройки HSV
cv2.namedWindow("Trackbars")
cv2.createTrackbar("LH", "Trackbars", 40, 179, nothing)  # Нижний Hue
cv2.createTrackbar("LS", "Trackbars", 50, 255, nothing)  # Нижняя Saturation
cv2.createTrackbar("LV", "Trackbars", 50, 255, nothing)  # Нижняя Value
cv2.createTrackbar("UH", "Trackbars", 80, 179, nothing)  # Верхний Hue
cv2.createTrackbar("US", "Trackbars", 255, 255, nothing)  # Верхняя Saturation
cv2.createTrackbar("UV", "Trackbars", 255, 255, nothing)  # Верхняя Value

while True:
    # Чтение кадра
    ret, frame = cap.read()
    if not ret:
        break

    # Преобразование кадра в цветовое пространство HSV
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Получение значений ползунков
    lh = cv2.getTrackbarPos("LH", "Trackbars")
    ls = cv2.getTrackbarPos("LS", "Trackbars")
    lv = cv2.getTrackbarPos("LV", "Trackbars")
    uh = cv2.getTrackbarPos("UH", "Trackbars")
    us = cv2.getTrackbarPos("US", "Trackbars")
    uv = cv2.getTrackbarPos("UV", "Trackbars")

    # Определение диапазона цвета
    lower_green = (lh, ls, lv)
    upper_green = (uh, us, uv)

    # Создание маски для зелёного цвета
    mask = cv2.inRange(hsv_frame, lower_green, upper_green)

    # Поиск контуров зелёных областей
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Отрисовка контуров
    for contour in contours:
        if cv2.contourArea(contour) > 100:  # Минимальная площадь контура
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Отображение маски и исходного кадра
    cv2.imshow("Mask", mask)
    cv2.imshow("Frame", frame)

    # Выход по нажатию 'q'
    if cv2.waitKey(1) == ord('q'):
        break

# Очистка ресурсов
cap.release()
cv2.destroyAllWindows()