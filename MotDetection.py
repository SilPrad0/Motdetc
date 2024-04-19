import cv2
import os
import time

# Definir la ruta de la carpeta de destino para los videos
carpeta_destino = r"C:\Users\Silvina\Practik\Deteccion mov\grabaciones"

# Verificar si la carpeta de destino existe, si no, crearla
if not os.path.exists(carpeta_destino):
    os.makedirs(carpeta_destino)

# Capturar video
cap = cv2.VideoCapture(0)  # Usar 0 para la cámara predeterminada

# Inicializar el detector de movimiento
detector = cv2.createBackgroundSubtractorMOG2()

# Inicializar el estado de grabación
grabando = False
grabador = None
tiempo_inicio_grabacion = None
duracion_minima_grabacion = 3  # Duración mínima de la grabación en segundos

while True:
    # Capturar frame
    ret, frame = cap.read()

    if not ret:
        print("Error al capturar el frame")
        break

    # Aplicar el detector de movimiento al frame
    mask = detector.apply(frame)

    # Filtrar el ruido y suavizar la máscara
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    mask = cv2.dilate(mask, None, iterations=2)

    # Encontrar contornos de la máscara de movimiento
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Determinar si hay movimiento
    movimiento_detectado = any(cv2.contourArea(contour) > 500 for contour in contours)

    # Iniciar o detener la grabación según el estado de movimiento
    if movimiento_detectado:
        if not grabando:
            print("Iniciando grabación")
            nombre_video = time.strftime("%Y%m%d-%H%M%S") + ".avi"
            ruta_video = os.path.join(carpeta_destino, nombre_video)
            grabador = cv2.VideoWriter(ruta_video, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 20.0, (frame.shape[1], frame.shape[0]))
            grabando = True
            tiempo_inicio_grabacion = time.time()  # Iniciar temporizador de grabación
    else:
        if grabando:
            # Verificar si se ha alcanzado la duración mínima de la grabación
            if time.time() - tiempo_inicio_grabacion >= duracion_minima_grabacion:
                print("Deteniendo grabación")
                grabador.release()
                grabando = False

    # Grabar frame si se está grabando
    if grabando:
        grabador.write(frame)

    # Dibujar rectángulos movimiento
    if movimiento_detectado:
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 155, 0), 2)

    # Mostrar el frame con los rectángulos dibujados
    cv2.imshow('frame', frame)

    # Salir del bucle si se presiona 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar la cámara y cerrar las ventanas
cap.release()
cv2.destroyAllWindows()
