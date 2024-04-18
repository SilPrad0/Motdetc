import cv2

# Capturar video de la cámara (cambiar el índice según sea necesario)
cap = cv2.VideoCapture(0)  # Usar 0 para la cámara predeterminada

# Inicializar el detector de movimiento
detector = cv2.createBackgroundSubtractorMOG2()

# Inicializar el estado de grabación
grabando = False
grabador = None

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
    movimiento_detectado = False
    for contour in contours:
        if cv2.contourArea(contour) > 500:  # Cambiar el umbral según sea necesario
            movimiento_detectado = True
            break

    # Dibujar rectángulos alrededor de los contornos solo si hay movimiento
    if movimiento_detectado:
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            # Iniciar grabación si no se está grabando
            if not grabando:
                print("Iniciando grabación")
                grabador = cv2.VideoWriter('grabacion.avc1', cv2.VideoWriter_fourcc(*'avc1'), 20.0, (frame.shape[1], frame.shape[0]))
                grabando = True

    else:
        # Detener la grabación si se está grabando
        if grabando:
            print("Deteniendo grabación")
            grabador.release()
            grabando = False

    # Mostrar el frame con los rectángulos dibujados
    cv2.imshow('frame', frame)

    # Salir del bucle si se presiona 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar la cámara y cerrar las ventanas
cap.release()
cv2.destroyAllWindows()
