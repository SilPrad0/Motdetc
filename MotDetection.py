import cv2
import os
import time
import logging


def setup_logger():
    logging.basicConfig(filename='detector_movimiento.log', level=logging.INFO,
                        format='%(asctime)s:%(levelname)s:%(message)s')


def verificar_o_crear_carpeta(carpeta):
    if not os.path.exists(carpeta):
        os.makedirs(carpeta)


def iniciar_captura_video():
    return cv2.VideoCapture(0)  # 0 para camara predeterminada


def iniciar_detector_movimiento():
    return cv2.createBackgroundSubtractorMOG2()


def iniciar_grabador(ruta_video, frame_shape):
    return cv2.VideoWriter(ruta_video, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 20.0,
                           (frame_shape[1], frame_shape[0]))


def detectar_movimiento(frame, detector):
    mask = detector.apply(frame)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    mask = cv2.dilate(mask, None, iterations=2)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return any(cv2.contourArea(contour) > 500 for contour in contours), contours


def dibujar_rectangulos(frame, contours):
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 255), 2)


def main():
    setup_logger()

    carpeta_destino = r"C:\Users\Silvina\Practik\Deteccion mov\grabaciones"
    verificar_o_crear_carpeta(carpeta_destino)

    cap = iniciar_captura_video()
    detector = iniciar_detector_movimiento()

    grabando = False
    grabador = None
    tiempo_inicio_grabacion = None
    duracion_minima_grabacion = 3  # Duracion mínima de la grabacion en segundos

    while True:
        ret, frame = cap.read()

        if not ret:
            logging.error("Error al capturar el frame")
            break

        movimiento_detectado, contours = detectar_movimiento(frame, detector)

        if movimiento_detectado:
            if not grabando:
                logging.info("Iniciando grabacion")
                nombre_video = time.strftime("%Y%m%d-%H%M%S") + ".avi"
                ruta_video = os.path.join(carpeta_destino, nombre_video)
                grabador = iniciar_grabador(ruta_video, frame.shape)
                grabando = True
                tiempo_inicio_grabacion = time.time()  # Iniciar temporizador
        else:
            if grabando:
                if time.time() - tiempo_inicio_grabacion >= duracion_minima_grabacion:
                    logging.info("Deteniendo grabacion")
                    grabador.release()
                    grabando = False

        if grabando:
            grabador.write(frame)

        if movimiento_detectado:
            dibujar_rectangulos(frame, contours)

        cv2.imshow('frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    logging.info("Closed app")


if __name__ == "__main__":
    main()
