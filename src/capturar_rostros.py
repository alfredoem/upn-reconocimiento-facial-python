import cv2
import os
import imutils


class CapturarRostros:
    DIR_NAME = os.path.dirname(os.path.realpath(__file__))
    DATASET_ROSTROS = DIR_NAME + os.sep + '..' + os.sep + 'data'
    # Detector de rostros
    CLASIFICADOR = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    LIMITE_CAPTURAS = 300

    def __init__(self, nombre_persona):
        self.nombre_persona = nombre_persona
        self.folder_persona = self.DATASET_ROSTROS + os.sep + nombre_persona
        self.contador_imagenes = 0
        self.crear_folder()

    def desde_camara(self):
        fuente = cv2.VideoCapture(0)
        self.capturar(fuente)

    def desde_video(self, video):
        fuente = cv2.VideoCapture(self.DIR_NAME + os.sep + '..' + os.sep + 'videos' + os.sep + video)
        self.capturar(fuente)

    def capturar(self, fuente):
        while True:
            # Captura los fotogramas de la fuente (video o camara)
            ret, frame_video = fuente.read()
            if not ret:
                # Cuando no hay mas fotogramas, terminamos el proceso
                break
            # reducimos el tamaño del fotograma
            frame_video = imutils.resize(frame_video, width=640)
            # convierte el fotograma a escala de grises
            frame_video_gris = cv2.cvtColor(frame_video, cv2.COLOR_BGR2GRAY)
            # guardamos el fotograma original
            copia_frame_video = frame_video.copy()
            # Devuelve las coordenadas de los rostros capturados del fotograma
            rostros = self.CLASIFICADOR.detectMultiScale(frame_video_gris, 1.3, 5)
            for (x, y, w, h) in rostros:
                # Dibuja un cuadro color verde en la posición del rostro
                cv2.rectangle(frame_video, (x, y), (x + w, y + h), (0, 255, 0), 2)
                # Obtenemos el rostro del fotograma original (a color) en base a las coordenadas
                rostro = copia_frame_video[y:y + h, x:x + w]
                # Reducimos el tamaño del rostro capturado aplicando una interpolación bicubica
                rostro = cv2.resize(rostro, (150, 150), interpolation=cv2.INTER_CUBIC)
                # Guardamos la imagen del rostro capturado en la ubicación indicada
                cv2.imwrite(self.folder_persona + os.sep + 'rotro_' + str(self.contador_imagenes) + '.jpg', rostro)
                self.contador_imagenes = self.contador_imagenes + 1
            # Mostramos todos los fotogramas que estamos procesando
            cv2.imshow('Capturando Rostros', frame_video)

            tecla = cv2.waitKey(1)
            if tecla == 27 or self.contador_imagenes >= self.LIMITE_CAPTURAS:
                # Si presionamos la tecla ESC o si llegamos a capturar mas de 300 rostros, terminamos el proceso
                break
        # Liberamos la fuente (video o camara) y eliminamos todas las ventanas
        fuente.release()
        cv2.destroyAllWindows()

    def crear_folder(self):
        if not os.path.exists(self.folder_persona):
            # Creamos el directorio donde almacenaremos las imagenes de los rostros capturados
            os.makedirs(self.folder_persona)


generador = CapturarRostros('Alfredo')
# generador.desde_camara()
generador.desde_video('Alfredo.mp4')
