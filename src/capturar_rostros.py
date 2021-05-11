import cv2
import os
import imutils


class CapturarRostros:
    DIR_NAME = os.path.dirname(os.path.realpath(__file__))
    DATASET_ROSTROS = '{}/../data'.format(DIR_NAME)
    CLASIFICADOR = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    LIMITE_CAPTURAS = 300

    def __init__(self, nombre_persona):
        self.nombre_persona = nombre_persona
        self.folder_persona = "{}/{}".format(self.DATASET_ROSTROS, nombre_persona)
        self.contador_imagenes = 0
        self.crear_folder()

    def desde_camara(self):
        fuente = cv2.VideoCapture(0)
        self.capturar(fuente)

    def desde_video(self, video):
        fuente = cv2.VideoCapture('{}/../videos/{}'.format(self.DIR_NAME, video))
        self.capturar(fuente)

    def capturar(self, fuente):
        while True:
            ret, frame_video = fuente.read()
            if not ret:
                break
            frame_video = imutils.resize(frame_video, width=640)
            rostro_gris = cv2.cvtColor(frame_video, cv2.COLOR_BGR2GRAY)
            copia_frame_video = frame_video.copy()
            rostros = self.CLASIFICADOR.detectMultiScale(rostro_gris, 1.3, 5)

            for (x, y, w, h) in rostros:
                cv2.rectangle(frame_video, (x, y), (x + w, y + h), (0, 255, 0), 2)
                rostro = copia_frame_video[y:y + h, x:x + w]
                rostro = cv2.resize(rostro, (150, 150), interpolation=cv2.INTER_CUBIC)
                cv2.imwrite(self.folder_persona + '/rotro_{}.jpg'.format(self.contador_imagenes), rostro)
                self.contador_imagenes = self.contador_imagenes + 1
            cv2.imshow('Capturando Rostros', frame_video)

            tecla = cv2.waitKey(1)
            if tecla == 27 or self.contador_imagenes >= self.LIMITE_CAPTURAS:
                break

        fuente.release()
        cv2.destroyAllWindows()

    def crear_folder(self):
        if not os.path.exists(self.folder_persona):
            os.makedirs(self.folder_persona)


generador = CapturarRostros('Alfredo')
# generador.desde_video('Gisell.mp4')
generador.desde_camara()