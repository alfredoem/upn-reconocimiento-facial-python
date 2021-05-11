import cv2
import os
import json
import datetime


class ReconocerPersonas:
    DIR_NAME = os.path.dirname(os.path.realpath(__file__))
    DATASET_ROSTROS = '{}/../data'.format(DIR_NAME)
    LISTA_PERSONAS = os.listdir(DATASET_ROSTROS)
    CLASIFICADOR = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    MODELO_RECONOCIMIENTO = '{}/../{}'.format(DIR_NAME, 'modelo_reco_entrenado.xml')
    BD_MARCACIONES = 'marcaciones.json'

    def __init__(self):
        self.iniciar_marcacion()

    def reducir_cuadro(self, cuadro, porcentaje):
        ancho = int(cuadro.shape[1] * porcentaje / 100)
        alto = int(cuadro.shape[0] * porcentaje / 100)
        return cv2.resize(cuadro, (ancho, alto), interpolation=cv2.INTER_AREA)

    def desde_camara(self):
        fuente = cv2.VideoCapture(0)
        self.__reconocer(fuente)

    def desde_video(self, video):
        fuente = cv2.VideoCapture('{}/../videos/{}'.format(self.DIR_NAME, video))
        self.__reconocer(fuente)

    def iniciar_marcacion(self):
        personas = {}
        if not os.path.exists(self.BD_MARCACIONES):
            for n in self.LISTA_PERSONAS:
                personas[n] = []
            with open(self.BD_MARCACIONES, 'w') as outfile:
                json.dump(personas, outfile)

    def marcacion(self, nombre, lista_marcaciones):
        if not lista_marcaciones[nombre]:
            hora_marcacion = datetime.datetime.now()
            print('Marcación de {}, registrada a las : {}'.format(nombre, hora_marcacion))
            lista_marcaciones[nombre].append(str(hora_marcacion))
            with open(self.BD_MARCACIONES, 'w') as outfile:
                json.dump(lista_marcaciones, outfile)
        else:
            print('{} ya tiene una mercación en el día!'.format(nombre))

        return lista_marcaciones

    def __reconocer(self, fuente):
        modelo = cv2.face.EigenFaceRecognizer_create()
        modelo.read(self.MODELO_RECONOCIMIENTO)
        with open('marcaciones.json') as archivo_marcaciones:
            marcaciones = json.load(archivo_marcaciones)

        while True:
            print(marcaciones)
            ret, frame_video = fuente.read()
            if not ret: break
            rostro_gris = cv2.cvtColor(frame_video, cv2.COLOR_BGR2GRAY)
            copia_frame_video = rostro_gris.copy()
            rostros = self.CLASIFICADOR.detectMultiScale(rostro_gris, 1.3, 5)

            for (x, y, w, h) in rostros:
                rostro = copia_frame_video[y:y + h, x:x + w]
                rostro = cv2.resize(rostro, (150, 150), interpolation=cv2.INTER_CUBIC)
                resultado = modelo.predict(rostro)
                cv2.putText(frame_video, '{}'.format(resultado), (x, y - 5), 1, 1.3, (255, 255, 0), 1, cv2.LINE_AA)

                if resultado[1] < 2300:
                    marcaciones = self.marcacion(self.LISTA_PERSONAS[resultado[0]], marcaciones)
                    cv2.putText(frame_video, '{}'.format(self.LISTA_PERSONAS[resultado[0]]), (x, y - 25), 2, 1.1, (0, 255, 0), 1,
                                cv2.LINE_AA)
                    cv2.rectangle(frame_video, (x, y), (x + w, y + h), (0, 255, 0), 2)
                else:
                    cv2.putText(frame_video, 'Desconocido', (x, y - 20), 2, 0.8, (0, 0, 255), 1, cv2.LINE_AA)
                    cv2.rectangle(frame_video, (x, y), (x + w, y + h), (0, 0, 255), 2)

            frame_video_reducido = self.reducir_cuadro(frame_video, 75)
            cv2.imshow("Marcacion Biometrica", frame_video_reducido)
            tecla = cv2.waitKey(1)
            if tecla == 27:
                break

        fuente.release()
        cv2.destroyAllWindows()
