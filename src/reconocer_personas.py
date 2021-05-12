import cv2
import os
import json
import datetime
import imutils


class ReconocerPersonas:
    DIR_NAME = os.path.dirname(os.path.realpath(__file__))
    DATASET_ROSTROS = DIR_NAME + os.sep + '..' + os.sep + 'data'
    LISTA_PERSONAS = os.listdir(DATASET_ROSTROS)
    # Detector de rostros
    CLASIFICADOR = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    # Modelo entrenado con las capturas de rostros de las personas indicadas para su reconocimiento
    MODELO_RECONOCIMIENTO = DIR_NAME + os.sep + '..' + os.sep + 'modelo_reco_entrenado.xml'
    # Archivo de "base de datos" en formato JSON para almacenar las marcaciones
    BD_MARCACIONES = 'marcaciones.json'

    def __init__(self):
        # Crear el archivo de "base de datos" de marcaciones
        self.iniciar_bd_marcacion()

    def desde_camara(self):
        fuente = cv2.VideoCapture(0)
        self.__reconocer(fuente)

    def desde_video(self, video):
        fuente = cv2.VideoCapture(self.DIR_NAME + os.sep + '..' + os.sep + 'videos' + os.sep + video)
        self.__reconocer(fuente)

    def iniciar_bd_marcacion(self):
        personas = {}
        if not os.path.exists(self.BD_MARCACIONES):
            # Si no existe el archivo, creamos un archivo con el registro de cada persona
            for n in self.LISTA_PERSONAS:
                personas[n] = []
            with open(self.BD_MARCACIONES, 'w') as outfile:
                # Almacenamos el archivo en formato JSON
                json.dump(personas, outfile)

    def marcacion(self, nombre, lista_marcaciones):
        if not lista_marcaciones[nombre]:
            # Si no se ha registrado una marcación de la persona detectada
            hora_marcacion = datetime.datetime.now()
            print('Marcación de {}, registrada a las : {}'.format(nombre, hora_marcacion))
            # Registramos la fecha y hora de la marcación
            lista_marcaciones[nombre].append(str(hora_marcacion))
            with open(self.BD_MARCACIONES, 'w') as outfile:
                # Guardamos el archivo en formato JSON
                json.dump(lista_marcaciones, outfile)
        else:
            print('{} ya tiene una mercación en el día!'.format(nombre))

        return lista_marcaciones

    def __reconocer(self, fuente):
        # Leemos el modelo entrenado para reconocer a las personas
        modelo = cv2.face.EigenFaceRecognizer_create()
        modelo.read(self.MODELO_RECONOCIMIENTO)
        # Leemos el archivo de "base de datos" de marcaciones
        with open('marcaciones.json') as archivo_marcaciones:
            marcaciones = json.load(archivo_marcaciones)

        while True:
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
                    cv2.putText(frame_video, '{}'.format(self.LISTA_PERSONAS[resultado[0]]), (x, y - 25), 2, 1.5, (0, 255, 0), 1,
                                cv2.LINE_AA)
                    cv2.rectangle(frame_video, (x, y), (x + w, y + h), (0, 255, 0), 4)
                else:
                    cv2.putText(frame_video, 'Desconocido', (x, y - 20), 2, 0.8, (0, 0, 255), 1, cv2.LINE_AA)
                    cv2.rectangle(frame_video, (x, y), (x + w, y + h), (0, 0, 255), 2)

            # reducimos el tamaño del fotograma
            frame_video_reducido = imutils.resize(frame_video, width=920)
            # Mostramos los fotogramas que estamos procesando
            cv2.imshow("Marcacion Biometrica", frame_video_reducido)
            tecla = cv2.waitKey(1)
            if tecla == 27:
                # Si presionamos la tecla ESC o si llegamos a capturar mas de 300 rostros, terminamos el proceso
                break

        # Liberamos la fuente (video o camara) y eliminamos todas las ventanas
        fuente.release()
        cv2.destroyAllWindows()
